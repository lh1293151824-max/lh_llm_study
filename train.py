import datetime
import math
import os
import random
import warnings
from types import SimpleNamespace

import config as default_config

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import torch
from torch.amp import autocast, GradScaler

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import PretrainDataset, SFTDataset
from k_model import (
    ModelConfig,
    Transformer,
    infer_output_head_type_from_state_dict,
    validate_state_dict_output_head_type,
)




def build_args_from_config(stage=None, mode=None):
    stage = stage or default_config.TRAIN_STAGE
    active_config = default_config.get_active_config(stage=stage, mode=mode)

    return SimpleNamespace(
        train_stage=stage,
        data_path=active_config["DATA_PATH"],
        tokenizer_path=default_config.TOKENIZER_NAME,
        seed=active_config["SEED"],
        epochs=active_config["EPOCHS"],
        batch_size=active_config["BATCH_SIZE"],
        max_seq_len=active_config["MAX_SEQ_LEN"],
        learning_rate=active_config["LEARNING_RATE"],
        accumulation_steps=active_config["ACCUMULATION_STEPS"],
        warmup_iters=active_config["WARMUP_ITERS"],
        grad_clip=active_config["GRAD_CLIP"],
        use_amp=active_config["USE_AMP"],
        num_workers=active_config["NUM_WORKERS"],
        dim=active_config["DIM_EMBEDDING"],
        n_layers=active_config["N_LAYERS"],
        n_heads=active_config["N_HEADS"],
        n_kv_heads=active_config["N_KV_HEADS"],
        norm_eps=active_config["NORM_EPS"],
        dropout=active_config["DROPOUT"],
        multiple_of=active_config["MULTIPLE_OF"],
        output_head_type=default_config.OUTPUT_HEAD_TYPE,
        operator_rank=default_config.OPERATOR_RANK,
        operator_alpha=default_config.OPERATOR_ALPHA,
        operator_scale_warning_ratio=(
            default_config.OPERATOR_SCALE_WARNING_RATIO
        ),
        checkpoint_dir=active_config["CHECKPOINT_DIR"],
        checkpoint_prefix=active_config["CHECKPOINT_PREFIX"],
        checkpoint_path=active_config["CHECKPOINT_PATH"],
        log_dir=default_config.LOG_DIR,
        text_log_dir=default_config.TEXT_LOG_DIR,
        log_interval=active_config["LOG_INTERVAL"],
        save_every_steps=active_config["SAVE_EVERY_STEPS"],
        save_every_epochs=active_config["SAVE_EVERY_EPOCHS"],
        val_ratio=default_config.VAL_RATIO,
        val_interval=default_config.VAL_INTERVAL,
        resume=default_config.RESUME,
        resume_checkpoint_path=default_config.RESUME_CHECKPOINT_PATH,
        sft_init_checkpoint_path=active_config["SFT_INIT_CHECKPOINT_PATH"],
    )


def ensure_stage_checkpoint_name(filename, train_stage):
    if train_stage not in default_config.STAGE_CONFIG_TABLE:
        raise ValueError(f"Unknown train_stage: {train_stage}")

    stem, ext = os.path.splitext(filename)
    stage_token = f"_{train_stage}"

    if stage_token in stem:
        return filename

    return f"{stem}{stage_token}{ext}"


def log_message(message, log_file=None):
    message = str(message)
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_lr(step, total_steps, args):
    min_lr = args.learning_rate * 0.1

    if args.warmup_iters > 0 and step < args.warmup_iters:
        return args.learning_rate * step / args.warmup_iters

    if step > total_steps:
        return min_lr

    if total_steps == args.warmup_iters:
        return min_lr

    decay_ratio = (step - args.warmup_iters) / (total_steps - args.warmup_iters)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)


def build_model_config(vocab_size, args, pad_token_id=None):
    return ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        hidden_dim=None,
        multiple_of=args.multiple_of,
        norm_eps=args.norm_eps,
        dropout=args.dropout,
        pad_token_id=pad_token_id,
        output_head_type=args.output_head_type,
        operator_rank=args.operator_rank,
        operator_alpha=args.operator_alpha,
        operator_scale_warning_ratio=args.operator_scale_warning_ratio,
    )


def split_train_val_dataset(dataset, val_ratio, seed, train_stage):
    if not 0 <= val_ratio < 1:
        raise ValueError(f"VAL_RATIO must be in [0, 1), got {val_ratio}")

    if train_stage != "pretrain":
        return dataset, None

    total_size = len(dataset)
    if total_size < 2 or val_ratio <= 0:
        return dataset, None

    val_size = max(1, int(total_size * val_ratio))
    val_size = min(val_size, total_size - 1)
    train_size = total_size - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def evaluate_loss(model, val_loader, device, args, stage_label, global_step):
    if val_loader is None:
        return None

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_batches = 0
    max_output_scale_ratio = None

    with torch.no_grad():
        progress = tqdm(
            val_loader,
            desc=f"{stage_label} val step={global_step}",
            leave=False,
        )
        for x, labels, loss_mask, attention_mask in progress:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            with autocast("cuda", enabled=args.use_amp and device.type == "cuda"):
                outputs = model(x, labels, attention_mask=attention_mask)
                loss = outputs["last_loss"]
                scale_ratio = outputs.get("output_scale_ratio")
                if scale_ratio is not None:
                    ratio_value = scale_ratio.item()
                    max_output_scale_ratio = (
                        ratio_value
                        if max_output_scale_ratio is None
                        else max(max_output_scale_ratio, ratio_value)
                    )
                loss_mask = loss_mask.view(-1)
                valid_tokens = loss_mask.sum()
                if valid_tokens.item() == 0:
                    continue
                loss = torch.sum(loss * loss_mask) / valid_tokens

            total_loss += loss.item()
            total_batches += 1
            progress.set_postfix(val_loss=f"{loss.item():.6f}")

    if was_training:
        model.train()

    if (
        max_output_scale_ratio is not None
        and max_output_scale_ratio
        > model.config.operator_scale_warning_ratio
    ):
        warnings.warn(
            "Linear and DeepONet logit RMS values differ by more than "
            f"{model.config.operator_scale_warning_ratio:g}x "
            f"(maximum observed ratio: {max_output_scale_ratio:.4g}x).",
            RuntimeWarning,
            stacklevel=2,
        )

    if total_batches == 0:
        return None
    return total_loss / total_batches


def _checkpoint_model_config(model):
    config = model.config
    return {
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
        "dim": config.dim,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_kv_heads": config.n_kv_heads,
        "hidden_dim": config.hidden_dim,
        "multiple_of": config.multiple_of,
        "norm_eps": config.norm_eps,
        "dropout": config.dropout,
        "pad_token_id": config.pad_token_id,
        "output_head_type": config.output_head_type,
        "operator_rank": config.operator_rank,
        "operator_alpha": config.operator_alpha,
        "operator_scale_warning_ratio": (
            config.operator_scale_warning_ratio
        ),
    }


def save_checkpoint(
    model,
    optimizer,
    epoch,
    avg_loss,
    save_path,
    global_step,
    scaler,
    args,
    log_file=None,
):
    checkpoint = {
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "global_step": global_step,
        "train_stage": args.train_stage,
        "model_config": _checkpoint_model_config(model),
        "tokenizer_name": args.tokenizer_path,
        "epoch": epoch,
        "avg_loss": avg_loss,
    }
    torch.save(checkpoint, save_path)
    log_message(f"checkpoint saved: {save_path}", log_file)


def _checkpoint_state_dict_and_mode(checkpoint):
    state_dict = checkpoint.get("model_state_dict")
    if isinstance(state_dict, dict):
        model_config = checkpoint.get("model_config")
        model_config = model_config if isinstance(model_config, dict) else {}
        return state_dict, model_config.get("output_head_type", "linear")

    if checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return checkpoint, infer_output_head_type_from_state_dict(checkpoint)

    raise KeyError('checkpoint missing a valid "model_state_dict"')


def validate_checkpoint_output_head(checkpoint, model, context):
    state_dict, checkpoint_mode = _checkpoint_state_dict_and_mode(checkpoint)
    if checkpoint_mode != model.config.output_head_type:
        raise ValueError(
            f"{context} output head mismatch: checkpoint={checkpoint_mode}, "
            f"model={model.config.output_head_type}"
        )
    validate_state_dict_output_head_type(
        state_dict,
        expected=checkpoint_mode,
        context=context,
    )
    return state_dict


def load_checkpoint_for_resume(checkpoint_path, model, optimizer, scaler, device, log_file=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = validate_checkpoint_output_head(
        checkpoint,
        model,
        context="resume checkpoint",
    )
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0) or 0

    log_message(f"resumed checkpoint: {checkpoint_path}", log_file)
    log_message(f"resume epoch: {start_epoch}", log_file)
    log_message(f"resume global_step: {global_step}", log_file)
    return start_epoch, global_step


def checkpoint_matches_stage(filename, stage):
    stem, ext = os.path.splitext(filename)
    return ext == ".pt" and f"_{stage}" in stem


def find_latest_checkpoint(checkpoint_dir, stage):
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    candidates = []
    for filename in os.listdir(checkpoint_dir):
        if not checkpoint_matches_stage(filename, stage):
            continue

        path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(path):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No {stage} .pt checkpoint found in {checkpoint_dir}")

    return max(candidates, key=os.path.getmtime)


def resolve_resume_checkpoint_path(args):
    if args.resume_checkpoint_path:
        return args.resume_checkpoint_path
    return find_latest_checkpoint(args.checkpoint_dir, args.train_stage)


def resolve_sft_init_checkpoint_path(args):
    if args.sft_init_checkpoint_path:
        return args.sft_init_checkpoint_path
    pretrain_config = default_config.get_active_config(stage="pretrain")
    return find_latest_checkpoint(pretrain_config["CHECKPOINT_DIR"], "pretrain")


def train(args):
    args.checkpoint_prefix = ensure_stage_checkpoint_name(
        args.checkpoint_prefix,
        args.train_stage,
    )
    args.checkpoint_path = ensure_stage_checkpoint_name(
        args.checkpoint_path,
        args.train_stage,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.text_log_dir, exist_ok=True)

    log_file = os.path.join(args.text_log_dir, "train.log")
    writer = SummaryWriter(log_dir=args.log_dir)

    log_message("=" * 80, log_file)
    log_message(f"start_time: {datetime.datetime.now().isoformat(timespec='seconds')}", log_file)
    log_message(f"train_stage: {args.train_stage}", log_file)
    log_message(f"data_path: {args.data_path}", log_file)
    log_message(f"tokenizer_path: {args.tokenizer_path}", log_file)
    log_message(f"checkpoint_dir: {args.checkpoint_dir}", log_file)
    log_message(f"tensorboard_dir: {args.log_dir}", log_file)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"device: {device}", log_file)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.padding_side = "left"

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"training data not found: {args.data_path}")

    if args.train_stage == "pretrain":
        dataset = PretrainDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            max_length=args.max_seq_len + 1,
        )
    elif args.train_stage == "sft":
        dataset = SFTDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            max_length=args.max_seq_len + 1,
        )
    else:
        raise ValueError(f"Unknown train_stage: {args.train_stage}")

    train_dataset, val_dataset = split_train_val_dataset(
        dataset=dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_stage=args.train_stage,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=args.num_workers,
        )

    log_message(f"samples: {len(dataset)}", log_file)
    log_message(f"train_samples: {len(train_dataset)}", log_file)
    log_message(f"val_samples: {len(val_dataset) if val_dataset is not None else 0}", log_file)
    log_message(f"train_batches: {len(train_loader)}", log_file)
    log_message(f"val_batches: {len(val_loader) if val_loader is not None else 0}", log_file)
    log_message(f"batch_size: {args.batch_size}", log_file)
    log_message(f"max_seq_len: {args.max_seq_len}", log_file)
    log_message(f"epochs: {args.epochs}", log_file)
    log_message(f"learning_rate: {args.learning_rate}", log_file)
    log_message(f"accumulation_steps: {args.accumulation_steps}", log_file)
    log_message(f"warmup_iters: {args.warmup_iters}", log_file)
    log_message(f"val_ratio: {args.val_ratio}", log_file)
    log_message(f"val_interval: {args.val_interval}", log_file)
    log_message(f"use_amp: {args.use_amp}", log_file)
    log_message(f"output_head_type: {args.output_head_type}", log_file)
    log_message(f"operator_rank: {args.operator_rank}", log_file)
    log_message(f"operator_alpha: {args.operator_alpha}", log_file)
    log_message(
        "operator_scale_warning_ratio: "
        f"{args.operator_scale_warning_ratio}",
        log_file,
    )
    assert args.dim % args.n_heads == 0, "dim must be divisible by n_heads"
    assert args.n_heads % args.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    model_config = build_model_config(
        vocab_size=len(tokenizer),
        args=args,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = Transformer(config=model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"total_params: {total_params / 1e9:.6f} B", log_file)
    log_message(f"trainable_params: {trainable_params / 1e9:.6f} B", log_file)

    if args.train_stage == "sft" and not args.resume:
        try:
            sft_init_checkpoint_path = resolve_sft_init_checkpoint_path(args)
        except FileNotFoundError:
            sft_init_checkpoint_path = ""

        if sft_init_checkpoint_path:
            if not os.path.exists(sft_init_checkpoint_path):
                raise FileNotFoundError(
                    f"sft init checkpoint not found: {sft_init_checkpoint_path}"
                )
            checkpoint = torch.load(
                sft_init_checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            state_dict = validate_checkpoint_output_head(
                checkpoint,
                model,
                context="SFT initialization checkpoint",
            )
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            log_message(f"sft init checkpoint: {sft_init_checkpoint_path}", log_file)
            if missing_keys:
                log_message(f"sft init missing_keys: {missing_keys}", log_file)
            if unexpected_keys:
                log_message(f"sft init unexpected_keys: {unexpected_keys}", log_file)
        else:
            log_message(
                "未设置 SFT 预训练权重文件，当前将随机初始化模型并用 SFT 数据从零训练。",
                log_file,
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")

    start_epoch = 0
    global_step = 0

    if args.resume:
        resume_checkpoint_path = resolve_resume_checkpoint_path(args)
        start_epoch, global_step = load_checkpoint_for_resume(
            checkpoint_path=resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            log_file=log_file,
        )
    model.train()
    if args.accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be greater than 0")
    total_steps = args.epochs * len(train_loader)
    avg_loss = 0.0
    stage_label = f"[{args.train_stage}]"
    last_step_checkpoint = global_step

    for epoch in range(start_epoch, args.epochs):
        total_loss = 0.0
        valid_batch_count = 0
        accumulated_batches = 0
        current_epoch = epoch + 1
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(
            train_loader,
            desc=f"{stage_label} epoch={current_epoch}/{args.epochs}",
        )

        for step, (x, labels, loss_mask, attention_mask) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            lr = get_lr(global_step, total_steps, args)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            with autocast("cuda", enabled=args.use_amp and device.type == "cuda"):
                outputs = model(x, labels, attention_mask=attention_mask)
                loss = outputs["last_loss"]
                loss_mask = loss_mask.view(-1)
                valid_tokens = loss_mask.sum()
                if valid_tokens.item() == 0:
                    continue
                loss = torch.sum(loss * loss_mask) / valid_tokens

            scaler.scale(loss).backward()
            accumulated_batches += 1
            valid_batch_count += 1

            if accumulated_batches == args.accumulation_steps:
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_batches)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0

            total_loss += loss.item()
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            writer.add_scalar("train/lr", lr, global_step)
            global_step += 1
            pbar.set_postfix(
                step=global_step,
                loss=f"{loss.item():.6f}",
                lr=f"{lr:.8f}",
            )

            if args.log_interval > 0 and global_step % args.log_interval == 0:
                message = (
                    f"{stage_label} epoch={current_epoch}/{args.epochs} | "
                    f"step={global_step} | "
                    f"loss={loss.item():.6f} | "
                    f"lr={lr:.8f}"
                )
                log_message(message, log_file)

            if (
                args.train_stage == "pretrain"
                and args.val_interval > 0
                and val_loader is not None
                and global_step % args.val_interval == 0
            ):
                val_loss = evaluate_loss(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    args=args,
                    stage_label=stage_label,
                    global_step=global_step,
                )
                if val_loss is not None:
                    writer.add_scalar("val/loss", val_loss, global_step)
                    val_message = (
                        f"{stage_label} val | "
                        f"step={global_step} | "
                        f"val_loss={val_loss:.6f}"
                    )
                    tqdm.write(val_message)
                    log_message(val_message, log_file)

            if (
                args.save_every_steps > 0
                and global_step - last_step_checkpoint >= args.save_every_steps
                and accumulated_batches == 0
            ):
                step_save_path = os.path.join(
                    args.checkpoint_dir,
                    f"{args.checkpoint_prefix}_step_{global_step}.pt",
                )
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=current_epoch,
                    avg_loss=loss.item(),
                    save_path=step_save_path,
                    global_step=global_step,
                    scaler=scaler,
                    args=args,
                    log_file=log_file,
                )
                last_step_checkpoint = global_step

        if accumulated_batches > 0:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_batches)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if valid_batch_count == 0:
            raise RuntimeError(
                f"No valid training batches found in epoch {current_epoch}. "
                "Check the dataset and loss masks."
            )

        avg_loss = total_loss / valid_batch_count
        writer.add_scalar("train/epoch_avg_loss", avg_loss, current_epoch)
        epoch_message = (
            f"{stage_label} epoch={current_epoch}/{args.epochs} | "
            f"avg_loss={avg_loss:.6f}"
        )
        # progress.set_description(epoch_message)
        log_message(epoch_message, log_file)

        if current_epoch % args.save_every_epochs == 0:
            epoch_save_path = os.path.join(
                args.checkpoint_dir,
                f"{args.checkpoint_prefix}_epoch_{current_epoch}.pt",
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=current_epoch,
                avg_loss=avg_loss,
                save_path=epoch_save_path,
                global_step=global_step,
                scaler=scaler,
                args=args,
                log_file=log_file,
            )

    final_save_path = os.path.join(args.checkpoint_dir, args.checkpoint_path)
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        avg_loss=avg_loss,
        save_path=final_save_path,
        global_step=global_step,
        scaler=scaler,
        args=args,
        log_file=log_file,
    )

    writer.close()
    log_message(f"training finished. final_checkpoint: {final_save_path}", log_file)
    log_message(f"end_time: {datetime.datetime.now().isoformat(timespec='seconds')}", log_file)


if __name__ == "__main__":

    train(build_args_from_config())
