import datetime
import math
import os
import random
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
from k_model import ModelConfig, Transformer




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
        flash_attn=active_config["FLASH_ATTN"],
        multiple_of=active_config["MULTIPLE_OF"],
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


def build_model_config(vocab_size, args):
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
        flash_attn=args.flash_attn,
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
                _, loss = model(x, labels, attention_mask=attention_mask)
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

    if total_batches == 0:
        return None
    return total_loss / total_batches


def save_checkpoint(model, optimizer, tokenizer, epoch, avg_loss, save_path, global_step, scaler, args, log_file=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "global_step": global_step,
        "train_stage": args.train_stage,
        "data_path": args.data_path,
        "model_config": {
            "vocab_size": len(tokenizer),
            "max_seq_len": args.max_seq_len,
            "dim": args.dim,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "n_kv_heads": args.n_kv_heads,
            "hidden_dim": None,
            "multiple_of": args.multiple_of,
            "norm_eps": args.norm_eps,
            "dropout": args.dropout,
            "flash_attn": args.flash_attn,
        },
        "vocab_size": len(tokenizer),
        "max_seq_len": args.max_seq_len,
        "dim_embedding": args.dim,
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "n_layers": args.n_layers,
        "tokenizer_name": args.tokenizer_path,
        "epoch": epoch,
        "avg_loss": avg_loss,
    }
    torch.save(checkpoint, save_path)
    log_message(f"checkpoint saved: {save_path}", log_file)


def load_checkpoint_for_resume(checkpoint_path, model, optimizer, scaler, device, log_file=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
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
    assert args.dim % args.n_heads == 0, "dim must be divisible by n_heads"
    assert args.n_heads % args.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    model_config = build_model_config(vocab_size=len(tokenizer), args=args)
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
            state_dict = checkpoint.get("model_state_dict", checkpoint)
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
    total_steps = args.epochs * len(train_loader)
    avg_loss = 0.0
    stage_label = f"[{args.train_stage}]"

    for epoch in range(start_epoch, args.epochs):
        total_loss = 0.0
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
                _, loss = model(x, labels, attention_mask=attention_mask)
                loss_mask = loss_mask.view(-1)
                valid_tokens = loss_mask.sum()
                if valid_tokens.item() == 0:
                    continue
                loss = torch.sum(loss * loss_mask) / valid_tokens
                loss_back = loss / args.accumulation_steps

            scaler.scale(loss_back).backward()

            is_update_step = (step + 1) % args.accumulation_steps == 0
            is_last_step = (step + 1) == len(train_loader)

            if is_update_step or is_last_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

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

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                step_save_path = os.path.join(
                    args.checkpoint_dir,
                    f"{args.checkpoint_prefix}_step_{global_step}.pt",
                )
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    epoch=current_epoch,
                    avg_loss=loss.item(),
                    save_path=step_save_path,
                    global_step=global_step,
                    scaler=scaler,
                    args=args,
                    log_file=log_file,
                )

        avg_loss = total_loss / len(train_loader)
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
                tokenizer=tokenizer,
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
        tokenizer=tokenizer,
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
