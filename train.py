import datetime
import math
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.amp import autocast, GradScaler

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

import config as default_config
from dataset import PretrainDataset, SFTDataset
from k_model import ModelConfig, Transformer




def build_args_from_config():
    return SimpleNamespace(
        train_stage=default_config.TRAIN_STAGE,
        data_path=default_config.DATA_PATH,
        tokenizer_path=default_config.TOKENIZER_NAME,
        seed=default_config.SEED,
        epochs=default_config.EPOCHS,
        batch_size=default_config.BATCH_SIZE,
        seq_len=default_config.SEQ_LEN,
        learning_rate=default_config.LEARNING_RATE,
        accumulation_steps=default_config.ACCUMULATION_STEPS,
        warmup_iters=default_config.WARMUP_ITERS,
        grad_clip=default_config.GRAD_CLIP,
        use_amp=default_config.USE_AMP,
        num_workers=default_config.NUM_WORKERS,
        dim=default_config.DIM_EMBEDDING,
        n_layers=default_config.N_LAYERS,
        n_heads=default_config.N_HEADS,
        n_kv_heads=default_config.N_KV_HEADS,
        norm_eps=default_config.NORM_EPS,
        dropout=default_config.DROPOUT,
        flash_attn=default_config.FLASH_ATTN,
        multiple_of=default_config.MULTIPLE_OF,
        checkpoint_dir=default_config.CHECKPOINT_DIR,
        checkpoint_prefix=default_config.CHECKPOINT_PREFIX,
        checkpoint_path=default_config.CHECKPOINT_PATH,
        log_dir=default_config.LOG_DIR,
        text_log_dir=default_config.TEXT_LOG_DIR,
        log_interval=default_config.LOG_INTERVAL,
        save_every_steps=default_config.SAVE_EVERY_STEPS,
        save_every_epochs=default_config.SAVE_EVERY_EPOCHS,
        generate_every_steps=default_config.GENERATE_EVERY_STEPS,
        generate_prompt=default_config.GENERATE_PROMPT,
        generate_max_new_tokens=default_config.GENERATE_MAX_NEW_TOKENS,
        generate_temperature=default_config.GENERATE_TEMPERATURE,
        generate_top_k=default_config.GENERATE_TOP_K,
        resume=default_config.RESUME,
        resume_checkpoint_path=default_config.RESUME_CHECKPOINT_PATH,
        sft_init_checkpoint_path=default_config.SFT_INIT_CHECKPOINT_PATH,
    )


def log_message(message, log_file=None):
    message = str(message)
    safe_message = message.encode("gbk", errors="replace").decode("gbk")
    # print(safe_message, flush=True)
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
    min_lr = args.learning_rate / 10

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
        max_seq_len=args.seq_len,
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
            "max_seq_len": args.seq_len,
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
        "seq_len": args.seq_len,
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


def generate_during_training(model, tokenizer, prompt, args, device):
    was_training = model.training
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    temperature = max(args.generate_temperature, 1e-5)
    eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(args.generate_max_new_tokens):
            x_cond = input_ids[:, -args.seq_len:]
            attention_mask = torch.ones_like(x_cond, device=device)
            logits = model(x_cond, attention_mask=attention_mask)
            next_token_logits = logits[:, -1, :] / temperature

            if args.generate_top_k is not None and args.generate_top_k > 0:
                top_k = min(args.generate_top_k, next_token_logits.size(-1))
                values, _ = torch.topk(next_token_logits, top_k)
                min_topk = values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_topk,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

    if was_training:
        model.train()

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def train(args):
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
            max_length=args.seq_len + 1,
        )
    elif args.train_stage == "sft":
        dataset = SFTDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            max_length=args.seq_len + 1,
        )
    else:
        raise ValueError(f"Unknown train_stage: {args.train_stage}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )

    log_message(f"samples: {len(dataset)}", log_file)
    log_message(f"batches: {len(loader)}", log_file)
    log_message(f"batch_size: {args.batch_size}", log_file)
    log_message(f"seq_len: {args.seq_len}", log_file)
    log_message(f"epochs: {args.epochs}", log_file)
    log_message(f"learning_rate: {args.learning_rate}", log_file)
    log_message(f"accumulation_steps: {args.accumulation_steps}", log_file)
    log_message(f"warmup_iters: {args.warmup_iters}", log_file)
    log_message(f"use_amp: {args.use_amp}", log_file)
    log_message(f"generate_every_steps: {args.generate_every_steps}", log_file)
    log_message(f"generate_prompt: {args.generate_prompt}", log_file)

    assert args.dim % args.n_heads == 0, "dim must be divisible by n_heads"
    assert args.n_heads % args.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    model_config = build_model_config(vocab_size=len(tokenizer), args=args)
    model = Transformer(config=model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"total_params: {total_params / 1e9:.6f} B", log_file)
    log_message(f"trainable_params: {trainable_params / 1e9:.6f} B", log_file)

    if args.train_stage == "sft" and not args.resume:
        if args.sft_init_checkpoint_path:
            if not os.path.exists(args.sft_init_checkpoint_path):
                raise FileNotFoundError(
                    f"sft init checkpoint not found: {args.sft_init_checkpoint_path}"
                )
            checkpoint = torch.load(
                args.sft_init_checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            log_message(f"sft init checkpoint: {args.sft_init_checkpoint_path}", log_file)
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
        start_epoch, global_step = load_checkpoint_for_resume(
            checkpoint_path=args.resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            log_file=log_file,
        )
    model.train()
    total_steps = args.epochs * len(loader)
    avg_loss = 0.0

    for epoch in range(start_epoch, args.epochs):
        total_loss = 0.0
        current_epoch = epoch + 1
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"epoch={current_epoch}/{args.epochs}")

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
            is_last_step = (step + 1) == len(loader)

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
                    f"epoch={current_epoch}/{args.epochs} | "
                    f"step={global_step} | "
                    f"loss={loss.item():.6f} | "
                    f"lr={lr:.8f}"
                )
                log_message(message, log_file)

            # if args.generate_every_steps > 0 and global_step % args.generate_every_steps == 0:
            #     sample_text = generate_during_training(
            #         model=model,
            #         tokenizer=tokenizer,
            #         prompt=args.generate_prompt,
            #         args=args,
            #         device=device,
            #     )
            #     log_message(f"sample step={global_step}: {sample_text}", log_file)

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

        avg_loss = total_loss / len(loader)
        writer.add_scalar("train/epoch_avg_loss", avg_loss, current_epoch)
        epoch_message = f"epoch={current_epoch}/{args.epochs} | avg_loss={avg_loss:.6f}"
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
