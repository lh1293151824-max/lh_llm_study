import os
import time

import torch
from transformers import AutoTokenizer

import config as cfg
from k_model import ModelConfig, Transformer


def get_default_generate_stage():
    if cfg.TRAIN_STAGE in cfg.STAGE_CONFIG_TABLE:
        return cfg.TRAIN_STAGE
    return "sft"


def checkpoint_matches_stage(filename, stage):
    stem, ext = os.path.splitext(filename)
    return ext == ".pt" and f"_{stage}" in stem


def find_latest_checkpoint(stage=None):
    stage = stage or get_default_generate_stage()
    if stage not in cfg.STAGE_CONFIG_TABLE:
        raise ValueError(f"Unknown TRAIN_STAGE: {stage}")

    candidates = []
    active_config = cfg.get_active_config(stage=stage)
    checkpoint_dir = active_config["CHECKPOINT_DIR"]

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    for filename in os.listdir(checkpoint_dir):
        if not checkpoint_matches_stage(filename, stage):
            continue

        path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(path):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No {stage} .pt checkpoint found in {checkpoint_dir}")

    return max(candidates, key=os.path.getmtime)


def load_model(checkpoint_path=None, stage=None):
    stage = stage or get_default_generate_stage()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(stage)

    print(f"checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    checkpoint_stage = checkpoint.get("train_stage")
    if checkpoint_stage is not None and checkpoint_stage != stage:
        raise ValueError(
            f"checkpoint stage mismatch: expected {stage}, got {checkpoint_stage}"
        )

    tokenizer_name = checkpoint.get("tokenizer_name", cfg.TOKENIZER_NAME)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if "model_config" in checkpoint:
        model_config = ModelConfig(**checkpoint["model_config"])
    else:
        model_config = ModelConfig(
            vocab_size=checkpoint["vocab_size"],
            max_seq_len=get_checkpoint_max_seq_len(checkpoint),
            dim=checkpoint["dim_embedding"],
            n_layers=checkpoint["n_layers"],
            n_heads=checkpoint["n_heads"],
            n_kv_heads=checkpoint.get("n_kv_heads", checkpoint["n_heads"]),
        )

    model = Transformer(config=model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, checkpoint, device


def get_checkpoint_max_seq_len(checkpoint):
    if "max_seq_len" in checkpoint:
        return checkpoint["max_seq_len"]
    if "seq_len" in checkpoint:
        return checkpoint["seq_len"]
    raise KeyError('checkpoint missing "max_seq_len"')


def generate_text_stream(
    model,
    tokenizer,
    prompt,
    stage=None,
    max_seq_len=None,
    max_token=None,
    device="cpu",
    temperature=1.0,
    top_k=None,
    delay=0.0,
):
    model.eval()
    stage = stage or get_default_generate_stage()
    active_config = cfg.get_active_config(stage=stage)
    max_seq_len = max_seq_len or active_config["MAX_SEQ_LEN"]
    max_token = max_token or active_config["MAX_NEW_TOKENS"]
    eos_token_id = tokenizer.eos_token_id
    prompt = build_prompt(prompt, tokenizer, stage)
    x = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    temperature = max(temperature, 1e-5)

    with torch.no_grad():
        for _ in range(max_token):
            x_cond = x[:, -max_seq_len:]
            attention_mask = torch.ones_like(x_cond, device=device)
            logits = model(x_cond, attention_mask=attention_mask)
            next_token_logits_total = logits[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, next_token_logits_total.size(-1))
                values, _ = torch.topk(next_token_logits_total, k)
                min_topk = values[:, -1].unsqueeze(-1)

                next_token_logits = torch.where(
                    next_token_logits_total < min_topk,
                    torch.full_like(next_token_logits_total, float("-inf")),
                    next_token_logits_total,
                )
            else:
                next_token_logits = next_token_logits_total

            next_token_prob = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_prob, num_samples=1)

            x = torch.cat([x, next_token_id], dim=1)

            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

            new_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            yield new_text

            if delay > 0:
                time.sleep(delay)


def build_prompt(text, tokenizer, stage):
    if stage == "sft":
        messages = [
            {"role": "system", "content": cfg.SYSTEM_PROMPT},
            {"role": "user", "content": text.strip()},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{tokenizer.bos_token}{text.strip()}"


def generate_text(
    model,
    tokenizer,
    prompt,
    stage=None,
    max_seq_len=None,
    max_token=None,
    device="cpu",
    temperature=1.0,
    top_k=None,
    stream=False,
    delay=None,
):
    full_text = ""
    stage = stage or get_default_generate_stage()
    active_config = cfg.get_active_config(stage=stage)
    max_seq_len = max_seq_len or active_config["MAX_SEQ_LEN"]
    max_token = max_token or active_config["MAX_NEW_TOKENS"]
    delay = 0.15 if delay is None else delay

    if stream:
        for chunk in generate_text_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            stage=stage,
            max_seq_len=max_seq_len,
            max_token=max_token,
            device=device,
            temperature=temperature,
            top_k=top_k,
            delay=delay,
        ):
            print(chunk, end="", flush=True)
            full_text += chunk
        return full_text.strip()

    for chunk in generate_text_stream(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        stage=stage,
        max_seq_len=max_seq_len,
        max_token=max_token,
        device=device,
        temperature=temperature,
        top_k=top_k,
        delay=0.0,
    ):
        full_text += chunk
    return full_text.strip()


def main():
    model, tokenizer, checkpoint, device = load_model()
    stage = checkpoint.get("train_stage", get_default_generate_stage())
    active_config = cfg.get_active_config(stage=stage)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params / 1e6:.3f} M parameters.")
    print("\n我是AI助手，请问有什么可以帮助你？\n")

    while True:
        prompt = input(">>> ").strip()

        if prompt.lower() in {"exit", "quit"}:
            print("Generation finished.")
            break

        if prompt == "":
            print("Input cannot be empty. Please try again.")
            continue

        print("🤖 ", end="", flush=True)
        answer = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            stage=stage,
            max_seq_len=model.max_seq_len,
            max_token=active_config["MAX_NEW_TOKENS"],
            device=device,
            temperature=active_config["TEMPERATURE"],
            top_k=active_config["TOP_K"],
            stream=active_config["STREAM"],
        )
        print("\n", end="", flush=True)

        if answer == "":
            print("(model did not generate visible text)")


if __name__ == "__main__":
    main()
