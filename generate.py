import os
import time

import torch
from transformers import AutoTokenizer

from config import *
from k_model import ModelConfig, Transformer


def find_latest_checkpoint():
    search_dirs = [CHECKPOINT_DIR, "."]
    candidates = []

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue

        for filename in os.listdir(search_dir):
            if not filename.endswith(".pt"):
                continue

            path = os.path.join(search_dir, filename)
            if os.path.isfile(path):
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No .pt checkpoint found in current directory or {CHECKPOINT_DIR}"
        )

    return max(candidates, key=os.path.getmtime)


def load_model(checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()

    print(f"checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    tokenizer_name = checkpoint.get("tokenizer_name", TOKENIZER_NAME)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if "model_config" in checkpoint:
        model_config = ModelConfig(**checkpoint["model_config"])
    else:
        model_config = ModelConfig(
            vocab_size=checkpoint["vocab_size"],
            max_seq_len=checkpoint["seq_len"],
            dim=checkpoint["dim_embedding"],
            n_layers=checkpoint["n_layers"],
            n_heads=checkpoint["n_heads"],
            n_kv_heads=checkpoint.get("n_kv_heads", checkpoint["n_heads"]),
        )

    model = Transformer(config=model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, checkpoint, device


def build_prompt(text):
    return text.strip()


def generate_text_stream(
    model,
    tokenizer,
    prompt,
    seq_len=128,
    max_token=100,
    device="cpu",
    temperature=1.0,
    top_k=None,
    delay=0.0,
):
    model.eval()
    eos_token_id = tokenizer.eos_token_id
    x = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    temperature = max(temperature, 1e-5)

    with torch.no_grad():
        for _ in range(max_token):
            x_cond = x[:, -seq_len:]
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


def generate_text(
    model,
    tokenizer,
    prompt,
    seq_len=128,
    max_token=100,
    device="cpu",
    temperature=1.0,
    top_k=None,
    stream=False,
    delay=0.15,
):
    full_text = ""

    if stream:
        for chunk in generate_text_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            seq_len=seq_len,
            max_token=max_token,
            device=device,
            temperature=temperature,
            top_k=top_k,
            delay=delay,
        ):
            print(chunk, end="", flush=True)
            full_text += chunk
        print()
        return full_text.strip()

    for chunk in generate_text_stream(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        seq_len=seq_len,
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

    print(
        "\nPretrain model loaded. Enter text to continue generation; "
        "enter exit or quit to stop.\n"
    )

    while True:
        prompt = input(">>> ").strip()

        if prompt.lower() in {"exit", "quit"}:
            print("Generation finished.")
            break

        if prompt == "":
            print("Input cannot be empty. Please try again.")
            continue

        answer = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=build_prompt(prompt),
            seq_len=model.max_seq_len,
            max_token=MAX_NEW_TOKENS,
            device=device,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            stream=STREAM,
        )

        if answer == "":
            print("(model did not generate visible text)")


if __name__ == "__main__":
    main()
