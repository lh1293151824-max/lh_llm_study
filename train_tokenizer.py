import json
import os
import random

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.normalizers import NFKC
from transformers import AutoTokenizer

import config as cfg

random.seed(42)


def read_texts_from_jsonl(file_path: str):
    """Read JSONL data and yield the text field."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if "text" not in data:
                    raise KeyError(f"Missing 'text' field in line {line_num}")
                yield data["text"]
            except json.JSONDecodeError:
                print(f"Error decoding JSON in line {line_num}")
                continue
            except KeyError as exc:
                print(exc)
                continue


def count_lines(file_path: str) -> int:
    with open(file_path, "rb") as f:
        return sum(1 for _ in f)

def train_tokenizer(
    data_path: str = None,
    save_dir: str = None,
    vocab_size: int = None,
) -> None:
    """Train and save the project tokenizer."""
    data_path = data_path or cfg.TOKENIZER_TRAIN_DATA_PATH
    save_dir = save_dir or cfg.TOKENIZER_SAVE_DIR
    vocab_size = vocab_size or cfg.VOCAB_SIZE

    os.makedirs(save_dir, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<unk>",
            "<s>",
            "</s>",
            "<|im_start|>",
            "<|im_end|>",
        ],
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"Tokenizer training data: {data_path}")
    texts = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=count_lines(data_path))

    # 验证特殊token映射
    try:
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        assert tokenizer.token_to_id("<|im_start|>") == 3
        assert tokenizer.token_to_id("<|im_end|>") == 4
    except AssertionError as e:
        print("Special tokens mapping error:", e)
        raise


    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")


def create_tokenizer_config(save_dir: str = None) -> None:
    """Create tokenizer config files used by AutoTokenizer."""
    save_dir = save_dir or cfg.TOKENIZER_SAVE_DIR

    tokenizer_config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "padding_side": "left",
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        ),
    }

    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=4)

    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"],
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)


def eval_tokenizer(tokenizer_path: str = None) -> None:
    """Evaluate tokenizer loading, chat template, and special token handling."""
    tokenizer_path = tokenizer_path or cfg.TOKENIZER_SAVE_DIR

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = "left"

    print("\n=== Tokenizer basic info ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    messages = [
        {"role": "system", "content": cfg.SYSTEM_PROMPT},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]

    print("\n=== Chat template test ===")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print("Generated prompt:\n", prompt, sep="")

    print("\n=== Encode and decode test ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    if decoded != prompt:
        raise ValueError("tokenizer encode/decode evaluation failed")
    print("Decoded text matches original: True")

    print("\n=== Special token test ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    if decoded != test_text:
        raise ValueError("tokenizer special-token preservation evaluation failed")
    print("Special tokens preserved: True")


def main():
    train_tokenizer()
    eval_tokenizer()


if __name__ == "__main__":
    main()
