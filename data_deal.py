import json
import os

from tqdm import tqdm

import config as cfg


def get_max_seq_len():
    return cfg.get_active_config(stage="pretrain")["MAX_SEQ_LEN"]


def split_text(text, max_seq_len=None):
    """Split text into fixed-size chunks for pretraining."""
    max_seq_len = max_seq_len or get_max_seq_len()
    return [text[i : i + max_seq_len] for i in range(0, len(text)//max_seq_len * max_seq_len, max_seq_len)]


def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def prepare_pretrain_data(
    origin_path=None,
    train_path=None,
    max_seq_len=None,
):
    origin_path = origin_path or cfg.PRETRAIN_ORIGIN_DATA
    train_path = train_path or cfg.PRETRAIN_TRAIN_DATA
    max_seq_len = max_seq_len or get_max_seq_len()

    ensure_parent_dir(train_path)
    total_lines = count_lines(origin_path)

    with open(train_path, "w", encoding="utf-8") as pretrain:
        with open(origin_path, "r", encoding="utf-8") as f:
            progress = tqdm(
                f,
                total=total_lines,
                desc=f"Processing lines in {origin_path}",
                unit="lines",
                leave=True,
            )
            for line_num, line in enumerate(progress, 1):
                try:
                    item = json.loads(line)
                    text = item["text"]
                    for chunk in split_text(text, max_seq_len):
                        pretrain.write(
                            json.dumps({"text": chunk}, ensure_ascii=False) + "\n"
                        )
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    print(f"Skip invalid JSON in {origin_path} line {line_num}: {exc}")
                    continue


def convert_message(data):
    """Convert original SFT conversations to chat-template message format."""
    message = [
        {"role": "system", "content": cfg.SYSTEM_PROMPT},
    ]
    for item in data:
        if item["from"] == "human":
            message.append(
                {"role": "user", "content": item["value"]}
            )
        elif item["from"] == "assistant":
            message.append(
                {
                    "role": "assistant",
                    "content": item["value"],
                }
            )
    return message


def prepare_sft_data(
    origin_path=None,
    train_path=None,
    max_records=None,
):
    origin_path = origin_path or cfg.SFT_ORIGIN_DATA
    train_path = train_path or cfg.SFT_TRAIN_DATA
    max_records = cfg.get_sft_data_limit() if max_records is None else max_records

    ensure_parent_dir(train_path)
    total_lines = count_lines(origin_path)
    if max_records is not None:
        print(f"SFT test mode limit: first {max_records} valid records")
    progress_total = max_records if max_records is not None else total_lines
    progress_desc = (
        "Processing SFT test records"
        if max_records is not None
        else f"Processing SFT records in {origin_path}"
    )

    with open(train_path,"w", encoding="utf-8") as sft:
        with open(origin_path, "r", encoding="utf-8") as f:
            progress = tqdm(
                f,
                total=progress_total,
                desc=progress_desc,
                unit="records",
            )
            written_records = 0
            for line_num, line in enumerate(progress, 1):
                if max_records is not None and written_records >= max_records:
                    break

                try:
                    item = json.loads(line)
                    message = convert_message(item["conversations"])
                    sft.write(json.dumps(message, ensure_ascii=False) + "\n")
                    written_records += 1
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    print(f"Skip invalid JSON in {origin_path} line {line_num}: {exc}")
                    continue


def prepare_training_data(stage=None):
    stage = stage or "all"

    if stage == "pretrain":
        print("[1/1] Preparing pretrain data...")
        prepare_pretrain_data()
    elif stage == "sft":
        print("[1/1] Preparing SFT data...")
        prepare_sft_data()
    elif stage == "all":
        print("[1/2] Preparing pretrain data...")
        prepare_pretrain_data()
        print("[2/2] Preparing SFT data...")
        prepare_sft_data()
    else:
        raise ValueError('stage must be "pretrain", "sft", or "all"')


def main():
    prepare_training_data("all")


if __name__ == "__main__":
    main()
