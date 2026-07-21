import json
from pathlib import Path

import config as cfg
from build_pretrain_original_from_docs import build_pretrain_original_from_docs
from data_deal import prepare_training_data
from export_model import export_model_from_checkpoint
from train import build_args_from_config, train
from train_tokenizer import eval_tokenizer, train_tokenizer


DOC_BUILD_STAGE = "build_pretrain_original_from_docs"
PRETRAIN_DATA_STAGES = {
    DOC_BUILD_STAGE,
    "tokenizer",
    "tokenizer_eval",
    "data_prep",
    "pretrain",
}


def copy_text_jsonl(source_path, output_path):
    source_path = Path(source_path)
    output_path = Path(output_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Happy-LLM pretrain data not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_lines = 0
    skipped_lines = 0

    with source_path.open("r", encoding="utf-8") as source_file:
        with output_path.open("w", encoding="utf-8") as output_file:
            for line_num, line in enumerate(source_file, 1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    skipped_lines += 1
                    print(
                        f"Warning: skip invalid JSON in {source_path} "
                        f"line {line_num}: {exc}"
                    )
                    continue

                text = str(item.get("text", "")).strip()
                if not text:
                    skipped_lines += 1
                    continue

                output_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                valid_lines += 1

    print(
        "Prepared pretrain original data from Happy-LLM only: "
        f"{source_path} -> {output_path}; "
        f"valid_lines={valid_lines}, skipped_lines={skipped_lines}"
    )


def ensure_pretrain_original_data(stage_names):
    stage_names = list(stage_names)

    if DOC_BUILD_STAGE in stage_names:
        print("Pretrain source: rebuild from Happy-LLM data plus document data.")
        return

    source_mode = cfg.PRETRAIN_ORIGINAL_SOURCE_WHEN_SKIP_DOCS
    pretrain_origin_path = Path(cfg.PRETRAIN_ORIGIN_DATA)
    happy_origin_path = Path(cfg.PRETRAIN_HAPPY_ORIGIN_DATA)

    if source_mode == "merged":
        if not pretrain_origin_path.exists():
            raise FileNotFoundError(
                f"{cfg.PRETRAIN_ORIGIN_DATA} does not exist. Please run from "
                "build_pretrain_original_from_docs first, or set "
                'PRETRAIN_ORIGINAL_SOURCE_WHEN_SKIP_DOCS = "happy".'
            )
        print(f"Pretrain source: existing merged data at {pretrain_origin_path}")
        return

    if source_mode == "happy":
        copy_text_jsonl(happy_origin_path, pretrain_origin_path)
        return

    raise ValueError(
        'PRETRAIN_ORIGINAL_SOURCE_WHEN_SKIP_DOCS must be "merged" or "happy"'
    )


def run_build_pretrain_original_from_docs_stage():
    build_pretrain_original_from_docs()


def run_tokenizer_stage():
    train_tokenizer()


def run_tokenizer_eval_stage():
    eval_tokenizer()


def run_data_prep_stage():
    prepare_training_data("all")


def run_train_stage(stage):
    args = build_args_from_config(stage=stage)
    train(args)
    checkpoint_path = Path(args.checkpoint_dir) / args.checkpoint_path
    export_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        stage=stage,
    )


def run_stage(stage):
    if stage == DOC_BUILD_STAGE:
        run_build_pretrain_original_from_docs_stage()
    elif stage == "tokenizer":
        run_tokenizer_stage()
    elif stage == "tokenizer_eval":
        run_tokenizer_eval_stage()
    elif stage == "data_prep":
        run_data_prep_stage()
    elif stage in {"pretrain", "sft"}:
        run_train_stage(stage)
    else:
        raise ValueError(f"TRAIN_STAGE must be one of {cfg.PIPELINE_STAGES}")


def get_start_stage():
    if cfg.RUN_FULL_PIPELINE:
        return DOC_BUILD_STAGE
    return cfg.TRAIN_STAGE


def main():
    start_stage = get_start_stage()
    pipeline_stages = cfg.PIPELINE_STAGES
    if start_stage not in pipeline_stages:
        raise ValueError(f"TRAIN_STAGE must be one of {pipeline_stages}")

    start_index = pipeline_stages.index(start_stage)
    selected_stages = pipeline_stages[start_index:]

    if any(stage in PRETRAIN_DATA_STAGES for stage in selected_stages):
        ensure_pretrain_original_data(selected_stages)

    for stage in selected_stages:
        run_stage(stage)

    print("训练完成，可以开始推理啦~")


if __name__ == "__main__":
    main()
