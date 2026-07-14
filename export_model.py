import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

import config as cfg
from k_model import ModelConfig, Transformer


DEFAULT_SAVE_ROOT = cfg.EXPORT_SAVE_ROOT


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_parameter_count(num_parameters):
    if num_parameters >= 1_000_000_000:
        return f"{num_parameters / 1_000_000_000:.1f}B".replace(".0B", "B")
    if num_parameters >= 1_000_000:
        return f"{round(num_parameters / 1_000_000)}M"
    if num_parameters >= 1_000:
        return f"{round(num_parameters / 1_000)}K"
    return str(num_parameters)


def clean_state_dict_prefix(state_dict, unwanted_prefix="_orig_mod."):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith(unwanted_prefix):
            key = key[len(unwanted_prefix) :]
        cleaned[key] = value
    return cleaned


def load_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def build_model_config_from_checkpoint(checkpoint):
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise KeyError('checkpoint missing complete "model_config" dict')
    return ModelConfig(**model_config)


def get_state_dict_from_checkpoint(checkpoint):
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint model state_dict must be a dict")
    return clean_state_dict_prefix(state_dict)


def infer_export_stage(checkpoint, fallback_stage=None):
    stage = checkpoint.get("train_stage") or fallback_stage
    if stage not in {"pretrain", "sft"}:
        raise ValueError(f'export stage must be "pretrain" or "sft", got: {stage}')
    return stage


def export_model_from_checkpoint(
    checkpoint_path,
    tokenizer_path=None,
    save_directory=None,
    stage=None,
):
    checkpoint = load_checkpoint(checkpoint_path)
    model_config = build_model_config_from_checkpoint(checkpoint)
    stage = infer_export_stage(checkpoint, fallback_stage=stage)
    tokenizer_path = tokenizer_path or checkpoint.get("tokenizer_name") or cfg.TOKENIZER_NAME

    ModelConfig.register_for_auto_class()
    Transformer.register_for_auto_class("AutoModelForCausalLM")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is not None:
        model_config.pad_token_id = tokenizer.pad_token_id

    model = Transformer(config=model_config)
    state_dict = get_state_dict_from_checkpoint(checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model.eval()

    num_parameters = count_parameters(model)
    parameter_label = format_parameter_count(num_parameters)
    save_directory = save_directory or str(
        Path(DEFAULT_SAVE_ROOT) / f"llm_{stage}_{parameter_label}"
    )
    model.save_pretrained(save_directory, safe_serialization=False)
    tokenizer.save_pretrained(save_directory)

    print(
        f"Exported {stage} model: params={num_parameters / 1e6:.2f}M "
        f"({num_parameters / 1e9:.2f}B), output={save_directory}"
    )
    if missing_keys:
        print(f"Warning: export missing_keys={missing_keys}")
    if unexpected_keys:
        print(f"Warning: export unexpected_keys={unexpected_keys}")

    return save_directory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a project checkpoint to Hugging Face save_pretrained format."
    )
    parser.add_argument("checkpoint_path", help="Path to a .pt checkpoint.")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help=f"Tokenizer path. Default: checkpoint tokenizer_name or {cfg.TOKENIZER_NAME}",
    )
    parser.add_argument(
        "--save-directory",
        default=None,
        help=(
            f'Output directory. Default: "{DEFAULT_SAVE_ROOT}/'
            'llm_{stage}_{parameter_count}".'
        ),
    )
    parser.add_argument(
        "--stage",
        default=None,
        choices=["pretrain", "sft"],
        help="Fallback stage if checkpoint does not contain train_stage.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    export_model_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        save_directory=args.save_directory,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()
