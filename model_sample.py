import os
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer

import config as cfg
from k_model import (
    ModelConfig,
    Transformer,
    infer_output_head_type_from_state_dict,
    validate_state_dict_output_head_type,
)





class TextGenerator:
    """Load a project checkpoint and generate pretrain or SFT samples."""

    SUPPORTED_DTYPES = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def __init__(
        self,
        checkpoint=None,
        tokenizer_model_path=None,
        seed=42,
        device=None,
        dtype="float32",
        stage=None,
    ):
        self.stage = stage or cfg.SAMPLE_STAGE
        if self.stage not in cfg.STAGE_CONFIG_TABLE:
            raise ValueError('stage must be "pretrain" or "sft"')
        if dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(
                f"dtype must be one of {tuple(self.SUPPORTED_DTYPES)}, got {dtype}"
            )

        self.seed = seed
        self.dtype = dtype
        self.torch_dtype = self.SUPPORTED_DTYPES[dtype]
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device_type = self.device.type

        self._set_seed()
        self.checkpoint = checkpoint or self.find_latest_checkpoint(self.stage)
        checkpoint_data = torch.load(
            self.checkpoint,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(checkpoint_data, dict):
            raise TypeError("checkpoint must contain a dictionary")

        checkpoint_stage = checkpoint_data.get("train_stage")
        if checkpoint_stage is not None and checkpoint_stage != self.stage:
            raise ValueError(
                f"checkpoint stage mismatch: expected {self.stage}, "
                f"got {checkpoint_stage}"
            )

        tokenizer_path = (
            tokenizer_model_path
            or checkpoint_data.get("tokenizer_name")
            or cfg.SAMPLE_TOKENIZER_PATH
        )
        self.tokenizer_model_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.padding_side = "left"

        model_config = self._build_model_config(checkpoint_data)
        if self.tokenizer.pad_token_id is not None:
            model_config.pad_token_id = self.tokenizer.pad_token_id

        self.model = Transformer(config=model_config).to(self.device)
        state_dict = self._extract_state_dict(checkpoint_data)
        state_dict = self._clean_state_dict_prefix(state_dict)
        validate_state_dict_output_head_type(
            state_dict,
            expected=model_config.output_head_type,
            context="sampling checkpoint",
        )
        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict,
            strict=False,
        )
        self.model.eval()

        num_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        print(
            f"stage={self.stage}, device={self.device}, "
            f"checkpoint={self.checkpoint}"
        )
        print(f"Model has {num_params / 1e6:.3f} M parameters.")
        if missing_keys:
            print(f"Warning: missing_keys={missing_keys}")
        if unexpected_keys:
            print(f"Warning: unexpected_keys={unexpected_keys}")

    def _set_seed(self):
        torch.manual_seed(self.seed)
        if self.device_type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _autocast_context(self):
        if self.device_type != "cuda" or self.torch_dtype == torch.float32:
            return nullcontext()
        return torch.amp.autocast(
            device_type=self.device_type,
            dtype=self.torch_dtype,
        )

    @staticmethod
    def checkpoint_matches_stage(filename, stage):
        stem, extension = os.path.splitext(filename)
        return extension == ".pt" and f"_{stage}" in stem

    @classmethod
    def find_latest_checkpoint(cls, stage):
        active_config = cfg.get_active_config(stage=stage)
        checkpoint_dir = active_config["CHECKPOINT_DIR"]
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(
                f"checkpoint directory not found: {checkpoint_dir}"
            )

        candidates = []
        for filename in os.listdir(checkpoint_dir):
            if not cls.checkpoint_matches_stage(filename, stage):
                continue
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            if os.path.isfile(checkpoint_path):
                candidates.append(checkpoint_path)

        if not candidates:
            raise FileNotFoundError(
                f"No {stage} .pt checkpoint found in {checkpoint_dir}"
            )
        return max(candidates, key=os.path.getmtime)

    @staticmethod
    def _get_checkpoint_max_seq_len(checkpoint_data):
        if "max_seq_len" in checkpoint_data:
            return checkpoint_data["max_seq_len"]
        if "seq_len" in checkpoint_data:
            return checkpoint_data["seq_len"]
        raise KeyError('checkpoint missing "max_seq_len"')

    def _build_model_config(self, checkpoint_data):
        model_config = checkpoint_data.get("model_config")
        if isinstance(model_config, dict):
            return ModelConfig(**model_config)

        if checkpoint_data and all(
            isinstance(value, torch.Tensor) for value in checkpoint_data.values()
        ):
            return self._build_model_config_from_state_dict(checkpoint_data)

        required_keys = {
            "vocab_size",
            "dim_embedding",
            "n_layers",
            "n_heads",
        }
        missing_keys = sorted(required_keys - checkpoint_data.keys())
        if missing_keys:
            raise KeyError(
                "legacy checkpoint missing model config fields: "
                f"{missing_keys}"
            )

        return ModelConfig(
            vocab_size=checkpoint_data["vocab_size"],
            max_seq_len=self._get_checkpoint_max_seq_len(checkpoint_data),
            dim=checkpoint_data["dim_embedding"],
            n_layers=checkpoint_data["n_layers"],
            n_heads=checkpoint_data["n_heads"],
            n_kv_heads=checkpoint_data.get(
                "n_kv_heads",
                checkpoint_data["n_heads"],
            ),
        )

    def _build_model_config_from_state_dict(self, state_dict):
        state_dict = self._clean_state_dict_prefix(state_dict)
        embedding_weight = state_dict.get("tok_embeddings.weight")
        if embedding_weight is None or embedding_weight.dim() != 2:
            raise KeyError(
                'raw state_dict missing valid "tok_embeddings.weight"'
            )

        vocab_size, dim = embedding_weight.shape
        layer_indices = []
        for key in state_dict:
            if not key.startswith("layers."):
                continue
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_indices.append(int(parts[1]))

        # Raw legacy state_dict files do not record the attention head layout.
        # They correspond to the original full training architecture.
        active_config = cfg.get_active_config(stage=self.stage, mode="train")
        n_layers = (
            max(layer_indices) + 1
            if layer_indices
            else active_config["N_LAYERS"]
        )
        hidden_weight = state_dict.get("layers.0.feed_forward.w1.weight")
        hidden_dim = (
            hidden_weight.size(0)
            if isinstance(hidden_weight, torch.Tensor)
            else None
        )
        output_head_type = infer_output_head_type_from_state_dict(state_dict)
        operator_weight = state_dict.get(
            "operator_output.branch_output.weight"
        )
        operator_rank = (
            operator_weight.size(0)
            if isinstance(operator_weight, torch.Tensor)
            and operator_weight.dim() == 2
            else cfg.OPERATOR_RANK
        )

        return ModelConfig(
            vocab_size=vocab_size,
            max_seq_len=active_config["MAX_SEQ_LEN"],
            dim=dim,
            n_layers=n_layers,
            n_heads=active_config["N_HEADS"],
            n_kv_heads=active_config["N_KV_HEADS"],
            hidden_dim=hidden_dim,
            multiple_of=active_config["MULTIPLE_OF"],
            norm_eps=active_config["NORM_EPS"],
            dropout=active_config["DROPOUT"],
            output_head_type=output_head_type,
            operator_rank=operator_rank,
            operator_alpha=cfg.OPERATOR_ALPHA,
            operator_scale_warning_ratio=(
                cfg.OPERATOR_SCALE_WARNING_RATIO
            ),
        )

    @staticmethod
    def _extract_state_dict(checkpoint_data):
        state_dict = checkpoint_data.get("model_state_dict")
        if isinstance(state_dict, dict):
            return state_dict
        if checkpoint_data and all(
            isinstance(value, torch.Tensor) for value in checkpoint_data.values()
        ):
            return checkpoint_data
        raise KeyError('checkpoint missing a valid "model_state_dict"')

    @staticmethod
    def _clean_state_dict_prefix(state_dict, unwanted_prefix="_orig_mod."):
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(unwanted_prefix):
                key = key[len(unwanted_prefix) :]
            cleaned_state_dict[key] = value
        return cleaned_state_dict

    def _build_prompt(self, start, stage):
        if stage == "sft":
            messages = [
                {"role": "system", "content": cfg.SYSTEM_PROMPT},
                {"role": "user", "content": start.strip()},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return start

    def _sample(
        self,
        start,
        stage,
        num_samples=None,
        max_new_tokens=None,
        temperature=None,
        top_k=None,
    ):
        if stage != self.stage:
            raise ValueError(
                f"TextGenerator loaded a {self.stage} checkpoint, "
                f"but {stage} sampling was requested"
            )

        sample_config = cfg.get_sample_config(stage)
        num_samples = (
            sample_config["NUM_SAMPLES"]
            if num_samples is None
            else num_samples
        )
        max_new_tokens = (
            sample_config["MAX_NEW_TOKENS"]
            if max_new_tokens is None
            else max_new_tokens
        )
        temperature = (
            sample_config["TEMPERATURE"]
            if temperature is None
            else temperature
        )
        top_k = sample_config["TOP_K"] if top_k is None else top_k

        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

        prompt = self._build_prompt(start, stage)
        input_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        generated_texts = []
        with self._autocast_context():
            for _ in range(num_samples):
                generated_ids = self.model.generate(
                    idx=input_ids,
                    stop_id=self.tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                generated_texts.append(
                    self.tokenizer.decode(
                        generated_ids[0].tolist(),
                        skip_special_tokens=True,
                    )
                )
        return generated_texts

    def pretrain_sample(
        self,
        start="Hello!",
        num_samples=None,
        max_new_tokens=None,
        temperature=None,
        top_k=None,
    ):
        return self._sample(
            start=start,
            stage="pretrain",
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    def sft_sample(
        self,
        start="Hello!",
        num_samples=None,
        max_new_tokens=None,
        temperature=None,
        top_k=None,
    ):
        return self._sample(
            start=start,
            stage="sft",
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )


def main():
    stage = cfg.SAMPLE_STAGE
    sample_config = cfg.get_sample_config(stage)
    generator = TextGenerator(
        checkpoint=cfg.SAMPLE_CHECKPOINT_PATH or None,
        tokenizer_model_path=cfg.SAMPLE_TOKENIZER_PATH or None,
        seed=cfg.SAMPLE_SEED,
        device=cfg.SAMPLE_DEVICE or None,
        dtype=cfg.SAMPLE_DTYPE,
        stage=stage,
    )

    if stage == "pretrain":
        print("------------------- Pretrain Sample -------------------")
        sample_method = generator.pretrain_sample
        prompts = PRETRAIN_PROMPTS
    else:
        print("------------------- SFT Sample -------------------")
        sample_method = generator.sft_sample
        prompts = SFT_PROMPTS

    for index, prompt in enumerate(prompts, 1):
        samples = sample_method(
            start=prompt,
            num_samples=sample_config["NUM_SAMPLES"],
            max_new_tokens=sample_config["MAX_NEW_TOKENS"],
            temperature=sample_config["TEMPERATURE"],
            top_k=sample_config["TOP_K"],
        )
        for sample_index, sample in enumerate(samples, 1):
            if stage == "pretrain":
                print(
                    f"\nPrompt {index}, Sample {sample_index}:\n"
                    f"{prompt}{sample}\n{'-' * 20}"
                )
            else:
                print(
                    f"\nQuestion {index}, Sample {sample_index}:\n"
                    f"{prompt}\nAI answer: {sample}\n{'-' * 20}"
                )


if __name__ == "__main__":
    PRETRAIN_PROMPTS = [
    "<|im_start|>北京大学是",
    "<|im_start|>中国矿业大学（北京）地球科学与测绘工程学院",
    ]

    SFT_PROMPTS = [
    "你好呀",
    "中国的首都是哪里？",
    "1+12等于多少？",
    "你是谁？",
    ]
    main()
