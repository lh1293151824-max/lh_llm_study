# config.py

# Options: "test", "train"
CONFIG_MODE = "train"

# Options: "pretrain", "sft"
TRAIN_STAGE = "pretrain"


# =============================================================================
# Common paths and switches
# =============================================================================

TOKENIZER_NAME = "tokenizer_k"
PRETRAIN_DATA_PATH = "data/seq_monkey_datawhale.jsonl"
SFT_DATA_PATH = "data/sft_train_3.5M_CN.json"

LOG_DIR = "runs"
TEXT_LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"

RESUME = False
RESUME_CHECKPOINT_PATH = ""

# Used by the formal SFT config when TRAIN_STAGE == "sft" and RESUME == False.
DEFAULT_SFT_INIT_CHECKPOINT_PATH = "checkpoints/transformer_final.pt"

SYSTEM_PROMPT = "你是一个AI助手"


# =============================================================================
# Shared model defaults: aligned with the happy-llm chapter 5 215M-style model.
# =============================================================================

MODEL_215M_CONFIG = {
    "DIM_EMBEDDING": 1024,
    "N_HEADS": 16,
    "N_LAYERS": 18,
    "N_KV_HEADS": 8,
    "NORM_EPS": 1e-5,
    "DROPOUT": 0.0,
    "FLASH_ATTN": False,
    "MULTIPLE_OF": 64,
}

MODEL_TEST_CONFIG = {
    "DIM_EMBEDDING": 128,
    "N_HEADS": 4,
    "N_LAYERS": 2,
    "N_KV_HEADS": 2,
    "NORM_EPS": 1e-5,
    "DROPOUT": 0.0,
    "FLASH_ATTN": False,
    "MULTIPLE_OF": 64,
}


# =============================================================================
# Pretrain configs
# =============================================================================

PRETRAIN_TEST_CONFIG = {
    # Runtime
    "SEED": 42,
    "NUM_WORKERS": 4,
    "USE_AMP": False,

    # Data and training
    "DATA_PATH": PRETRAIN_DATA_PATH,
    "BATCH_SIZE": 1,
    "SEQ_LEN": 64,
    "EPOCHS": 1,
    "LEARNING_RATE": 1e-4,
    "ACCUMULATION_STEPS": 1,
    "WARMUP_ITERS": 0,
    "GRAD_CLIP": 1.0,

    # Model
    **MODEL_TEST_CONFIG,

    # Logging and checkpoints
    "CHECKPOINT_PREFIX": "transformer_pretrain_test",
    "CHECKPOINT_PATH": "transformer_pretrain_test_final.pt",
    "LOG_INTERVAL": 5,
    "SAVE_EVERY_STEPS": 20,
    "SAVE_EVERY_EPOCHS": 1,

    # Generation during training
    "GENERATE_EVERY_STEPS": 10,
    "GENERATE_PROMPT": "人工智能",
    "GENERATE_MAX_NEW_TOKENS": 20,
    "GENERATE_TEMPERATURE": 0.9,
    "GENERATE_TOP_K": 5,

    # Standalone generate.py defaults
    "MAX_NEW_TOKENS": 64,
    "TEMPERATURE": 0.9,
    "TOP_K": 3,
    "STREAM": True,
}

PRETRAIN_TRAIN_CONFIG = {
    # Runtime
    "SEED": 42,
    "NUM_WORKERS": 4,
    "USE_AMP": True,

    # Data and training
    "DATA_PATH": PRETRAIN_DATA_PATH,
    "BATCH_SIZE": 12,
    "SEQ_LEN": 512,
    "EPOCHS": 2,
    "LEARNING_RATE": 1e-4,
    "ACCUMULATION_STEPS": 8,
    "WARMUP_ITERS": 0,
    "GRAD_CLIP": 1.0,

    # Model
    **MODEL_215M_CONFIG,

    # Logging and checkpoints
    "CHECKPOINT_PREFIX": "transformer_pretrain",
    "CHECKPOINT_PATH": "transformer_final.pt",
    "LOG_INTERVAL": 100,
    "SAVE_EVERY_STEPS": 5000,
    "SAVE_EVERY_EPOCHS": 1,

    # Generation during training
    "GENERATE_EVERY_STEPS": 500,
    "GENERATE_PROMPT": "人工智能",
    "GENERATE_MAX_NEW_TOKENS": 80,
    "GENERATE_TEMPERATURE": 0.9,
    "GENERATE_TOP_K": 5,

    # Standalone generate.py defaults
    "MAX_NEW_TOKENS": 512,
    "TEMPERATURE": 0.9,
    "TOP_K": 3,
    "STREAM": True,
}


# =============================================================================
# SFT configs
# =============================================================================

SFT_TEST_CONFIG = {
    # Runtime
    "SEED": 42,
    "NUM_WORKERS": 4,
    "USE_AMP": False,

    # Data and training
    "DATA_PATH": SFT_DATA_PATH,
    "SFT_INIT_CHECKPOINT_PATH": "",
    "BATCH_SIZE": 1,
    "SEQ_LEN": 64,
    "EPOCHS": 1,
    "LEARNING_RATE": 1e-5,
    "ACCUMULATION_STEPS": 1,
    "WARMUP_ITERS": 0,
    "GRAD_CLIP": 1.0,

    # Model
    **MODEL_TEST_CONFIG,

    # Logging and checkpoints
    "CHECKPOINT_PREFIX": "transformer_sft_test",
    "CHECKPOINT_PATH": "transformer_sft_test_final.pt",
    "LOG_INTERVAL": 5,
    "SAVE_EVERY_STEPS": 20,
    "SAVE_EVERY_EPOCHS": 1,

    # Generation during training
    "GENERATE_EVERY_STEPS": 10,
    "GENERATE_PROMPT": "请介绍一下人工智能。",
    "GENERATE_MAX_NEW_TOKENS": 20,
    "GENERATE_TEMPERATURE": 0.7,
    "GENERATE_TOP_K": 5,

    # Standalone generate.py defaults
    "MAX_NEW_TOKENS": 128,
    "TEMPERATURE": 0.7,
    "TOP_K": 5,
    "STREAM": True,
}

SFT_TRAIN_CONFIG = {
    # Runtime
    "SEED": 42,
    "NUM_WORKERS": 4,
    "USE_AMP": True,

    # Data and training
    "DATA_PATH": SFT_DATA_PATH,
    "SFT_INIT_CHECKPOINT_PATH": DEFAULT_SFT_INIT_CHECKPOINT_PATH,
    "BATCH_SIZE": 12,
    "SEQ_LEN": 512,
    "EPOCHS": 1,
    "LEARNING_RATE": 1e-5,
    "ACCUMULATION_STEPS": 8,
    "WARMUP_ITERS": 0,
    "GRAD_CLIP": 1.0,

    # Model
    **MODEL_215M_CONFIG,

    # Logging and checkpoints
    "CHECKPOINT_PREFIX": "transformer_sft",
    "CHECKPOINT_PATH": "transformer_sft_final.pt",
    "LOG_INTERVAL": 100,
    "SAVE_EVERY_STEPS": 5000,
    "SAVE_EVERY_EPOCHS": 1,

    # Generation during training
    "GENERATE_EVERY_STEPS": 500,
    "GENERATE_PROMPT": "请介绍一下人工智能。",
    "GENERATE_MAX_NEW_TOKENS": 80,
    "GENERATE_TEMPERATURE": 0.7,
    "GENERATE_TOP_K": 5,

    # Standalone generate.py defaults
    "MAX_NEW_TOKENS": 512,
    "TEMPERATURE": 0.7,
    "TOP_K": 5,
    "STREAM": True,
}


# =============================================================================
# Export active config with the variable names used by train.py and generate.py.
# =============================================================================

CONFIG_TABLE = {
    ("pretrain", "test"): PRETRAIN_TEST_CONFIG,
    ("pretrain", "train"): PRETRAIN_TRAIN_CONFIG,
    ("sft", "test"): SFT_TEST_CONFIG,
    ("sft", "train"): SFT_TRAIN_CONFIG,
}

try:
    ACTIVE_CONFIG = CONFIG_TABLE[(TRAIN_STAGE, CONFIG_MODE)]
except KeyError as exc:
    raise ValueError(
        'TRAIN_STAGE must be "pretrain" or "sft", and CONFIG_MODE must be "test" or "train"'
    ) from exc

DATA_PATH = ACTIVE_CONFIG["DATA_PATH"]
TRAIN_TEXT_PATH = DATA_PATH
SFT_INIT_CHECKPOINT_PATH = ACTIVE_CONFIG.get("SFT_INIT_CHECKPOINT_PATH", "")

SEED = ACTIVE_CONFIG["SEED"]
NUM_WORKERS = ACTIVE_CONFIG["NUM_WORKERS"]
USE_AMP = ACTIVE_CONFIG["USE_AMP"]

BATCH_SIZE = ACTIVE_CONFIG["BATCH_SIZE"]
SEQ_LEN = ACTIVE_CONFIG["SEQ_LEN"]
EPOCHS = ACTIVE_CONFIG["EPOCHS"]
LEARNING_RATE = ACTIVE_CONFIG["LEARNING_RATE"]
ACCUMULATION_STEPS = ACTIVE_CONFIG["ACCUMULATION_STEPS"]
WARMUP_ITERS = ACTIVE_CONFIG["WARMUP_ITERS"]
GRAD_CLIP = ACTIVE_CONFIG["GRAD_CLIP"]

DIM_EMBEDDING = ACTIVE_CONFIG["DIM_EMBEDDING"]
N_HEADS = ACTIVE_CONFIG["N_HEADS"]
N_LAYERS = ACTIVE_CONFIG["N_LAYERS"]
N_KV_HEADS = ACTIVE_CONFIG["N_KV_HEADS"]
NORM_EPS = ACTIVE_CONFIG["NORM_EPS"]
DROPOUT = ACTIVE_CONFIG["DROPOUT"]
FLASH_ATTN = ACTIVE_CONFIG["FLASH_ATTN"]
MULTIPLE_OF = ACTIVE_CONFIG["MULTIPLE_OF"]

CHECKPOINT_PREFIX = ACTIVE_CONFIG["CHECKPOINT_PREFIX"]
CHECKPOINT_PATH = ACTIVE_CONFIG["CHECKPOINT_PATH"]
LOG_INTERVAL = ACTIVE_CONFIG["LOG_INTERVAL"]
SAVE_EVERY_STEPS = ACTIVE_CONFIG["SAVE_EVERY_STEPS"]
SAVE_EVERY_EPOCHS = ACTIVE_CONFIG["SAVE_EVERY_EPOCHS"]

GENERATE_EVERY_STEPS = ACTIVE_CONFIG["GENERATE_EVERY_STEPS"]
GENERATE_PROMPT = ACTIVE_CONFIG["GENERATE_PROMPT"]
GENERATE_MAX_NEW_TOKENS = ACTIVE_CONFIG["GENERATE_MAX_NEW_TOKENS"]
GENERATE_TEMPERATURE = ACTIVE_CONFIG["GENERATE_TEMPERATURE"]
GENERATE_TOP_K = ACTIVE_CONFIG["GENERATE_TOP_K"]

MAX_NEW_TOKENS = ACTIVE_CONFIG["MAX_NEW_TOKENS"]
TEMPERATURE = ACTIVE_CONFIG["TEMPERATURE"]
TOP_K = ACTIVE_CONFIG["TOP_K"]
STREAM = ACTIVE_CONFIG["STREAM"]
