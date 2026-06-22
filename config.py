# config.py

# Switch between quick local smoke tests and happy-llm style pretraining192819.
# Options: "test", "train"
CONFIG_MODE = "train"


# =============================================================================
# Common paths
# =============================================================================

TOKENIZER_NAME = "tokenizer_k"
TRAIN_TEXT_PATH = "data/seq_monkey_datawhale.jsonl"

LOG_DIR = "runs"
TEXT_LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = "transformer"
CHECKPOINT_PATH = "transformer_final.pt"

RESUME = False
RESUME_CHECKPOINT_PATH = ""


# =============================================================================
# Test config: small CPU-friendly run for checking that the code path works.
# =============================================================================

TEST_CONFIG = {
    # Runtime
    "SEED": 42,
    "NUM_WORKERS": 4,
    "USE_AMP": False,

    # Data and training
    "BATCH_SIZE": 1,
    "SEQ_LEN": 64,
    "EPOCHS": 1,
    "LEARNING_RATE": 1e-4,
    "ACCUMULATION_STEPS": 1,
    "WARMUP_ITERS": 0,
    "GRAD_CLIP": 1.0,

    # Model
    "DIM_EMBEDDING": 128,
    "N_HEADS": 4,
    "N_LAYERS": 2,
    "N_KV_HEADS": 2,
    "NORM_EPS": 1e-5,
    "DROPOUT": 0.0,
    "FLASH_ATTN": False,
    "MULTIPLE_OF": 64,

    # Logging and checkpoints
    "LOG_INTERVAL": 5,
    "SAVE_EVERY_STEPS": 20,
    "SAVE_EVERY_EPOCHS": 1,

    # Generation during training
    "GENERATE_EVERY_STEPS": 10,
    "GENERATE_PROMPT": "\u4eba\u5de5\u667a\u80fd",
    "GENERATE_MAX_NEW_TOKENS": 20,
    "GENERATE_TEMPERATURE": 0.9,
    "GENERATE_TOP_K": 5,

    # Standalone generate.py defaults
    "MAX_NEW_TOKENS": 64,
    "TEMPERATURE": 0.9,
    "TOP_K": 3,
    "STREAM": True,
}


# =============================================================================
# Train config: aligned with happy-llm chapter 5 pretraining defaults.
# =============================================================================

TRAIN_CONFIG = {
    # Runtime
    "SEED": 42,
    "NUM_WORKERS": 4,
    "USE_AMP": True,

    # Data and training
    "BATCH_SIZE": 12,
    "SEQ_LEN": 512,
    "EPOCHS": 2,
    "LEARNING_RATE": 1e-4,
    "ACCUMULATION_STEPS": 8,
    "WARMUP_ITERS": 0,
    "GRAD_CLIP": 1.0,

    # Model
    "DIM_EMBEDDING": 1024,
    "N_HEADS": 16,
    "N_LAYERS": 18,
    "N_KV_HEADS": 8,
    "NORM_EPS": 1e-5,
    "DROPOUT": 0.0,
    "FLASH_ATTN": False,
    "MULTIPLE_OF": 64,

    # Logging and checkpoints
    "LOG_INTERVAL": 100,
    "SAVE_EVERY_STEPS": 5000,
    "SAVE_EVERY_EPOCHS": 1,

    # Generation during training
    "GENERATE_EVERY_STEPS": 500,
    "GENERATE_PROMPT": "\u4eba\u5de5\u667a\u80fd",
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
# Export active config with the variable names used by train.py and generate.py.
# =============================================================================

if CONFIG_MODE == "test":
    ACTIVE_CONFIG = TEST_CONFIG
elif CONFIG_MODE == "train":
    ACTIVE_CONFIG = TRAIN_CONFIG
else:
    raise ValueError('CONFIG_MODE must be "test" or "train"')

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
