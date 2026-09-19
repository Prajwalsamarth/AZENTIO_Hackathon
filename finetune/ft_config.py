"""Fine-tune pipeline configuration. Contains no secrets.

The API key is read from the OPENAI_API_KEY environment variable at call time
and is never stored here, passed on a command line, or written to disk.
"""

from pathlib import Path

FT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = FT_DIR / "artifacts"
PAYLOAD_DIR = ARTIFACT_DIR / "payloads"
DATA_DIR = ARTIFACT_DIR / "data"
ADAPTER_DIR = ARTIFACT_DIR / "adapters"
REPORT_PATH = ARTIFACT_DIR / "ft_report.json"

SEED = 42

# --- dataset composition -------------------------------------------------
N_FRAUD = 10          # counterfactuals, constructed by intervention
N_CLEAN = 10          # real, low score, no rules triggered
N_NEAR_MISS = 10      # real, 1-2 rules triggered, still below triage
VALID_FRACTION = 0.2  # 24 train / 6 valid, stratified

# --- generation ----------------------------------------------------------
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TIMEOUT_S = 60
OPENAI_MAX_RETRIES = 2

# --- LoRA ----------------------------------------------------------------
# 4 layers / 150 iters over 24 examples is ~12 epochs. Deliberately small:
# mlx-lm's defaults (16 layers, 1000 iters) would memorise a set this size.
BASE_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
# The first run (150 iters, eval every 25) showed validation loss bottoming at
# iter 25 (0.988) and rising monotonically to 1.160 by iter 150 while train loss
# fell to 0.006 -- memorisation of 24 examples, exactly as expected at this
# dataset size. Iterations are now cut to 60 and both eval and checkpointing run
# every 10 steps, so the optimum is measured finely and an early checkpoint can
# actually be recovered rather than merely recommended.
LORA_ARGS = {
    "num_layers": 4,
    "iters": 60,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "steps_per_eval": 10,
    "steps_per_report": 10,
    "save_every": 10,
    "val_batches": -1,
    "max_seq_length": 2048,
}
MASK_PROMPT = True     # loss on the verdict only, not the prompt template

# --- serving -------------------------------------------------------------
MLX_MAX_TOKENS = 220
MLX_TEMPERATURE = 0.0
