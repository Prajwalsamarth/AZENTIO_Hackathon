"""Tunable configuration for the fraud detector.

Everything a reviewer would want to argue about lives here as data, so tuning
is never a code change.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "cleaned_data"
OUT_DIR = BASE_DIR / "outputs"
ARTIFACT_DIR = OUT_DIR / "artifacts"
SFT_DIR = OUT_DIR / "sft"
ADAPTER_DIR = OUT_DIR / "adapters"

RANDOM_SEED = 42

# --- score fusion ---------------------------------------------------------
# Rules weigh most because they are the only interpretable component.
FUSION_WEIGHTS = {"rule": 0.4, "iforest": 0.3, "autoencoder": 0.3}

# Transactions at or above this fused percentile go to the SLM. Hard-rule hits
# go regardless. ~15% of 988 -> ~150 SLM calls.
TRIAGE_PERCENTILE = 0.85

# Fused score at/above which the deterministic fallback calls a row fraud,
# used when the SLM is unavailable or returns unusable output.
FALLBACK_FRAUD_THRESHOLD = 0.75

# --- anomaly models -------------------------------------------------------
IFOREST_PARAMS = {"n_estimators": 300, "contamination": "auto",
                  "random_state": RANDOM_SEED, "n_jobs": -1}
# Fitted on the ~48 continuous/derived features only (not the one-hots), so the
# bottleneck is sized for that subspace rather than the full 94-dim vector.
AUTOENCODER_PARAMS = {"hidden_layer_sizes": (16, 4, 16), "random_state": RANDOM_SEED,
                      "max_iter": 800, "early_stopping": False}

# --- SLM ------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434"
SLM_MODEL = "llama3.2:1b"        # 1.2B -- under the brief's 3B cap
TEACHER_MODEL = "qwen3:14b"      # local, used only to build training labels
SLM_OPTIONS = {"temperature": 0.0, "num_predict": 220, "top_p": 1.0, "seed": RANDOM_SEED}
SLM_TIMEOUT_S = 120
SLM_MAX_RETRIES = 1              # one repair attempt, then deterministic fallback

# The brief's required output schema, enforced by Ollama structured outputs.
#
# Field order matters and is deliberate. Structured decoding emits properties in
# schema order, so putting `is_fraud` first forces the model to commit to a
# boolean before it has written a single word of analysis -- it is guessing, and
# the justification then rationalises whatever it already said. Emitting the
# justification first makes that text serve as chain-of-thought the boolean is
# conditioned on.
#
# Measured on 5 transactions with llama3.2:1b: is_fraud-first got 1 of 3
# hard-rule fraud cases right and produced verdicts that flatly contradicted
# their own justification; justification-first got 3 of 3 and stayed coherent.
# The consumer reads fields by name, so the order is invisible downstream.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "justification": {"type": "string"},
        "is_fraud": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["justification", "is_fraud", "confidence"],
}

# --- domain knowledge -----------------------------------------------------
HIGH_RISK_CATEGORIES = {"GAMBLING", "CRYPTO", "GIFT_CARD", "MONEY_TRANSFER"}
HOME_COUNTRY = "IN"
NIGHT_HOURS = (0, 5)
FAR_FROM_HOME_KM = 500.0
HIGH_VALUE_AMOUNT = 10000.0
