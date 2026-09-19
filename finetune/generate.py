"""Dataset construction: select rows, build counterfactuals, write the JSONL.

Two payloads exist here and they must not be confused:

  * the OPENAI payload -- built by redact.project(), carries only allowlisted
    non-identifying facts, and is the ONLY thing that leaves this machine.
  * the TRAINING prompt -- built by slm.build_messages(), identical to what the
    detector sends at inference, and never transmitted anywhere.

Keeping the training prompt byte-identical to the inference prompt matters:
prompt skew between training and serving is the most common way a small
fine-tune silently fails to transfer.
"""

import json
import os
import random
import urllib.error
import urllib.request

import pandas as pd

import ft_config
import redact
import counterfactual as cf

import slm as slm_mod

SYSTEM_PROMPT = (
    "You write one-sentence justifications for bank fraud decisions. "
    "You are given de-identified transaction facts, the detector findings, and "
    "the CORRECT verdict. Your job is only to explain that verdict in one plain "
    "sentence citing the specific facts that support it, and to give a "
    "calibrated confidence. Do not dispute the verdict. Do not invent facts."
)

JUSTIFICATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "justification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "justification": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["justification", "confidence"],
            "additionalProperties": False,
        },
    },
}


# Rules that define the fraud class -- the signals the counterfactual
# interventions create. A negative example must never carry one of these, or it
# becomes a direct contradiction of the positives.
STRONG_RULE_CODES = {
    "FOREIGN_MERCHANT", "HIGH_RISK_CATEGORY", "IMPOSSIBLE_TRAVEL",
    "AMOUNT_RATIO_EXTREME", "AMOUNT_RATIO_HIGH", "NO_AUTH_HIGH_VALUE",
    "VELOCITY_24H", "RAPID_SUCCESSION", "ORPHAN_ACCOUNT",
}

# Weak signals a legitimate transaction can plausibly trip: a new phone, a late
# night, a pending KYC renewal. These are what near-miss negatives are made of.

# --- row selection --------------------------------------------------------

def select_rows(pipeline, transactions: pd.DataFrame, rng: random.Random):
    """Pick fraud factuals, clean negatives, and near-miss negatives."""
    prepared = []
    for record in transactions.to_dict(orient="records"):
        try:
            prepared.append((record, pipeline._prepare(record)))
        except Exception:
            continue

    threshold = pipeline.triage_threshold
    median = pd.Series([p["scores"]["fused"] for _, p in prepared]).median()

    # Factuals for intervention: ordinary, well-formed, low-risk transactions.
    factual_pool = [
        (r, p) for r, p in prepared
        if p["scores"]["fused"] < median
        and not p["rule_info"]["rule_codes"]
        and r.get("amount") and r.get("amount_to_account_avg_ratio")
    ]
    clean_pool = list(factual_pool)

    # Near-misses: real rows that DO trip a rule or two but stay below triage.
    # These are the examples that teach "one weak signal is not fraud" -- the
    # direct antidote to the rubber-stamping seen in stage 2.
    #
    # They must trip ONLY weak rules. A row tripping FOREIGN_MERCHANT and
    # IMPOSSIBLE_TRAVEL is below the triage threshold on fused score, so it
    # passes a naive filter -- but those are the very signals the counterfactual
    # fraud rows are built from. Labelling such a row "not fraud" teaches the
    # model the exact opposite of the positives and poisons the contrast the
    # whole counterfactual method depends on.
    near_miss_pool = [
        (r, p) for r, p in prepared
        if 1 <= len(p["rule_info"]["rule_codes"]) <= 2
        and p["scores"]["fused"] < threshold
        and not (set(p["rule_info"]["rule_codes"]) & STRONG_RULE_CODES)
    ]

    rng.shuffle(factual_pool)
    rng.shuffle(near_miss_pool)
    factuals = factual_pool[:ft_config.N_FRAUD]
    clean = [x for x in clean_pool if x not in factuals][:ft_config.N_CLEAN]
    near_miss = near_miss_pool[:ft_config.N_NEAR_MISS]
    return factuals, clean, near_miss


# --- item construction ----------------------------------------------------

def _item(pipeline, record, prepared, label, kind, extra=None):
    """One dataset row: local training prompt + redacted outbound payload."""
    return {
        "synthetic_id": None,
        "kind": kind,
        "is_fraud": bool(label),
        "rule_codes": prepared["rule_info"]["rule_codes"],
        "rule_reasons": prepared["rule_info"]["rule_reasons"],
        "scores": prepared["scores"],
        "triaged": prepared["triaged"],
        # Stays local -- identical to what the detector builds at inference.
        "messages": slm_mod.build_messages(
            prepared["clean"], prepared["rule_info"], prepared["scores"],
            prepared["injection_suspected"]),
        # Leaves the machine -- allowlisted fields only.
        "payload": redact.project(prepared["clean"]),
        **(extra or {}),
    }


def build_items(pipeline, transactions: pd.DataFrame, rng: random.Random):
    factuals, clean, near_miss = select_rows(pipeline, transactions, rng)
    averages = cf.account_average_amounts(transactions)
    items = []

    for i, (record, _) in enumerate(factuals, 1):
        average = averages.get(record.get("account_id")) or 1000.0
        row = cf.make_counterfactual(record, average, rng)
        row["transaction_id"] = f"SYN_F_{i:03d}"
        prepared = pipeline._prepare(row)
        item = _item(pipeline, row, prepared, True, "counterfactual_fraud",
                     extra={"recipes": row["_recipes"],
                            "factual_scores": None,
                            "diff": {k: v for k, v in
                                     cf.changed_fields(record, row).items()
                                     if k != "transaction_id"}})
        item["synthetic_id"] = row["transaction_id"]
        items.append(item)

    for i, (record, prepared) in enumerate(clean, 1):
        item = _item(pipeline, record, prepared, False, "clean_negative")
        item["synthetic_id"] = f"SYN_N_{i:03d}"
        items.append(item)

    for i, (record, prepared) in enumerate(near_miss, 1):
        item = _item(pipeline, record, prepared, False, "near_miss_negative")
        item["synthetic_id"] = f"SYN_M_{i:03d}"
        items.append(item)

    return items


# --- OpenAI justification -------------------------------------------------

def build_generation_prompt(item: dict) -> str:
    verdict = "FRAUD" if item["is_fraud"] else "NOT FRAUD"
    lines = [f"Correct verdict: {verdict}", "", "De-identified transaction facts:"]
    lines += [f"- {k}: {v}" for k, v in item["payload"].items()]
    lines += ["", "Detector findings:"]
    lines += [f"- {r}" for r in item["rule_reasons"]] or ["- none"]
    scores = item["scores"]
    lines += ["", "Anomaly percentiles: "
              f"rule {scores['rule']:.2f}, isolation forest {scores['iforest']:.2f}, "
              f"autoencoder {scores['autoencoder']:.2f}, combined {scores['fused']:.2f}",
              "",
              "Write the one-sentence justification for the verdict above, and a "
              "confidence between 0 and 1."]
    return "\n".join(lines)


def request_justification(prompt: str, api_key: str, name_pattern) -> dict:
    """POST to OpenAI over stdlib urllib. No SDK, no extra dependency."""
    # Fail-closed check, immediately before the bytes leave the process.
    redact.assert_clean(prompt, name_pattern, context="OpenAI generation request")

    body = json.dumps({
        "model": ft_config.OPENAI_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
        "temperature": 0.3,
        "response_format": JUSTIFICATION_SCHEMA,
    }).encode()

    request = urllib.request.Request(
        ft_config.OPENAI_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    last_error = None
    for _ in range(ft_config.OPENAI_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=ft_config.OPENAI_TIMEOUT_S) as response:
                payload = json.loads(response.read())
            content = payload["choices"][0]["message"]["content"]
            return json.loads(content)
        except (urllib.error.URLError, OSError, KeyError,
                json.JSONDecodeError, IndexError) as exc:
            last_error = exc
    raise RuntimeError(f"OpenAI generation failed: {type(last_error).__name__}: {last_error}")


def fallback_justification(item: dict) -> dict:
    """Deterministic text, used for --dry-run and if generation is unavailable."""
    reasons = item["rule_reasons"][:3]
    if item["is_fraud"]:
        text = ("Fraudulent: " + "; ".join(reasons) + ".") if reasons else \
               "Fraudulent: the transaction is a severe outlier against the account's history."
        confidence = round(min(0.97, 0.75 + item["scores"]["fused"] / 5), 2)
    else:
        text = ("Legitimate: only minor signals present (" + "; ".join(reasons) + ").") \
            if reasons else "Legitimate: consistent with the account's normal pattern."
        confidence = round(min(0.95, 0.70 + (1 - item["scores"]["fused"]) / 5), 2)
    return {"justification": text, "confidence": confidence}


# --- output ---------------------------------------------------------------

def write_payloads(items):
    ft_config.PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for item in items:
        path = ft_config.PAYLOAD_DIR / f"{item['synthetic_id']}.json"
        path.write_text(json.dumps({
            "synthetic_id": item["synthetic_id"],
            "kind": item["kind"],
            "is_fraud": item["is_fraud"],
            "outbound_prompt": build_generation_prompt(item),
        }, indent=2, default=str))
    return ft_config.PAYLOAD_DIR


def write_jsonl(items, rng: random.Random):
    """Stratified 80/20 split in mlx-lm chat format."""
    ft_config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in items:
        completion = json.dumps({
            "justification": item["justification"],
            "is_fraud": item["is_fraud"],
            "confidence": item["confidence"],
        })
        rows.append({"is_fraud": item["is_fraud"], "item": item,
                     "row": {"messages": item["messages"]
                             + [{"role": "assistant", "content": completion}]}})

    # Stratify so the 1:2 fraud ratio survives into both splits -- with 6
    # validation rows, an unstratified split could contain zero fraud.
    fraud = [r for r in rows if r["is_fraud"]]
    legit = [r for r in rows if not r["is_fraud"]]
    rng.shuffle(fraud)
    rng.shuffle(legit)

    n_fraud_valid = max(1, round(len(fraud) * ft_config.VALID_FRACTION))
    n_legit_valid = max(1, round(len(legit) * ft_config.VALID_FRACTION))
    valid = fraud[:n_fraud_valid] + legit[:n_legit_valid]
    train = fraud[n_fraud_valid:] + legit[n_legit_valid:]
    rng.shuffle(valid)
    rng.shuffle(train)

    # Record which split each row landed in. Without this, "held-out
    # performance" has to be reconstructed by string-matching justifications
    # back to the JSONL, which over-matches and quietly inflates the sample.
    for entry in valid:
        entry["item"]["split"] = "valid"
    for entry in train:
        entry["item"]["split"] = "train"

    for name, chunk in (("train", train), ("valid", valid), ("test", valid)):
        with open(ft_config.DATA_DIR / f"{name}.jsonl", "w") as handle:
            for entry in chunk:
                handle.write(json.dumps(entry["row"]) + "\n")

    return {"train": len(train), "valid": len(valid), "dir": str(ft_config.DATA_DIR)}


def api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in the shell before running "
            "generation, or use --dry-run to build the dataset without any "
            "network call."
        )
    return key
