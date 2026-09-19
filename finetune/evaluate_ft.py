"""Base vs fine-tuned comparison.

The headline metric is DISCRIMINATION, not accuracy. Stage 2's base model
called fraud at 0.88 in the lowest triaged score band and 0.97 in the highest
-- barely a gradient, which is the rubber-stamping this fine-tune targets. A
flatter gradient after tuning means it made things worse.
"""

import json

import pandas as pd

import ft_config
import slm as slm_mod


def _verdict(client, item):
    """One adjudication attempt -> (verdict or None, raw_json_valid, latency)."""
    rule_info = {"rule_reasons": item["rule_reasons"], "rule_codes": item["rule_codes"],
                 "hard_rule": False}
    messages = item["messages"]
    try:
        content, elapsed = client.chat(messages)
    except Exception as exc:
        return None, False, 0.0, f"{type(exc).__name__}: {exc}"
    try:
        payload = json.loads(content)
        return slm_mod._coerce(payload), True, elapsed, None
    except Exception as exc:
        return None, False, elapsed, f"{type(exc).__name__}: {exc}"


def score_client(client, items, label: str, progress: int = 10) -> dict:
    valid, coherent, latencies, verdicts = 0, 0, [], []
    for i, item in enumerate(items, 1):
        verdict, ok, elapsed, _ = _verdict(client, item)
        latencies.append(elapsed)
        if ok and verdict is not None:
            valid += 1
            verdicts.append({"id": item.get("synthetic_id") or item.get("transaction_id"),
                             "is_fraud": verdict["is_fraud"],
                             "confidence": verdict["confidence"],
                             "justification": verdict["justification"],
                             "expected": item.get("is_fraud")})
            if _coherent(verdict):
                coherent += 1
        if progress and i % progress == 0:
            print(f"    {label}: {i}/{len(items)}", flush=True)

    agreed = [v for v in verdicts if v["expected"] is not None
              and v["is_fraud"] == v["expected"]]
    return {
        "n": len(items),
        "raw_json_validity": round(valid / max(1, len(items)), 4),
        "verdict_justification_coherence": round(coherent / max(1, valid), 4),
        "label_agreement": round(len(agreed) / max(1, len([v for v in verdicts
                                                           if v["expected"] is not None])), 4),
        "mean_latency_s": round(sum(latencies) / max(1, len(latencies)), 3),
        "verdicts": verdicts,
    }


FRAUD_WORDS = ("fraud", "suspicious", "unauthorized", "unauthorised", "illegitimate")
SAFE_WORDS = ("legitimate", "normal", "consistent", "not fraud", "benign", "cleared")


def _coherent(verdict: dict) -> bool:
    """Does the prose agree with the boolean? A 1B model often disagrees with itself."""
    text = verdict["justification"].lower()
    says_fraud = any(w in text for w in FRAUD_WORDS) and "not fraud" not in text
    says_safe = any(w in text for w in SAFE_WORDS)
    if says_fraud and not says_safe:
        return verdict["is_fraud"]
    if says_safe and not says_fraud:
        return not verdict["is_fraud"]
    return True  # ambiguous prose is not counted against the model


def discrimination(scores_csv, verdicts) -> dict:
    """Fraud rate by fused-score band. A steeper gradient is the goal."""
    frame = pd.read_csv(scores_csv)
    lookup = dict(zip(frame.transaction_id, frame.fused_score))
    rows = [{"fused": lookup.get(v["id"]), "is_fraud": v["is_fraud"]}
            for v in verdicts if lookup.get(v["id"]) is not None]
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    bands = pd.cut(frame.fused, [0, 0.6, 0.7, 0.8, 1.0])
    table = frame.groupby(bands, observed=True).is_fraud.agg(["size", "mean"])
    result = {str(k): {"n": int(v["size"]), "fraud_rate": round(float(v["mean"]), 3)}
              for k, v in table.iterrows()}
    rates = [v["fraud_rate"] for v in result.values()]
    result["_gradient"] = round(max(rates) - min(rates), 3) if rates else 0.0
    return result


def write_report(payload: dict):
    ft_config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ft_config.REPORT_PATH.write_text(json.dumps(payload, indent=2, default=str))
    return ft_config.REPORT_PATH
