#!/usr/bin/env python3
"""Verification suite for the fine-tune pipeline."""

import json
import random
import sys
from pathlib import Path

FT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FT_DIR))
sys.path.insert(0, str(FT_DIR.parent))

import pandas as pd

import config
import counterfactual as cf
import ft_config
import generate as gen
import redact
from detector import FraudPipeline

results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"{'  [ok  ]' if condition else '  [FAIL]'} {name}"
          + (f" -- {detail}" if detail else ""))


def main():
    customers = pd.read_csv(config.CLEAN_DIR / "cleaned_customers.csv")
    transactions = pd.read_csv(config.CLEAN_DIR / "cleaned_transactions.csv",
                               parse_dates=["transaction_timestamp"])
    name_pattern = redact.build_name_scanner(customers)
    manifest = json.loads((ft_config.ARTIFACT_DIR / "dataset_manifest.json").read_text())

    print("\n-- PII boundary --")
    payloads = sorted(ft_config.PAYLOAD_DIR.glob("*.json"))
    check("a payload file exists per row", len(payloads) == len(manifest),
          f"{len(payloads)} files")

    blob = "\n".join(p.read_text() for p in payloads)
    findings = redact.scan(blob, name_pattern)
    check("no PII in any outbound payload", not findings,
          str(findings[:5]) if findings else "clean")

    for field in ("first_name", "email", "phone_number", "ip_address",
                  "postal_code", "cust_city", "cust_age", "cust_annual_income",
                  "merchant_name", "customer_id"):
        check(f"withheld field absent: {field}", field not in blob)

    # The scanner must actually fire -- a clean result is only meaningful if
    # the detector is known to work.
    planted = "Correct verdict: FRAUD\namount: 500\nreviewed by Siddharth Rao"
    try:
        redact.assert_clean(planted, name_pattern)
        check("scanner raises on a planted customer name", False, "did NOT raise")
    except redact.PIILeakError:
        check("scanner raises on a planted customer name", True)

    for label, text in (("email", "contact a@b.com"), ("ipv4", "49.57.192.223"),
                        ("identifier", "see TXN_0000176")):
        try:
            redact.assert_clean(text, name_pattern)
            check(f"scanner raises on {label}", False, "did NOT raise")
        except redact.PIILeakError:
            check(f"scanner raises on {label}", True)

    print("\n-- counterfactual construction --")
    fraud = [r for r in manifest if r["kind"] == "counterfactual_fraud"]
    near_miss = [r for r in manifest if r["kind"] == "near_miss_negative"]
    clean = [r for r in manifest if r["kind"] == "clean_negative"]

    check("dataset is 30 rows: 10 fraud / 20 non-fraud",
          len(manifest) == 30 and len(fraud) == 10
          and len(near_miss) + len(clean) == 20,
          f"{len(fraud)}F/{len(near_miss)}M/{len(clean)}N")

    pipeline = FraudPipeline.load(use_slm=False)
    threshold = pipeline.triage_threshold

    check("every fraud row clears the triage threshold",
          all(r["scores"]["fused"] >= threshold for r in fraud),
          f"min {min(r['scores']['fused'] for r in fraud):.2f} vs {threshold:.2f}")
    check("every negative stays below the triage threshold",
          all(r["scores"]["fused"] < threshold for r in near_miss + clean),
          f"max {max(r['scores']['fused'] for r in near_miss + clean):.2f}")

    # The contrast the whole method rests on: negatives must not carry the
    # signals that define the positive class.
    leaked = [r["synthetic_id"] for r in near_miss + clean
              if set(r["rule_codes"]) & gen.STRONG_RULE_CODES]
    check("no negative carries a fraud-defining rule", not leaked, str(leaked))

    check("clean negatives trigger no rules at all",
          all(not r["rule_codes"] for r in clean))
    check("near-miss negatives each trigger 1-2 weak rules",
          all(1 <= len(r["rule_codes"]) <= 2 for r in near_miss))

    check("every fraud row records its recipes",
          all(r.get("recipes") for r in fraud))
    check("every fraud row records a non-empty diff from its factual",
          all(r.get("diff") for r in fraud))

    # Invariants: regenerate and assert each constructed row satisfies the same
    # consistency rules stage 1 verified on the cleaned data.
    rng = random.Random(ft_config.SEED)
    averages = cf.account_average_amounts(transactions)
    factuals, _, _ = gen.select_rows(pipeline, transactions, rng)
    violations = []
    for record, _ in factuals:
        average = averages.get(record.get("account_id")) or 1000.0
        try:
            row = cf.make_counterfactual(record, average, rng)
            cf.validate(row, average)
        except cf.InconsistentRow as exc:
            violations.append(str(exc))
    check("all generated rows satisfy stage-1 consistency invariants",
          not violations, str(violations[:2]))

    print("\n-- dataset files --")
    counts = {}
    for name in ("train", "valid"):
        path = ft_config.DATA_DIR / f"{name}.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        counts[name] = rows
        check(f"{name}.jsonl parses as chat-format JSONL",
              all("messages" in r for r in rows), f"{len(rows)} rows")

    check("split is 24 train / 6 valid",
          len(counts["train"]) == 24 and len(counts["valid"]) == 6,
          f"{len(counts['train'])}/{len(counts['valid'])}")

    def fraud_rate(rows):
        labels = [json.loads(r["messages"][-1]["content"])["is_fraud"] for r in rows]
        return sum(labels) / max(1, len(labels))

    check("the fraud ratio survives into both splits (stratified)",
          0 < fraud_rate(counts["valid"]) < 1 and 0 < fraud_rate(counts["train"]) < 1,
          f"train {fraud_rate(counts['train']):.2f}, valid {fraud_rate(counts['valid']):.2f}")

    # Prompt skew between training and inference is the classic silent failure.
    sample = counts["train"][0]
    check("training prompts carry the detector's own system prompt",
          sample["messages"][0]["role"] == "system"
          and "fraud analyst" in sample["messages"][0]["content"])
    check("every training row ends with an assistant verdict",
          all(r["messages"][-1]["role"] == "assistant" for r in counts["train"]))
    check("every assistant turn is valid JSON with the required keys",
          all(set(json.loads(r["messages"][-1]["content"]))
              == {"justification", "is_fraud", "confidence"}
              for r in counts["train"] + counts["valid"]))

    passed, total = sum(results), len(results)
    print(f"\n  {passed}/{total} checks passed\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
