#!/usr/bin/env python3
"""Fine-tune pipeline CLI (separate from the scoring pipeline).

  run_finetune.py generate --dry-run   build the 30-row dataset, NO network
  run_finetune.py generate             same, with OpenAI-written justifications
  run_finetune.py train                LoRA-tune and save the adapter
  run_finetune.py evaluate             base vs fine-tuned comparison
"""

import argparse
import json
import random
import sys
from pathlib import Path

FT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FT_DIR))          # finetune modules
sys.path.insert(0, str(FT_DIR.parent))   # stage 1 + 2 modules

import pandas as pd

import config
import ft_config
import generate as gen
import redact
from detector import FraudPipeline


def _load(table):
    dates = {"transactions": ["transaction_timestamp"],
             "accounts": ["open_date", "close_date", "last_login_date"],
             "customers": ["date_of_birth", "customer_since"]}
    return pd.read_csv(config.CLEAN_DIR / f"cleaned_{table}.csv",
                       parse_dates=dates[table])


def cmd_generate(args):
    rng = random.Random(ft_config.SEED)
    pipeline = FraudPipeline.load(use_slm=False)
    transactions = _load("transactions")
    customers = _load("customers")
    name_pattern = redact.build_name_scanner(customers)

    print(f"  building {ft_config.N_FRAUD} counterfactual fraud + "
          f"{ft_config.N_CLEAN} clean + {ft_config.N_NEAR_MISS} near-miss rows")
    items = gen.build_items(pipeline, transactions, rng)

    path = gen.write_payloads(items)
    print(f"  outbound payloads written for inspection -> {path}")

    # Verify the PII boundary on every payload before any of them can be sent.
    for item in items:
        redact.assert_clean(gen.build_generation_prompt(item), name_pattern,
                            context=f"payload {item['synthetic_id']}")
    print(f"  PII scan: {len(items)} payloads clean "
          f"(allowlist {len(redact.ALLOWED_FIELDS)} fields, "
          f"{len(redact.WITHHELD)} withheld)")

    if args.dry_run:
        print("  --dry-run: no network call made; using deterministic justifications")
        for item in items:
            item.update(gen.fallback_justification(item))
    else:
        key = gen.api_key()
        print(f"  generating justifications with {ft_config.OPENAI_MODEL}")
        for i, item in enumerate(items, 1):
            prompt = gen.build_generation_prompt(item)
            result = gen.request_justification(prompt, key, name_pattern)
            item["justification"] = result["justification"].strip()
            item["confidence"] = round(min(1.0, max(0.0, float(result["confidence"]))), 3)
            if i % 10 == 0:
                print(f"    {i}/{len(items)}", flush=True)

    counts = gen.write_jsonl(items, rng)
    manifest = [{k: v for k, v in item.items() if k != "messages"} for item in items]
    (ft_config.ARTIFACT_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))

    fraud = sum(i["is_fraud"] for i in items)
    print(f"\n  {len(items)} rows ({fraud} fraud / {len(items) - fraud} non-fraud)")
    print(f"  train {counts['train']} / valid {counts['valid']} -> {counts['dir']}")


def cmd_train(args):
    import train as trainer
    result = trainer.train()
    report_path = ft_config.ARTIFACT_DIR / "train_result.json"
    report_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n  exit {result['returncode']}")
    print(f"  {result['overfit_check']}")
    print(f"  adapter saved -> {result['adapter_path']}")


def cmd_evaluate_real(args):
    """Score REAL triaged transactions with base vs fine-tuned.

    This is where the discrimination metric belongs. The synthetic rows cannot
    answer "did the rubber-stamping improve" -- they are 10 constructed frauds
    and 20 constructed negatives, so any model that separates them at all looks
    good. The 153 rows stage 2 actually triaged are the real test.
    """
    import evaluate_ft as ev
    import serve

    pipeline = FraudPipeline.load(use_slm=False)
    transactions = _load("transactions")
    scores = pd.read_csv(config.OUT_DIR / "fraud_scores.csv")
    triaged_ids = set(scores[scores.triaged.fillna(False)].transaction_id)

    rows = [r for r in transactions.to_dict(orient="records")
            if r["transaction_id"] in triaged_ids]
    if args.limit:
        rows = rows[:args.limit]

    items = []
    for record in rows:
        prepared = pipeline._prepare(record)
        items.append({
            "transaction_id": record["transaction_id"],
            "synthetic_id": record["transaction_id"],
            "is_fraud": None,                 # no ground truth on real data
            "rule_codes": prepared["rule_info"]["rule_codes"],
            "rule_reasons": prepared["rule_info"]["rule_reasons"],
            "messages": __import__("slm").build_messages(
                prepared["clean"], prepared["rule_info"], prepared["scores"],
                prepared["injection_suspected"]),
        })

    report = {"n_real_triaged": len(items)}
    for name, client in (("base", serve.base_client()), ("tuned", serve.tuned_client())):
        print(f"  scoring {len(items)} real triaged rows with the {name} model")
        block = ev.score_client(client, items, name, progress=25)
        block["discrimination"] = ev.discrimination(
            config.OUT_DIR / "fraud_scores.csv", block["verdicts"])
        report[name] = block

    path = ft_config.ARTIFACT_DIR / "ft_report_real.json"
    path.write_text(json.dumps(report, indent=2, default=str))

    print(f"\n  {'model':6} {'JSON ok':>8} {'fraud rate':>11} {'gradient':>9} {'s/rec':>7}")
    for name in ("base", "tuned"):
        block = report[name]
        verdicts = block["verdicts"]
        rate = sum(v["is_fraud"] for v in verdicts) / max(1, len(verdicts))
        gradient = block["discrimination"].get("_gradient", 0.0)
        print(f"  {name:6} {block['raw_json_validity']:>7.0%} {rate:>11.0%} "
              f"{gradient:>9.2f} {block['mean_latency_s']:>7.2f}")
    print(f"\n  report -> {path}")


def cmd_evaluate(args):
    import evaluate_ft as ev
    import serve

    manifest = json.loads((ft_config.ARTIFACT_DIR / "dataset_manifest.json").read_text())
    pipeline = FraudPipeline.load(use_slm=False)
    transactions = _load("transactions")

    # Held-out rows are rebuilt through the same path so prompts match exactly.
    rng = random.Random(ft_config.SEED)
    items = gen.build_items(pipeline, transactions, rng)
    by_id = {i["synthetic_id"]: i for i in items}
    for entry in manifest:
        item = by_id.get(entry["synthetic_id"])
        if item:
            item["is_fraud"] = entry["is_fraud"]

    sample = list(by_id.values())[:args.limit] if args.limit else list(by_id.values())

    report = {"dataset": {"n": len(sample)}}
    print("  scoring with the BASE model")
    report["base"] = ev.score_client(serve.base_client(), sample, "base")
    print("  scoring with the FINE-TUNED model")
    report["tuned"] = ev.score_client(serve.tuned_client(), sample, "tuned")

    for key in ("base", "tuned"):
        report[key]["discrimination"] = ev.discrimination(
            config.OUT_DIR / "fraud_scores.csv", report[key]["verdicts"])

    path = ev.write_report(report)
    held_out = {e["synthetic_id"] for e in manifest if e.get("split") == "valid"}
    for key in ("base", "tuned"):
        block = report[key]
        verdicts = [v for v in block["verdicts"] if v["expected"] is not None]

        def agreement(rows):
            if not rows:
                return None
            return sum(v["is_fraud"] == v["expected"] for v in rows) / len(rows)

        ho = [v for v in verdicts if v["id"] in held_out]
        tr = [v for v in verdicts if v["id"] not in held_out]
        block["held_out_agreement"] = agreement(ho)
        block["held_out_n"] = len(ho)
        block["trained_on_agreement"] = agreement(tr)
        block["trained_on_n"] = len(tr)

        ho_text = (f"{block['held_out_agreement']:.0%} on {len(ho)} held-out"
                   if ho else "no parseable held-out rows")
        tr_text = (f"{block['trained_on_agreement']:.0%} on {len(tr)} trained-on"
                   if tr else "no parseable trained-on rows")
        print(f"\n  {key}: JSON validity {block['raw_json_validity']:.0%} | "
              f"coherence {block['verdict_justification_coherence']:.0%} | "
              f"{block['mean_latency_s']}s/record")
        print(f"        agreement: {ho_text}  /  {tr_text} (contaminated)")
    print(f"\n  report -> {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--dry-run", action="store_true",
                                 help="build the dataset with no network call")
    generate_parser.set_defaults(func=cmd_generate)

    sub.add_parser("train").set_defaults(func=cmd_train)

    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--limit", type=int, default=None)
    evaluate_parser.set_defaults(func=cmd_evaluate)

    real_parser = sub.add_parser("evaluate-real")
    real_parser.add_argument("--limit", type=int, default=None)
    real_parser.set_defaults(func=cmd_evaluate_real)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
