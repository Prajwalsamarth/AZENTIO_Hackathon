#!/usr/bin/env python3
"""Fraud detector CLI.

  run_fraud.py fit                      fit the ensemble, persist artifacts
  run_fraud.py batch                    score every transaction -> predictions.json
  run_fraud.py single --id TXN_0000176  score one record -> JSON on stdout
  run_fraud.py single --json '{...}'    score an ad-hoc record -> JSON on stdout
  run_fraud.py evaluate                 rebuild fraud_report.json from the scores

Progress goes to stderr, results to stdout, so `single ... | jq` stays clean.
Add -v for per-record scores and every SLM call, -q for warnings only.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config
import log
from detector import FraudPipeline

LOG = log.get("run")

DATES = {
    "transactions": ["transaction_timestamp"],
    "accounts": ["open_date", "close_date", "last_login_date"],
    "customers": ["date_of_birth", "customer_since"],
}


def load_clean(table):
    path = config.CLEAN_DIR / f"cleaned_{table}.csv"
    if not path.exists():
        sys.exit(f"{path} is missing -- run stage 1 first: "
                 f"{sys.executable} run.py")
    frame = pd.read_csv(path, parse_dates=DATES[table])
    LOG.info(f"loaded {len(frame)} {table} from {path.name}")
    return frame


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=str))
    LOG.info(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB)")


def cmd_fit(args):
    LOG.info("fitting the ensemble (no SLM involved in this step)")
    transactions, accounts, customers = (load_clean(t) for t in
                                         ("transactions", "accounts", "customers"))
    pipeline = FraudPipeline(use_slm=False).fit(transactions, accounts, customers)
    directory = pipeline.save()
    LOG.info(f"fit complete in {log.elapsed():.1f}s -- artifacts in {directory}")


def cmd_batch(args):
    pipeline = FraudPipeline.load(use_slm=not args.no_slm)
    transactions = load_clean("transactions")
    if args.limit:
        transactions = transactions.head(args.limit)
        LOG.info(f"--limit {args.limit}: scoring the first {len(transactions)} only")

    pipeline.slm_ready()  # logs ready / unavailable before the run starts
    results = pipeline.process_batch(transactions, detail=True,
                                     progress=args.progress)

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = [{k: v for k, v in r.items() if not k.startswith("_")}
                   for r in results]
    write_json(config.OUT_DIR / "predictions.json", predictions)

    rows = []
    for result in results:
        detail = result.get("_detail", {})
        scores = detail.get("scores", {})
        rows.append({
            "transaction_id": result["transaction_id"],
            "is_fraud": result["is_fraud"],
            "confidence": result["confidence"],
            "rule_score": scores.get("rule"),
            "iforest_score": scores.get("iforest"),
            "autoencoder_score": scores.get("autoencoder"),
            "fused_score": scores.get("fused"),
            "triaged": detail.get("triaged"),
            "source": detail.get("source"),
            "hard_rule": detail.get("hard_rule"),
            "injection_suspected": detail.get("injection_suspected"),
            "rule_slm_conflict": detail.get("rule_slm_conflict"),
            "rule_codes": "|".join(detail.get("rule_codes", [])),
            "justification": result["justification"],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(config.OUT_DIR / "fraud_scores.csv", index=False)
    LOG.info(f"wrote fraud_scores.csv ({len(frame)} rows)")

    stats = pipeline.stats
    flagged = int(frame.is_fraud.sum())
    triaged = int(frame.triaged.fillna(False).sum())
    LOG.info(f"{len(frame)} scored | {triaged} triaged to the SLM | "
             f"{flagged} flagged fraud")
    mean_latency = stats["slm_latency_s"] / max(1, stats["slm_calls"])
    LOG.info(f"SLM calls {stats['slm_calls']} (mean {mean_latency:.1f}s), "
             f"failures {stats['slm_failures']}, fallbacks {stats['fallbacks']}, "
             f"raw JSON invalid {stats['raw_json_invalid']}, "
             f"rule/SLM conflicts {stats['rule_slm_conflicts']}")
    if stats["slm_failures"]:
        LOG.warning(f"{stats['slm_failures']} record(s) fell back because the SLM "
                    f"returned nothing usable -- see the lines above, or re-run "
                    f"with -v")

    import evaluate
    report = evaluate.build_report(pipeline_stats=stats)
    write_json(config.OUT_DIR / "fraud_report.json", report)
    LOG.info(f"batch complete in {log.elapsed():.1f}s -- outputs in {config.OUT_DIR}")


def cmd_evaluate(args):
    import evaluate
    scores_csv = config.OUT_DIR / "fraud_scores.csv"
    if not scores_csv.exists():
        sys.exit(f"{scores_csv} is missing -- run `batch` first")
    report = evaluate.build_report()
    write_json(config.OUT_DIR / "fraud_report.json", report)
    print(json.dumps(report, indent=2, default=str))


def cmd_single(args):
    pipeline = FraudPipeline.load(use_slm=not args.no_slm)
    if args.json:
        record = json.loads(args.json)
        LOG.info(f"scoring the record supplied on the command line "
                 f"({record.get('transaction_id', 'no transaction_id')})")
    else:
        transactions = load_clean("transactions")
        match = transactions[transactions.transaction_id == args.id]
        if match.empty:
            sys.exit(f"transaction {args.id} not found in cleaned_transactions.csv")
        record = match.iloc[0].to_dict()
        LOG.info(f"scoring {args.id}")

    result = pipeline.process(record, detail=args.detail)
    LOG.info(f"verdict: fraud={result['is_fraud']} "
             f"confidence={result['confidence']} (in {log.elapsed():.1f}s)")
    # The verdict itself is the only thing on stdout.
    print(json.dumps(result, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="per-record scores and every SLM call")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="warnings and errors only")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fit").set_defaults(func=cmd_fit)

    sub.add_parser("evaluate").set_defaults(func=cmd_evaluate)

    batch = sub.add_parser("batch")
    batch.add_argument("--limit", type=int, default=None)
    batch.add_argument("--no-slm", action="store_true")
    batch.add_argument("--progress", type=int, default=100,
                       help="log progress every N transactions (default 100)")
    batch.set_defaults(func=cmd_batch)

    single = sub.add_parser("single")
    group = single.add_mutually_exclusive_group(required=True)
    group.add_argument("--id")
    group.add_argument("--json")
    single.add_argument("--detail", action="store_true")
    single.add_argument("--no-slm", action="store_true")
    single.set_defaults(func=cmd_single)

    args = parser.parse_args()
    log.configure(verbose=args.verbose, quiet=args.quiet)
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit("\n  interrupted -- nothing further was written")


if __name__ == "__main__":
    main()
