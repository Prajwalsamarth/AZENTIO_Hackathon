"""Evaluation.

Important framing: there is no ground truth in this dataset. Every accuracy
figure below is agreement with a *constructed reference* (qwen3:14b teacher
labels), not correctness. Treat these as consistency metrics.
"""

import json

import numpy as np
import pandas as pd

import config
import log

LOG = log.get("evaluate")


def _rates(predicted, reference):
    predicted = np.asarray(predicted, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    tp = int((predicted & reference).sum())
    fp = int((predicted & ~reference).sum())
    fn = int((~predicted & reference).sum())
    tn = int((~predicted & ~reference).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4),
            "agreement": round((tp + tn) / max(1, len(predicted)), 4)}


def detector_agreement(scores_csv=None):
    scores = pd.read_csv(scores_csv or config.OUT_DIR / "fraud_scores.csv")
    columns = ["rule_score", "iforest_score", "autoencoder_score"]
    correlation = scores[columns].corr(method="spearman").round(3)
    return {
        "spearman": correlation.to_dict(),
        "max_offdiagonal": round(float(
            correlation.where(~np.eye(3, dtype=bool)).max().max()), 3),
    }


def score_summary(scores_csv=None):
    scores = pd.read_csv(scores_csv or config.OUT_DIR / "fraud_scores.csv")
    return {
        "n": len(scores),
        "triaged": int(scores.triaged.fillna(False).sum()),
        "flagged_fraud": int(scores.is_fraud.sum()),
        "hard_rule": int(scores.hard_rule.fillna(False).sum()),
        "injection_suspected": int(scores.injection_suspected.fillna(False).sum()),
        "verdict_source": scores.source.value_counts().to_dict(),
        "fused_quantiles": {f"p{int(q*100)}": round(float(scores.fused_score.quantile(q)), 4)
                            for q in (0.5, 0.75, 0.85, 0.95, 0.99)},
    }


def build_report(pipeline_stats=None, extra=None):
    LOG.info(f"building the report from {config.OUT_DIR / 'fraud_scores.csv'}")
    report = {
        "disclaimer": ("This dataset has no ground-truth fraud label. Figures "
                       "below describe the detector's own behaviour -- score "
                       "distributions, detector agreement, throughput -- not "
                       "correctness. The only labelled data in this project is "
                       "the fine-tune set, whose labels are constructed by "
                       "counterfactual intervention (see finetune/)."),
        "scores": score_summary(),
        "detector_agreement": detector_agreement(),
    }
    if pipeline_stats:
        calls = max(1, pipeline_stats.get("slm_calls", 0))
        report["slm"] = {
            **pipeline_stats,
            "mean_latency_s": round(pipeline_stats.get("slm_latency_s", 0.0) / calls, 3),
            "raw_json_validity": round(
                1 - pipeline_stats.get("raw_json_invalid", 0) / calls, 4),
        }
    if extra:
        report.update(extra)
    summary = report["scores"]
    LOG.info(f"{summary['n']} scored | {summary['triaged']} triaged | "
             f"{summary['flagged_fraud']} flagged fraud | "
             f"{summary['hard_rule']} hard-rule hits | "
             f"max off-diagonal detector correlation "
             f"{report['detector_agreement']['max_offdiagonal']}")
    return report
