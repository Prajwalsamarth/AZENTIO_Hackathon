"""FraudPipeline -- the single path every transaction takes.

  enrich -> sanitize -> featurize -> score -> triage -> reason -> validate

process() handles ONE record. process_batch() maps the same stages over many
and is asserted (in verification) to be element-wise identical to looping
process(), so there is no second implementation that can drift.

Every stage degrades rather than raises: a malformed record, a missing account,
or an unreachable Ollama yields a deterministic rule-based verdict, never an
exception. A fraud detector that crashes on bad input is not a fraud detector.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import anomaly
import config
import features
import log
import rules
import sanitize as sanitize_mod
import slm as slm_mod

ARTIFACT_FILE = "pipeline.pkl"
META_FILE = "pipeline_meta.json"

LOG = log.get("detector")


def _describe(client) -> str:
    """Name a client for the log without assuming which one it is.

    finetune/serve.py's MLXClient mirrors OllamaClient's .available()/.chat()
    and nothing else, so logging reaches for attributes through getattr or the
    drop-in contract breaks.
    """
    name = (getattr(client, "model", None) or getattr(client, "model_name", None)
            or type(client).__name__)
    url = getattr(client, "url", None)
    return f"{name} at {url}" if url else f"{name} (in-process)"


class FraudPipeline:
    def __init__(self, feature_builder=None, ensemble=None, sanitizer=None,
                 accounts_idx=None, customers_idx=None, triage_threshold=None,
                 client=None, use_slm=True):
        self.feature_builder = feature_builder
        self.ensemble = ensemble
        self.sanitizer = sanitizer
        self.accounts_idx = accounts_idx or {}
        self.customers_idx = customers_idx or {}
        self.triage_threshold = triage_threshold
        self.use_slm = use_slm
        self._client = client
        self._slm_ready = None
        self.stats = {"slm_calls": 0, "slm_failures": 0, "fallbacks": 0,
                      "raw_json_invalid": 0, "slm_latency_s": 0.0,
                      # SLM cleared a row the hard rules called fraud. Recorded,
                      # NOT overridden: clearing false positives is exactly the
                      # job the SLM is here to do. But a rising count means the
                      # rules or the model need attention, so it must be visible.
                      "rule_slm_conflicts": 0}

    # --- SLM wiring -------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            self._client = slm_mod.OllamaClient()
        return self._client

    def slm_ready(self) -> bool:
        if not self.use_slm:
            return False
        if self._slm_ready is None:
            LOG.debug(f"probing {_describe(self.client)}")
            self._slm_ready = self.client.available()
            if self._slm_ready:
                LOG.info(f"SLM ready: {_describe(self.client)}")
            else:
                LOG.warning(f"SLM unavailable ({_describe(self.client)}) -- every "
                            f"triaged record will fall back to the deterministic "
                            f"verdict")
        return self._slm_ready

    # --- fit --------------------------------------------------------------
    def fit(self, transactions: pd.DataFrame, accounts: pd.DataFrame,
            customers: pd.DataFrame):
        LOG.info(f"[1/6] indexing {len(accounts)} accounts, {len(customers)} customers")
        self.accounts_idx = features.build_index(accounts, "account_id")
        self.customers_idx = features.build_index(customers, "customer_id")

        LOG.info("[2/6] fitting sanitizer vocabularies")
        self.sanitizer = sanitize_mod.Sanitizer.fit(
            {"transactions": transactions, "accounts": accounts, "customers": customers}
        )
        LOG.debug("vocabulary sizes: "
                  + ", ".join(f"{k}={len(v)}" for k, v in self.sanitizer.vocabularies.items()))

        LOG.info(f"[3/6] enriching {len(transactions)} transactions with account "
                 f"+ customer context")
        enriched = [features.enrich(r, self.accounts_idx, self.customers_idx)
                    for r in transactions.to_dict(orient="records")]

        LOG.info("[4/6] building the feature matrix")
        self.feature_builder = features.FeatureBuilder().fit(enriched)
        X, _, _ = self.feature_builder.transform(enriched)
        LOG.info(f"        feature matrix {X.shape[0]}x{X.shape[1]} "
                 f"({int(np.sum(features.CONTINUOUS_MASK))} of them continuous)")

        LOG.info("[5/6] fitting the anomaly ensemble (isolation forest + autoencoder)")
        self.ensemble = anomaly.AnomalyEnsemble().fit(
            X, autoencoder_mask=np.array(features.CONTINUOUS_MASK))

        # Fix the triage cut against the training distribution so that a single
        # record scored later is judged by the same bar as the corpus was.
        LOG.info("[6/6] fixing the triage threshold against the fitted distribution")
        scored = self.ensemble.score(X)
        rule_scores = np.array([rules.evaluate(r)["rule_score"] for r in enriched])
        fused = anomaly.fuse(rule_scores, scored["iforest"], scored["autoencoder"])
        self.triage_threshold = float(np.quantile(fused, config.TRIAGE_PERCENTILE))
        triaged = int((fused >= self.triage_threshold).sum())
        LOG.info(f"        threshold {self.triage_threshold:.4f} "
                 f"(P{config.TRIAGE_PERCENTILE:.0%}) -> {triaged} of {len(fused)} "
                 f"would triage to the SLM")
        return self

    # --- persistence ------------------------------------------------------
    def save(self, artifacts_dir=None):
        directory = Path(artifacts_dir or config.ARTIFACT_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / ARTIFACT_FILE, "wb") as handle:
            pickle.dump({
                "feature_builder": self.feature_builder.to_dict(),
                "ensemble": self.ensemble,
                "sanitizer": self.sanitizer.to_dict(),
                "accounts_idx": self.accounts_idx,
                "customers_idx": self.customers_idx,
                "triage_threshold": self.triage_threshold,
            }, handle)
        (directory / META_FILE).write_text(json.dumps({
            "n_features": len(features.FEATURE_NAMES),
            "feature_names": features.FEATURE_NAMES,
            "triage_threshold": self.triage_threshold,
            "fusion_weights": config.FUSION_WEIGHTS,
            "triage_percentile": config.TRIAGE_PERCENTILE,
            "slm_model": config.SLM_MODEL,
            "accounts_indexed": len(self.accounts_idx),
            "customers_indexed": len(self.customers_idx),
            "vocabulary_sizes": {k: len(v) for k, v in self.sanitizer.vocabularies.items()},
        }, indent=2))
        LOG.info(f"artifacts written to {directory}")
        return directory

    @classmethod
    def load(cls, artifacts_dir=None, use_slm=True):
        directory = Path(artifacts_dir or config.ARTIFACT_DIR)
        LOG.info(f"loading fitted artifacts from {directory}")
        with open(directory / ARTIFACT_FILE, "rb") as handle:
            payload = pickle.load(handle)

        pipeline = cls(
            feature_builder=features.FeatureBuilder.from_dict(payload["feature_builder"]),
            ensemble=payload["ensemble"],
            accounts_idx=payload["accounts_idx"],
            customers_idx=payload["customers_idx"],
            triage_threshold=payload["triage_threshold"],
            use_slm=use_slm,
        )
        gate = slm_mod.make_gate(pipeline.client) if use_slm else None
        pipeline.sanitizer = sanitize_mod.Sanitizer.from_dict(payload["sanitizer"], gate=gate)
        LOG.info(f"ready: {len(pipeline.accounts_idx)} accounts / "
                 f"{len(pipeline.customers_idx)} customers indexed, "
                 f"triage threshold {pipeline.triage_threshold:.4f}, "
                 f"SLM {'enabled' if use_slm else 'disabled (--no-slm)'}")
        return pipeline

    # --- stages -----------------------------------------------------------
    def _prepare(self, transaction: dict) -> dict:
        """Stages 1-4: enrich, sanitize, featurize, score. No SLM, no I/O."""
        txn_id = transaction.get("transaction_id")
        LOG.debug(f"{txn_id}: enriching")
        enriched = features.enrich(transaction, self.accounts_idx, self.customers_idx)
        clean, findings = self.sanitizer.sanitize_record(enriched)
        suspected = sanitize_mod.Sanitizer.is_suspected(findings)
        if suspected:
            # Security-relevant and rare: visible at the default level.
            tripped = ", ".join(f"{field} ({'/'.join(reasons)})"
                                for field, reasons in sorted(findings.items()))
            LOG.warning(f"{txn_id}: prompt-injection filter fired on {tripped} "
                        f"-- value withheld from the prompt")
        elif findings:
            LOG.debug(f"{txn_id}: sanitizer tidied {sorted(findings)}")

        # Features come from the ENRICHED record, not the sanitized one: the
        # numeric signal is ours and was never attacker-controlled, and
        # redaction must not blind the anomaly models.
        X, _, _ = self.feature_builder.transform([enriched])
        scored = self.ensemble.score(X)
        rule_info = rules.evaluate(enriched)

        fused = float(anomaly.fuse(
            [rule_info["rule_score"]], scored["iforest"], scored["autoencoder"])[0])

        scores = {
            "rule": rule_info["rule_score"],
            "iforest": float(scored["iforest"][0]),
            "autoencoder": float(scored["autoencoder"][0]),
            "fused": fused,
        }
        triaged = fused >= self.triage_threshold or rule_info["hard_rule"]
        LOG.debug(f"{txn_id}: rule {scores['rule']:.2f} iforest {scores['iforest']:.2f} "
                  f"autoencoder {scores['autoencoder']:.2f} -> fused {fused:.2f} "
                  f"(threshold {self.triage_threshold:.2f}"
                  f"{', hard rule' if rule_info['hard_rule'] else ''}) "
                  f"-> {'triage to SLM' if triaged else 'deterministic'}")
        if rule_info["rule_codes"]:
            LOG.debug(f"{txn_id}: rules fired {rule_info['rule_codes']}")
        return {
            "enriched": enriched,
            "clean": clean,
            "findings": findings,
            "injection_suspected": suspected,
            "rule_info": rule_info,
            "scores": scores,
            "triaged": triaged,
        }

    def _fallback_verdict(self, prepared: dict) -> dict:
        """Deterministic verdict -- used below threshold and whenever the SLM fails."""
        scores, rule_info = prepared["scores"], prepared["rule_info"]
        fused = scores["fused"]
        is_fraud = fused >= config.FALLBACK_FRAUD_THRESHOLD or rule_info["hard_rule"]

        reasons = rule_info["rule_reasons"]
        if is_fraud and reasons:
            justification = ("Flagged by deterministic rules: "
                             + "; ".join(reasons[:3]) + ".")
        elif is_fraud:
            justification = (f"Flagged as anomalous by the ensemble "
                             f"(combined score {fused:.2f}) without a specific rule match.")
        elif reasons:
            justification = ("Reviewed and cleared; minor signals only ("
                             + "; ".join(reasons[:2]) + ").")
        else:
            justification = ("Consistent with the account's normal transaction "
                             "pattern; no fraud indicators triggered.")

        # Confidence tracks distance from the decision boundary, so a borderline
        # call is reported as borderline rather than dressed up as certainty.
        confidence = min(0.99, 0.5 + abs(fused - config.FALLBACK_FRAUD_THRESHOLD))
        return {"is_fraud": bool(is_fraud), "confidence": round(confidence, 3),
                "justification": justification}

    def _reason(self, prepared: dict) -> tuple:
        """Stage 6: SLM adjudication for triaged records, fallback otherwise."""
        if not prepared["triaged"]:
            return self._fallback_verdict(prepared), "deterministic_below_threshold"

        if not self.slm_ready():
            self.stats["fallbacks"] += 1
            return self._fallback_verdict(prepared), "deterministic_slm_unavailable"

        txn_id = prepared["enriched"].get("transaction_id")
        LOG.debug(f"{txn_id}: asking {_describe(self.client)} to adjudicate")
        verdict, meta = slm_mod.adjudicate(
            self.client, prepared["clean"], prepared["rule_info"],
            prepared["scores"], prepared["injection_suspected"],
        )
        self.stats["slm_calls"] += 1
        self.stats["slm_latency_s"] += meta.get("latency_s", 0.0)
        if meta.get("raw_json_valid") is False:
            self.stats["raw_json_invalid"] += 1

        if verdict is None:
            self.stats["slm_failures"] += 1
            self.stats["fallbacks"] += 1
            LOG.warning(f"{txn_id}: SLM produced no usable verdict after "
                        f"{meta['attempts']} attempt(s) ({meta.get('error')}) -- "
                        f"falling back to the deterministic verdict")
            return self._fallback_verdict(prepared), "deterministic_slm_failed"

        LOG.debug(f"{txn_id}: SLM says fraud={verdict['is_fraud']} "
                  f"confidence={verdict['confidence']} in {meta['latency_s']:.1f}s"
                  f"{' (after a repair attempt)' if meta['attempts'] > 1 else ''}")
        return verdict, "slm"

    def _apply_backstop(self, verdict: dict, prepared: dict) -> tuple:
        """Output-side guard: an injection can never talk the model down.

        If a field was redacted by the sanitizer, the SLM is not permitted to
        downgrade a high ensemble score to 'safe'. The deterministic layer wins.
        """
        if not prepared["injection_suspected"]:
            return verdict, False
        if verdict["is_fraud"]:
            return verdict, False
        if prepared["scores"]["fused"] < config.FALLBACK_FRAUD_THRESHOLD \
                and not prepared["rule_info"]["hard_rule"]:
            return verdict, False

        LOG.warning(f"{prepared['enriched'].get('transaction_id')}: injection backstop "
                    f"engaged -- the SLM cleared a record whose fields were redacted "
                    f"and whose fused score is {prepared['scores']['fused']:.2f}; "
                    f"keeping the deterministic verdict")
        override = self._fallback_verdict(prepared)
        override["justification"] = (
            "Prompt-injection content was detected and removed from this record; "
            "the deterministic verdict is used instead. " + override["justification"]
        )
        return override, True

    # --- public API -------------------------------------------------------
    def process(self, transaction: dict, detail: bool = False) -> dict:
        """ONE transaction -> the brief's exact JSON verdict."""
        transaction_id = transaction.get("transaction_id")
        try:
            prepared = self._prepare(transaction)
            verdict, source = self._reason(prepared)
            verdict, overridden = self._apply_backstop(verdict, prepared)
            conflict = (source == "slm"
                        and prepared["rule_info"]["hard_rule"]
                        and not verdict["is_fraud"])
            if conflict:
                self.stats["rule_slm_conflicts"] += 1
                LOG.info(f"{transaction_id}: rule/SLM conflict -- hard rule flagged it, "
                         f"the SLM cleared it; recorded, not overridden")
        except Exception as exc:  # never let one bad record kill a batch
            LOG.error(f"{transaction_id}: could not be scored "
                      f"({type(exc).__name__}: {exc}) -- routed for manual review",
                      exc_info=LOG.isEnabledFor(logging.DEBUG))
            return self._error_verdict(transaction_id, exc, detail)

        result = {"transaction_id": transaction_id, **verdict}
        if detail:
            result["_detail"] = {
                "source": source,
                "triaged": prepared["triaged"],
                "injection_suspected": prepared["injection_suspected"],
                "injection_overrode_slm": overridden,
                "rule_slm_conflict": conflict,
                "findings": prepared["findings"],
                "scores": prepared["scores"],
                "rule_codes": prepared["rule_info"]["rule_codes"],
                "rule_reasons": prepared["rule_info"]["rule_reasons"],
                "hard_rule": prepared["rule_info"]["hard_rule"],
            }
        return result

    @staticmethod
    def _error_verdict(transaction_id, exc, detail):
        result = {
            "transaction_id": transaction_id,
            "is_fraud": False,
            "confidence": 0.0,
            "justification": ("Could not be scored because the record failed "
                              "processing; routed for manual review."),
        }
        if detail:
            result["_detail"] = {"source": "error", "error": f"{type(exc).__name__}: {exc}"}
        return result

    def process_batch(self, transactions, detail: bool = False, progress=None) -> list:
        """Many transactions -> many verdicts, via the same per-record path."""
        records = (transactions.to_dict(orient="records")
                   if isinstance(transactions, pd.DataFrame) else list(transactions))
        tracker = log.Progress(LOG, len(records), every=progress or 100,
                               unit="transactions") if progress else None
        if tracker:
            LOG.info(f"scoring {len(records)} transactions "
                     f"(~{1 - config.TRIAGE_PERCENTILE:.0%} expected to reach the SLM, "
                     f"plus any hard-rule hit)")
        results = []
        flagged = 0
        for i, record in enumerate(records, 1):
            result = self.process(record, detail=detail)
            results.append(result)
            flagged += bool(result["is_fraud"])
            if tracker:
                tracker.update(i, suffix=f" | {self.stats['slm_calls']} SLM calls, "
                                         f"{flagged} flagged")
        if tracker:
            LOG.info(f"scoring finished in {tracker.done():.1f}s")
        return results
