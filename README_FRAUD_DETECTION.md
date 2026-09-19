# Fraud Detection — Usage Guide

One transaction in, one verdict out:

```json
{
  "transaction_id": "TXN_0000176",
  "is_fraud": true,
  "confidence": 0.99,
  "justification": "The transaction from BetKing Online, a high-risk merchant category, ..."
}
```

This document is about **using** the detector. For why it is built the way it
is — the three-detector ensemble, the triage cut, the injection defense, the
field-order finding — read `README_FRAUD.md`.

> **It is a library and a CLI, not a network service.** Nothing here listens on
> a port. You call it in-process (`FraudPipeline.process`) or from the command
> line (`run_fraud.py`). If you need HTTP, wrap `process()` yourself — load the
> pipeline once at startup, not per request; see [Using it from Python](#using-it-from-python).

## Prerequisites

| | Needed for | Check |
|---|---|---|
| The project venv | everything | `.venv/bin/python -c "import sklearn, pandas"` |
| `cleaned_data/*.csv` | everything | `ls cleaned_data/cleaned_transactions.csv` |
| `outputs/artifacts/pipeline.pkl` | everything except `fit` | `ls outputs/artifacts/` |
| Ollama + `llama3.2:1b` | the SLM reasoning stage | `curl -s localhost:11434/api/tags` |

Missing stage-1 data? Regenerate it with `.venv/bin/python run.py`. Missing
artifacts? `run_fraud.py fit`. Missing Ollama? Nothing breaks — every triaged
record falls back to a deterministic verdict, and the run says so loudly.

Run everything from the project root with the venv interpreter. The commands
below drop the `.venv/bin/` prefix for readability.

## Quick start

```bash
python run_fraud.py fit                       # once: fit the ensemble (~1s, no SLM)
python run_fraud.py single --id TXN_0000176   # one record  -> JSON on stdout
python run_fraud.py batch                     # all 988     -> outputs/
python tests.py                               # 25-check verification suite
```

## The CLI

```
run_fraud.py [-v | -q] {fit,batch,single,evaluate} [options]
```

| Command | What it does | Time |
|---|---|---|
| `fit` | Fits the ensemble, fixes the triage threshold, writes `outputs/artifacts/` | ~1s |
| `batch` | Scores every transaction, writes all four outputs | ~2min warm, ~5min cold |
| `single --id TXN_x` | Scores one record from the cleaned data | ~1-8s |
| `single --json '{...}'` | Scores an ad-hoc record you supply | ~1-8s |
| `evaluate` | Rebuilds `fraud_report.json` from an existing `fraud_scores.csv` | instant |

Options:

| Flag | Applies to | Effect |
|---|---|---|
| `--no-slm` | `batch`, `single` | Fully deterministic. No Ollama, no network; a full batch drops from minutes to seconds |
| `--limit N` | `batch` | Score only the first N transactions |
| `--progress N` | `batch` | Progress line every N records (default 100) |
| `--detail` | `single` | Adds a `_detail` block: per-detector scores, rule codes, triage decision, sanitizer findings |
| `-v` | all | Per-record scores and every SLM call |
| `-q` | all | Warnings and errors only |

`fit` is only needed once, or after changing anything in `config.py` that
affects features, fusion weights, or the triage percentile. Everything else
loads the persisted artifacts.

### Results and progress go to different streams

Verdicts go to **stdout**, logs to **stderr**, so this is safe:

```bash
python run_fraud.py single --id TXN_0000176 | jq .is_fraud
python run_fraud.py -q single --id TXN_0000176 > verdict.json
```

## Outputs

`batch` writes four files to `outputs/`:

| File | Contents |
|---|---|
| `predictions.json` | The contract: one `{transaction_id, is_fraud, confidence, justification}` per transaction, and nothing else |
| `fraud_scores.csv` | The audit trail: every verdict plus its three component scores, fused score, rule codes, triage flag, verdict source, and conflict/injection flags |
| `fraud_report.json` | Run metrics — score distribution, detector agreement, SLM throughput |
| `artifacts/` | The fitted models (written by `fit`, not `batch`) |

**Read `fraud_report.json` with its own disclaimer in hand.** This dataset has
no ground-truth `is_fraud` column, so nothing in that file measures
correctness — it describes the detector's own behaviour. The report says so in
its `disclaimer` field.

## Using it from Python

This is the integration path. Load once, then call `process()` per record:

```python
import sys; sys.path.insert(0, "/Users/psamarth/datapipeline")
from detector import FraudPipeline

pipeline = FraudPipeline.load()             # ~1s: unpickles models + indexes
verdict = pipeline.process(transaction)     # dict, same shape as the JSON above
```

Loading is the expensive part — unpickling the forest, the autoencoder, and the
account/customer indexes. In a server, do it once at startup and keep the
instance; `process()` is then a few milliseconds plus the SLM call, when the
record triages.

```python
pipeline.process(transaction, detail=True)          # + a _detail block
pipeline.process_batch(records, progress=100)       # a list of dicts
FraudPipeline.load(use_slm=False)                   # deterministic only
```

`process_batch` is asserted in `tests.py` to be element-wise identical to
looping `process`, so there is no second code path that can drift. Use it for
the progress logging, not for speed — it is the same per-record work in a loop.

Two things to know about the instance:

- **It accumulates.** `pipeline.stats` counts SLM calls, failures, fallbacks and
  rule/SLM conflicts across everything that instance has scored. Read it after a
  batch; reset it yourself if you want per-request counters.
- **It is not thread-safe.** Nothing guards `stats`, and the scikit-learn models
  are shared. One instance per worker process, or serialize your calls.

### What one record needs

A record is a plain dict — the columns of `cleaned_transactions.csv`. The
fields that carry the most signal:

```python
{
  "transaction_id": "TXN_0000176",   # used for the ID grammar check and the output
  "account_id": "ACC_000175",        # the join key for account + customer context
  "amount": 213707.14,
  "transaction_timestamp": "2026-04-25 04:49:50",
  "transaction_type": "TRANSFER", "channel": "ONLINE", "status": "SUCCESS",
  "merchant_name": "BetKing Online", "merchant_category": "GAMBLING",
  "merchant_country": "UA", "auth_method": "NONE",
  "is_new_device": True, "distance_from_home_km": 9183.7,
  "amount_to_account_avg_ratio": 213.707, "balance_after_txn": -56058.31,
}
```

Missing fields are not an error. An unknown `account_id` is an orphan: the
account and customer fields stay empty and missing-indicator features carry
that fact into the model — being an orphan is itself a signal. A record that
fails outright still returns a verdict (`is_fraud: false`, `confidence: 0.0`,
"routed for manual review") and logs an `ERROR` naming the record. **Nothing
raises.** A fraud detector that crashes on bad input is not a fraud detector.

## Watching a run

Default verbosity narrates stages, progress with an ETA, and final counts:

```
  [   0.8s] info   detector  SLM ready: llama3.2:1b at http://localhost:11434
  [   0.8s] info   detector  scoring 988 transactions (~15% expected to reach the SLM...)
  [  93.5s] info   detector  scored 900/988 transactions (9.7/s, eta 9s) | 139 SLM calls, 128 flagged
  [ 102.4s] info   run       988 scored | 153 triaged to the SLM | 141 flagged fraud
```

Four things surface as `WARN` at **every** verbosity, including `-q`, because a
run that hides them is misrepresenting what it did:

- the prompt-injection filter firing on a record (named field, named pattern)
- Ollama unreachable, or up but missing the model
- an SLM call returning nothing usable, so the record fell back
- the injection backstop overriding an SLM verdict

`log.py` holds the formatter and the level policy. Importing a module emits
nothing until a CLI calls `log.configure()`, so embedding the pipeline does not
put logs in your application's output unless you ask for them:

```python
import log; log.configure(verbose=1)   # opt in
```

## Swapping the reasoning model

The deterministic stages never change. Only stage 6 — adjudication — is
pluggable.

**A different Ollama model:** set `SLM_MODEL` in `config.py`. Stay under the
brief's 3B cap. No refit needed; the ensemble does not know or care.

**No model at all:** `--no-slm`, or `FraudPipeline.load(use_slm=False)`. Every
record gets the deterministic rule verdict. This is also the fallback path, so
it is exercised on every run.

**The LoRA fine-tuned model** (`finetune/`, MLX, requires a trained adapter):

```python
import sys; sys.path.insert(0, "finetune")
import serve
from detector import FraudPipeline

pipeline = FraudPipeline.load(use_slm=True)
pipeline._client = serve.tuned_client()   # MLXClient mirrors OllamaClient's interface
pipeline._slm_ready = None                # force a re-probe of the new client
```

Two caveats, both verified rather than assumed:

- `_client` is private. There is no public setter today; `FraudPipeline(client=...)`
  takes one but `load()` does not forward it.
- The **injection gate stays on Ollama**. `load()` binds the gate to the client
  that existed at construction time, so swapping `_client` afterwards does not
  move it. With Ollama down, the gate fails closed and unfamiliar strings stay
  redacted — safe, but stricter than you may expect.

Read `finetune/artifacts/ft_report.json` for the base-vs-tuned comparison
before choosing the tuned model for anything real — it is a 1B model LoRA-tuned
on 24 examples, and the report is the only evidence of what that bought.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `WARN ollama unreachable ... is 'ollama serve' running?` | No Ollama | Start it, or accept the deterministic fallback |
| `WARN ollama is up ... but has no llama3.2:1b` | Model not pulled | `ollama pull llama3.2:1b` |
| `cleaned_data/... is missing` | Stage 1 has not run | `python run.py` |
| `FileNotFoundError: .../pipeline.pkl` | Never fitted | `python run_fraud.py fit` |
| `fraud_scores.csv is missing -- run 'batch' first` | `evaluate` before `batch` | `python run_fraud.py batch` |
| Batch is much slower than 2 minutes | Cold model | Expected on the first run; the model stays resident afterwards |
| Every verdict is deterministic | Ollama down, or `--no-slm` | Check the `SLM ready` / `SLM unavailable` line at the top of the run |

## Tuning

Everything adjustable lives in `config.py` as data, so tuning is never a code
change. The knobs you are most likely to reach for:

| Setting | Default | Effect |
|---|---|---|
| `TRIAGE_PERCENTILE` | `0.85` | Raise to send fewer records to the SLM, lower to send more |
| `FALLBACK_FRAUD_THRESHOLD` | `0.75` | The fused score at which the deterministic path calls fraud |
| `FUSION_WEIGHTS` | `0.4 / 0.3 / 0.3` | Rule, isolation forest, autoencoder shares of the fused score |
| `SLM_MODEL` | `llama3.2:1b` | The adjudicating model |
| `SLM_TIMEOUT_S`, `SLM_MAX_RETRIES` | `120`, `1` | Patience per call, and repair attempts before falling back |

Changing `FUSION_WEIGHTS` or `TRIAGE_PERCENTILE` requires a re-`fit`: the
threshold is a quantile of the fitted score distribution, so a single record
scored later is judged by the same bar the corpus set.

## Related documents

| | |
|---|---|
| `README.md` | Stage 1 — cleaning, schema, validation |
| `README_FRAUD.md` | Stage 2 design — why the ensemble, the triage cut, the injection defense |
| `finetune/` | Stage 3 — counterfactual dataset, LoRA training, base-vs-tuned evaluation |
