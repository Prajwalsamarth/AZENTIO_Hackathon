# Azentio Hackathon — Relational Data Wrangler & Fraud Sentinel

An end-to-end pipeline over three messy relational CSVs: deterministic cleaning,
then ensemble anomaly triage feeding a sub-3B SLM adjudicator, then a LoRA
fine-tune trained on counterfactual data.

| Stage | What it does | Entry point |
|---|---|---|
| **1 — Clean** | Schema-driven coercion of the raw CSVs → `cleaned_data/` | `run.py` |
| **2 — Detect** | Rules + Isolation Forest + autoencoder → triage → SLM verdict | `run_fraud.py` |
| **3 — Tune** | Counterfactual dataset → QLoRA on Llama-3.2-1B | `finetune/run_finetune.py` |

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install pandas scikit-learn mlx-lm
ollama pull llama3.2:1b        # 1.2B — under the brief's 3B cap
```

Verified with pandas 3.0.6, scikit-learn 1.9.1, mlx-lm 0.31.3, Ollama 0.32.15 on
an M4 Pro. Stage 1 needs only pandas. Raw files are never modified.

## Run

```bash
# Stage 1 — clean
.venv/bin/python run.py

# Stage 2 — detect
.venv/bin/python run_fraud.py fit                       # fit ensemble, persist artifacts
.venv/bin/python run_fraud.py batch                     # all 988 → predictions.json
.venv/bin/python run_fraud.py single --id TXN_0000176   # one record → JSON on stdout
.venv/bin/python run_fraud.py single --json '{...}'     # ad-hoc record → JSON
.venv/bin/python run_fraud.py evaluate                  # metrics → fraud_report.json

# Stage 3 — fine-tune
.venv/bin/python finetune/run_finetune.py generate --dry-run   # dataset, NO network
.venv/bin/python finetune/run_finetune.py generate             # + OpenAI justifications
.venv/bin/python finetune/run_finetune.py train                # QLoRA → saved adapter
.venv/bin/python finetune/run_finetune.py evaluate             # on 30 synthetic rows
.venv/bin/python finetune/run_finetune.py evaluate-real        # on 153 real triaged rows

# Verification
.venv/bin/python tests.py                 # 25 checks, stages 1–2
.venv/bin/python finetune/ft_tests.py     # 32 checks, stage 3
```

Add `--no-slm` to any scoring command to run fully deterministically.

---

# Stage 1 — Cleaning

Deterministic and schema-driven. No imputation guessing, no LLM in the loop: the
same input always produces the same output, and every transformation is counted
in an audit report.

| File | Role |
|---|---|
| `schema.py` | `ColumnSpec` per column — single source of truth for dtype, vocabulary, ranges, null tokens |
| `cleaners.py` | Pure coercion primitives (`clean_category`, `clean_numeric`, `clean_datetime`, …) |
| `pipeline.py` | load → clean → dedupe → repair → flag → validate → write |

Adding or retyping a column is a one-line change in `schema.py`.

## Outputs

- `cleaned_customers.csv` — 124 rows × 27 cols
- `cleaned_accounts.csv` — 178 rows × 23 cols
- `cleaned_transactions.csv` — 988 rows × 33 cols
- `rejects_transactions.csv` — the 18 quarantined rows, with `reject_reason`
- `cleaning_report.json` — per-column audit trail

Cleaned tables carry boolean flag columns (`is_orphan_account`,
`has_invalid_transaction_timestamp`, `is_quarantined`, …). Flagged rows are
**kept** in the clean output *and* copied to the rejects file, so the fraud stage
filters on its own terms — nothing is silently dropped.

## What it fixes

| Fix | Count |
|---|---|
| Exact duplicate rows dropped | 12 |
| Timestamps re-parsed from non-ISO formats | 80 (53 slash, 12 dash, 15 ISO-`T`) |
| Timestamps unparseable → NA + flagged | 10 (7 blank, 3 `NOT_AVAILABLE`) |
| `amount` currency formatting stripped (`INR 62146.26`, `41,677.41`) | 17 |
| `merchant_category` repaired from `merchant_id` | 83 → 0 missing |
| `merchant_category` vocabulary collapsed | 74 → 22 |
| `channel` / `device_type` / `transaction_type` / `status` collapsed | 23→5, 21→6, 17→4, 12→4 |
| `is_foreign_transaction` boolean encodings unified | 8 → 2 |
| Orphan `account_id` rows flagged | 8 |

### The one genuinely ambiguous decision

Mixed date orders were **resolved from evidence, not convention**. Both day-first
and month-first readings were parsed for every non-ISO row and scored against the
independent `is_weekend` and `transaction_hour` columns:

- Slash rows (`29/07/2026 18:07`) are **day-first** — 53/53 weekend match vs 14/53
- Dash rows (`05-24-2026 21:49:39`) are **month-first** — 12/12 vs 3/12

A day-first assumption on the dash rows (the intuitive default) silently
mis-dates 6 of 12 and drops a 7th. `cleaning_report.json` records
`ambiguous_day_month_decided_by_format_order` — the 30 rows where both orders
parse and disagree — so the choice stays auditable.

## Validation

Four consistency invariants run after cleaning. They are **checks, not rewrites**
— a violation means parsing went wrong, which makes them a regression test on the
riskiest step. All pass at 0 violations:

- `age` ↔ `date_of_birth`
- `transaction_hour` ↔ `transaction_timestamp`
- `is_weekend` ↔ `transaction_timestamp` ← this one caught the dash-format bug
- `is_foreign_transaction` ↔ `merchant_country != 'IN'`

Plus referential integrity. `transactions_account_id_orphans: 8` is expected and
flagged, not a failure — those `ACC_9000xx` ids are a synthetic block absent from
`accounts.csv`. Stage 3 reuses all four invariants as *generators*, to gate the
synthetic rows it constructs.

## Deliberately not done

- **Categorical imputation.** Blanks stay NA (`auth_method` 144,
  `transaction_type` 75, `risk_rating` 18, …). Only provably-recoverable values
  are filled: `merchant_category` from `merchant_id` (verified 1:1) and
  `customer_id` from the account owner.
- **Negative values are kept.** `balance_after_txn` has 69 negatives and `amount`
  6 — overdrafts and reversals, not corruption.

### Two notes on the brief vs. the actual data

- The brief says injections hide in "transaction notes", but **`transactions.csv`
  has no notes column**. See *Injection defense* below.
- `accounts.csv` has no duplicate entries and no corrupt credit limits —
  `credit_limit` is a clean 7-value ladder. Its only defect is blank cells.

### Report caveat

Per-column `nulls_after` in `cleaning_report.json` is measured **before** the
repair pass. `merchant_category` reads `nulls_after: 83`; the `_repairs` block
shows those 83 were then filled to 0. Check `_repairs` for post-repair counts.

---

# Stage 2 — Detection: Ensemble Triage + SLM Adjudication

Three independent anomaly signals score every transaction; their fused score
triages a candidate set; a sub-3B SLM reasons over only those candidates. One
transaction in, one verdict out.

```
 ONE transaction record
          │
  [1] enrich      join its account + customer  ──────────┐
  [2] sanitize    deterministic filter → SLM gate        │ fitted artifacts
  [3] featurize   → 94-dim vector ───────────────────────┤ loaded from disk
  [4] score       ┌─ Isolation Forest ─┐                 │ (scaler, iforest,
                  ├─ Autoencoder ──────┼→ fused ─────────┘  autoencoder,
                  └─ Rule engine ──────┘                     score references)
  [5] triage      fused ≥ threshold OR hard-rule hit?
  [6] reason      yes → SLM      no → deterministic verdict
  [7] validate → {"transaction_id","is_fraud","confidence","justification"}
```

| File | Role |
|---|---|
| `config.py` | Every tunable: fusion weights, thresholds, model names, output schema |
| `features.py` | Enrichment + the 94-feature vector |
| `rules.py` | 15 weighted rules → score, reason codes, hard-rule flag |
| `anomaly.py` | Isolation Forest + autoencoder + percentile fusion |
| `sanitize.py` | Injection defense (6 layers) |
| `slm.py` | Ollama client, prompt construction, strict JSON, retry |
| `detector.py` | `FraudPipeline` — the single per-record path |
| `log.py` | Run logging: stderr handler, three verbosities, batch progress |
| `evaluate.py` | Metrics |

Full run: 988 scored, 153 triaged, 153 SLM calls, 0 failures, 100% JSON validity,
0.62 s mean latency.

## Watching a run

Progress goes to **stderr**, results to **stdout**, so `single --id X | jq` stays
clean however chatty the run is.

```bash
run_fraud.py batch                  # default: stages, progress + eta, counts
run_fraud.py -v batch               # every record's scores and every SLM call
run_fraud.py -q batch               # warnings and errors only
run_fraud.py batch --progress 25    # progress line every 25 records
```

A default `batch` reads like this — the slow leg is the ~15% of records that
reach the SLM, so the progress line carries a rate and an ETA:

```
  [   0.8s] info   detector  loading fitted artifacts from outputs/artifacts
  [   0.8s] info   detector  SLM ready: llama3.2:1b at http://localhost:11434
  [   0.8s] info   detector  scoring 988 transactions (~15% expected to reach the SLM...)
  [  12.8s] info   detector  scored 100/988 transactions (0.8/s, eta 36s) | 14 SLM calls, 9 flagged
```

Three things surface as `WARN` at every verbosity, because a run that hides them
is lying about what it did: the prompt-injection filter firing, the SLM being
unreachable or returning nothing usable, and the injection backstop overriding an
SLM verdict. `log.py` holds the formatter and level policy; importing a module
never emits anything until a CLI calls `log.configure()`.

## Design decisions worth knowing

### The two models see different feature spaces — deliberately

Isolation Forest gets all 94 features; the autoencoder gets only the ~48
continuous/derived ones. A standardized one-hot column that is 1 for 2% of rows
is nearly unpredictable, so an 8-unit bottleneck can only emit the mean and bank
~1.0 error on it every time. With all 60 one-hots included, reconstruction error
was mostly a constant noise floor.

Measured: the known fraud case moved from **rank #69 → #12** of 988, and rank
correlation with the forest fell **0.60 → 0.48**. Final pairwise Spearman is
0.33–0.48 across all three detectors — genuinely independent views rather than
three versions of the same one.

### JSON field order is load-bearing

Structured decoding emits properties in schema order. With `is_fraud` first, the
model commits to a boolean before writing a word of analysis — it guesses, then
rationalises. On a 5-transaction A/B with `llama3.2:1b`:

| Schema order | Hard-rule fraud cases correct |
|---|---|
| `is_fraud` first | **1 of 3** — and justifications contradicted their own verdict |
| `justification` first | **3 of 3** — coherent throughout |

The justification now acts as chain-of-thought the boolean is conditioned on.
Consumers read fields by name, so the order is invisible downstream.

### Features come from the enriched record, not the sanitized one

Redaction protects the *prompt*. The numeric signal was never attacker-controlled,
so blinding the anomaly models to it would let an injection reduce its own
anomaly score — the opposite of what the defense is for.

### Nothing raises

Every stage degrades to a deterministic rule verdict: a malformed record, a
missing account, a dead Ollama. A fraud detector that crashes on bad input is not
a fraud detector.

### Rule/SLM conflicts are recorded, not overridden

When the SLM clears a row the hard rules flagged, that is **counted**
(`rule_slm_conflicts`) but **not reversed**. Clearing false positives is the job
the SLM is there to do. A rising count means the rules or the model need
attention, so it must be visible rather than silently suppressed. The one
exception is the injection backstop, where the deterministic layer does win.

---

# Stage 3 — Counterfactual Dataset + LoRA Fine-Tune

A **separate pipeline** in `finetune/`. Stage 2 is untouched.

## Why counterfactuals instead of a teacher model

Stage 2 showed the 1.2B model **rubber-stamping**: 88% fraud rate in the lowest
triaged band vs 97% in the highest — it mostly agreed with whatever the ensemble
had already decided.

Teacher distillation would have inherited another model's errors. Instead, fraud
rows are built by **intervening on real transactions**: take a low-risk record,
apply 2–3 named interventions (`GEO_IMPOSSIBLE`, `HIGH_RISK_MCC`,
`AMOUNT_ESCALATION`, `AUTH_STRIPPED`, `VELOCITY_BURST`, `NIGHT_SHIFT`), and
recompute every dependent field. The label is **known by construction**, not
inferred, so generation error can corrupt wording but never ground truth.

**Consistency repair is what makes the rows usable.** Raising an amount without
recomputing the account ratio and resulting balance produces a transaction that
cannot physically exist. Stage 1's four validation invariants are reused here as
generators, and every constructed row is asserted against them.

### Negatives are half hard, on purpose

10 clean (no rules triggered) + **10 near-miss** (1–2 *weak* rules: new device,
night hours, KYC pending). Near-misses are the direct antidote to "any rule hit →
fraud".

A first attempt filtered near-misses only on "below triage threshold", which let
through a foreign, card-present transaction 8,061 km from home labelled *not
fraud* — the exact pattern the positives are built from. Negatives now exclude
the nine fraud-defining rule codes (`STRONG_RULE_CODES`). Classes separate
cleanly: fraud 0.58–0.98, negatives 0.06–0.50.

## Results

Training (24 train / 6 valid, stratified):

| Run | Outcome |
|---|---|
| 150 iters | Val loss bottomed at 0.988 (iter 25) then rose to 1.160 while train loss fell to 0.006 — memorization |
| 60 iters, eval/save every 10 | Best val loss **0.945 @ iter 20**, final 1.029 |

`select_best_checkpoint()` **promotes** the best checkpoint rather than printing
advice and shipping the final (most overfit) weights — mlx-lm leaves
`adapters.safetensors` at the last iteration by default.

Base vs fine-tuned, both served through MLX (**no schema-constrained decoding**):

| Metric | Base | Fine-tuned |
|---|---|---|
| Raw JSON validity (153 real rows) | **8%** | **97%** |
| Verdict/justification coherence | 50% | 93% |
| Held-out label agreement | no parseable rows | 80% (n=5) |
| Fraud rate on triaged rows | — (unusable) | 28% |
| Score gradient | — | **0.22** |
| Latency | 0.58 s | 0.70 s |

### What this does and does not show

**Real:** the format win. Unconstrained, the base model emits prose and fails to
produce parseable JSON 92% of the time; the fine-tune fixed that. 30 examples is
ample to teach an output format.

**Real but noisy:** the rubber-stamping did improve. Stage 2's base model flagged
92% of triaged rows with a 0.09 gradient; the tuned model flags 28% with a 0.22
gradient. But the gradient is **not monotonic** (0.24 / 0.27 / 0.19 / 0.41) — it
is not a reliable ranker, and it may now *under*-flag.

**Not shown:** any improvement in fraud detection accuracy. Held-out agreement of
80% is measured on **5 rows**. That number is an anecdote, not a metric.

**Not like-for-like:** stage 2's 92%/0.09 baseline came from the same model served
through Ollama *with* schema-constrained decoding. The 8% vs 97% column compares
base vs tuned on the MLX path only.

---

# Injection defense

Deterministic first, SLM second — six layers, in `sanitize.py`:

1. **Field allowlist** — only named fields reach the prompt
2. **ID grammar** — identifiers must match `TXN_\d+` etc., never free text
3. **Vocabulary allowlist** — `merchant_name` has exactly 35 known values; anything else becomes `[UNRECOGNIZED]`
4. **Normalization** — NFKC, strip control/zero-width/bidi, 64-char cap
5. **Pattern detection** — role markers, override phrasing, verdict steering, schema steering, URLs, base64 blobs
6. **SLM gate** — clean-but-unfamiliar strings are classified DATA vs INSTRUCTION in isolation, never seeing the transaction

Plus structural hardening (untrusted values inside a delimited block, system
prompt declaring it data) and an **output-side backstop**: if a field was
redacted, the SLM may not downgrade a high ensemble score. Rules win. The gate
**fails closed** — an unreachable gate leaves the value redacted.

`cleaners.sanitize_text()` in stage 1 delegates here, so the same rules apply at
cleaning time and at prompt-construction time.

### There are no injections in the shipped data

Verified in stage 1: no notes column exists, the longest value across all three
files is 36 characters, and keyword sweeps return nothing. The defense is built
and unit-tested against crafted payloads because the brief requires it and real
deployments need it — not because this dataset exercises it.

# PII boundary (stage 3 only)

Stage 3 is the only stage that sends anything off the machine — 30 justification
requests to OpenAI. Three independent mechanisms guard it, because one is not a
guarantee:

1. **Allowlist** — payloads are *built* from 25 named fields, never filtered down
   from a record, so a new PII column upstream is excluded by default.
2. **Scanner** — emails, phones, IPv4, ID grammars, and every real customer name
   are matched against the serialized payload, which **raises** on a hit.
3. **`--dry-run`** — writes every payload to `finetune/artifacts/payloads/` for
   reading, so the claim is verifiable rather than trusted.

Withheld and why: direct identifiers; `merchant_name` (not PII, but the one
attacker-controlled free-text field); `cust_city` / `cust_state` / `cust_age` /
`cust_annual_income` (quasi-identifiers — in a 124-customer population those four
are close to re-identifying, and the justification only needs the *distance*).
`OPENAI_API_KEY` is read from the environment only, never a CLI argument.

`ft_tests.py` asserts both directions: no PII in any payload, **and** that the
scanner actually raises on planted names, emails, IPs, and IDs.

# Metrics are not accuracy

This dataset has **no ground-truth `is_fraud` label**. Figures in
`fraud_report.json` describe the detector's own behaviour — score distributions,
detector agreement, throughput — not correctness. The report says so in its own
`disclaimer` field.

The only labelled data in the project is the 30-row fine-tune set, whose labels
are constructed by counterfactual intervention. Even there, 30 examples is
demonstration scale: it teaches output format and calibration, not fraud
detection. Counterfactuals also inherit the generator's assumptions — those rows
are fraud because *our rule engine* says those interventions constitute fraud, so
the model learns that worldview, blind spots included.

The dataset is regenerable from a seed, so scaling to 300 rows is a parameter
change (`ft_config.N_FRAUD` etc.), not a rewrite.
