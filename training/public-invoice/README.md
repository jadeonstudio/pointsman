# Pinned public invoice inputs

This tool archives only `typesafe/evalsafe-invoice-processing` revision
`6beeb2d2acd65c086c835022f5f4d7434114cafc` from anonymous official Hugging Face
URLs. Its actual LICENSE, dataset card and `dataset.json` declare Apache-2.0.
No Jev API call, training or publication is performed.

## Completed narrow comparison

The blind source-only oracle accepts all 150 `discount_days` questions: 136 `none`
and 14 `10`. `oracle.py` fixes the grammar and checks complete target-question/state
identity before answers are opened. The fixed comparison was run once after
the lossless local cohort completed. [Aggregate evidence](evidence.json) preserves
the exact artifact identities and frozen numerical rules.

| Metric | Jev 1.13.0 published run | Local Clef-Flash 8-bit | Always-none |
| --- | ---: | ---: | ---: |
| Served / fixed cases | 150/150 | 150/150 | 150/150 |
| Correct / accuracy | 144/150, 96% | 144/150, 96% | 136/150, 90.67% |
| Observed-gold-class macro F1 | 0.90049 | 0.90049 | 0.47552 |
| All-six-class macro F1, zero for unobserved gold classes | 0.30016 | 0.30016 | 0.15851 |
| NLL | 0.36935 | 0.20120 | 3.22362 |
| Brier, sum across classes | 0.07739 | 0.07292 | 0.18667 |
| ECE, ten fixed bins | 0.03840 | 0.04730 | 0.09333 |

Clef and Jev have the same 144 correct cases and the same six wrong cases:
improved 0, regressed 0, accuracy difference 0 percentage points, paired case
bootstrap 95% interval [0, 0], exact two-sided McNemar p=1. This describes this
sample, not population equivalence. Against always-none, each improves 14 cases
and regresses six: +5.33 percentage points, interval [-0.67, +11.33], p=0.11532.
The frozen bootstrap uses 2,000 case resamples, seed 42, within this one family.
No superiority gate passes. Only `none` and `10` have gold support; the other
four classes have no recall evidence. All-six F1 explicitly includes their zeros.
The fixed grammar parser is correct by construction, not a learned-model or
task-efficiency result. These are discount-term results, not invoice approval,
fraud detection or general decision quality.

NLL/Brier use normalized rounded probability mass; ECE uses the probability of
the actual selected choice. Provider confidence is kept separately and is never
substituted. The six classification errors give a one-sided exact 95% error upper
bound of 7.74% (iid-case assumption). Critical severity was not adjudicated;
even hypothetical zero critical errors in 150 cases has upper bound 1.98%, above
the required 1%. Whole-task A, above-Jev B and unseen-generalization C remain
unpassed; no checkpoint or production mode is promoted.

`predict_clef.py` initially rejected all 150 complete inputs at the earlier 2048
experiment cap, with zero forward calls. Actual lengths are 4459–10567 tokens.
A separate frozen 16384-token pass preserves every input; its longest-input
resource probe passed at 12.282 GiB sampled combined Metal allocation. The cohort
completed all 150 forward calls with 150 valid outputs. Its stage ran from
2026-10-03 06:52:38.669181 UTC to 09:24:13.226946 UTC (9,094.56 seconds).
Model load was 3.38835 seconds; first forward 41.41693 seconds; warm median
61.88683 seconds and p95 83.91355 seconds. MLX peak allocation was
12,986,770,920 bytes (12.095 GiB). Metal driver/current and RSS counters have
overlapping scopes and their separate peaks must not be summed.

CPU verification overlapped this cohort at approximately 08:42:47.171618–
08:43:27.310777 UTC and 08:44:18.495514–08:44:19.216495 UTC, based on gate-log
filesystem creation/write times. These are ambient-load observations, not exact
child start/end timestamps. OS process absence plus the immutable completion
manifest establish completion; the old tool session was inaccessible and its
shell exit code remains UNKNOWN. No inference was repeated for scoring.

The Jev manifest says `question_mode=all`. Per-question records prove matching
target questions and states, but do not attest the original complete multi-question
API envelope or batching independence. This is a narrow target comparison,
not a proven identical-full-request experiment. Local and hosted timings are
different conditions. Original refusals, preregistrations and status-protocol
clarifications remain immutable; no production provider or mode is changed.

Keep the archive in a private directory outside Git. `source/` contains the full
public Parquet files, including model outputs and reference labels; do not give
that directory to a blind assessor. `source-hashes.json` records the exact URLs,
byte counts and SHA256 values. `schema.json` contains schemas only.

`blinded/` is the independent-assessment handoff:

| File | Rows | Allowlisted columns |
| --- | ---: | --- |
| `blinded-cases.jsonl` | 150 | `case_id`, `input_json` |
| `blinded-inputs.jsonl` | 6,874 | `question_instance_id`, `case_id`, `node_id`, `question_id`, `kind`, `question_json`, `state_json` |

Every source row is retained, with no label, answer, status, confidence, action,
timing or run-based eligibility filter. JSON text is preserved exactly. Only
allowlisted Parquet columns are decoded; the prediction table is inspected for
schema/counts without reading values. `blind` verifies source hashes, JSON
syntax, case linkage, unique IDs and exact export readback. Its manifest records
input hashes, runtime, policy IDs and the script identity.

The published questions originate from reference workflow traces. They are not
a newly preregistered unseen cohort. Legitimate business-source fields such as
delivery evidence `outcome` remain part of the exact input. The dataset provides
policy IDs/titles (`startup`, `enterprise`, `high_volume_retailer`), but not
executable policy rules. Independent gold requires separately verified policy
code and an input-only eligibility rule. Input preparation alone proves no B/C
quality gate.

The frozen independent oracle supplies only this narrow question's gold. It does
not turn the published model-consensus workflow references into independent
invoice-action or critical-risk labels.

## Reproduce

Use Python 3.12 with `pyarrow==20.0.0` in a separate environment. Do not change
the Clef/Laya numerical experiment environments.

```sh
INVOICE_ROOT="<private-directory>"
python3.12 -m venv "$INVOICE_ROOT/.venv"
"$INVOICE_ROOT/.venv/bin/python" -m pip install pyarrow==20.0.0
python3 training/public-invoice/archive.py self-test
"$INVOICE_ROOT/.venv/bin/python" training/public-invoice/archive.py acquire --root "$INVOICE_ROOT"
"$INVOICE_ROOT/.venv/bin/python" training/public-invoice/archive.py inspect --root "$INVOICE_ROOT"
"$INVOICE_ROOT/.venv/bin/python" training/public-invoice/archive.py blind --root "$INVOICE_ROOT"
```

Acquisition uses stdlib HTTPS with no authorization header or host credential
lookup. The six fixed files total 16,087,532 bytes. No other dataset is acquired.

Scoring replay requires the immutable private archive, including its independent
gold, original pre-unblind contract, approved status-protocol clarification,
separate 16k admission receipt, producer contract and frozen predictions. Use a
fresh output copy with no `comparison/report.json`; preserve the original archive.

```sh
python3 training/public-invoice/compare.py score --root "$INVOICE_ROOT" \
  --clef "$INVOICE_ROOT/clef8-invoice-16k/predictions.jsonl"
```

This command performs no model inference and refuses an existing report. The
repo contains sanitized aggregates; individual cases and predictions stay private.
The local model uses the frozen 8-bit/group-64 weights and BF16 head reference
with a distinct `invoice-recorded-payload-v1`, `admission_revision=16k` producer.
It is not the publisher's full-precision endpoint or an unchanged earlier DEV
input contract. Published Jev per-case workflow timing is not a comparable local
single-question latency baseline.
