# Pinned public invoice inputs

This tool archives only `typesafe/evalsafe-invoice-processing` revision
`6beeb2d2acd65c086c835022f5f4d7434114cafc` from anonymous official Hugging Face
URLs. Its actual LICENSE, dataset card and `dataset.json` declare Apache-2.0.
No Jev API call, training or publication is performed.

## Current experiment status

The blind source-only oracle accepts all 150 `discount_days` questions: 136 `none`
and 14 `10`. `oracle.py` fixes the grammar and checks complete target-question/state
identity before answers are opened. The recorded Jev 1.13.0 run is 144/150 correct;
the always-none control is 136/150. These are narrow canonical discount-term
results, not invoice approval, fraud detection or general decision quality.

`predict_clef.py` initially rejected all 150 complete inputs at the earlier 2048
experiment cap, with zero forward calls. Actual lengths are 4459–10567 tokens.
A separate frozen 16384-token pass preserves every input; its longest-input
resource probe passed at 12.282 GiB sampled combined Metal allocation. The cohort
is still running; no completed Clef accuracy or paired comparison is published yet.

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
code and an input-only eligibility rule. This preparation proves no B/C quality
gate or independent accuracy.

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
