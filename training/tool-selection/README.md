# Pinned BFCL tool selection: preparation only

The CPU-only builder prepares a private Choice wire pack from
[Gorilla@916260dfc116bf06793a1af79b4ec8195b0453b6](https://github.com/gorilla-llm/gorilla/tree/916260dfc116bf06793a1af79b4ec8195b0453b6/berkeley-function-call-leaderboard/data).
The pinned LICENSE is Apache-2.0; the official
[dataset card](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
also identifies that license. Preserve license/attribution notices and mark
modified files if redistributing derivatives. Raw rows and references remain
private; public evidence contains aggregate admission facts and hashes only.
No downloads, Jev calls, model inference, weights, GPU execution or training are
part of this builder.

The native `multiple` source has 200 cases with 2–4 offered tools, exactly one
selected name in each independent published `possible_answer` reference. The
native `irrelevance` source has 240 one-tool cases whose empty selection is
explicitly defined by the pinned official `relevance_file_runner`. Missing or
malformed multiple references fail preparation; they never become no-call gold.
This is a function-name projection, not argument/AST/execution correctness.

The original input-only admission froze before downloading the reference file.
Cases sharing any exact canonical full named-function schema form one component.
The existing SHA256 bucket split is preserved: 440 original cases / 363 schema
components, split train228/dev127/test85. Components are schema groups, not
363 domains or semantic task families. No source input, function or component
crosses splits. Pretraining exposure of published BFCL remains UNKNOWN.

Every multiple origin has all-allowed, deny-first-lexicographic-tool and deny-all
variants. Conversation and function schemas remain unchanged; allowed tools
change. Gold is the original selected names intersect allowed tools, with an
explicit prohibition on alternative substitution. Empty selection means no
authorized required tool; it does not mean the user request was fulfilled.
These are local policy derivatives, not the publisher's native evaluation.
The native no-call cases remain once each. Variants stay with their source case.

| Split | Origins | Schema components | Case variants | Binary fields |
|---|---:|---:|---:|---:|
| Train | 228 | 225 | 392 | 776 |
| Dev | 127 | 76 | 287 | 755 |
| Test | 85 | 62 | 161 | 380 |
| Total | 440 | 363 | 840 | 1,911 |

The canonical independent-dataset writer currently requires objective authored
oracle provenance and reassigns its own 70/10/10/10 split. It cannot honestly
preserve this published benchmark reference and frozen split. The builder uses
an explicit wire pack and leaves the shared store unchanged. It reuses the real
portable `validateRequest`, runtime `wireRequest` data wrapper and
`assertSplitIsolation`; it invents no `RULES`, human, runner or confidence labels.

Each split has separate `cases.jsonl`, `fields.jsonl` and `reference.jsonl` files.
Cases contain the full joint per-candidate questions. Fields contain one common
`tool_name` Choice question with yes/no criteria and the full case state. These
are distinct envelopes. Consumers send only `wire`, never source/category/split
metadata or reference records. `request_identity` identifies the unwrapped
request; manifest file hashes bind the actual wrapped wires. Dev inputs are
already separated from dev references. Test files are hash sealed for future
evaluation; the builder refuses existing output directories. Freeze future
predictions before consuming test references.

To join a field with a reference, match `field.case_id` to `reference.sample_id`;
the final segment of `field.sample_id` identifies its entry in
`reference.question_values`. The model-facing question ID remains `tool_name`.

Reference records explicitly distinguish 440 `benchmark_reference` cases from
400 `derived_policy_oracle` cases, retaining source/category/evidence hashes
and the deterministic intersection formula. All source cases are retained;
840 variants and 1,911 fields are not independent sample counts.

Actual full encoding with the existing pinned Clef tokenizer and official
encoder passed for all 2,751 envelopes, without truncation or weight loading.
Case requests range308–1,338 tokens (p95 1,160); field requests306–872 (p95 726).
All fit both 2,048 and 16,384 input caps. This establishes input admission only,
not model support, correctness, memory use or latency.

Recorded source controls: on native multiple, none/invoke-all both score0/200,
first lexicographic80/200. Across all440 originals, none240/440 (54.55%) and
invoke-all0%; across840 policy variants, none520/840 (61.90%) and invoke-all0%.
Derived test none105/161 (65.22%); first allowed lexicographic69/161 (42.86%).
Invoke-all means every offered tool, including prohibited tools. Native no-call
has a candidate-count shortcut because all240 cases offer one tool; report it
separately from the primary200 semantic-choice origins. Field majority accuracy
alone is misleading. There are no native repeated complete candidate sets in
these two files; controlled policy variants supply the same-input contrast.

Future evaluation must freeze case exact-set correctness, coverage failures,
F1, calibration, false-positive selection and policy violations before inference.
Resample source cases/schema components, not correlated fields or variants.
Even zero errors over85 heldout origins gives a one-sided95% upper bound3.46%
(38 multiple origins:7.58%), insufficient for a1% critical floor. No A/B/C,
task completion, operational action severity or Jev superiority claim follows.

Reproduce using the existing private source archive and existing Clef venv,
choosing a fresh output directory. No packages or network access are needed:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_tool_selection.py -v
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  "$CLEF_PYTHON" training/tool-selection/build.py --archive "$BFCL_ARCHIVE" \
  --output "$BFCL_OUTPUT" --clef-upstream "$CLEF_UPSTREAM"
```

Omitting `--clef-upstream` produces the same wires with token admission explicitly
`NOT_RUN`. Source pins, frozen admission/contrast/wording receipts, split counts,
wire hashes, tokenizer assets and focused-test results are in
[admission-evidence.json](admission-evidence.json). Five focused tests passed;
the full builder exited0. Test/model quality evaluation was not run.

## DEV producer and evaluator checkpoint

The as-shipped baselines `base`, `d6`, `typed` and `clef8` each admit all 755 DEV
fields. The three Laya runs and their frozen scoring are complete; the original
uncached Clef run is still in progress. No calibration, promotion or test
evaluation occurs at this checkpoint. The accepted analysis
specification SHA256 is
`ba3cfedf5ebaa67aff0b93744f21d5627f0d78291b27479fa78f1bc601ee16e3`.

The fixed scope is 755 fields / 287 variants / 127 origins / 76 schema components:
80 native multiple/all-allowed cases are primary and 47 native irrelevance cases
are auxiliary. Missing or malformed fields fail complete cases. Actual emitted
choices determine quality; normalized selected-choice probabilities determine
calibration. Component/source bootstrap uses 2,000 draws and seed 42.
Research GO requires primary exact-set accuracy ≥90%, all three policy variants
correct together for ≥90% of multiple origins, zero policy violations and 100%
valid field coverage. It does not establish A/B/C or authorize promotion.

A separate code-projection control reuses each origin's all-allowed semantic
selection and intersects it with the variant's allowed tools. It traces base
prediction IDs, adds zero inference calls and invents no derived confidence.
Its zero policy violations follow from code; they are not raw model rule quality.
All 755 raw field evaluations and their GO gates remain required.

Use the appropriate existing environment and model root for one of the four
model names. Producer manifest hashes below are placeholders until the refreshed
manifests are accepted. Run inference only after resource/manifest release;
score only after its completion receipt and prediction hash are frozen:

```sh
"$MODEL_PYTHON" training/tool-selection/predict.py prepare --model "$MODEL_NAME" \
  --root "$MODEL_ROOT" --production "$BFCL_PRODUCTION" --output "$PREDICTION_OUTPUT"
"$MODEL_PYTHON" training/tool-selection/predict.py run --model "$MODEL_NAME" \
  --root "$MODEL_ROOT" --production "$BFCL_PRODUCTION" --output "$PREDICTION_OUTPUT" \
  --manifest-sha "$ACCEPTED_PRODUCER_MANIFEST_SHA"
node training/tool-selection/evaluate.mjs --root "$BFCL_PRODUCTION" --spec "$DEV_SPEC" \
  --predictions "$PREDICTION_OUTPUT/predictions.jsonl" --prediction-sha "$CLOSED_PREDICTIONS_SHA" \
  --out "$FRESH_DEV_REPORT"
```

The producer opens blind DEV fields only; the evaluator opens frozen DEV
references only. Test references remain sealed. Each model is scored once after
its completion receipt and prediction hash have been verified.

| Completed model | Primary exact set, 80 cases | All three policy variants correct | Forbidden selections | Research GO |
|---|---:|---:|---:|---|
| Laya base | 1/80 | 0/80 | 284 | NO |
| Laya d6 | 0/80 | 0/80 | 191 | NO |
| Laya typed | 0/80 | 0/80 | 308 | NO |

Each completed model served all 755 fields. Separate deterministic projection
removed forbidden selections by construction but left primary semantic accuracy
unchanged. A posthoc diagnostic forced one selection by highest normalized
P(yes): base/d6/typed scored 30/28/45 of 80, versus 31/80 for first-lexicographic.
This is a representation hypothesis, not a replacement gate or no-call evidence.
The original comparison remains frozen; unfinished Clef predictions are neither
scored nor extrapolated.
