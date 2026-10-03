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
fields. The three Laya runs and their frozen scoring are complete. The original
uncached Clef8 arm was interrupted for observed shared-machine memory/I/O
contention before its quality results were opened. No calibration, promotion or test
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

Use the appropriate existing environment and model root for `base`, `d6`,
`typed`, `clef8` or `clef4`. Completed manifest and result identities are recorded
in [evidence.json](evidence.json). Run inference only after resource/manifest release;
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

| Model | Served fields | Primary exact set | Auxiliary no-call | All three policy variants correct | Forbidden selections | Research GO |
|---|---:|---:|---:|---:|---:|---|
| Laya base | 755/755 | 1/80 | 1/47 | 0/80 | 284 | NO |
| Laya d6 | 755/755 | 0/80 | 41/47 | 0/80 | 191 | NO |
| Laya typed | 755/755 | 0/80 | 0/47 | 0/80 | 308 | NO |
| Clef-Flash 8-bit, interrupted | 402/755 | 49/80 | 0/47, all unserved | 22/80 | 30 | NO, incomplete |
| Clef-Flash 4-bit | 755/755 | **80/80** | 24/47 | 36/80 | 48 | NO |

Each completed model served all 755 fields. Separate deterministic projection
removed forbidden selections by construction but left primary semantic accuracy
unchanged. A posthoc diagnostic forced one selection by highest normalized
P(yes): base/d6/typed scored 30/28/45 of 80, versus 31/80 for first-lexicographic.
This is a representation hypothesis, not a replacement gate or no-call evidence.
The original comparison remains frozen. Clef8 scored 49/49 on completely served
primary cases, but 31 primary and all 47 auxiliary cases are unserved. That
conditional result is not an unbiased estimate of the whole cohort. Original
missing fields count as failures; no producer completion or complete-arm latency
was fabricated. The original process exited 143 after root's scoped SIGTERM and
was not restarted. Completed-field median/p95 was 2,284/44,602 ms under variable
shared-machine load. The source of all system pressure and true peak allocation
remain UNKNOWN; this is not an OOM or intrinsic model-speed conclusion.

The completed 4-bit arm demonstrates strong primary semantic selection on this
DEV cohort. Code projection preserves those 80 choices and enforces all derived
allowed-tool restrictions, giving 264/287 correct variants and zero forbidden
selections by construction. It does not fix the 23 no-call errors, prove learned
rule compliance, or establish task savings. Its all-request median/p95 was
1,763.016/2,345.354 ms (755 requests); excluding the first request gives
1,761.878/2,306.842 ms (754 requests). Model loading is separate at 2.899 seconds;
the complete stage took 1,346.728 seconds. Observed latency includes ambient load.

The separate typed-Laya enum arm chooses one tool or none in one request per
origin. It served 127/127 cases and scored **51/80 primary, 0/47 auxiliary**,
versus its original binary 0/80 and 0/47. It fails the preregistered enum screen
requiring full coverage and at least 90% correctness in each slice. Code
projection gives 182/287 correct variants with zero additional calls. Categorical
NLL/Brier/ECE are 1.100321/0.713257/0.418360 in a different option space from the
binary metrics. All-request median/p95 was 109.329/171.949 ms (127 requests);
loading took 1.144 seconds and the complete stage 15.529 seconds.

These two additional arms made exactly 882 calls and exited normally. Each was
scored once; the original reports and sealed test remain unchanged. The enum
scorer has six focused tests, and the producer has five. See the complete
aggregate and closed artifact hashes in [evidence.json](evidence.json), and the
accepted specification identities in [PLAN.md](../../PLAN.md).

The subsequent Clef4 enum arm also completed 127/127 requests. It retained
**80/80 primary**, but no-call accuracy fell from 24/47 to **23/47**, with no
improved cases and one regression. It therefore fails the same enum screen;
changing representation did not solve the no-call error. Code projection gives
263/287 correct variants. Categorical NLL/Brier/ECE are
0.683664/0.335462/0.142773. Observed median/p95 is 1,567.871/2,501.775 ms (N=127),
with separate loading of 3.368 seconds and a full stage of 208.877 seconds.
This closed arm adds 127 calls to the previous 882; no further prompt tuning,
training or model activation follows from its failed screen.

Its 127 requests must be compared with the 283 original all-allowed binary
fields, not the 755 fields including diagnostic policy variants. A joint binary
request could also share the state in one forward per origin and has not been
measured; no optimized-client or whole-task speedup is established here.

The original typed enum scorer/specification is preserved at `cd55ee2`. Version
2 adds `--model typed|clef4` and verifies the matching original report and
checkpoint before paired comparisons; its seven focused tests passed. The old
report remains unchanged. Score a closed enum arm with the matching frozen v2
specification and source revision:

```sh
node training/tool-selection/evaluate-enum.mjs --model "$MODEL_NAME" \
  --root "$ENUM_PROTOTYPE" --original "$ORIGINAL_MODEL_REPORT" --spec "$ENUM_SPEC" \
  --predictions "$PREDICTION_OUTPUT/predictions.jsonl" \
  --prediction-sha "$CLOSED_PREDICTIONS_SHA" --out "$FRESH_DEV_REPORT"
```
