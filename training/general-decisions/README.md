# Independent workflow decision corpus

This local MIT corpus is authored in `corpus.mjs`. It uses no TypeSafe/Jev
outputs, AI reference labels, private captures, network calls or paid compute.
It does not manufacture runner/human events. Each objective label comes from
six independently specified true/false/unknown facts and a public rule oracle.
Source/oracle revisions bind the generator's exact SHA-256, and the separately
stored `oracle_specs.json` retains authored English/Korean policies and licenses.
This is a rule feasibility corpus, not evidence of general task quality.

The 2026-10-03 frozen pilot predates `independent-workflow-rules-v3-explicit-priority`.
Later inspection found eight DEV Noul rows with overlapping failure conditions
whose text omitted explicit first-match priority. Historical accuracy fields are
frozen-oracle agreement; raw predictions and labels remain unchanged. The nine
changed-rule Choice pairs are unaffected. The current English/Korean policy makes
the existing oracle order explicit; this changes new input/provenance identities,
not the oracle logic or family split. Regenerating the same cases is not fresh
independent confirmation. See the caveat in [pilot evidence](pilot-evidence.json).

## Generate through the canonical dataset/export path

Node >=22, standard library only. Use an explicit private home outside Git;
resolve macOS `/tmp` symlinks to `/private/tmp` before passing the path.

```sh
node training/general-decisions/corpus.mjs --home /private/tmp/pointsman-independent --count 2000 --seed 42
node training/general-decisions/corpus.mjs --home /private/tmp/pointsman-independent --count 20000 --seed 42
node --test tests/general-decisions.test.mjs
```

Each run writes the existing `training/datasets/<version>/canonical.jsonl`,
`corpus_report.json`, `oracle_specs.json`, and `training/manifests/<version>.json`.
The existing exporter writes `training/exports/<version>/laya/` containing
train/dev/calibration/test JSONL, metadata JSONL and a hash manifest. No model
training, provider activation, promotion, uploads or captures are performed.

The design covers four actual bounded workflow judgment targets, four distinct
rules per target: evidence relevance, next branch, failure class, and whether to
continue/escalate. Eight workflow domain contexts, KR/EN/mixed prose, Choice,
Noul and ordered evidence-strength Score questions provide presentation slices.
Choice meanings/keys/orders swap; Noul propositions reverse; Score ordering
reverses while retaining zero-based semantics. Quoted instructions remain data.
Full authored rules live in `state.policy`, whose official serializer preserves
Unicode. Questions contain only the rule reference, changing exception and query;
the official question serializer otherwise escapes Korean Unicode and can exceed
the unchanged 256-token question head. Every oracle fact and full rule remains
available; no state or question is truncated to fit.
Strict missing-metadata counterfactual policies can change gold on identical
states. The policy never grants permissions or executes actions.

There are **16 semantic rule families**, not 60+ independent schemas. Each fact
case has four deliberately correlated sibling renderings. About 2,000 rows
represent 512 distinct structured cases, and 20,000 rows represent 5,008 cases.
These counts are reported separately; rendering variants must not count as
unseen-family diversity. Class distribution is intentionally exposed: some
positive complete-evidence outcomes are rare in exhaustive ternary enumeration.
Training methods must check these counts before drawing learning conclusions.

All siblings, translations, domains and rule changes of a base semantic rule
stay in one source/template/family group. Canonical schema2 splits connected
groups into train/dev/calibration/test; adding rows for the same 16 families
preserves assignment. Adding families creates a new split revision and cannot
redraw an already frozen holdout. Some targets are absent from sealed splits;
absent family/slice quality stays UNKNOWN. Domain/language transfer is not an
independent unseen-domain/language evaluation here. Byte bounds do not prove
tokenizer admission or preservation of all required evidence.

The optional `admission.py` uses the installed official tokenizer locally and
calls both the trainer's `preprocess_split(input_fit="lossless")` and worker's
`assert_lossless` on all four splits. It reads tokenizer/config assets only,
loads no weights, makes no model calls and never edits the installed checkpoint.
It reports separate head/option/whole-sequence causes and pinned asset/script
hashes. Use the existing local Laya Python environment; install nothing for it:

```sh
/path/to/existing/laya/venv/bin/python training/general-decisions/admission.py --export-dir /private/tmp/pointsman-independent/training/exports/VERSION/laya --model-dir /path/to/pinned/local/checkpoint --out /private/tmp/admission.json
```

Version `independent-workflow-rules-v1` and pilot
`112f9a47982364a75b17b4cfb357b5ba2005f3848d26650e499a51bf80bd9f34`
are superseded: 1,296/2,000 rows failed the question-head budget, including
1,053/1,625 train/dev/calibration rows. No learning happened on that input.

## TRAIN-only measured curriculum follow-up

`curriculum.mjs` selects 130 informative existing TRAIN fact cases plus 12
non-overlapping rare-class anchors: 142 unique cases, 568 correlated question
rows. For each Choice rule pair it inherits the base question's exact keys,
meanings and insertion order; only the exception instruction changes. Original
oracle rules/facts stay unchanged. It records the transform revision separately
from the original oracle/source revision and never oversamples canonical rows.

```sh
node training/general-decisions/curriculum.mjs --home /private/tmp/pointsman-independent-admitted-pilot --source-version ba27f11ee6e5bf296c536db112db6ecafa961b686c05edcc3d02ea13a728c070
```

The canonical home must remain outside Git, including ignored repository paths.
The helper writes a new 1,193-row dataset with train/dev/calibration/test counts
568/125/125/375, a curriculum report and a dev specification. Every original held
canonical/export row byte, ID, hash and family assignment is preserved. Held
content is used only for byte/hash and canonical integrity checks, never rule
selection or oracle/model evaluation. Use `admission.py --split train` to check
the 568 changed rows; the unchanged held admission evidence remains reusable.
The intended two-epoch follow-up changes both sample exposure and update count;
it is a measured curriculum experiment, not an equal-update causal claim.

## Completed local evidence (2026-10-03)

[pilot-evidence.json](pilot-evidence.json) preserves the initial pilot record and
adds sanitized aggregates for six completed candidates. Every comparison uses
the same 125 dev rows from **one semantic family**; dev also selected epochs, so
these results are not sealed confirmation or broad generalization evidence.

| Candidate | Accuracy | Macro F1 | Rule/order | Noul negation | KO |
|---|---:|---:|---:|---:|---:|
| Base | 53.6% | .329167 | 4/9 | 0/20 | 22/41 |
| d6 | 52.0% | .282408 | 2/9 | 0/20 | 18/41 |
| Initial pilot | 64.0% | .339148 | 0/9 | 5/20 | 19/41 |
| Curriculum CE | 40.0% | .318053 | 1/9 | 0/20 | 19/41 |
| Curriculum RLCD/GRPO | 45.6% | .325878 | 0/9 | 0/20 | 25/41 |
| Cached typed baseline | 32.0% | .295629 | 0/9 | 3/20 | 14/41 |

The five preregistered follow-up floors were accuracy >=53.6%, macro F1
>=.329167, rule/order >=4/9, Noul negation >=5/20 and KO >=22/41.
**Curriculum CE failed all five; RLCD/GRPO passed KO only. Both are NO-GO.**
The initial pilot's aggregate gain did not justify scaling its unchanged recipe.
The dev rule/order probe changes option insertion order as well as the rule;
it is not a pure isolated rule-change measurement. TRAIN curriculum pairs were
corrected without changing any held row bytes or IDs.

CE and RL used identical 568 rows/142 fact cases, two epochs, 1,136 question
exposures and 36 optimizer updates, with batch 2/accumulation 16 on local MPS.
Their local-run records report 167.649s and 190.394s of training wall time;
these fields are not estimates of total resume overhead. Dev collection took
9.229s and 10.009s respectively, compared with 16.105s for typed. Prior baseline
predictions were reused. Checkpoint IDs, all six measured timings and full
aggregate metrics are retained in the JSON.

The [typed specialist's publisher card](https://huggingface.co/convaiinnovations/laya-typed-decisions)
lists Apache-2.0, 421M parameters and English-only support. Its existing shipped
option-bucket temperatures were retained. EN was 14/44; KO/mixed observations
fall outside the stated language support. This is an as-shipped unqualified
baseline, not a newly calibrated or universal comparison. No test inference,
comparison-time calibration decisions, paid calls or model promotion occurred.

## Freeze and evaluate model-neutral predictions

Freeze a specification before any model evaluation, substituting actual paths:

```sh
node training/general-decisions/evaluate.mjs --dataset /private/tmp/pointsman-independent/training/datasets/VERSION/canonical.jsonl --manifest /private/tmp/pointsman-independent/training/manifests/VERSION.json --freeze > /private/tmp/evaluation-spec.json
node training/general-decisions/evaluate.mjs --dataset /private/tmp/pointsman-independent/training/datasets/VERSION/canonical.jsonl --manifest /private/tmp/pointsman-independent/training/manifests/VERSION.json --spec /private/tmp/evaluation-spec.json --predictions /private/tmp/predictions.jsonl
```

Predictions must bind exported metadata's `sample_id` and exact candidate
identity. An OK row has `{sample_id, model_id, checkpoint, status: "ok",
probabilities: {OPTION_KEY: PROBABILITY}}`. The keys must exactly match the
question's distribution (Noul: `false/true`; Score: zero-based positions).
Non-OK statuses are `timeout`, `rejected`, `unsupported_input`, or `error`. Missing predictions are
also counted. Duplicate/foreign joins and mixed candidate identities fail.
Only well-formed probability vectors receive distribution scores; every
rejection/timeout/malformed/missing item fails full-envelope correctness.
As in the runtime contract, finite bounded vectors with positive mass and sum
error at most 0.02 are accepted. Their rounded mass is normalized before scoring;
reports count corrections and malformed vectors separately. Raw prediction files
remain unchanged. Evaluation revision v2 replaces v1's overly strict 1e-6 mass
check, which incorrectly rejected some runtime-rounded valid predictions.

Reports include full-envelope and served accuracy, semantically normalized
macro F1, NLL/Brier, reliability/ECE, expected Score ordinal/severe errors,
threshold risk/coverage, and family/language/domain/type/target slices with
effective family counts. Thresholds are reported, never selected on test.
`comparePredictions()` pairs common frozen samples and bootstraps differences
at the semantic-family level, reporting INCONCLUSIVE with fewer than five
families. No synthetic-row confidence interval can establish executor utility,
broad model quality or Jev superiority. Approved PLAN criteria and critical
slice floors remain separate acceptance gates, and unavailable comparisons
remain UNKNOWN.
