# Independent workflow decision corpus

This local MIT corpus is authored in `corpus.mjs`. It uses no TypeSafe/Jev
outputs, AI reference labels, private captures, network calls or paid compute.
It does not manufacture runner/human events. Each objective label comes from
six independently specified true/false/unknown facts and a public rule oracle.
Source/oracle revisions bind the generator's exact SHA-256, and the separately
stored `oracle_specs.json` retains authored English/Korean policies and licenses.
This is a rule feasibility corpus, not evidence of general task quality.

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
