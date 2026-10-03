# Learning a general decision model for Pointsman

Status: proposed research plan, 2026-10-03; source baseline `39bf5a4`.
This delivery changes documentation only. Training, paid comparisons and
deployment require a subsequent execution scope and budget. The integration
boundary is defined in [ARCHITECTURE.md](ARCHITECTURE.md).

## Three independent targets

1. Improve agent task outcomes and total cost/latency relative to the normal host.
2. Outperform a pinned Jev baseline on identical decision inputs and independent gold.
3. Generalize across unseen domains, instructions, schemas, languages and hosts.

All three are research targets. A win on one is not evidence of the others.
Current reference-label agreement and warm inference measurements do not prove
Jev superiority or live task savings. Broad superiority remains unestablished.

## Feasibility verdict

| Target | Feasibility assessment | What would establish success |
|---|---|---|
| A. Completed-task quality, cost and latency | Plausible on frequent bounded decisions; not established | Independent task-success non-regression plus lower total cost and latency against a competent normal host and, where permitted, a Jev-assisted host |
| B. Same-input decision quality above Jev | Plausible on specialized rules, Korean/English and selected domains; unmeasured | A positive paired difference on independent gold against a fixed Jev version, with calibrated risk/coverage and scope held comparable |
| C. Broad decision generalization above Jev | A research possibility, not a supportable promise for the current 322M model | Superiority across preregistered unseen domain/schema/rule/language/host suites, including failures and unsupported requests |

A finite benchmark cannot prove superiority on every possible decision. Define
the intended broad operating envelope before testing, then report exactly that
scope. All three ledgers must pass for an overall “meets all targets” result.
Passing A alone supports an efficient system, not a stronger general model.
Without an authorized Jev comparison, Jev superiority stays **UNKNOWN**.

Training only to copy Jev would inherit its mistakes and would not supply an
independent measure of truth. A student can sometimes outperform a noisy teacher
on a distribution through better supervision or inductive bias; this is neither
a universal impossibility nor a guaranteed consequence of distillation. Our
route to improvement is independently justified labels, local task evidence,
better question-conditioned learning and calibrated abstention.

## Current evidence and missing capabilities

The committed [model card](model-card/README.md) records a multilingual mmBERT
base of about 322M parameters with **all parameters trained**, not a frozen
encoder with only a new classifier head. Training used 7,219 synthetic coding
tasks with Claude reference labels. Its 308-task test reports reference agreement:
intent 0.880, exact difficulty 0.718, risk 0.692. Of 101 routed cases, 12 were
unsafe relative to those reference labels (11.9%; reported upper bound 19.6%).
These are historical measurements, not newly reproduced results or live task
success. They do not justify unrestricted routing.

The historical warm three-question latency was 101 ms p50 / 167 ms p95 and cold
load about 5.5 s on M4 Pro. The training record is six epochs in about 12.8 hours.
New data, longer inputs, hardware availability and concurrent-agent performance
have not been measured here. Local inference avoids provider charges but still
uses compute, memory and maintenance effort.

| Existing component | Reuse | Required change before broad qualification |
|---|---|---|
| [Dataset grouping](src/training/dataset.mjs) | Task/request/state grouping and immutable exports | Add source/template/semantic-family groups; split translations, paraphrases and counterfactual siblings together |
| [Distillation grouping](src/training/laya-distill.mjs) | Explicit grouping concept | Do not treat old model labels as new independent gold or an automatic permission to reuse them |
| [Trainer](training/laya-kit/train_from_export.py) | Full encoder/head optimization; local MPS path; resumable workflow | Separate development from calibration; add independent dataset/provenance support |
| [Evidence evaluator](src/training/evaluate.mjs) | Question-specific evidence, separate shadow/active, observational comparisons | New utility targets for actual model×effort outcomes; do not relabel route intent/difficulty/risk as task success |
| [Lifecycle](src/training/laya-lifecycle.mjs) | Register, qualify, compare, explicit promote/rollback | Bind qualifications to decision family, data/preprocessing and runtime identity; add prospective holdout evidence |

Today the general exporter has train/calibration/test splits. The trainer uses
calibration to select the best epoch and then fits temperature on it, so that
calibration estimate has been exposed to model selection. The frozen regression
holdout is a copy of test, not an independent second sample. Preserve its value
as a regression floor while correcting these boundaries for new research.

## What the model must learn

Learn `P(answer | state, question, option meanings, current rules)`, rather than
memorizing the three routing questions or stable label IDs. Reuse Laya's
question/option-conditioned decision architecture and train encoder and scoring
parameters. A separate fixed head for every new business schema is not the
default design. Host names and role labels must not become shortcuts for quality.

| Curriculum | Required examples and counterexamples |
|---|---|
| Grounding | Entailed, contradicted and missing facts; evidence IDs; quoted instructions treated as data |
| Dynamic schema | Randomized label keys/order; new option descriptions; no valid option; overlapping candidates with an explicit rule |
| Rules | Same state with changed policy and changed gold; AND/OR, exceptions, priority, negation, correction and retraction |
| Agent decisions | Tool/skill selection, relevant evidence, authorized retry choices, escalation and next verification check |
| Domain transfer | Coding, support, documents, retrieval and workflow rules; Korean, English and mixed language; later new languages |
| Stress | Distractors, contradictions, stale evidence, missing fields, long inputs, many options and prompt injection in state |

Example: a request about an account maps to `alpha` when `alpha=account` and
`beta=delivery`, but must map to `beta` when those meanings are exchanged.
Renaming keys with meanings preserved should leave the semantic answer unchanged.
Changing an applicable business rule should change the answer when warranted.
Both tests matter; paraphrase agreement alone does not show rule following.

The [GLiClass paper](https://arxiv.org/abs/2508.07662) motivates learning from
text and label meanings, and [T0](https://arxiv.org/abs/2110.08207) motivates
training across tasks and prompt forms while holding tasks out. These are design
precedents, not performance evidence for this model.

## Data rights and paid work

As checked on 2026-10-03, the [TypeSafe MCA](https://typesafe.ai/legal/mca),
updated 2026-09-23, restricts using the service or its outputs for distillation,
imitation training or facilitating a similar/competing product in section
2.3(b). This plan therefore does not assume permission to collect Jev labels or
run competitive-development API comparisons. Resolve the applicable agreement
and any required permission before that paid stage. A permitted production
integration is not automatically permission for competitor training/evaluation.

Start with independently constructed rule gold, appropriately licensed public
datasets and independently verified task outcomes. Keep records of source,
license/terms, revision, transformation and redistribution permission. Existing
Claude reference labels retain `ai_reference` provenance and their original
model identity; a weight license does not establish rights to all training data.
Any new model-generated data needs its own permitted use. No new external teacher
is required for the first pilot.

Use three label classes explicitly:

- **Rule/source gold:** a reproducible oracle or licensed source establishes the
  label. The input must require semantic interpretation; do not train a model to
  replace arithmetic or other exact checks that code already handles better.
- **Independent adjudication/outcome:** task acceptance comes from an actual
  runner or independent review with evidence. Subjective judgments carry rater
  disagreement. A parser passing is not evidence for an unrelated semantic label.
- **AI reference/teacher:** auxiliary supervision and agreement measurements,
  never renamed human truth. Model explanations and self-confidence are not gold.

If AI is used to verbalize a rule-generated example, verify preservation of all
facts and conditions; exclude cases without a valid oracle. Do not invent uniform
soft targets just because a sentence sounds ambiguous. Missing evidence can be
an explicit answer; uncertain probability targets need a justified distribution.

Public commits contain schemas, methodology and approved aggregate results only.
Private conversations, raw captures, keys and unreviewed labels remain outside
Git. Capture, model download, training, paid labeling and publication are separate
operator actions; this documentation does not turn any of them on.

## Split and evaluation design before training

1. Deduplicate and form source/template/semantic-family groups before splitting.
   Keep all translated, paraphrased and counterfactual variants in one group.
2. Create train, **dev**, calibration and sealed test manifests. Dev selects
   architecture/loss/epoch; calibration fits probability transforms and gates;
   test is opened only for a frozen candidate decision.
3. Add whole unseen domain/schema/rule/language/host slices. Hide semantics and
   rule families, not just different names for the same template. Reserve a later
   time window and new families as a prospective confirmation set.
4. Audit label reproducibility, option balance, evidence retention and cross-split
   near-duplicates. Version tokenizer, state builder, questions and transformations.
5. Preserve the existing route regression set. If a test is repeatedly used to
   choose candidates, it has become development evidence; use a fresh sealed set
   for the final generalization claim.

Unknown labels stay unknown. Report how much of each slice has independent gold,
reference-only labels or no labels. The current store/export schemas do not
necessarily accept every proposed provenance field; implement and validate those
changes before importing, rather than forging a trusted label source.

## Training sequence and bounded experiments

| Phase | Work and starting budget | Exit / no-go |
|---|---|---|
| 0. Contract and data | Fix four-way splits, provenance, decision-family identities and evaluation manifests | Reproducible labels and no known split leakage; otherwise no training |
| 1. Local pilot | About 2,000 question rows; permitted base checkpoint; short inputs; one supervised run | Valid admission/resume, dev learning signal, rule-swap response, measured memory and throughput |
| 2. Multitask supervised candidate | About 20,000 question rows across roughly 8 domains/60 schemas, expanded only as needed | Better unseen-family dev results than base/d6 without route regression; quantities are starting designs, not sufficient-data claims |
| 3. Targeted expansion | Mine dev failures and newly adjudicated disagreement; consider up to 100,000 rows only if learning curves justify it | Improvement from genuinely new families; stop growth if rule/OOD performance stalls |
| 4. Loss ablation | Supervised baseline versus existing RL recipe with equal data, updates and reported compute | Retain RL only if independent quality/calibration gains justify added time/complexity |
| 5. Calibration and sealed evaluation | Fit gates once on calibration; freeze model/question/preprocessing revisions; run B/C comparisons when permitted | All preregistered quality, risk and coverage bounds pass; otherwise retain host fallback |
| 6. Host utility | Frozen sandbox tasks and then explicitly scoped native application/A-B evidence | Pass A using independently verified completed tasks, not recommendation agreement |
| 7. Release | Qualification, comparison against active, fixed regression plus prospective holdout; explicit promote/rollback | Only approved family/host capabilities enter the package; no automatic provider selection |

Begin a new general candidate from a permitted multilingual base, preserving d6
as a route reference. Do not inherit unreviewed teacher-data provenance by default.
Use CE for Choice and binary CE for Noul; equivalent two-option CE is acceptable.
Use distributional CE for Score, evaluating ordinal errors; compare a ranked
probability loss only if severe ordinal errors remain a measured failure mode.
KL/soft targets require justified distributions and data rights.

The current trainer already has a noise-perturbed probability reward with an
RL/GRPO-style term and CE. [TypeSafe describes RLCD](https://docs.typesafe.ai/introduction/machine-learning-primer)
as training for calibrated decisions, but that public description does not
establish an identical algorithm, corpus or reward implementation. “Add RLCD”
is not a substitute for a working supervised baseline and independent evaluation.

Initial tuning candidates can stay near the existing recipe: encoder learning
rate 1e-5–3e-5, head 3e-5–1e-4, dropout around 0.1, one to three epochs, selected
on dev. The historical d6 recipe is a reproducibility reference, not evidence
that six epochs or its temperature is optimal for general tasks. Change one
factor per bounded ablation; cap pilot attempts at three with a new hypothesis
for each. Stop and inspect when label integrity or admission fails.

Full fine-tuning at the historical short-input size has prior execution evidence.
New runtime estimates must come from the pilot's actual tokens/second, sequences/
second, peak memory, optimizer state and measured resume overhead. Estimate total
hours from those measurements and disclose variability. Do not promise a larger
run's cost from the old 12.8-hour result. External GPUs require a separate budget;
do not add infrastructure before local results establish the need.

Long context and cardinality are separate research axes. The current checkpoint
uses a 1024-token input / 256-token head budget; 255 meaningful choices cannot be
enabled by changing a validator constant. Extend tokenizer admission, model
input construction, training distribution, inference and calibration together.
Progressively test longer contexts with decisive evidence at the start/middle/end.
Candidate retrieval plus a small classifier is valid, but its end-to-end score
includes shortlist recall and it is not native 255-way classification.

Only consider a larger encoder, adapters/LoRA, quantization or shared-state
encoding after the failure analysis identifies capacity, memory or repeated
encoding as the bottleneck. Quantization is a separately calibrated candidate.
There is no current evidence that a 322M local model can match Jev throughout its
full input envelope at the desired latency.

## Evaluation and adoption

The [evaluation plan](EVALUATION.md) defines the three scorecards, actual routing
utility, statistical gates, full cost accounting and release/stop conditions.
