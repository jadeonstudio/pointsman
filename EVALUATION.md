# Evaluating Pointsman against Jev and the normal host

Status: proposed acceptance design, 2026-10-03; no new results claimed.
Read the [training plan](TRAINING_PLAN.md) for feasibility, data rights and
learning stages, and the [architecture](ARCHITECTURE.md) for host boundaries.

## Metrics and explicit gates

Freeze an evaluation specification before a paid or sealed run: corpus hashes,
model revisions, prompts/questions, supported envelope, timeout, seeds, hardware,
primary metric, risk budget, non-inferiority margins and planned comparisons.
Select margins using operational tolerance and pilot variance, not final results.
Insufficient statistical power yields INCONCLUSIVE rather than PASS.

| Ledger | Baselines | Required measurements and pass rule |
|---|---|---|
| A: tasks | Normal competent host; rule-only policy; current d6; new candidate; permitted Jev-assisted host; all-strong diagnostic arm | Verified success non-inferiority, cost per successful task and end-to-end p50/p95. For the requested combined win, cost and latency must improve; a tradeoff is reported separately. |
| B: decisions | Rule baseline where applicable; unmodified base; d6; supervised candidate; RL candidate; permitted pinned Jev | Same frozen state/questions/options and independent gold. Primary paired quality difference must have a positive lower 95% confidence bound; critical error/calibration/coverage gates must also pass. |
| C: generalization | Same candidates on preregistered unseen slices | Positive lower 95% confidence bound for the unseen-family macro difference against permitted pinned Jev, plus every critical slice floor. Unavailable comparison = UNKNOWN. Count rejections/timeouts; report common-envelope quality separately from full target-envelope coverage. No aggregate win hiding a critical regression. |

Choice/Noul: accuracy, macro F1, NLL and Brier score; reliability diagrams with
sample counts and uncertainty. Score: ordinal MAE, severe-distance error and
appropriate distribution scoring. Use task/source/family clustered comparisons
or cluster bootstrap, since sibling questions are not independent samples.
Predeclare critical comparisons to avoid picking a favorable subgroup afterward.

Compare error at matched coverage and coverage at matched risk; evaluate selective
error upper bounds at the proposed application threshold. A model that abstains
on everything has not won. Report effective sample sizes per gate; zero observed
errors in a small sample does not prove zero risk. A simple independent-sample
approximation needs roughly 300 accepted zero-error cases to put a one-sided 95%
upper bound near 1%; clustering can require more. This is a planning illustration,
not a universal acceptance rule.

Fit temperature on calibration separately from epoch selection. It may improve
probability estimates but cannot fix wrong argmax answers or guarantee OOD
calibration. [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html)
provides the calibration method, not a task-specific guarantee. Preserve
[Choice confidence versus probability and Noul/Score semantics](ARCHITECTURE.md#probability-and-input-compatibility).

For a hypothetical `primary_quality` metric, write the actual numbers into the
preflight specification: `delta_success`, `max_critical_error`,
`min_coverage`, `max_p95_ratio`, and `min_cost_reduction`. They are planning
variables, not existing config keys. Until these and sample-size requirements
are fixed, there is no objective go/no-go decision for a live experiment.

## Learn actual routing utility separately

Intent/difficulty/risk agreement does not identify the cheapest successful model
or its best reasoning effort. The [runner](src/training/runner.mjs) already
rejects task-success labels for those route questions, and the evaluator prevents
shadow advice from inheriting an active execution's outcome. Preserve that rule.

Use a separate action-utility target: task and available model×effort candidates
→ observed success probability, total cost and duration. Choose only among
actions meeting a quality/risk floor; abstain when unsupported. Obtain evidence
from safe, controlled runs of the same frozen task/repository/checks, including
retries, escalation and independent acceptance. Randomize order and use repeated
runs where stochastic variation matters. Keep all descendants of the same task
in one split. Record action-selection propensity if using randomized allocation.

Do not infer unexecuted alternatives from selected-arm success. Observational
preferences and human/AI suggestions are weaker evidence than comparable runs.
Paired replay proves performance on the replay, not natural deployment. A native
A/B phase needs independently captured completion quality; the current hook
metrics aggregate cost/time and cannot by themselves close that requirement.
[RouteLLM](https://arxiv.org/abs/2406.18665) is relevant routing research, not a
substitute for our model/effort/task outcome measurements.

## Economic and latency model

For N attempted tasks with S independently verified successes:

```text
total_cost = allocated_training_and_maintenance_cost
           + sum(decision + agent + retry + escalation + verification costs)
cost_per_success = total_cost / S
break_even_requests = fixed_incremental_cost / (jev_path_cost - local_path_cost)
```

The last formula applies only when comparable-quality net per-request savings
are positive. Include local energy/equipment allocation, labeling, failed runs,
cache read/write charges, extra model turns and human review. Count missing
success evidence separately; do not put UNKNOWN cases in the success denominator.
For subscriptions, API-equivalent token cost is an estimate, not measured savings
in the subscription bill or quota.

The supplied article's $0.042 per million input tokens would imply $0.42 for
10,000 requests of 1,000 tokens, before any other costs. This is arithmetic using
the article's quoted rate, **not a verified current tariff or our measured bill**.
It illustrates why avoiding Jev charges alone may not repay model development.
Local privacy/offline availability, task-specific quality and expensive downstream
work avoided can be stronger benefits. Fetch current contracted rates only when
preparing an authorized paid comparison.

Measure full completion latency plus decision warm/cold p50/p95, queueing under
multiple agents, serialization, startup and fallback. A warm local 101 ms figure
cannot be compared directly with someone else's network median. If an extra MCP
turn erases the savings, move the decision into an owned dispatch boundary or
do not use that decision at all; do not claim gains from synthetic fixtures.

## Stop conditions and release evidence

Stop expansion on split leakage, invalid gold, missing usage rights, unsafe
downsizing, critical-slice regression, sustained unseen-rule stagnation, excessive
input loss or non-positive net efficiency. Diagnose that boundary before spending
more tokens/epochs. The existing host and prior checkpoint remain available.

An accepted release needs dataset/split hashes and provenance, code/model/runtime
identities, independent B/C results, task utility evidence for A, thresholds and
coverage, cold/warm/concurrent measurements, limitations, qualification, active
comparison and tested rollback. Publish only permitted artifacts with a revised
model card. Provider selection and checkpoint promotion remain explicit actions.

Current result: the architecture and a feasible staged investigation are defined;
new training, native verification and Jev superiority have **not** been performed
or established by this documentation change.
