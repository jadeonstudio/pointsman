import { INTENTS, TIERS } from './feature-policy.mjs';
import { ID, MODEL_ID, RESERVED, fail, isObject } from './constants.mjs';

// Original implementation inspired by classifier.dev's separated dimensions.
// A five-level Score is zero-based on the wire; displayed difficulty is score + 1.
export const ROUTE_QUESTIONS = {
  intent: { type: 'choice', instructions: 'Classify the work actually required. Use other when the request is unclear or outside these categories.', criteria: {
    explain: 'Explain or locate existing code; do not change behavior', edit: 'Write or change a bounded piece of code or documentation',
    debug: 'Investigate the cause of a failure', operate: 'Operate infrastructure, deploy, or modify live data',
    research: 'Research information outside the repository', architecture: 'Cross-module design, whole-repository audit or major refactoring',
    other: 'Insufficient evidence or none of these intents' } },
  difficulty: { type: 'score', instructions: 'Estimate the reasoning needed to finish, not prompt length. Account for unresolved dependencies and unknown scope. Do not infer that a short request is easy.', criteria: [
    '1: Mechanical, exact edit with explicit location and no behavioral change',
    '2: Small local change with explicit requirements and known validation',
    '3: Moderate implementation requiring several related steps',
    '4: Difficult debugging or interacting components needing substantial investigation',
    '5: Deep reasoning, unknown repository-wide impact, architecture or long-horizon planning' ] },
  risk: { type: 'choice', instructions: 'Classify consequence and uncertainty, not permission. Safe requires evidence of local, reversible impact. Missing context is unknown. Risk is independent of difficulty.', criteria: {
    safe: 'Known local, reversible work with no security, production, payment or financial impact',
    caution: 'Material uncertainty or changes needing additional review',
    high: 'Irreversible operation, production, credentials, permissions, financial or payment consequences',
    unknown: 'Not enough context to assess impact' } },
};
function keys(value, allowed) {
  if (!isObject(value) || Object.keys(value).some(k => !allowed.includes(k) || RESERVED.has(k))) fail('INVALID_ROUTE_REQUEST');
}
function profileTargets(policy, host) {
  const profile = policy.profiles[host] ?? {};
  return [...TIERS.map(tier => profile[tier]).filter(Boolean), ...Object.values(profile.intents ?? {})];
}
function targetAvailable(target, input) {
  if (!target || !input.availableRoles.includes(target.role)) return false;
  if (target.model !== undefined && input.availableModels !== undefined && !input.availableModels.includes(target.model)) return false;
  return true;
}
export function validateRouteInput(input) {
  keys(input, ['task', 'host', 'risk', 'context', 'availableRoles', 'availableModels', 'availableSkills']);
  if (typeof input.task !== 'string' || !input.task.trim() || Buffer.byteLength(input.task) > 8000 ||
      !['codex', 'claude'].includes(input.host) || !['routine', 'sensitive'].includes(input.risk)) fail('INVALID_ROUTE_REQUEST');
  keys(input.context, ['complete', 'scope', 'previousFailures', 'highImpact', 'modelLocked', 'exhaustive']);
  const c = input.context;
  for (const key of ['complete', 'highImpact', 'modelLocked', 'exhaustive']) if (typeof c[key] !== 'boolean') fail('INVALID_ROUTE_REQUEST');
  if (!['local', 'cross-module', 'repository', 'unknown'].includes(c.scope) || !Number.isInteger(c.previousFailures) || c.previousFailures < 0 || c.previousFailures > 100) fail('INVALID_ROUTE_REQUEST');
  if (!Array.isArray(input.availableRoles) || input.availableRoles.length > 32 || new Set(input.availableRoles).size !== input.availableRoles.length ||
      input.availableRoles.some(r => typeof r !== 'string' || !ID.test(r))) fail('INVALID_ROUTE_REQUEST');
  if (input.availableModels !== undefined && (!Array.isArray(input.availableModels) || input.availableModels.length > 32 ||
      new Set(input.availableModels).size !== input.availableModels.length ||
      input.availableModels.some(m => typeof m !== 'string' || !MODEL_ID.test(m)))) fail('INVALID_ROUTE_REQUEST');
  const skills = input.availableSkills ?? [];
  if (!Array.isArray(skills) || skills.length > 128 || new Set(skills).size !== skills.length || skills.some(s => typeof s !== 'string' || !ID.test(s) || RESERVED.has(s))) fail('INVALID_ROUTE_REQUEST');
  return structuredClone({ ...input, availableSkills: skills });
}
export function routeGuard(input, policy) {
  const c = input.context;
  if (input.risk === 'sensitive' || c.highImpact) return 'SENSITIVE_SCOPE';
  if (c.modelLocked) return 'MODEL_LOCKED';
  if (c.exhaustive || c.scope === 'repository' || c.scope === 'cross-module') return 'SCOPE_REQUIRES_HOST';
  if (!c.complete || c.scope === 'unknown') return 'INCOMPLETE_CONTEXT';
  if (c.previousFailures > 0) return 'PRIOR_FAILURE';
  const available = new Set(profileTargets(policy, input.host).map(t => t.role).filter(role => input.availableRoles.includes(role)));
  return available.size < 2 ? 'INSUFFICIENT_TARGETS' : null;
}
export function routeRequest(input) {
  // Models, skill IDs and local policy are not sent to TypeSafe.
  return { purpose: 'route', risk: input.risk,
    state: { task: input.task, context: input.context }, questions: structuredClone(ROUTE_QUESTIONS) };
}
// Shared tail: resolve a decided tier into a host target. A weak-role intent override never
// applies to 'strong': difficult (or, for the decision gate, uncertain-but-strong) work must
// never be quietly handed to a lightweight role.
function resolveTarget(tier, intentValue, rule, features, input, policy) {
  const profile = policy.profiles[input.host] ?? {};
  let target = profile[tier], intentOverride = false;
  if (tier !== 'strong' && profile.intents?.[intentValue]) { target = profile.intents[intentValue]; intentOverride = true; }
  if (!targetAvailable(target, input)) return { reason: 'TARGET_UNAVAILABLE', features };
  const skills = (target.skills?.[intentValue] ?? []).filter(id => input.availableSkills.includes(id));
  return { reason: rule, features, route: { tier, role: target.role, ...(target.model !== undefined ? { model: target.model } : {}),
    ...(target.reasoning ? { reasoning: target.reasoning } : {}), skills, ...(intentOverride ? { intentOverride: true } : {}) } };
}
export function chooseRoute(answers, input, policy) {
  const { intent, difficulty, risk } = answers;
  if (!intent || !difficulty || !risk) return { reason: 'MISSING_DIMENSIONS' };
  const mean = difficulty.value + 1;
  const hardTail = (difficulty.probabilities['3'] ?? 0) + (difficulty.probabilities['4'] ?? 0);
  const deepTail = difficulty.probabilities['4'] ?? 0;
  const features = { intent: intent.value, difficulty: mean, hardTail, deepTail, risk: risk.value };
  // Check uncertainty and risk BEFORE cheap-route rules (unlike the source demo).
  if ([intent, difficulty, risk].some(a => a.confidence < policy.minConfidence) ||
      intent.selectedProbability < policy.minProbability || risk.selectedProbability < policy.minProbability) return { reason: 'UNCERTAIN_DIMENSIONS', features };
  if (intent.value === 'other' || (intent.probabilities.other ?? 0) > policy.maxUnknownProbability) return { reason: 'UNKNOWN_INTENT', features };
  if (risk.value !== 'safe' || (risk.probabilities.high ?? 0) > policy.maxRiskProbability ||
      (risk.probabilities.unknown ?? 0) > policy.maxUnknownProbability ||
      (risk.probabilities.caution ?? 0) > policy.maxUnknownProbability) return { reason: 'RISK_REQUIRES_HOST', features };
  if (['operate', 'architecture'].includes(intent.value)) return { reason: 'INTENT_REQUIRES_HOST', features };
  let tier = 'strong', rule = 'STRONG_DEFAULT';
  if (['explain', 'edit'].includes(intent.value) && mean <= policy.economyMaxDifficulty && hardTail <= policy.economyMaxHardProbability) {
    tier = 'economy'; rule = 'BOUNDED_EASY';
  } else if (intent.value !== 'debug' && mean <= policy.standardMaxDifficulty && deepTail <= policy.standardMaxDeepProbability) {
    tier = 'standard'; rule = 'BOUNDED_MODERATE';
  }
  return resolveTarget(tier, intent.value, rule, features, input, policy);
}
// Deterministic label for a single (intent, difficulty, risk) triple, independent of any
// per-answer confidence/probability distribution. Must agree with chooseRoute on one-hot inputs
// (see tests/features.test.mjs); used both by chooseRouteByDecision (via tierDistribution) and by
// laya-lifecycle qualify to compute each task's reference tier from its recorded target labels.
export function labelTier(intent, difficultyIndex, risk, policy) {
  if (risk !== 'safe' || ['other', 'operate', 'architecture'].includes(intent)) return 'host';
  const displayed = difficultyIndex + 1;
  if (['explain', 'edit'].includes(intent) && displayed <= policy.economyMaxDifficulty) return 'economy';
  if (intent !== 'debug' && displayed <= policy.standardMaxDifficulty) return 'standard';
  return 'strong';
}
// Joint tier distribution from the three independent answer distributions (intent x difficulty x
// risk). The three heads are treated as independent -- their joint probability is the product.
export function tierDistribution(answers, policy) {
  const { intent, difficulty, risk } = answers;
  const dist = { host: 0, economy: 0, standard: 0, strong: 0 };
  for (const i of Object.keys(ROUTE_QUESTIONS.intent.criteria)) {
    const pi = intent.probabilities[i] ?? 0;
    if (!pi) continue;
    for (let d = 0; d <= 4; d++) {
      const pd = difficulty.probabilities[String(d)] ?? 0;
      if (!pd) continue;
      for (const r of Object.keys(ROUTE_QUESTIONS.risk.criteria)) {
        const pr = risk.probabilities[r] ?? 0;
        if (!pr) continue;
        dist[labelTier(i, d, r, policy)] += pi * pd * pr;
      }
    }
  }
  return dist;
}
function round4(dist) { return Object.fromEntries(Object.entries(dist).map(([k, v]) => [k, Math.round(v * 10000) / 10000])); }
// Pure tier-selection over a joint distribution: HOST_PROBABLE when the host share exceeds the
// gate's maxHostProbability; otherwise the cheapest tier (economy -> standard -> strong) whose
// cumulative probability first reaches tierCoverage; UNCERTAIN_TIER if none does (only possible
// when tierCoverage > 1 - P.host). Exported so laya-lifecycle qualify reuses the exact same
// decision rule the runtime gate applies, instead of a second copy that could drift.
export function decideTier(distribution, gate) {
  if (distribution.host > gate.maxHostProbability) return { applied: false, reason: 'HOST_PROBABLE' };
  let cumulative = 0;
  for (const tier of TIERS) {
    cumulative += distribution[tier];
    if (cumulative >= gate.tierCoverage) return { applied: true, tier, reason: `DECISION_${tier.toUpperCase()}` };
  }
  return { applied: false, reason: 'UNCERTAIN_TIER' };
}
// Decision-level routing gate (see providers.json laya.qualification.routeGate): gates on the
// tier decision the policy actually needs instead of on
// each answer's own confidence/selected-probability, with a threshold fitted offline by
// laya-lifecycle qualify. Never used unless the active checkpoint's qualification carries a
// fitted routeGate for purpose 'route'.
export function chooseRouteByDecision(answers, input, policy, gate) {
  const { intent, difficulty, risk } = answers;
  if (!intent || !difficulty || !risk) return { reason: 'MISSING_DIMENSIONS' };
  const distribution = tierDistribution(answers, policy);
  const features = { tierProbabilities: round4(distribution), intent: intent.value, difficulty: difficulty.value + 1, risk: risk.value };
  const d = decideTier(distribution, gate);
  if (!d.applied) return { reason: d.reason, features };
  return resolveTarget(d.tier, intent.value, d.reason, features, input, policy);
}
export async function routeOrDelegate(layer, input, { use, delegate, signal } = {}) {
  if (typeof use !== 'function' || typeof delegate !== 'function') fail('HANDLERS_REQUIRED');
  const result = await layer.route(input, { signal });
  // Host still owns supported model/effort validation, execution, permissions and tests.
  return result.apply ? use(result.route, result) : delegate(result);
}
export const routeSchema = {
  type: 'object', additionalProperties: false, required: ['task', 'host', 'risk', 'context', 'availableRoles'], properties: {
    task: { type: 'string', minLength: 1, maxLength: 8000 }, host: { type: 'string', enum: ['codex', 'claude'] },
    risk: { type: 'string', enum: ['routine', 'sensitive'], description: 'sensitive keeps the host choice (no cheaper route) — use it for production, credentials, permissions, payments or other hard-to-reverse work.' },
    context: { type: 'object', additionalProperties: false, required: ['complete', 'scope', 'previousFailures', 'highImpact', 'modelLocked', 'exhaustive'], properties: {
      complete: { type: 'boolean', description: 'true only when the task statement already contains what the worker needs (files, acceptance criteria); false keeps the host choice.' },
      scope: { type: 'string', enum: ['local', 'cross-module', 'repository', 'unknown'], description: 'Extent of the change. cross-module, repository and unknown keep the host choice.' },
      previousFailures: { type: 'integer', minimum: 0, maximum: 100, description: 'Earlier failed attempts at this same task; any value above 0 keeps the host choice.' },
      highImpact: { type: 'boolean', description: 'true for work whose mistakes are costly to undo; keeps the host choice.' },
      modelLocked: { type: 'boolean', description: 'true when the user or caller fixed the model; routing never overrides it.' },
      exhaustive: { type: 'boolean', description: 'true when every input must be covered (audits, full reviews); keeps the host choice.' },
    } },
    availableRoles: { type: 'array', maxItems: 32, uniqueItems: true, items: { type: 'string' }, description: 'Role names the host actually has agent definitions for (e.g. ~/.codex/agents/*.toml or ~/.claude/agents/*.md stems), not guessed names.' },
    availableModels: { type: 'array', maxItems: 32, uniqueItems: true, items: { type: 'string' }, description: 'Optional; actual model IDs available to this host, not guessed names. Codex ignores a spawn_agent model argument, so this rarely applies there.' },
    availableSkills: { type: 'array', maxItems: 128, uniqueItems: true, items: { type: 'string' } },
  },
};
