import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { MODES, ID, MODEL_ID, RESERVED, fail, isObject } from './constants.mjs';
import { atomicWrite, ensureDir, readText } from './storage.mjs';
import { containsSensitiveData } from './contracts.mjs';
import { discoverHostRoles } from './host-roles.mjs';

export const INTENTS = Object.freeze(['explain', 'edit', 'debug', 'operate', 'research', 'architecture', 'other']);
export const TIERS = Object.freeze(['economy', 'standard', 'strong']);
// Only these intents may carry a per-intent target override; operate/architecture/other are already delegated to the host before a target is chosen.
export const PROFILE_INTENTS = Object.freeze(['explain', 'research', 'edit', 'debug']);
// Codex's spawn_agent silently ignores a `model` argument and applies the role TOML's own model/effort (verified 2026-09-23,
// codex-cli 0.154.0 multi_agent_v2), so `model` there is display-only metadata, never something that changes execution.
// Claude's Agent tool does accept a model override, but only these three IDs; `fable` is intentionally excluded from profiles (owner decision).
export const CLAUDE_MODELS = Object.freeze(['haiku', 'sonnet', 'opus']);
// Default role assignment used both by v1->v2 migration and by `policy roles` preset generation.
export const DEFAULT_TIER_ROLES = Object.freeze({
  codex: { economy: 'lightweight_worker', standard: 'implementer', strong: 'specialist' },
  claude: { economy: 'lightweight-worker', standard: 'implementer', strong: 'specialist' },
});
export const DEFAULT_EXPLAIN_ROLE = Object.freeze({ codex: 'scout', claude: 'scout' });
const CLAUDE_TIER_MODEL = Object.freeze({ economy: 'haiku', standard: 'sonnet', strong: 'opus', explain: 'haiku' });

export const FEATURE_DEFAULTS = Object.freeze({
  version: 2,
  router: { mode: 'off', expectedModel: 'jev-1.13.0', minConfidence: 0.90, minProbability: 0.85,
    maxRiskProbability: 0.01, maxUnknownProbability: 0.05,
    economyMaxDifficulty: 2, standardMaxDifficulty: 3.5,
    economyMaxHardProbability: 0.02, standardMaxDeepProbability: 0.05,
    // Fraction (0..0.5) of otherwise-rewritten spawns randomly held back as an unmodified 'control'
    // arm instead of applying the route (see `pointsman router ab`); 0 means always apply (no A/B split).
    abControlShare: 0,
    profiles: { codex: {}, claude: {} } },
  bulk: { mode: 'off', expectedModel: 'jev-1.13.0', maxItems: 128, batchSize: 8,
    maxRequests: 16, maxTotalMs: 10000, minRejectConfidence: 0.97, minRejectProbability: 0.99 },
  // Phase 1 (shadow): attaches pointsman's existing route judgment to the Claude Code main loop so a
  // turn's reasoning effort can follow task difficulty. No new heads, no training; reuses the same
  // laya 'route' checkpoint/questions as `router`. See docs/plan/2026-09-27-pointsman-effort-mod.md.
  effort: { mode: 'off', input: 'prompt', contextChars: 600, lowerTo: 'medium', raiseTo: 'high',
    minLowerProbability: 0.7, maxHighRiskForLower: 0.2, minRaiseProbability: 0.5, raiseRiskProbability: 0.7,
    // Fraction (0..0.5) of otherwise-applied ON-mode changes randomly held back as an unmodified
    // 'control' arm instead of applying the recommended effort (see `pointsman effort ab`).
    abControlShare: 0,
    // Measured 2026-09-27: an effort change rewrites the whole MESSAGES part of the prompt cache
    // (Claude Code 2.1.280), and the owner's real main-loop requests run large enough (p50 ~443k
    // tokens, 1h cache TTL) that one avoidable rewrite costs far more than a turn's thinking-token
    // savings. 'cold-only' only ever applies a main-loop change when the previous main-loop request
    // of the session was more than coldAfterSeconds ago (or there was none); 'off' never applies to
    // the main loop. SHADOW always records the main-loop recommendation regardless of this setting.
    mainLoop: 'off', coldAfterSeconds: 3600,
    // A subagent starts with a fresh context, so fixing its effort from its own first step has no
    // cache penalty; 'on' allows applying it (only meaningful once `mode` is 'on' -- SHADOW always
    // records regardless of this setting; never changes a subagent's model or role).
    subagents: 'off' },
  evidence: { mode: 'off' },
  workflow: { mode: 'off', nativeMode: 'off', maxActions: 128, maxMs: 10000,
    maxDecisionCalls: 2, maxOutputBytes: 24000 },
});
function fields(value, allowed) {
  if (!isObject(value) || ![Object.prototype, null].includes(Object.getPrototypeOf(value)) ||
      Object.keys(value).some(k => RESERVED.has(k) || !allowed.includes(k))) fail('INVALID_FEATURE_POLICY');
}
function probability(value) { if (!Number.isFinite(value) || value < 0 || value > 1) fail('INVALID_FEATURE_POLICY'); }
function model(value) {
  if (typeof value !== 'string' || !MODEL_ID.test(value)) fail('INVALID_FEATURE_POLICY');
}
function claudeModel(value) {
  if (typeof value !== 'string' || !CLAUDE_MODELS.includes(value)) fail('INVALID_FEATURE_POLICY');
}
function role(value) {
  if (typeof value !== 'string' || !ID.test(value) || RESERVED.has(value)) fail('INVALID_FEATURE_POLICY');
}
function checkModel(value, host) { if (host === 'claude') claudeModel(value); else model(value); }
function reasoningAndSkills(value) {
  if (value.reasoning !== undefined && (typeof value.reasoning !== 'string' || !ID.test(value.reasoning))) fail('INVALID_FEATURE_POLICY');
  if (value.skills !== undefined) {
    fields(value.skills, INTENTS);
    for (const ids of Object.values(value.skills)) {
      if (!Array.isArray(ids) || ids.length > 4 || new Set(ids).size !== ids.length ||
          ids.some(id => typeof id !== 'string' || !ID.test(id) || RESERVED.has(id))) fail('INVALID_FEATURE_POLICY');
    }
  }
}
// v1 target: {model (required), reasoning?, skills?}. No role, no intents.
function targetV1(value) {
  fields(value, ['model', 'reasoning', 'skills']);
  model(value.model); // legacy shape never restricted Claude to a fixed model set
  reasoningAndSkills(value);
}
function profileV1(value) {
  fields(value, TIERS);
  for (const target of Object.values(value)) targetV1(target);
}
// v2 target: {role (required), model?, reasoning?, skills?}.
function targetV2(value, host) {
  fields(value, ['role', 'model', 'reasoning', 'skills']);
  role(value.role);
  if (value.model !== undefined) checkModel(value.model, host);
  reasoningAndSkills(value);
}
function profileV2(value, host) {
  fields(value, [...TIERS, 'intents']);
  for (const tier of TIERS) if (value[tier] !== undefined) targetV2(value[tier], host);
  if (value.intents !== undefined) {
    fields(value.intents, PROFILE_INTENTS);
    for (const target of Object.values(value.intents)) targetV2(target, host);
  }
}
function migrateTargetV1(target, host, tier) {
  const built = { role: DEFAULT_TIER_ROLES[host][tier], model: target.model,
    ...(target.reasoning !== undefined ? { reasoning: target.reasoning } : {}),
    ...(target.skills !== undefined ? { skills: target.skills } : {}) };
  targetV2(built, host); // fail-closed: a v1 target that cannot satisfy v2 (e.g. a non-preset Claude model) blocks the whole load
  return built;
}
function migrateProfileV1(value, host) {
  const migrated = {};
  for (const tier of TIERS) if (value[tier] !== undefined) migrated[tier] = migrateTargetV1(value[tier], host, tier);
  return migrated;
}
export function validateFeaturePolicy(raw) {
  fields(raw, ['version', 'router', 'bulk', 'effort', 'workflow', 'evidence']);
  if (Object.hasOwn(raw, 'version') && raw.version !== 1 && raw.version !== 2) fail('INVALID_FEATURE_POLICY');
  const version = raw.version ?? 1;
  fields(Object.hasOwn(raw, 'router') ? raw.router : {}, Object.keys(FEATURE_DEFAULTS.router));
  fields(Object.hasOwn(raw, 'bulk') ? raw.bulk : {}, Object.keys(FEATURE_DEFAULTS.bulk));
  // An existing features.json written before `effort` existed has no `effort` key at all; it must
  // keep loading with the OFF defaults below rather than failing INVALID_FEATURE_POLICY.
  fields(Object.hasOwn(raw, 'effort') ? raw.effort : {}, Object.keys(FEATURE_DEFAULTS.effort));
  fields(Object.hasOwn(raw, 'workflow') ? raw.workflow : {}, Object.keys(FEATURE_DEFAULTS.workflow));
  fields(Object.hasOwn(raw, 'evidence') ? raw.evidence : {}, Object.keys(FEATURE_DEFAULTS.evidence));
  // The in-memory result is always current-schema v2, even when `raw` was v1; the file itself is
  // rewritten as v2 only the next time a write path (setFeatureMode, presetHostRoles, ...) saves it.
  const policy = { version: 2,
    router: { ...structuredClone(FEATURE_DEFAULTS.router), ...raw.router },
    bulk: { ...FEATURE_DEFAULTS.bulk, ...raw.bulk },
    effort: { ...FEATURE_DEFAULTS.effort, ...raw.effort },
    workflow: { ...FEATURE_DEFAULTS.workflow, ...raw.workflow },
    evidence: { ...FEATURE_DEFAULTS.evidence, ...raw.evidence } };
  if (!MODES.includes(policy.evidence.mode)) fail('INVALID_FEATURE_POLICY');
  for (const feature of [policy.router, policy.bulk]) {
    if (!MODES.includes(feature.mode) || !/^jev-\d+\.\d+\.\d+$/.test(feature.expectedModel)) fail('INVALID_FEATURE_POLICY');
  }
  const r = policy.router, b = policy.bulk;
  for (const key of ['minConfidence', 'minProbability', 'maxRiskProbability', 'maxUnknownProbability', 'economyMaxHardProbability', 'standardMaxDeepProbability']) probability(r[key]);
  if (r.minConfidence < 0.5 || r.minProbability < 0.5 || r.maxRiskProbability > 0.1 || r.maxUnknownProbability > 0.2) fail('INVALID_FEATURE_POLICY');
  if (!Number.isFinite(r.abControlShare) || r.abControlShare < 0 || r.abControlShare > 0.5) fail('INVALID_FEATURE_POLICY');
  if (!Number.isFinite(r.economyMaxDifficulty) || !Number.isFinite(r.standardMaxDifficulty) || !(r.economyMaxDifficulty >= 1 && r.economyMaxDifficulty <= r.standardMaxDifficulty && r.standardMaxDifficulty <= 5)) fail('INVALID_FEATURE_POLICY');
  fields(r.profiles, ['codex', 'claude']);
  for (const host of Object.keys(r.profiles)) {
    const p = r.profiles[host];
    if (version === 1) { profileV1(p); r.profiles[host] = migrateProfileV1(p, host); }
    else profileV2(p, host);
  }
  for (const [key, min, max] of [['maxItems', 1, 128], ['batchSize', 1, 8], ['maxRequests', 1, 32], ['maxTotalMs', 100, 10000]]) {
    if (!Number.isInteger(b[key]) || b[key] < min || b[key] > max) fail('INVALID_FEATURE_POLICY');
  }
  for (const key of ['minRejectConfidence', 'minRejectProbability']) { probability(b[key]); if (b[key] < 0.9) fail('INVALID_FEATURE_POLICY'); }
  const ef = policy.effort;
  if (!MODES.includes(ef.mode)) fail('INVALID_FEATURE_POLICY');
  if (!['prompt', 'prompt+context'].includes(ef.input)) fail('INVALID_FEATURE_POLICY');
  if (!Number.isInteger(ef.contextChars) || ef.contextChars < 0 || ef.contextChars > 2000) fail('INVALID_FEATURE_POLICY');
  if (!['low', 'medium'].includes(ef.lowerTo)) fail('INVALID_FEATURE_POLICY');
  if (!['high', 'xhigh', 'max'].includes(ef.raiseTo)) fail('INVALID_FEATURE_POLICY');
  for (const key of ['minLowerProbability', 'maxHighRiskForLower', 'minRaiseProbability', 'raiseRiskProbability']) probability(ef[key]);
  if (!Number.isFinite(ef.abControlShare) || ef.abControlShare < 0 || ef.abControlShare > 0.5) fail('INVALID_FEATURE_POLICY');
  if (!['off', 'cold-only'].includes(ef.mainLoop)) fail('INVALID_FEATURE_POLICY');
  if (!Number.isInteger(ef.coldAfterSeconds) || ef.coldAfterSeconds < 300 || ef.coldAfterSeconds > 86400) fail('INVALID_FEATURE_POLICY');
  if (!['off', 'on'].includes(ef.subagents)) fail('INVALID_FEATURE_POLICY');
  const wf = policy.workflow;
  if (!MODES.includes(wf.mode) || !MODES.includes(wf.nativeMode)) fail('INVALID_FEATURE_POLICY');
  for (const [key, min, max] of [['maxActions', 1, 1024], ['maxMs', 100, 120000],
    ['maxDecisionCalls', 0, 16], ['maxOutputBytes', 1024, 49152]]) {
    if (!Number.isSafeInteger(wf[key]) || wf[key] < min || wf[key] > max) fail('INVALID_FEATURE_POLICY');
  }
  if (containsSensitiveData(policy)) fail('SENSITIVE_FEATURE_POLICY');
  return structuredClone(policy);
}
export function loadFeaturePolicy(home) {
  const text = readText(path.join(home, 'features.json'), { optional: true, privateFile: true, maxBytes: 16384 });
  let raw; try { raw = text === null ? {} : JSON.parse(text); } catch { fail('INVALID_FEATURE_POLICY'); }
  return validateFeaturePolicy(raw);
}
export function policyFingerprint(policy) { return createHash('sha256').update(JSON.stringify(policy)).digest('hex'); }
export function setFeatureMode(home, feature, mode) {
  if (!['router', 'bulk', 'effort', 'workflow', 'evidence'].includes(feature) || !MODES.includes(mode)) fail('INVALID_FEATURE_MODE');
  ensureDir(home, true);
  const file = path.join(home, 'features.json');
  const previous = readText(file, { optional: true, privateFile: true });
  let policy;
  try { policy = loadFeaturePolicy(home); }
  catch (error) { if (mode !== 'off') throw error; policy = validateFeaturePolicy({}); }
  policy[feature].mode = mode;
  atomicWrite(file, JSON.stringify(policy, null, 2) + '\n', { expected: previous });
  return policy;
}
export function setWorkflowNativeMode(home, mode) {
  if (!MODES.includes(mode)) fail('INVALID_FEATURE_MODE');
  ensureDir(home, true);
  const file = path.join(home, 'features.json');
  const previous = readText(file, { optional: true, privateFile: true });
  const policy = loadFeaturePolicy(home);
  policy.workflow.nativeMode = mode;
  atomicWrite(file, JSON.stringify(validateFeaturePolicy(policy), null, 2) + '\n', { expected: previous });
  return policy;
}
export function evidencePolicy(home, globalMode) {
  const policy = loadFeaturePolicy(home);
  return { ...policy.workflow, mode: effectiveMode(globalMode, policy.evidence.mode),
    nativeMode: 'off', revision: policyFingerprint(policy) };
}
export function workflowPolicy(home, globalMode) {
  const policy = loadFeaturePolicy(home);
  const workflow = policy.workflow;
  const mode = effectiveMode(globalMode, workflow.mode);
  return { ...workflow, mode, nativeMode: effectiveMode(mode, workflow.nativeMode),
    revision: policyFingerprint(policy) };
}
/** `pointsman router ab <share>|off` / `pointsman effort ab <share>|off`: writes only that feature's
 * abControlShare; every other field (including the other feature's) is preserved. */
export function setAbControlShare(home, value, feature = 'router') {
  if (!['router', 'effort'].includes(feature)) fail('INVALID_FEATURE_MODE');
  const share = value === 'off' ? 0 : Number(value);
  if (!Number.isFinite(share) || share < 0 || share > 0.5) fail('INVALID_AB_CONTROL_SHARE');
  ensureDir(home, true);
  const file = path.join(home, 'features.json');
  const previous = readText(file, { optional: true, privateFile: true });
  const policy = loadFeaturePolicy(home);
  policy[feature].abControlShare = share;
  const validated = validateFeaturePolicy(policy);
  atomicWrite(file, JSON.stringify(validated, null, 2) + '\n', { expected: previous });
  return validated;
}
export function initializeFeaturePolicy(home) {
  ensureDir(home, true);
  const file = path.join(home, 'features.json');
  const previous = readText(file, { optional: true, privateFile: true });
  if (previous !== null) return { created: false, path: file, policy: loadFeaturePolicy(home) };
  const policy = validateFeaturePolicy({});
  atomicWrite(file, JSON.stringify(policy, null, 2) + '\n', { expected: null });
  return { created: true, path: file, policy };
}
export function effectiveMode(globalMode, featureMode) {
  if (globalMode === 'off' || featureMode === 'off') return 'off';
  return globalMode === 'shadow' || featureMode === 'shadow' ? 'shadow' : 'on';
}
/** Pure preset builder: only default roles the host actually has agent definitions for are included. */
export function buildRolePreset(host, discoveredRoles) {
  const tierRoles = DEFAULT_TIER_ROLES[host], explainRole = DEFAULT_EXPLAIN_ROLE[host];
  const missingRoles = [];
  const profile = {};
  for (const tier of TIERS) {
    const roleId = tierRoles[tier];
    if (!discoveredRoles.includes(roleId)) { missingRoles.push(roleId); continue; }
    profile[tier] = { role: roleId, ...(host === 'claude' ? { model: CLAUDE_TIER_MODEL[tier] } : {}) };
  }
  if (discoveredRoles.includes(explainRole)) {
    profile.intents = { explain: { role: explainRole, ...(host === 'claude' ? { model: CLAUDE_TIER_MODEL.explain } : {}) } };
  } else if (!missingRoles.includes(explainRole)) missingRoles.push(explainRole);
  return { profile, missingRoles };
}
/** `pointsman policy roles`: writes only the requested host's profile; every other host, mode and threshold is preserved. */
export function presetHostRoles(home, host, { env = process.env, dryRun = false, replace = false, discover = discoverHostRoles } = {}) {
  if (!['codex', 'claude'].includes(host)) fail('INVALID_HOST');
  const discovered = discover(host, { env, userHome: env.HOME || os.homedir() });
  const { profile, missingRoles } = buildRolePreset(host, discovered);
  const roleCount = new Set(TIERS.map(tier => profile[tier]?.role).filter(Boolean)).size;
  if (roleCount < 2) fail('INSUFFICIENT_DISCOVERED_ROLES');
  if (dryRun) return { written: false, host, profile, missingRoles };
  ensureDir(home, true);
  const file = path.join(home, 'features.json');
  const previous = readText(file, { optional: true, privateFile: true });
  const policy = loadFeaturePolicy(home);
  const existing = policy.router.profiles[host] ?? {};
  const existingNonEmpty = TIERS.some(tier => existing[tier] !== undefined) || existing.intents !== undefined;
  if (existingNonEmpty && !replace) fail('PROFILE_EXISTS');
  policy.version = 2;
  policy.router.profiles[host] = profile;
  const validated = validateFeaturePolicy(policy);
  atomicWrite(file, JSON.stringify(validated, null, 2) + '\n', { expected: previous });
  return { written: true, host, profile: validated.router.profiles[host], missingRoles };
}
