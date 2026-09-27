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
    profiles: { codex: {}, claude: {} } },
  bulk: { mode: 'off', expectedModel: 'jev-1.13.0', maxItems: 128, batchSize: 8,
    maxRequests: 16, maxTotalMs: 10000, minRejectConfidence: 0.97, minRejectProbability: 0.99 },
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
  fields(raw, ['version', 'router', 'bulk']);
  if (Object.hasOwn(raw, 'version') && raw.version !== 1 && raw.version !== 2) fail('INVALID_FEATURE_POLICY');
  const version = raw.version ?? 1;
  fields(Object.hasOwn(raw, 'router') ? raw.router : {}, Object.keys(FEATURE_DEFAULTS.router));
  fields(Object.hasOwn(raw, 'bulk') ? raw.bulk : {}, Object.keys(FEATURE_DEFAULTS.bulk));
  // The in-memory result is always current-schema v2, even when `raw` was v1; the file itself is
  // rewritten as v2 only the next time a write path (setFeatureMode, presetHostRoles, ...) saves it.
  const policy = { version: 2,
    router: { ...structuredClone(FEATURE_DEFAULTS.router), ...raw.router },
    bulk: { ...FEATURE_DEFAULTS.bulk, ...raw.bulk } };
  for (const feature of [policy.router, policy.bulk]) {
    if (!MODES.includes(feature.mode) || !/^jev-\d+\.\d+\.\d+$/.test(feature.expectedModel)) fail('INVALID_FEATURE_POLICY');
  }
  const r = policy.router, b = policy.bulk;
  for (const key of ['minConfidence', 'minProbability', 'maxRiskProbability', 'maxUnknownProbability', 'economyMaxHardProbability', 'standardMaxDeepProbability']) probability(r[key]);
  if (r.minConfidence < 0.5 || r.minProbability < 0.5 || r.maxRiskProbability > 0.1 || r.maxUnknownProbability > 0.2) fail('INVALID_FEATURE_POLICY');
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
  if (!['router', 'bulk'].includes(feature) || !MODES.includes(mode)) fail('INVALID_FEATURE_MODE');
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
