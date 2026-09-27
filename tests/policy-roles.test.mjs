import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { presetHostRoles, loadFeaturePolicy, validateFeaturePolicy, setFeatureMode } from '../src/feature-policy.mjs';
import { atomicWrite } from '../src/storage.mjs';
import { ControlError } from '../src/constants.mjs';

function tempHome() {
  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-policy-roles-')));
  fs.chmodSync(home, 0o700);
  return { home, cleanup: () => fs.rmSync(home, { recursive: true, force: true }) };
}
const codexDiscover = () => ['lightweight_worker', 'implementer', 'specialist', 'scout'];
const claudeDiscover = () => ['lightweight-worker', 'implementer', 'specialist', 'scout'];

test('policy roles preset assigns the default role table when every default role is present', () => {
  const { home, cleanup } = tempHome();
  try {
    const result = presetHostRoles(home, 'codex', { discover: codexDiscover });
    assert.equal(result.written, true);
    assert.deepEqual(result.missingRoles, []);
    assert.equal(result.profile.economy.role, 'lightweight_worker');
    assert.equal(result.profile.standard.role, 'implementer');
    assert.equal(result.profile.strong.role, 'specialist');
    assert.equal(result.profile.intents.explain.role, 'scout');
    assert.equal(loadFeaturePolicy(home).router.profiles.codex.strong.role, 'specialist');
  } finally { cleanup(); }
});

test('claude preset attaches haiku/sonnet/opus/haiku model hints; codex preset attaches no model', () => {
  const { home, cleanup } = tempHome();
  try {
    const codex = presetHostRoles(home, 'codex', { discover: codexDiscover });
    assert.equal(codex.profile.economy.model, undefined);
    const { home: home2, cleanup: cleanup2 } = tempHome();
    try {
      const claude = presetHostRoles(home2, 'claude', { discover: claudeDiscover });
      assert.equal(claude.profile.economy.model, 'haiku');
      assert.equal(claude.profile.standard.model, 'sonnet');
      assert.equal(claude.profile.strong.model, 'opus');
      assert.equal(claude.profile.intents.explain.model, 'haiku');
    } finally { cleanup2(); }
  } finally { cleanup(); }
});

test('a role the host does not have is left out and reported in missingRoles', () => {
  const { home, cleanup } = tempHome();
  try {
    const result = presetHostRoles(home, 'codex', { discover: () => ['implementer', 'specialist'] });
    assert.equal(result.profile.economy, undefined);
    assert.ok(result.missingRoles.includes('lightweight_worker'));
    assert.ok(result.missingRoles.includes('scout'));
    assert.equal(result.profile.intents, undefined);
  } finally { cleanup(); }
});

test('fewer than two available default roles refuses without writing', () => {
  const { home, cleanup } = tempHome();
  try {
    assert.throws(() => presetHostRoles(home, 'codex', { discover: () => ['specialist'] }), (e) => e instanceof ControlError && e.code === 'INSUFFICIENT_AVAILABLE_ROLES' || e.code === 'INSUFFICIENT_DISCOVERED_ROLES');
    assert.equal(loadFeaturePolicy(home).router.profiles.codex.economy, undefined);
  } finally { cleanup(); }
});

test('dry-run computes the preset but writes nothing to features.json', () => {
  const { home, cleanup } = tempHome();
  try {
    const before = fs.existsSync(path.join(home, 'features.json')) ? fs.readFileSync(path.join(home, 'features.json'), 'utf8') : null;
    const result = presetHostRoles(home, 'codex', { discover: codexDiscover, dryRun: true });
    assert.equal(result.written, false);
    assert.equal(result.profile.strong.role, 'specialist');
    const after = fs.existsSync(path.join(home, 'features.json')) ? fs.readFileSync(path.join(home, 'features.json'), 'utf8') : null;
    assert.equal(after, before);
  } finally { cleanup(); }
});

test('a non-empty existing profile is refused without --replace and kept without --replace', () => {
  const { home, cleanup } = tempHome();
  try {
    presetHostRoles(home, 'codex', { discover: codexDiscover });
    const before = fs.readFileSync(path.join(home, 'features.json'), 'utf8');
    assert.throws(() => presetHostRoles(home, 'codex', { discover: () => ['implementer', 'specialist'] }), (e) => e instanceof ControlError && e.code === 'PROFILE_EXISTS');
    assert.equal(fs.readFileSync(path.join(home, 'features.json'), 'utf8'), before);
  } finally { cleanup(); }
});

test('--replace overwrites only the requested host profile and preserves the other host, mode and thresholds', () => {
  const { home, cleanup } = tempHome();
  try {
    presetHostRoles(home, 'codex', { discover: codexDiscover });
    presetHostRoles(home, 'claude', { discover: claudeDiscover });
    setFeatureMode(home, 'router', 'shadow');
    const beforeClaude = loadFeaturePolicy(home).router.profiles.claude;
    const result = presetHostRoles(home, 'codex', { discover: () => ['implementer', 'specialist'], replace: true });
    assert.equal(result.written, true);
    assert.equal(result.profile.economy, undefined);
    const after = loadFeaturePolicy(home);
    assert.deepEqual(after.router.profiles.claude, beforeClaude);
    assert.equal(after.router.mode, 'shadow');
  } finally { cleanup(); }
});

test('an invalid host is refused', () => {
  const { home, cleanup } = tempHome();
  try { assert.throws(() => presetHostRoles(home, 'gpt', { discover: codexDiscover })); }
  finally { cleanup(); }
});

test('CLI: policy roles writes a preset, refuses a second call without --replace, and --dry-run never touches the file', () => {
  const { home, cleanup } = tempHome();
  try {
    const bin = fileURLToPath(new URL('../bin/pointsman.mjs', import.meta.url));
    const codexHome = path.join(home, 'codex-agents');
    fs.mkdirSync(path.join(codexHome, 'agents'), { recursive: true });
    for (const role of ['lightweight_worker', 'implementer', 'specialist', 'scout']) fs.writeFileSync(path.join(codexHome, 'agents', `${role}.toml`), '');
    const env = { PATH: process.env.PATH, HOME: home, CODEX_HOME: codexHome };
    const exec = args => spawnSync(process.execPath, [bin, ...args, '--home', home], { encoding: 'utf8', env });

    const dry = exec(['policy', 'roles', '--host', 'codex', '--dry-run']);
    assert.equal(dry.status, 0);
    assert.equal(JSON.parse(dry.stdout).written, false);
    assert.equal(loadFeaturePolicy(home).router.profiles.codex.economy, undefined);

    const first = exec(['policy', 'roles', '--host', 'codex']);
    assert.equal(first.status, 0);
    assert.equal(JSON.parse(first.stdout).profile.strong.role, 'specialist');

    const second = exec(['policy', 'roles', '--host', 'codex']);
    assert.notEqual(second.status, 0);
    assert.match(second.stderr, /PROFILE_EXISTS/);

    const replaced = exec(['policy', 'roles', '--host', 'codex', '--replace']);
    assert.equal(replaced.status, 0);
  } finally { cleanup(); }
});

test('CLI: an unknown --host value fails without writing', () => {
  const { home, cleanup } = tempHome();
  try {
    const bin = fileURLToPath(new URL('../bin/pointsman.mjs', import.meta.url));
    const env = { PATH: process.env.PATH, HOME: home };
    const r = spawnSync(process.execPath, [bin, 'policy', 'roles', '--host', 'gemini', '--home', home], { encoding: 'utf8', env });
    assert.notEqual(r.status, 0);
    assert.equal(fs.existsSync(path.join(home, 'features.json')), false);
  } finally { cleanup(); }
});

test('CLI: --host/--dry-run/--replace on an unrelated command are rejected', () => {
  const { home, cleanup } = tempHome();
  try {
    const bin = fileURLToPath(new URL('../bin/pointsman.mjs', import.meta.url));
    const env = { PATH: process.env.PATH, HOME: home };
    const r = spawnSync(process.execPath, [bin, 'status', '--host', 'codex', '--home', home], { encoding: 'utf8', env });
    assert.notEqual(r.status, 0);
  } finally { cleanup(); }
});

// --- v1 -> v2 migration ---
test('a v1 codex profile migrates in memory to v2 default roles', () => {
  const raw = { version: 1, router: { profiles: { codex: {
    economy: { model: 'gpt-mini' }, standard: { model: 'gpt-mid', reasoning: 'medium' }, strong: { model: 'gpt-max', skills: { debug: ['fixture-search'] } },
  }, claude: {} } } };
  const policy = validateFeaturePolicy(raw);
  assert.equal(policy.version, 2);
  assert.deepEqual(policy.router.profiles.codex.economy, { role: 'lightweight_worker', model: 'gpt-mini' });
  assert.equal(policy.router.profiles.codex.standard.role, 'implementer');
  assert.equal(policy.router.profiles.codex.strong.role, 'specialist');
  assert.deepEqual(policy.router.profiles.codex.strong.skills, { debug: ['fixture-search'] });
});

test('a v1 claude profile with a valid preset model migrates to v2', () => {
  const raw = { version: 1, router: { profiles: { claude: { economy: { model: 'haiku' }, standard: { model: 'sonnet' } }, codex: {} } } };
  const policy = validateFeaturePolicy(raw);
  assert.equal(policy.router.profiles.claude.economy.role, 'lightweight-worker');
  assert.equal(policy.router.profiles.claude.economy.model, 'haiku');
});

test('a v1 claude profile whose model is not haiku/sonnet/opus fails closed on load', () => {
  const { home, cleanup } = tempHome();
  try {
    atomicWrite(path.join(home, 'features.json'), JSON.stringify({ version: 1, router: { mode: 'on',
      profiles: { claude: { economy: { model: 'claude-3-opus-20240229' } }, codex: {} } }, bulk: { mode: 'on' } }));
    assert.throws(() => loadFeaturePolicy(home), (e) => e instanceof ControlError && e.code === 'INVALID_FEATURE_POLICY');
    // Fail-closed exactly like other malformed-policy cases: setFeatureMode router off still succeeds and control-layer status treats router as off.
    setFeatureMode(home, 'router', 'off');
    assert.equal(loadFeaturePolicy(home).router.mode, 'off');
  } finally { cleanup(); }
});

test('claude model "fable" is always rejected from a profile target', () => {
  assert.throws(() => validateFeaturePolicy({ version: 2, router: { profiles: { claude: { economy: { role: 'implementer', model: 'fable' } }, codex: {} } } }),
    (e) => e instanceof ControlError && e.code === 'INVALID_FEATURE_POLICY');
  // Also rejected on the v1 migration path when the legacy file already carried it.
  assert.throws(() => validateFeaturePolicy({ version: 1, router: { profiles: { claude: { economy: { model: 'fable' } }, codex: {} } } }),
    (e) => e instanceof ControlError && e.code === 'INVALID_FEATURE_POLICY');
});
