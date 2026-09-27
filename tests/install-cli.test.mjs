import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { installationPlan, applyInstallation, REPO_ROOT } from '../src/installer.mjs';
import { atomicWrite, loadConfig, saveCredential } from '../src/storage.mjs';
import { tempHome, KEY } from './helpers.mjs';
function setup(t) {
  const user = tempHome(t); const home = path.join(user, "shared state 'with spaces'");
  const env = { ...process.env, HOME: user, TYPESAFE_API_KEY: '', POINTSMAN_HOME: '', CODEX_HOME: path.join(user, 'custom-codex'), CLAUDE_CONFIG_DIR: '', POINTSMAN_DISABLE: '0' };
  const opts = { home, env, project: user };
  return { user, home, env, opts, install: (changes = {}) => applyInstallation(installationPlan({ ...opts, ...changes })) };
}
function cli(args, env, input) { return spawnSync(process.execPath, [path.join(REPO_ROOT, 'bin/pointsman.mjs'), ...args], { env, input, encoding: 'utf8', timeout: 5000 }); }

test('dry-run is side-effect-free and initial installation is OFF', t => {
  const f = setup(t); const plan = installationPlan(f.opts); assert.ok(plan.actions.length > 10);
  assert.equal(applyInstallation(plan, { dryRun: true }).dryRun, true); assert.deepEqual(fs.readdirSync(f.user), []);
  f.install(); assert.equal(loadConfig(f.home, {}).mode, 'off');
});
test('both installs preserve unrelated config, permissions, skills and are idempotent', t => {
  const f = setup(t); const codex = path.join(f.env.CODEX_HOME, 'config.toml');
  const baseline = 'model = "keep-existing"\n[features]\nmulti_agent = true\n';
  atomicWrite(codex, baseline); atomicWrite(path.join(f.user, '.claude.json'), JSON.stringify({ theme: 'dark', mcpServers: { existing: { command: 'keep' } } }));
  const first = f.install(); assert.ok(first.backupDirectory);
  assert.ok(fs.readFileSync(codex, 'utf8').startsWith(baseline)); assert.ok(fs.readFileSync(codex, 'utf8').includes('env_vars = ["TYPESAFE_API_KEY", "POINTSMAN_DISABLE"]'));
  const c = JSON.parse(fs.readFileSync(path.join(f.user, '.claude.json'))); assert.equal(c.theme, 'dark'); assert.equal(c.mcpServers.existing.command, 'keep');
  assert.equal(c.mcpServers.pointsman.env.POINTSMAN_HOME, f.home); assert.equal(JSON.stringify(c).includes(KEY), false);
  assert.equal(f.install().changes.length, 0);
  for (const s of ['.agents/skills', '.claude/skills']) assert.equal(fs.existsSync(path.join(f.user, s, 'pointsman-decisions/SKILL.md')), true);
});
test('shared CLI shim and installed skill use the configured custom home', t => {
  const f = setup(t); f.install();
  const env = { ...f.env }; delete env.POINTSMAN_HOME;
  const shim = spawnSync(path.join(f.user, '.local/bin/pointsman'), ['status'], { env, encoding: 'utf8' });
  assert.equal(shim.status, 0, shim.stderr); assert.equal(JSON.parse(shim.stdout).home, f.home);
  const run = spawnSync(process.execPath, [path.join(f.user, '.agents/skills/pointsman-control/scripts/run.mjs'), 'status'], { env, encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr); assert.equal(JSON.parse(run.stdout).home, f.home);
});
test('project scope writes only project host configs and skills, not user settings', t => {
  const f = setup(t); const project = path.join(f.user, 'project'); fs.mkdirSync(project);
  f.install({ scope: 'project', project });
  assert.equal(fs.existsSync(path.join(project, '.codex/config.toml')), true); assert.equal(fs.existsSync(path.join(project, '.mcp.json')), true);
  assert.equal(fs.existsSync(path.join(f.user, '.claude.json')), false); assert.equal(fs.existsSync(path.join(f.user, '.local/bin/pointsman')), false);
});
test('uninstall removes only owned entries; partial removal leaves the shared runtime', t => {
  const f = setup(t); const codex = path.join(f.env.CODEX_HOME, 'config.toml'); atomicWrite(codex, 'model = "existing"\n');
  atomicWrite(path.join(f.user, '.claude.json'), '{"theme":"existing"}'); f.install(); saveCredential(f.home, KEY);
  f.install({ target: 'codex', remove: true }); assert.equal(fs.existsSync(path.join(f.user, '.local/bin/pointsman')), true);
  assert.equal(fs.readFileSync(codex, 'utf8'), 'model = "existing"\n');
  f.install({ remove: true }); assert.deepEqual(JSON.parse(fs.readFileSync(path.join(f.user, '.claude.json'))), { theme: 'existing' });
  assert.equal(fs.existsSync(path.join(f.user, '.local/bin/pointsman')), false); assert.equal(fs.existsSync(path.join(f.home, 'credentials.env')), true);
  assert.equal(loadConfig(f.home, {}).mode, 'off'); assert.equal(f.install({ remove: true }).changes.length, 0);
});
test('edited skill files, server-name collisions, and malformed host configs are not overwritten', t => {
  const f = setup(t); f.install(); const skill = path.join(f.user, '.agents/skills/pointsman-decisions/SKILL.md'); fs.appendFileSync(skill, '\nlocal edit\n');
  assert.throws(() => f.install(), /SKILL_CHANGED_OR_COLLISION/); assert.ok(fs.readFileSync(skill, 'utf8').includes('local edit'));
});
test('existing unowned server refuses install', t => {
  const f = setup(t); atomicWrite(path.join(f.user, '.claude.json'), '{"mcpServers":{"pointsman":{"command":"other"}}}');
  assert.throws(() => f.install({ target: 'claude' }), /SERVER_NAME_COLLISION/);
});
test('concurrent configuration modification prevents committing the installation', t => {
  const f = setup(t); const plan = installationPlan(f.opts); const first = plan.actions[0]; atomicWrite(first.file, 'concurrent');
  assert.throws(() => applyInstallation(plan), /WRITE_CONFLICT/); assert.equal(fs.readFileSync(first.file, 'utf8'), 'concurrent');
});
test('install mutex refuses concurrent execution', t => {
  const f = setup(t); atomicWrite(path.join(f.home, 'install.lock'), 'owned');
  assert.throws(() => f.install(), /INSTALL_LOCKED/); assert.equal(fs.readFileSync(path.join(f.home, 'install.lock'), 'utf8'), 'owned');
});
test('CLI has no secret argument, refuses noninteractive key entry, and OFF smoke is offline', t => {
  const f = setup(t); const env = { ...f.env, POINTSMAN_HOME: f.home, TYPESAFE_API_KEY: KEY };
  const status = cli(['status'], env); assert.equal(status.status, 0); assert.equal(status.stdout.includes(KEY), false);
  const key = cli(['key', 'set'], env, `${KEY}\n`); assert.equal(key.status, 2); assert.ok(key.stderr.includes('KEY_REQUIRES_YOUR_INTERACTIVE_TERMINAL')); assert.equal(key.stderr.includes(KEY), false);
  const arg = cli(['key', 'set', KEY], env); assert.equal(arg.status, 2); assert.equal((arg.stderr + arg.stdout).includes(KEY), false);
  const smoke = cli(['smoke'], env); assert.equal(smoke.status, 0, smoke.stderr); assert.equal(JSON.parse(smoke.stdout).externalApiCalls, 0);
  const live = cli(['smoke', '--live'], env); assert.equal(live.status, 2); assert.ok(live.stderr.includes('ENABLE_SHADOW_BEFORE_LIVE_SMOKE'));
});
test('CLI on/off updates shared state and does not echo environment key', t => {
  const f = setup(t); const env = { ...f.env, POINTSMAN_HOME: f.home, TYPESAFE_API_KEY: KEY };
  for (const mode of ['shadow', 'on', 'off']) { const out = cli([mode], env); assert.equal(out.status, 0, out.stderr); assert.equal(JSON.parse(out.stdout).mode, mode); assert.equal(out.stdout.includes(KEY), false); }
});
test('moving a reviewed runtime can update only previously recorded managed entries', t => {
  const f = setup(t); f.install(); const relocated = path.join(f.user, 'relocated-runtime');
  fs.mkdirSync(path.join(relocated, 'bin'), { recursive: true }); fs.writeFileSync(path.join(relocated, 'bin/pointsman.mjs'), '// test runtime');
  fs.cpSync(path.join(REPO_ROOT, 'skills'), path.join(relocated, 'skills'), { recursive: true });
  f.install({ root: relocated });
  const config = JSON.parse(fs.readFileSync(path.join(f.user, '.claude.json'))); assert.equal(config.mcpServers.pointsman.args[0], path.join(relocated, 'bin/pointsman.mjs'));
  f.install({ root: relocated, remove: true }); assert.equal(fs.existsSync(path.join(f.user, '.claude.json')), false);
});
test('a failed write rolls back earlier managed writes', t => {
  const f = setup(t); const plan = installationPlan(f.opts); const original = fs.renameSync;
  let calls = 0;
  t.mock.method(fs, 'renameSync', (...args) => { if (++calls === 3) throw new Error('simulated disk failure'); return original(...args); });
  assert.throws(() => applyInstallation(plan), /simulated disk failure/);
  for (const action of plan.actions) assert.equal(fs.existsSync(action.file), false, action.file);
});

test('custom Claude profiles do not silently write the wrong user configuration', t => {
  const f = setup(t); f.env.CLAUDE_CONFIG_DIR = path.join(f.user, 'alternate-profile');
  assert.throws(() => f.install(), /CUSTOM_CLAUDE_PROFILE_USE_PROJECT_SCOPE/);
  assert.deepEqual(fs.readdirSync(f.user), []);
  f.install({ scope: 'project' }); assert.equal(fs.existsSync(path.join(f.user, '.mcp.json')), true);
});
