import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { installationPlan, applyInstallation, describeHookStatus, REPO_ROOT } from '../src/installer.mjs';
import { atomicWrite } from '../src/storage.mjs';
import { tempHome } from './helpers.mjs';

function setup(t) {
  const user = tempHome(t); const home = path.join(user, 'pm-home');
  const env = { ...process.env, HOME: user, TYPESAFE_API_KEY: '', POINTSMAN_HOME: '', CODEX_HOME: path.join(user, '.codex'), CLAUDE_CONFIG_DIR: '', POINTSMAN_DISABLE: '0' };
  const opts = { home, env, project: user };
  return {
    user, home, env, opts,
    install: (changes = {}) => applyInstallation(installationPlan({ ...opts, ...changes })),
    dry: (changes = {}) => applyInstallation(installationPlan({ ...opts, ...changes }), { dryRun: true }),
  };
}
const claudeSettings = user => path.join(user, '.claude/settings.json');
const claudeMd = user => path.join(user, '.claude/CLAUDE.md');
const codexHooks = env => path.join(env.CODEX_HOME, 'hooks.json');
const codexAgents = env => path.join(env.CODEX_HOME, 'AGENTS.md');

test('install --hooks preserves foreign hook groups, appends ours at the end, and never leaks env-shaped values', t => {
  const f = setup(t);
  const foreignSettings = { env: { SOME_SECRET: 'sk-do-not-leak-12345' }, hooks: {
    PreToolUse: [{ matcher: 'Bash', hooks: [{ type: 'command', command: 'echo pre', timeout: 5 }] }],
    PostToolUse: [{ matcher: 'Agent', hooks: [{ type: 'command', command: 'echo post', timeout: 5 }] }],
  } };
  fs.mkdirSync(path.dirname(claudeSettings(f.user)), { recursive: true });
  atomicWrite(claudeSettings(f.user), JSON.stringify(foreignSettings, null, 2) + '\n');
  const foreignHooksJson = { hooks: { PreToolUse: [{ matcher: 'Agent', hooks: [{ type: 'command', command: 'existing-codex-hook', timeout: 3 }] }] } };
  fs.mkdirSync(f.env.CODEX_HOME, { recursive: true });
  atomicWrite(codexHooks(f.env), JSON.stringify(foreignHooksJson, null, 2) + '\n');

  const report = f.install({ hooks: true, target: 'both' });
  assert.equal(JSON.stringify(report).includes('sk-do-not-leak-12345'), false);

  const settings = JSON.parse(fs.readFileSync(claudeSettings(f.user), 'utf8'));
  assert.equal(settings.env.SOME_SECRET, 'sk-do-not-leak-12345');
  assert.equal(settings.hooks.PreToolUse.length, 2);
  assert.deepEqual(settings.hooks.PreToolUse[0], foreignSettings.hooks.PreToolUse[0]);
  assert.equal(settings.hooks.PreToolUse[1].matcher, 'Agent');
  assert.ok(settings.hooks.PreToolUse[1].hooks[0].command.includes('hook'));
  assert.equal(settings.hooks.PostToolUse.length, 2);
  assert.deepEqual(settings.hooks.PostToolUse[0], foreignSettings.hooks.PostToolUse[0]);
  assert.equal(settings.hooks.SubagentStop.length, 1);

  const hooksJson = JSON.parse(fs.readFileSync(codexHooks(f.env), 'utf8'));
  assert.equal(hooksJson.hooks.PreToolUse.length, 2);
  assert.deepEqual(hooksJson.hooks.PreToolUse[0], foreignHooksJson.hooks.PreToolUse[0]);
  assert.equal(hooksJson.hooks.SubagentStart.length, 1);
  assert.equal(hooksJson.hooks.SubagentStop.length, 1);

  const hc = report.hookChanges.find(h => h.path === claudeSettings(f.user));
  assert.equal(hc.added.length, 3);
  assert.deepEqual(hc.existingGroupHashes.before, hc.existingGroupHashes.after);

  assert.ok(report.instructionBlocks.find(b => b.path === claudeMd(f.user) && b.action === 'installed'));
  assert.ok(report.instructionBlocks.find(b => b.path === codexAgents(f.env) && b.action === 'installed'));
  const claudeBlock = fs.readFileSync(claudeMd(f.user), 'utf8');
  assert.ok(claudeBlock.includes('lightweight-worker'));
  const codexBlock = fs.readFileSync(codexAgents(f.env), 'utf8');
  assert.ok(codexBlock.includes('only records')); // Codex hooks only record (opaque spawn message, plan B4)

  assert.deepEqual(report.hostTrustRequired, ['codex']);
  assert.equal(report.hostApprovalsChanged, false);
});

test('reinstalling --hooks is idempotent (zero actions)', t => {
  const f = setup(t);
  f.install({ hooks: true });
  const second = f.install({ hooks: true });
  assert.equal(second.changes.length, 0);
  for (const hc of second.hookChanges) { assert.equal(hc.added.length, 0); assert.equal(hc.removed.length, 0); }
});

test('a user-modified managed hook group is refused, not silently overwritten', t => {
  const f = setup(t);
  f.install({ hooks: true });
  const settings = JSON.parse(fs.readFileSync(claudeSettings(f.user), 'utf8'));
  settings.hooks.PreToolUse[settings.hooks.PreToolUse.length - 1].hooks[0].command = 'tampered';
  atomicWrite(claudeSettings(f.user), JSON.stringify(settings, null, 2) + '\n');
  assert.throws(() => f.install({ hooks: true }), /MANAGED_HOOK_CHANGED/);
});

test('uninstall --hooks-only removes only hooks and the instruction block; MCP, skills, shim and mode are untouched', t => {
  const f = setup(t);
  f.install({ hooks: true });
  const beforeStatus = fs.readFileSync(path.join(f.home, 'config.json'), 'utf8');
  const shimBefore = fs.readFileSync(path.join(f.user, '.local/bin/pointsman'), 'utf8');
  const claudeJsonBefore = fs.readFileSync(path.join(f.user, '.claude.json'), 'utf8');

  const report = f.install({ remove: true, hooksOnly: true });
  assert.equal(fs.existsSync(claudeSettings(f.user)), false);
  assert.equal(fs.existsSync(claudeMd(f.user)), false);
  assert.equal(fs.existsSync(codexHooks(f.env)), false);
  assert.equal(fs.existsSync(codexAgents(f.env)), false);

  assert.equal(fs.readFileSync(path.join(f.home, 'config.json'), 'utf8'), beforeStatus);
  assert.equal(fs.readFileSync(path.join(f.user, '.local/bin/pointsman'), 'utf8'), shimBefore);
  assert.equal(fs.readFileSync(path.join(f.user, '.claude.json'), 'utf8'), claudeJsonBefore);
  assert.ok(fs.existsSync(path.join(f.user, '.agents/skills/pointsman-decisions/SKILL.md')));
  assert.ok(fs.existsSync(path.join(f.user, '.claude/skills/pointsman-decisions/SKILL.md')));

  assert.ok(report.hookChanges.find(h => h.path === claudeSettings(f.user) && h.removed.length === 3));
  assert.ok(report.instructionBlocks.find(b => b.path === claudeMd(f.user) && b.action === 'removed'));

  const reapplied = f.install({ hooks: true });
  assert.ok(reapplied.hookChanges.find(h => h.path === claudeSettings(f.user) && h.added.length === 3));
});

test('Codex hooks.json is created fresh and removed entirely on --hooks-only uninstall; config.toml hooks.state is never written', t => {
  const f = setup(t);
  f.install({ hooks: true, target: 'codex' });
  assert.ok(fs.existsSync(codexHooks(f.env)));
  const configToml = fs.readFileSync(path.join(f.env.CODEX_HOME, 'config.toml'), 'utf8');
  assert.equal(configToml.includes('hooks.state'), false);
  f.install({ remove: true, hooksOnly: true, target: 'codex' });
  assert.equal(fs.existsSync(codexHooks(f.env)), false);
});

test('instruction block: append, idempotent, damage and drift detection, clean removal', t => {
  const f = setup(t);
  fs.mkdirSync(path.dirname(claudeMd(f.user)), { recursive: true });
  atomicWrite(claudeMd(f.user), '# My existing notes\nkeep me\n');
  f.install({ hooks: true, target: 'claude' });
  const afterInstall = fs.readFileSync(claudeMd(f.user), 'utf8');
  assert.ok(afterInstall.startsWith('# My existing notes\nkeep me\n'));
  assert.ok(afterInstall.includes('pointsman'));

  const idempotent = f.install({ hooks: true, target: 'claude' });
  assert.equal(idempotent.changes.length, 0);

  const damaged = afterInstall.replace('<!-- <<< pointsman managed -->', '');
  atomicWrite(claudeMd(f.user), damaged);
  assert.throws(() => f.install({ hooks: true, target: 'claude' }), /MANAGED_BLOCK_DAMAGED/);

  atomicWrite(claudeMd(f.user), afterInstall.replace('decide', 'tampered_call'));
  assert.throws(() => f.install({ hooks: true, target: 'claude' }), /MANAGED_CONFIG_CHANGED/);

  atomicWrite(claudeMd(f.user), afterInstall);
  f.install({ remove: true, hooksOnly: true, target: 'claude' });
  assert.equal(fs.readFileSync(claudeMd(f.user), 'utf8'), '# My existing notes\nkeep me\n');
});

test('dry-run for hooks writes nothing and reports what would change', t => {
  const f = setup(t);
  const before = fs.readdirSync(f.user);
  const report = f.dry({ hooks: true });
  assert.equal(report.dryRun, true);
  assert.deepEqual(fs.readdirSync(f.user), before);
  assert.ok(report.hookChanges.some(h => h.added.length > 0));
  assert.ok(report.instructionBlocks.some(b => b.action === 'installed'));
});

test('project scope: hook files live under the project, and the instruction block is skipped', t => {
  const f = setup(t);
  const project = path.join(f.user, 'proj'); fs.mkdirSync(project);
  const report = f.install({ scope: 'project', project, hooks: true });
  assert.ok(fs.existsSync(path.join(project, '.claude/settings.json')));
  assert.ok(fs.existsSync(path.join(project, '.codex/hooks.json')));
  assert.equal(fs.existsSync(path.join(f.user, '.claude/CLAUDE.md')), false);
  assert.ok(report.instructionBlocks.every(b => b.action === 'skipped-project-scope'));
});

test('install without --hooks never touches hook files, even if already installed', t => {
  const f = setup(t);
  f.install({ hooks: true });
  const before = fs.readFileSync(claudeSettings(f.user), 'utf8');
  const report = f.install({ hooks: false });
  assert.equal(fs.readFileSync(claudeSettings(f.user), 'utf8'), before);
  assert.equal(report.hookChanges.length, 0);
});

test('installed hook command actually runs pointsman hook and is a no-op under global off', t => {
  const f = setup(t);
  f.install({ hooks: true, target: 'claude' });
  const settings = JSON.parse(fs.readFileSync(claudeSettings(f.user), 'utf8'));
  const command = settings.hooks.PreToolUse[0].hooks[0].command;
  const result = spawnSync('/bin/sh', ['-c', command], { env: { ...f.env, HOME: f.user }, input: '{}', encoding: 'utf8', timeout: 5000 });
  assert.equal(result.status, 0);
  assert.equal(result.stdout, '');
});

test('doctor-facing describeHookStatus reports missing, installed and changed', t => {
  const f = setup(t);
  let status = describeHookStatus({ home: f.home, env: f.env, scope: 'user', project: f.user });
  assert.equal(status.claude.events.PreToolUse, 'missing');
  assert.equal(status.codex.trust, 'trust-entry-absent');

  f.install({ hooks: true });
  status = describeHookStatus({ home: f.home, env: f.env, scope: 'user', project: f.user });
  assert.equal(status.claude.events.PreToolUse, 'installed');
  assert.equal(status.claude.instructionBlock, 'installed');
  assert.equal(status.codex.events.SubagentStart, 'installed');

  const settings = JSON.parse(fs.readFileSync(claudeSettings(f.user), 'utf8'));
  settings.hooks.PreToolUse[0].hooks[0].command = 'tampered';
  atomicWrite(claudeSettings(f.user), JSON.stringify(settings, null, 2) + '\n');
  status = describeHookStatus({ home: f.home, env: f.env, scope: 'user', project: f.user });
  assert.equal(status.claude.events.PreToolUse, 'changed');
});

test('doctor CLI includes a hooks section and native-hook-execution as not performed', t => {
  const f = setup(t);
  f.install({ hooks: true });
  const result = spawnSync(process.execPath, [path.join(REPO_ROOT, 'bin/pointsman.mjs'), 'doctor'], { env: { ...f.env, POINTSMAN_HOME: f.home }, encoding: 'utf8' });
  const out = JSON.parse(result.stdout);
  assert.ok(out.hooks.claude);
  assert.ok(out.checksNotPerformed.includes('native-hook-execution'));
});

test('original 2-space JSON formatting is restored byte-for-byte after --hooks-only uninstall', t => {
  const f = setup(t);
  const original = JSON.stringify({ permissions: { allow: ['Bash'] } }, null, 2) + '\n';
  fs.mkdirSync(path.dirname(claudeSettings(f.user)), { recursive: true });
  atomicWrite(claudeSettings(f.user), original);
  f.install({ hooks: true, target: 'claude' });
  const hc = f.dry({ target: 'claude' }); // no-op dry run just to ensure nothing pending
  f.install({ remove: true, hooksOnly: true, target: 'claude' });
  assert.equal(fs.readFileSync(claudeSettings(f.user), 'utf8'), original);
});

test('an identical unowned hook group (no install record) is refused instead of being duplicated', t => {
  const f = setup(t);
  // Produce our exact group once, then forget the installation records so the group looks foreign.
  f.install({ hooks: true });
  const settings = fs.readFileSync(claudeSettings(f.user), 'utf8');
  fs.rmSync(path.join(f.home, 'installations'), { recursive: true, force: true });
  assert.throws(() => f.install({ hooks: true }), /HOOK_COLLISION/);
  assert.equal(fs.readFileSync(claudeSettings(f.user), 'utf8'), settings);
});

test('--no-skills installs MCP, hooks and block for a host whose skills directory is a symlink, without writing through it', t => {
  const f = setup(t);
  const outside = path.join(f.user, 'forge-skills'); fs.mkdirSync(outside, { recursive: true });
  fs.mkdirSync(path.join(f.user, '.claude'), { recursive: true });
  fs.symlinkSync(outside, path.join(f.user, '.claude/skills'));
  assert.throws(() => f.install({ target: 'claude', hooks: true }), /UNSAFE_SYMLINK/);
  const r = f.install({ target: 'claude', hooks: true, skills: false });
  assert.equal(r.ok, true);
  assert.deepEqual(fs.readdirSync(outside), []);
  assert.ok(JSON.parse(fs.readFileSync(path.join(f.user, '.claude.json'), 'utf8')).mcpServers['pointsman']);
  assert.ok(JSON.parse(fs.readFileSync(claudeSettings(f.user), 'utf8')).hooks.PreToolUse.length >= 1);
  assert.match(fs.readFileSync(claudeMd(f.user), 'utf8'), /pointsman managed/);
  const u = f.install({ target: 'claude', remove: true, skills: false });
  assert.equal(u.ok, true);
  assert.deepEqual(fs.readdirSync(outside), []);
  assert.equal(fs.existsSync(claudeSettings(f.user)), false);
});
test('CLI accepts --no-skills for install and uninstall only', t => {
  const f = setup(t);
  const bin = path.join(REPO_ROOT, 'bin/pointsman.mjs');
  const run = args => spawnSync(process.execPath, [bin, ...args, '--home', f.home], { encoding: 'utf8', env: f.env });
  assert.equal(run(['install', '--target', 'claude', '--no-skills', '--dry-run']).status, 0);
  assert.notEqual(run(['status', '--no-skills']).status, 0);
});

test('Codex trust status reads per-hook [hooks.state."<file>:<event>:<group>:<handler>"] headers for our own groups', t => {
  const f = setup(t);
  const hooksFile = codexHooks(f.env);
  fs.mkdirSync(f.env.CODEX_HOME, { recursive: true });
  // One foreign PreToolUse group first, so our group lands at index 1 there and index 0 elsewhere.
  atomicWrite(hooksFile, JSON.stringify({ hooks: { PreToolUse: [{ matcher: 'Bash', hooks: [{ type: 'command', command: 'foreign' }] }] } }, null, 2) + '\n');
  f.install({ target: 'codex', hooks: true });
  const config = path.join(f.env.CODEX_HOME, 'config.toml');
  const base = fs.readFileSync(config, 'utf8');
  const entry = key => `\n[hooks]\n[hooks.state]\n\n[hooks.state."${hooksFile}:${key}"]\ntrusted_hash = "sha256:00"\n`;
  const status = () => describeHookStatus({ home: f.home, env: f.env, scope: 'user', project: f.user }).codex.trust;
  assert.equal(status(), 'trust-entry-absent');
  // A trust entry for the FOREIGN group only does not count as ours.
  fs.writeFileSync(config, base + entry('pre_tool_use:0:0'));
  assert.equal(status(), 'trust-entry-absent');
  fs.writeFileSync(config, base + entry('pre_tool_use:1:0'));
  assert.equal(status(), 'trust-entry-partial-unverified');
  fs.writeFileSync(config, base + entry('pre_tool_use:1:0') + `\n[hooks.state."${hooksFile}:subagent_start:0:0"]\ntrusted_hash = "sha256:00"\n[hooks.state."${hooksFile}:subagent_stop:0:0"]\ntrusted_hash = "sha256:00"\n`);
  assert.equal(status(), 'trust-entry-present-unverified');
});

test('Codex pre-spawn matcher covers the namespaced spawn_agent tool (0.154.0 did not fire PreToolUse for matcher "Agent")', t => {
  const f = setup(t);
  f.install({ target: 'codex', hooks: true });
  const groups = JSON.parse(fs.readFileSync(codexHooks(f.env), 'utf8')).hooks.PreToolUse;
  const re = new RegExp(`^(?:${groups.at(-1).matcher})$`);
  for (const name of ['Agent', 'spawn_agent', 'agents.spawn_agent']) assert.ok(re.test(name), name);
});
test('changing the managed matcher upgrades an existing owned group in place', t => {
  const f = setup(t);
  f.install({ target: 'codex', hooks: true });
  const file = codexHooks(f.env);
  const doc = JSON.parse(fs.readFileSync(file, 'utf8'));
  doc.hooks.PreToolUse.at(-1).matcher = 'Agent';
  // Simulate the previous release's recorded group so the upgrade path is exercised.
  const recordDir = path.join(f.home, 'installations');
  for (const name of fs.readdirSync(recordDir)) {
    const rec = JSON.parse(fs.readFileSync(path.join(recordDir, name), 'utf8'));
    if (rec.path === `${file}::hooks::PreToolUse`) { rec.value.matcher = 'Agent'; atomicWrite(path.join(recordDir, name), JSON.stringify(rec, null, 2) + '\n'); }
  }
  atomicWrite(file, JSON.stringify(doc, null, 2) + '\n');
  const r = f.install({ target: 'codex', hooks: true });
  assert.equal(r.ok, true);
  assert.notEqual(JSON.parse(fs.readFileSync(file, 'utf8')).hooks.PreToolUse.at(-1).matcher, 'Agent');
  assert.equal(JSON.parse(fs.readFileSync(file, 'utf8')).hooks.PreToolUse.length, 1);
});

test('instruction blocks match what each host hook can actually see', t => {
  const f = setup(t);
  f.install({ hooks: true });
  const codexBlock = fs.readFileSync(codexAgents(f.env), 'utf8');
  const claudeBlock = fs.readFileSync(claudeMd(f.user), 'utf8');
  // codex-cli 0.154.0 hands hooks an opaque spawn message, and an explicit route before each spawn measured
  // +8.6s with no saving (plan B4), so the Codex block must tell agents NOT to route and state that hooks only record.
  assert.match(codexBlock, /Do not call the pointsman `route` tool before spawning/);
  assert.match(codexBlock, /only records/);
  assert.doesNotMatch(codexBlock, /route\.role/);
  assert.doesNotMatch(codexBlock, /\[route scope=/);
  assert.match(claudeBlock, /\[route scope=/);
});
