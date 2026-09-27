import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { installationPlan, applyInstallation, layaAgentPlistPath } from '../src/installer.mjs';
import { tempHome } from './helpers.mjs';

function setup(t) {
  const user = tempHome(t); const home = path.join(user, 'pm-home');
  const env = { ...process.env, HOME: user, TYPESAFE_API_KEY: '', POINTSMAN_HOME: '', CODEX_HOME: path.join(user, '.codex'), CLAUDE_CONFIG_DIR: '', POINTSMAN_DISABLE: '0' };
  const opts = { home, env, project: user };
  return { user, home, env, opts,
    install: (changes = {}) => applyInstallation(installationPlan({ ...opts, ...changes })),
    dry: (changes = {}) => applyInstallation(installationPlan({ ...opts, ...changes }), { dryRun: true }) };
}
function withPlatform(name, fn) {
  const original = Object.getOwnPropertyDescriptor(process, 'platform');
  Object.defineProperty(process, 'platform', { value: name, configurable: true });
  try { return fn(); } finally { Object.defineProperty(process, 'platform', original); }
}

test('install --laya-agent --dry-run makes no filesystem change and reports the launchctl commands', t => {
  if (process.platform !== 'darwin') return;
  const f = setup(t);
  const report = f.dry({ layaAgent: true });
  assert.equal(fs.existsSync(layaAgentPlistPath(f.user)), false);
  assert.equal(report.layaAgent.action, 'installed');
  assert.ok(report.launchctl.bootstrap.includes('launchctl bootstrap'));
  assert.ok(report.launchctl.bootstrap.includes(layaAgentPlistPath(f.user)));
  assert.ok(report.launchctl.bootout.includes('com.pointsman.laya'));
});

test('install --laya-agent writes a plist with only POINTSMAN_HOME as an env var, KeepAlive on abnormal exit only, Interactive process type', t => {
  if (process.platform !== 'darwin') return;
  const f = setup(t);
  const report = f.install({ layaAgent: true });
  assert.equal(report.layaAgent.action, 'installed');
  const plistPath = layaAgentPlistPath(f.user);
  const text = fs.readFileSync(plistPath, 'utf8');
  assert.ok(text.includes('<key>Label</key>'));
  assert.ok(text.includes('com.pointsman.laya'));
  assert.ok(text.includes('laya'));
  assert.ok(text.includes('serve'));
  assert.ok(text.includes(f.home));
  assert.ok(text.includes('<key>KeepAlive</key>'));
  assert.ok(text.includes('<key>SuccessfulExit</key>'));
  assert.ok(text.includes('<false/>'));
  assert.ok(text.includes('ProcessType'));
  assert.ok(text.includes('Interactive'));
  const envSection = text.slice(text.indexOf('EnvironmentVariables'));
  assert.ok(envSection.includes('POINTSMAN_HOME'));
  assert.equal(envSection.includes('TYPESAFE_API_KEY'), false);
  const stat = fs.statSync(plistPath);
  assert.equal(stat.mode & 0o777, 0o600);
});

test('a second install --laya-agent is idempotent (no diff, unchanged action)', t => {
  if (process.platform !== 'darwin') return;
  const f = setup(t);
  f.install({ layaAgent: true });
  const before = fs.readFileSync(layaAgentPlistPath(f.user), 'utf8');
  const report = f.install({ layaAgent: true });
  assert.equal(report.layaAgent.action, 'unchanged');
  assert.equal(fs.readFileSync(layaAgentPlistPath(f.user), 'utf8'), before);
});

test('a user-modified plist is refused, not silently overwritten', t => {
  if (process.platform !== 'darwin') return;
  const f = setup(t);
  f.install({ layaAgent: true });
  fs.writeFileSync(layaAgentPlistPath(f.user), '<plist>tampered by the user</plist>\n');
  assert.throws(() => f.install({ layaAgent: true }), /LAYA_AGENT_PLIST_CHANGED/);
});

test('uninstall --laya-agent removes the plist it installed', t => {
  if (process.platform !== 'darwin') return;
  const f = setup(t);
  f.install({ layaAgent: true });
  const report = f.install({ layaAgent: true, remove: true });
  assert.equal(report.layaAgent.action, 'removed');
  assert.equal(fs.existsSync(layaAgentPlistPath(f.user)), false);
});

test('on a non-macOS platform, --laya-agent is refused with UNSUPPORTED_PLATFORM', t => {
  const f = setup(t);
  withPlatform('linux', () => { assert.throws(() => f.install({ layaAgent: true }), /UNSUPPORTED_PLATFORM/); });
});

test('a plain install/uninstall without --laya-agent never touches the plist', t => {
  if (process.platform !== 'darwin') return;
  const f = setup(t);
  f.install({ layaAgent: true });
  f.install({});
  assert.ok(fs.existsSync(layaAgentPlistPath(f.user)));
});

test('doctor reports a layaServer section: plist status, socket existence and live status', async t => {
  const { spawn } = await import('node:child_process');
  const { REPO_ROOT } = await import('../src/installer.mjs');
  const { startLayaServer } = await import('../src/laya-server.mjs');
  const { atomicWrite } = await import('../src/storage.mjs');
  const f = setup(t);
  const bin = path.join(REPO_ROOT, 'bin/pointsman.mjs');
  // The resident server in this process is async (event-loop driven), so the doctor subprocess must
  // be awaited asynchronously too -- spawnSync would block this process's own event loop and the
  // server could never accept the doctor subprocess's connection (a real cross-process deadlock, not
  // a production concern since `laya serve` and `doctor` are always separate OS processes).
  const run = () => new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [bin, 'doctor'], { env: { ...f.env, POINTSMAN_HOME: f.home, HOME: f.user } });
    let stdout = ''; child.stdout.on('data', d => { stdout += d; });
    child.on('error', reject); child.on('exit', () => resolve({ stdout }));
  });

  let out = JSON.parse((await run()).stdout);
  assert.equal(out.layaServer.socketExists, false);
  assert.equal(out.layaServer.responding, false);
  if (process.platform === 'darwin') assert.equal(out.layaServer.plist, 'missing');

  if (process.platform === 'darwin') { f.install({ layaAgent: true }); out = JSON.parse((await run()).stdout); assert.equal(out.layaServer.plist, 'installed'); }

  atomicWrite(path.join(f.home, 'providers.json'), JSON.stringify({ version: 1, provider: 'laya', laya: {
    python: process.execPath, modelPath: f.home, model: 'laya/base', checkpoint: 'a'.repeat(64), runtimeVersion: '0.3.4', device: 'cpu' } }));
  const server = await startLayaServer({ home: f.home, env: {}, preload: false, spawnImpl: () => { throw new Error('never spawned in this test'); } });
  t.after(() => server.close());
  out = JSON.parse((await run()).stdout);
  assert.equal(out.layaServer.socketExists, true);
  assert.equal(out.layaServer.responding, true);
  assert.equal(out.layaServer.ready, false);
});
