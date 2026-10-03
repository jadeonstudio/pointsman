// Light offline child-process probe; no installed/global configuration or real provider is used.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { setup } from '../../tests/features-helpers.mjs';
import { HOOK_TIMEOUT_MS } from '../../src/hooks.mjs';

const started = performance.now(), repo = fileURLToPath(new URL('../../', import.meta.url)), state = setup();
state.engine.close();
const env = { PATH: process.env.PATH, HOME: state.home, POINTSMAN_HOME: state.home };
const input = { session_id: 'offline-fixture', tool_use_id: 'offline-tool', tool_name: 'Agent',
  tool_input: { subagent_type: 'fixture-standard-role', description: 'fixture', prompt: 'No route marker.' } };
const argv = ['bin/pointsman.mjs', 'hook', '--host', 'claude', '--event', 'pre-spawn', '--home', state.home];
const rows = [], round = value => Math.round(value * 1000) / 1000;
function child(args, data, { keepStdinOpen = false, childEnv = env } = {}) {
  return new Promise((resolve, reject) => {
    const start = performance.now(), p = spawn(process.execPath, args, { cwd: repo, env: childEnv, stdio: ['pipe', 'pipe', 'pipe'] });
    let stdout = '', stderr = '';
    const watchdog = setTimeout(() => p.kill('SIGKILL'), HOOK_TIMEOUT_MS + 2500);
    p.stdin.on('error', () => {}); p.stdout.on('data', data => { stdout += data; }); p.stderr.on('data', data => { stderr += data; });
    p.on('error', error => { clearTimeout(watchdog); reject(error); });
    p.on('close', (code, signal) => { clearTimeout(watchdog); resolve({ elapsedMs: round(performance.now() - start), code, signal, stdout, stderr }); });
    if (keepStdinOpen) p.stdin.write(data); else p.stdin.end(data);
  });
}
let report;
try {
  for (const width of [1, 6]) for (let attempt = 0; attempt < 3; attempt++) {
    const values = await Promise.all(Array.from({ length: width }, () => child(argv, JSON.stringify(input))));
    for (const value of values) {
      assert.equal(value.code, 0); assert.equal(value.signal, null); assert.equal(value.stdout, ''); assert.equal(value.stderr, '');
      rows.push({ concurrency: width, attempt, elapsedMs: value.elapsedMs, exitCode: value.code, stdoutBytes: 0 });
    }
  }
  const stalled = await child(argv, '{', { keepStdinOpen: true });
  assert.equal(stalled.code, 0); assert.equal(stalled.signal, null); assert.equal(stalled.stdout, ''); assert.equal(stalled.stderr, '');
  assert(stalled.elapsedMs < HOOK_TIMEOUT_MS + 2000, 'Open stdin must not keep the actual child alive after the owned timeout.');
  const roleDir = path.join(state.home, '.claude', 'agents'); fs.mkdirSync(roleDir, { recursive: true });
  for (const role of ['fixture-economy-role', 'fixture-standard-role', 'fixture-strong-role']) fs.writeFileSync(path.join(roleDir, `${role}.md`), '', { mode: 0o600 });
  const latePath = path.join(state.home, 'late-fixture.mjs');
  const url = name => new URL(`../../${name}`, import.meta.url).href;
  fs.writeFileSync(latePath, `import fs from 'node:fs'; import {performance} from 'node:perf_hooks';
import {runHookCli} from ${JSON.stringify(url('src/hooks.mjs'))};
import {createDecisionEngine} from ${JSON.stringify(url('src/engine.mjs'))};
import {response} from ${JSON.stringify(url('tests/features-helpers.mjs'))};
const started=performance.now(); await runHookCli({host:'claude',event:'pre-spawn',timeoutMs:100,
engineFactory:options=>createDecisionEngine({...options,provider:async payload=>{
 fs.appendFileSync(${JSON.stringify(path.join(state.home, 'fixture-calls'))},'1\\n');
 await new Promise(resolve=>setTimeout(resolve,200)); return response(payload);
}})}); process.stderr.write(JSON.stringify({hookReturnMs:performance.now()-started}));`, { mode: 0o600 });
  const late = await child([latePath], JSON.stringify({ ...input, tool_input: { ...input.tool_input, prompt: '[route scope=local complete=yes failures=0]\nFixture only.' } }),
    { childEnv: { ...env, TYPESAFE_API_KEY: state.env.TYPESAFE_API_KEY } });
  assert.equal(late.code, 0); assert.equal(late.signal, null); assert.equal(late.stdout, '');
  const lateReturn = JSON.parse(late.stderr); assert(lateReturn.hookReturnMs < 1500); assert(late.elapsedMs < 2000);
  assert.equal(fs.readFileSync(path.join(state.home, 'fixture-calls'), 'utf8'), '1\n');
  const files = dir => fs.existsSync(dir) ? fs.readdirSync(dir, { withFileTypes: true }).flatMap(e => e.isDirectory() ? files(path.join(dir, e.name)) : [path.join(dir, e.name)]) : [];
  assert.deepEqual(files(path.join(state.home, 'links')), [], 'A discarded late completion must leave no successful link/pending record.');
  const events = files(path.join(state.home, 'logs')).flatMap(name => fs.readFileSync(name, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse));
  const hooks = events.filter(e => e.kind === 'hook'); assert.equal(hooks.length, rows.length + 2); assert(hooks.every(e => e.applied === false));
  const timings = width => { const values = rows.filter(r => r.concurrency === width).map(r => r.elapsedMs).sort((a, b) => a - b); return {
    processes: values.length, p50Ms: values[Math.ceil(values.length * .5) - 1], p95Ms: values[Math.ceil(values.length * .95) - 1], maxMs: values.at(-1) }; };
  const sources = ['src/hooks.mjs', 'mods/pointsman-workflows/hooks/pointsman-workflows.ts', 'examples/workflow-hosts/hook-deadline-probe.mjs'];
  report = { version: 1, observedAt: new Date().toISOString(), node: process.version, grade: 'offline actual CLI children plus one delayed injected decision fixture',
    sourceHashes: Object.fromEntries(sources.map(name => [name, createHash('sha256').update(fs.readFileSync(path.join(repo, name))).digest('hex')])),
    scope: 'Same temporary private home: process startup, policy/status reads and content-free hook logging. No provider admission/global rate limit measurement.',
    installedConfigurationChanged: false, actualGpuInference: 0, externalProviderRequests: 0, injectedFixtureCalls: 1,
    contentionCaveat: 'Ambient workload, including separate local training, is uncontrolled; these light timings establish observed bounds, not a speedup or provider-contention claim.',
    perProcessLimitScope: 'Existing limits remain per-process; no global limiter is introduced or claimed.', serial: timings(1), concurrent: timings(6), rows,
    openStdin: { configuredTimeoutMs: HOOK_TIMEOUT_MS, processExitMs: stalled.elapsedMs, exitCode: stalled.code, stdoutBytes: 0 },
    delayedCompletion: { configuredTimeoutMs: 100, fixtureDelayMs: 200, hookReturnMs: round(lateReturn.hookReturnMs), processExitMs: late.elapsedMs, exitCode: late.code, stdoutBytes: 0, successfulLinkOrPendingFiles: 0 },
    hookEvents: hooks.length, appliedEvents: 0, timeoutEvents: hooks.filter(e => e.reason === 'HOOK_TIMEOUT').length };
} finally { state.cleanup(); }
report.cleanup = 'PASS: owned temporary home and every child removed/closed';
report.totalRuntimeIncludingCleanupMs = round(performance.now() - started);
fs.writeFileSync(new URL('hook-deadline-evidence.json', import.meta.url), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report));
