#!/usr/bin/env node
// L3 real-weights end-to-end reproduction script.
// Offline, no network, writes only under a throwaway temp POINTSMAN_HOME. Never touches the real
// POINTSMAN_HOME/~/.codex/~/.claude, never modifies the checkpoint directory it is pointed at, never runs
// launchctl. Not part of `node --test`; run it by hand when real weights are available.
//
// Usage:
//   node scripts/laya-server-e2e.mjs --python /abs/laya-venv/bin/python --model-path /abs/checkpoint-dir \
//     [--precision fp32|fp16] [--device mps|cpu|cuda] [--hook-calls 20] [--idle-unload-ms 60000]
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync, spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { startLayaServer } from '../src/laya-server.mjs';
import { layaSocketPath } from '../src/inference.mjs';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const CONTEXT_LINE = '[route scope=local complete=yes failures=0]';
const cliBin = path.join(ROOT, 'bin/pointsman.mjs');

function summarize(ts) {
  const sorted = [...ts].sort((a, b) => a - b);
  const n = sorted.length;
  return { n, p50: sorted[Math.floor(0.5 * n)], p95: sorted[Math.min(n - 1, Math.floor(0.95 * n))], min: sorted[0], max: sorted[n - 1] };
}
// The Python worker is this Node process's own child, not the RSS-relevant process to sample when
// measuring the memory an idle unload returns; sum the actual `laya_worker.py` process(es) instead.
// `ps` reports the venv python's resolved framework binary, not the venv symlink path passed as
// --python, so match on this repo's own worker script path instead (unique per invocation).
function workerRssMiB() {
  try {
    const marker = path.join(ROOT, 'workers/laya_worker.py');
    const out = spawnSync('ps', ['-axo', 'pid=,rss=,command='], { encoding: 'utf8' }).stdout;
    let total = 0, found = false;
    for (const line of out.trim().split('\n')) {
      const m = /^\s*(\d+)\s+(\d+)\s+(.*)$/.exec(line);
      if (!m) continue;
      if (m[3].includes(marker)) { total += Number(m[2]); found = true; }
    }
    return found ? Math.round(total / 1024) : 0;
  } catch { return null; }
}
// Async spawn, not spawnSync: this script's own `startLayaServer` runs in this same process's event
// loop, and spawnSync would block that loop while the child hook process waits on the server -- a
// harness deadlock that has no production equivalent (`laya serve` and a hook are always separate
// OS processes there). Discovered and confirmed 2026-09-23 while building this script (see the
// cross-process test fix in tests/install-laya-agent.test.mjs for the same root cause).
function runHook(home, input) {
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const child = spawn(process.execPath, [cliBin, 'hook', '--host', 'claude', '--event', 'pre-spawn', '--home', home],
      { env: { ...process.env, POINTSMAN_HOME: home, HOME: home } });
    let out = '';
    child.stdout.on('data', d => { out += d; });
    child.on('error', reject);
    child.on('exit', () => resolve({ ms: Date.now() - t0, out }));
    child.stdin.end(input);
  });
}

async function main() {
  const { values } = parseArgs({ options: {
    python: { type: 'string' }, 'model-path': { type: 'string' }, precision: { type: 'string', default: 'fp32' },
    device: { type: 'string', default: 'mps' }, 'hook-calls': { type: 'string', default: '20' }, 'idle-unload-ms': { type: 'string', default: '60000' },
  } });
  if (!values.python || !values['model-path']) {
    process.stderr.write('usage: node scripts/laya-server-e2e.mjs --python /abs/path --model-path /abs/checkpoint-dir [--precision fp32|fp16] [--device mps|cpu|cuda] [--hook-calls 20] [--idle-unload-ms 60000]\n');
    process.exitCode = 2; return;
  }
  const python = path.resolve(values.python), modelPath = path.resolve(values['model-path']);
  const fp = spawnSync(python, ['-I', path.join(ROOT, 'workers/laya_worker.py'), '--fingerprint', modelPath], { encoding: 'utf8', timeout: 60000 });
  if (fp.status !== 0) { process.stderr.write(`fingerprint failed: ${fp.stderr}\n`); process.exitCode = 1; return; }
  const checkpoint = JSON.parse(fp.stdout.trim().split('\n').pop()).checkpoint;

  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-server-e2e-')));
  const report = { python, modelPath, checkpoint, precision: values.precision, device: values.device };
  try {
    fs.mkdirSync(path.join(home, '.claude', 'agents'), { recursive: true, mode: 0o700 });
    for (const role of ['lightweight-worker', 'implementer', 'specialist', 'scout']) fs.writeFileSync(path.join(home, '.claude', 'agents', `${role}.md`), '', { mode: 0o600 });
    fs.writeFileSync(path.join(home, 'providers.json'), JSON.stringify({ version: 1, provider: 'laya', laya: {
      python, modelPath, model: 'laya/e2e-server', checkpoint, runtimeVersion: '0.3.4', device: values.device,
      startupTimeoutMs: 60000, idleTimeoutMs: 60000, precision: values.precision, serverIdleUnloadMs: Number(values['idle-unload-ms']) } }, null, 2) + '\n', { mode: 0o600 });
    fs.writeFileSync(path.join(home, 'config.json'), JSON.stringify({ version: 1, mode: 'on', model: 'jev-latest', timeoutMs: 10000,
      maxInputBytes: 24000, maxQuestions: 8, minConfidence: 0.85, minChoiceProbability: 0.8, noulCertainty: 0.95,
      maxCallsPerMinute: 60, maxInFlight: 4, circuitFailureThreshold: 3, circuitCooldownMs: 30000, telemetry: true }, null, 2) + '\n', { mode: 0o600 });
    fs.writeFileSync(path.join(home, 'features.json'), JSON.stringify({ version: 2, router: { mode: 'on', profiles: { claude: {
      economy: { role: 'lightweight-worker' }, standard: { role: 'implementer' }, strong: { role: 'specialist' } } } }, bulk: { mode: 'off' } }, null, 2) + '\n', { mode: 0o600 });

    // 1) Server start -> ready time, and a status-op round trip time.
    const t0 = Date.now();
    const server = await startLayaServer({ home, env: {}, preload: true });
    let readyMs = null;
    for (let i = 0; i < 600 && readyMs === null; i++) {
      await new Promise(r => setTimeout(r, 100));
      const s = await import('../src/inference.mjs').then(m => m.createLayaSocketClient({ socketPath: layaSocketPath(home) }).status({ timeoutMs: 2000 }));
      if (s.ready) readyMs = Date.now() - t0;
    }
    report.server_start_to_ready_ms = readyMs;
    const statusT0 = Date.now();
    await (await import('../src/inference.mjs')).createLayaSocketClient({ socketPath: layaSocketPath(home) }).status({ timeoutMs: 2000 });
    report.status_roundtrip_ms = Date.now() - statusT0;

    // 2) Claude pre-spawn hook wall-clock p50/p95 (Node startup included), server up vs down.
    const n = Number(values['hook-calls']);
    const prompt = each => `${CONTEXT_LINE}\nFix an isolated, bounded issue #${each} with an explicit location and no behavioral change.`;
    const hookInput = i => JSON.stringify({ session_id: 's1', tool_use_id: `tu-${i}`, cwd: ROOT, tool_name: 'Agent',
      tool_input: { subagent_type: 'implementer', description: 'bounded fix', prompt: prompt(i) } });
    const timedWithServer = [];
    for (let i = 0; i < n; i++) timedWithServer.push((await runHook(home, hookInput(i))).ms);
    report.hook_p50_p95_ms_server_up = summarize(timedWithServer);

    await server.close();
    const timedNoServer = [];
    for (let i = 0; i < n; i++) timedNoServer.push((await runHook(home, hookInput(n + i))).ms);
    report.hook_p50_p95_ms_server_down = summarize(timedNoServer);

    // 3) Idle unload with a short window, then reload timing.
    const shortIdleHome = home;
    const idleServer = await startLayaServer({ home: shortIdleHome, env: {}, preload: true });
    for (let i = 0; i < 600; i++) { await new Promise(r => setTimeout(r, 100)); if (idleServer.layaClient.status().ready) break; }
    const rssBeforeUnload = workerRssMiB();
    await new Promise(r => setTimeout(r, Number(values['idle-unload-ms']) + 3000));
    const rssAfterUnload = workerRssMiB();
    report.idle_unload = { ready_before: true, ready_after_unload: idleServer.layaClient.status().ready, rss_mib_before_unload: rssBeforeUnload, rss_mib_after_unload: rssAfterUnload };
    const reloadT0 = Date.now();
    await (await import('../src/inference.mjs')).createLayaSocketClient({ socketPath: layaSocketPath(shortIdleHome) }).infer(
      { state: { task: 'reload probe' }, questions: { ok: { type: 'noul', instructions: 'Is this reachable?' } } },
      JSON.parse(fs.readFileSync(path.join(home, 'providers.json'), 'utf8')), { wait: true, timeoutMs: 60000 });
    report.reload_after_idle_unload_ms = Date.now() - reloadT0;
    await idleServer.close();
    report.socket_removed_after_close = !fs.existsSync(layaSocketPath(shortIdleHome));

    process.stdout.write(JSON.stringify(report, null, 2) + '\n');
  } finally {
    fs.rmSync(home, { recursive: true, force: true });
  }
}
await main();
