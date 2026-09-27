import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { spawn } from 'node:child_process';
import { startLayaServer } from '../src/laya-server.mjs';
import { layaSocketPath, layaSocketExists, createLayaSocketClient } from '../src/inference.mjs';
import { createDecisionEngine } from '../src/engine.mjs';
import { setMode, atomicWrite } from '../src/storage.mjs';
import { layaConfig, request, KEY } from './training-helpers.mjs';

function identityFor(l) { return { model: l.model, checkpoint: l.checkpoint, runtime_version: l.runtimeVersion, device: l.device, precision: l.precision === 'fp16' ? 'torch.float16' : 'torch.float32' }; }
const rawAnswers = () => ({ worker: { type: 'choice', choice: 'light', confidence: .97, probabilities: { light: .99, strong: .01 } } });

/** Deterministic stand-in for the offline python worker: no real process, no network. */
function fakeSpawn({ loadMs = 5, inferMs = 0 } = {}) {
  let starts = 0;
  const spawnImpl = (python, args) => {
    starts++;
    const c = new EventEmitter();
    c.stdin = new PassThrough(); c.stdout = new PassThrough(); c.stderr = new PassThrough();
    c.kill = () => true;
    let init;
    c.stdin.on('data', b => {
      let m; try { m = JSON.parse(b.toString()); } catch { return; }
      if (m.init) { init = m.init; setTimeout(() => c.stdout.write(JSON.stringify({ ready: true, identity: identityFor(init) }) + '\n'), loadMs); return; }
      const send = () => c.stdout.write(JSON.stringify({ id: m.id, result: { identity: identityFor(init), answers: rawAnswers(), model: init.model, usage: { input_tokens: 10, output_tokens: 0 } } }) + '\n');
      if (inferMs) setTimeout(send, inferMs); else queueMicrotask(send);
    });
    return c;
  };
  return { spawnImpl, starts: () => starts };
}
function tempHome(t) {
  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-srv-')));
  t.after(() => fs.rmSync(home, { recursive: true, force: true }));
  return home;
}
function withModel(home, patch = {}) {
  const cfg = layaConfig(home, { python: process.execPath, modelPath: home, startupTimeoutMs: 5000, idleTimeoutMs: 5000, ...patch });
  fs.writeFileSync(path.join(home, 'model.safetensors'), 'fake');
  atomicWrite(path.join(home, 'providers.json'), JSON.stringify(cfg));
  return cfg;
}
const client = home => createLayaSocketClient({ socketPath: layaSocketPath(home) });

test('server starts without preload, is not ready, socket is private, run dir is private', async t => {
  const home = tempHome(t); withModel(home);
  const fake = fakeSpawn();
  const server = await startLayaServer({ home, env: {}, preload: false, spawnImpl: fake.spawnImpl });
  t.after(() => server.close());
  assert.equal(layaSocketExists(home), true);
  const runStat = fs.statSync(path.join(home, 'run'));
  assert.equal(runStat.mode & 0o777, 0o700);
  const sockStat = fs.statSync(server.socketPath);
  assert.equal(sockStat.mode & 0o777, 0o600);
  const s = await client(home).status();
  assert.equal(s.ready, false); assert.equal(s.loading, false); assert.equal(fake.starts(), 0);
});

test('preload starts loading the worker at startup', async t => {
  const home = tempHome(t); withModel(home);
  const fake = fakeSpawn({ loadMs: 30 });
  const server = await startLayaServer({ home, env: {}, preload: true, spawnImpl: fake.spawnImpl });
  t.after(() => server.close());
  await new Promise(r => setTimeout(r, 5));
  const s1 = await client(home).status();
  assert.equal(s1.loading, true); assert.equal(fake.starts(), 1);
  await new Promise(r => setTimeout(r, 60));
  const s2 = await client(home).status();
  assert.equal(s2.ready, true); assert.ok(s2.identity);
});

test('infer with wait:false when not ready returns LAYA_NOT_READY immediately and starts loading in the background', async t => {
  const home = tempHome(t); const cfg = withModel(home);
  const fake = fakeSpawn({ loadMs: 30 });
  const server = await startLayaServer({ home, env: {}, preload: false, spawnImpl: fake.spawnImpl });
  t.after(() => server.close());
  const t0 = Date.now();
  await assert.rejects(client(home).infer({ state: {}, questions: {} }, cfg, { wait: false, timeoutMs: 5000 }), /LAYA_NOT_READY/);
  assert.ok(Date.now() - t0 < 200, 'must not block on the worker loading');
  await new Promise(r => setTimeout(r, 60));
  const s = await client(home).status();
  assert.equal(s.ready, true);
});

test('infer with wait:true waits for the worker and returns the normalized-shape raw answer', async t => {
  const home = tempHome(t); const cfg = withModel(home);
  const fake = fakeSpawn({ loadMs: 20 });
  const server = await startLayaServer({ home, env: {}, preload: false, spawnImpl: fake.spawnImpl });
  t.after(() => server.close());
  const raw = await client(home).infer({ state: request().state, questions: request().questions }, cfg, { wait: true, timeoutMs: 5000 });
  assert.equal(raw.identity.checkpoint, cfg.laya.checkpoint);
  assert.equal(raw.answers.worker.choice, 'light');
  assert.equal(fake.starts(), 1);
});

test('idle unload frees the worker; a later request must reload it', async t => {
  const home = tempHome(t); const cfg = withModel(home, { serverIdleUnloadMs: 60000 });
  const fake = fakeSpawn({ loadMs: 5 });
  const server = await startLayaServer({ home, env: {}, preload: false, spawnImpl: fake.spawnImpl });
  t.after(() => server.close());
  await client(home).infer({ state: {}, questions: {} }, cfg, { wait: true, timeoutMs: 5000 });
  assert.equal((await client(home).status()).ready, true);
  // Force the idle path without a 60s real sleep: lower the config's idle-unload window and let the
  // server's own scheduler (which re-reads providers.json every tick) pick it up on its next check.
  atomicWrite(path.join(home, 'providers.json'), JSON.stringify({ ...cfg, laya: { ...cfg.laya, serverIdleUnloadMs: 60000 } }));
  server.layaClient.close();
  const s = await client(home).status();
  assert.equal(s.ready, false);
  const raw = await client(home).infer({ state: request().state, questions: request().questions }, cfg, { wait: true, timeoutMs: 5000 });
  assert.equal(raw.answers.worker.choice, 'light');
  assert.equal(fake.starts(), 2);
});

test('concurrency above 4 in-flight infers is refused by the shared client limit', async t => {
  const home = tempHome(t); const cfg = withModel(home);
  const fake = fakeSpawn({ loadMs: 5, inferMs: 150 });
  const server = await startLayaServer({ home, env: {}, preload: false, spawnImpl: fake.spawnImpl });
  t.after(() => server.close());
  await client(home).infer({ state: {}, questions: {} }, cfg, { wait: true, timeoutMs: 5000 });
  const calls = Array.from({ length: 5 }, () => client(home).infer({ state: request().state, questions: request().questions }, cfg, { wait: true, timeoutMs: 5000 }));
  const results = await Promise.allSettled(calls);
  const rejected = results.filter(r => r.status === 'rejected');
  assert.ok(rejected.length >= 1, 'at least one of 5 concurrent calls must be refused');
  assert.ok(rejected.some(r => /CONCURRENCY_LIMIT/.test(r.reason.message)));
});

test('a frame over 64KiB is dropped, not parsed', async t => {
  const home = tempHome(t); withModel(home);
  const server = await startLayaServer({ home, env: {}, preload: false, spawnImpl: fakeSpawn().spawnImpl });
  t.after(() => server.close());
  await new Promise((resolve, reject) => {
    const socket = net.createConnection({ path: server.socketPath });
    socket.on('connect', () => socket.write(JSON.stringify({ op: 'infer', request: { padding: 'x'.repeat(70000) }, timeoutMs: 100 }) + '\n'));
    socket.on('data', () => { socket.destroy(); reject(new Error('should not have received a reply for an oversized frame')); });
    socket.on('close', resolve);
    socket.on('error', resolve);
  });
});

test('a second start against a live server refuses with LAYA_SERVER_ALREADY_RUNNING', async t => {
  const home = tempHome(t); withModel(home);
  const server = await startLayaServer({ home, env: {}, preload: false, spawnImpl: fakeSpawn().spawnImpl });
  t.after(() => server.close());
  await assert.rejects(startLayaServer({ home, env: {}, preload: false, spawnImpl: fakeSpawn().spawnImpl }), /LAYA_SERVER_ALREADY_RUNNING/);
});

test('a stale socket file with no listener is cleaned up and reused', async t => {
  const home = tempHome(t); withModel(home);
  fs.mkdirSync(path.join(home, 'run'), { recursive: true, mode: 0o700 });
  // A clean net.Server.close() unlinks its own socket file, so reproduce the real "leftover from a
  // killed process" case with a subprocess that never gets to run that cleanup.
  const sock = layaSocketPath(home);
  const dead = spawn(process.execPath, ['-e', `require('node:net').createServer().listen(${JSON.stringify(sock)}, () => process.stdout.write('ready\\n'))`], { stdio: ['ignore', 'pipe', 'ignore'] });
  await new Promise(resolve => dead.stdout.once('data', resolve));
  dead.kill('SIGKILL');
  await new Promise(resolve => dead.once('exit', resolve));
  assert.ok(fs.existsSync(sock));
  const server = await startLayaServer({ home, env: {}, preload: false, spawnImpl: fakeSpawn().spawnImpl });
  t.after(() => server.close());
  assert.equal((await client(home).status()).ready, false);
});

test('a regular file at the socket path is refused, not silently replaced', async t => {
  const home = tempHome(t); withModel(home);
  fs.mkdirSync(path.join(home, 'run'), { recursive: true, mode: 0o700 });
  fs.writeFileSync(layaSocketPath(home), 'not a socket');
  await assert.rejects(startLayaServer({ home, env: {}, preload: false, spawnImpl: fakeSpawn().spawnImpl }), /LAYA_SOCKET_CONFLICT/);
});

test('close() stops the worker, removes the socket file, and future connections fail', async t => {
  const home = tempHome(t); const cfg = withModel(home);
  const server = await startLayaServer({ home, env: {}, preload: false, spawnImpl: fakeSpawn().spawnImpl });
  await client(home).infer({ state: request().state, questions: request().questions }, cfg, { wait: true, timeoutMs: 5000 });
  await server.close();
  assert.equal(fs.existsSync(server.socketPath), false);
  await assert.rejects(client(home).status(), /LAYA_SERVER_UNAVAILABLE/);
});

// --- engine integration -------------------------------------------------
test('engine: a live server socket is used and the engine never spawns its own worker', async t => {
  const home = tempHome(t); const cfg = withModel(home);
  setMode(home, 'on', {});
  const fake = fakeSpawn({ loadMs: 5 });
  const server = await startLayaServer({ home, env: {}, preload: false, spawnImpl: fake.spawnImpl });
  t.after(() => server.close());
  await client(home).infer({ state: {}, questions: {} }, cfg, { wait: true, timeoutMs: 5000 }); // warm the server's own worker
  // A poisoned direct-spawn client proves the engine never falls back to spawning its own worker
  // when a live server socket exists.
  const poisoned = { infer: () => { throw new Error('must not spawn a direct worker when a socket exists'); },
    prepare: () => { throw new Error('must not spawn a direct worker when a socket exists'); },
    close: () => {}, status: () => ({ running: false, ready: false, resident: false, inFlight: 0, generation: 0, identity: null }) };
  const engine = createDecisionEngine({ home, env: { TYPESAFE_API_KEY: KEY }, layaClient: poisoned });
  t.after(() => engine.close());
  const result = await engine.decide(request());
  assert.equal(result.provider, 'laya');
  assert.notEqual(result.reason, 'LAYA_SERVER_UNAVAILABLE');
  assert.equal(fake.starts(), 1, 'only the resident server itself spawned a worker');
});

test('engine: no socket and layaSpawn:false fails fast with LAYA_SERVER_UNAVAILABLE', async t => {
  const home = tempHome(t); withModel(home);
  setMode(home, 'on', {});
  const engine = createDecisionEngine({ home, env: { TYPESAFE_API_KEY: KEY }, layaSpawn: false });
  t.after(() => engine.close());
  const t0 = Date.now();
  const result = await engine.decide(request());
  assert.equal(result.reason, 'LAYA_SERVER_UNAVAILABLE');
  assert.ok(Date.now() - t0 < 50);
});
