// L3 resident Laya server (owner decision 2026-09-23):
// one user launchd agent + a local Unix socket under POINTSMAN_HOME. No network port, no HTTP. Idle unload
// frees the worker's memory after inactivity; the server process itself stays up. A hook talking to
// this server never waits: `wait:false` (the default) returns LAYA_NOT_READY immediately while a load
// starts in the background. Every logged event is content-free (kind, kind-specific reason, latency,
// counters only) per SECURITY.md.
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { fail, errorCode, isObject } from './constants.mjs';
import { ensureDir, noSymlinks, appendEvent, readText } from './storage.mjs';
import { loadProviderConfig, layaSocketPath, createLayaClient, createLayaSocketClient } from './inference.mjs';
import { MAX_FRAME_BYTES } from './constants.mjs';

const CONNECT_PROBE_MS = 200;

function log(home, event, fields = {}) {
  appendEvent(home, { kind: 'laya-server', at: new Date().toISOString(), event, ...fields });
}
/** Best-effort liveness probe of an existing socket path; never throws. */
async function probeExisting(sock) {
  const client = createLayaSocketClient({ socketPath: sock });
  try { await client.status({ timeoutMs: CONNECT_PROBE_MS }); return true; }
  catch { return false; }
}
/** Inspect (never follow) an existing socket path and decide whether it is safe to reclaim. */
function inspectExistingSocketFile(sock) {
  let st;
  try { st = fs.lstatSync(sock); } catch (e) { if (e.code === 'ENOENT') return 'absent'; throw e; }
  if (st.isSymbolicLink()) fail('UNSAFE_SYMLINK');
  if (!st.isSocket()) fail('LAYA_SOCKET_CONFLICT');
  if (typeof process.getuid === 'function' && st.uid !== process.getuid()) fail('WRONG_OWNER');
  return 'socket';
}
async function reclaimSocketPath(home, sock) {
  const kind = inspectExistingSocketFile(sock);
  if (kind === 'absent') return;
  if (await probeExisting(sock)) fail('LAYA_SERVER_ALREADY_RUNNING');
  fs.unlinkSync(sock);
}
function readFrame(socket, onLine) {
  let buffer = '', done = false;
  socket.on('data', chunk => {
    if (done) return;
    buffer += chunk.toString('utf8');
    if (Buffer.byteLength(buffer) > MAX_FRAME_BYTES) { done = true; socket.destroy(); return; }
    const end = buffer.indexOf('\n');
    if (end === -1) return;
    done = true;
    onLine(buffer.slice(0, end));
  });
}
/**
 * Starts the resident server: listens on `${home}/run/laya.sock` (dir 0700, socket 0600), serves
 * status/infer/prepare over short-lived connections, and unloads the worker after `serverIdleUnloadMs`
 * of inactivity while the server process itself stays up. Returns `{ close(), socketPath }`.
 */
export async function startLayaServer({ home, env = process.env, preload = true, spawnImpl, createServerImpl = net.createServer } = {}) {
  ensureDir(home, true);
  const runDir = path.join(home, 'run'); ensureDir(runDir, true);
  const sock = layaSocketPath(home);
  noSymlinks(sock);
  await reclaimSocketPath(home, sock);

  const layaClient = createLayaClient(spawnImpl ? { spawnImpl } : {});
  let idleTimer = null, lastUsedAt = null, closing = false, settled = false;

  function currentIdleUnloadMs() {
    let settings; try { settings = loadProviderConfig(home); } catch { return 1800000; }
    return settings.laya?.serverIdleUnloadMs ?? 1800000;
  }
  function scheduleIdleUnload() {
    clearTimeout(idleTimer);
    if (closing) return;
    idleTimer = setTimeout(() => {
      if (closing) return;
      if (layaClient.status().inFlight > 0) { scheduleIdleUnload(); return; }
      if (layaClient.status().running) { layaClient.close(); log(home, 'idle-unload', {}); }
      scheduleIdleUnload();
    }, currentIdleUnloadMs());
    idleTimer.unref();
  }

  async function handleInfer(msg, socket) {
    if (!isObject(msg.request) || !Number.isInteger(msg.timeoutMs) || msg.timeoutMs < 1 || msg.timeoutMs > 60000) {
      socket.end(JSON.stringify({ error: 'INVALID_LAYA_SERVER_REQUEST' }) + '\n'); return;
    }
    let settings;
    try { settings = loadProviderConfig(home); }
    catch (e) { socket.end(JSON.stringify({ error: errorCode(e) }) + '\n'); return; }
    if (!settings.laya) { socket.end(JSON.stringify({ error: 'LAYA_NOT_CONFIGURED' }) + '\n'); return; }
    const wait = Boolean(msg.wait);
    if (!wait && !layaClient.status().ready) {
      layaClient.prepare(settings, { resident: true, timeoutMs: settings.laya.startupTimeoutMs, env }).then(
        () => { lastUsedAt = new Date().toISOString(); scheduleIdleUnload(); log(home, 'worker-ready', {}); },
        e => log(home, 'worker-start-failed', { reason: errorCode(e) }),
      );
      socket.end(JSON.stringify({ error: 'LAYA_NOT_READY' }) + '\n');
      return;
    }
    const t0 = Date.now();
    try {
      const raw = await layaClient.infer(msg.request, settings, { timeoutMs: msg.timeoutMs, env });
      lastUsedAt = new Date().toISOString(); scheduleIdleUnload();
      log(home, 'infer-ok', { latencyMs: Date.now() - t0 });
      socket.end(JSON.stringify({ ok: true, raw }) + '\n');
    } catch (e) {
      log(home, 'infer-error', { latencyMs: Date.now() - t0, reason: errorCode(e) });
      socket.end(JSON.stringify({ error: errorCode(e) }) + '\n');
    }
  }
  function handleStatus(socket) {
    const s = layaClient.status();
    socket.end(JSON.stringify({ ready: s.ready, loading: s.running && !s.ready, identity: s.identity,
      lastUsedAt, idleUnloadMs: currentIdleUnloadMs(), generation: s.generation }) + '\n');
  }
  async function handlePrepare(socket) {
    let settings;
    try { settings = loadProviderConfig(home); }
    catch (e) { socket.end(JSON.stringify({ error: errorCode(e) }) + '\n'); return; }
    if (!settings.laya) { socket.end(JSON.stringify({ error: 'LAYA_NOT_CONFIGURED' }) + '\n'); return; }
    if (layaClient.status().ready) { socket.end(JSON.stringify({ ready: true }) + '\n'); return; }
    layaClient.prepare(settings, { resident: true, timeoutMs: settings.laya.startupTimeoutMs, env }).then(
      () => { lastUsedAt = new Date().toISOString(); scheduleIdleUnload(); log(home, 'worker-ready', {}); },
      e => log(home, 'worker-start-failed', { reason: errorCode(e) }),
    );
    socket.end(JSON.stringify({ loading: true }) + '\n');
  }
  const server = createServerImpl(socket => {
    socket.on('error', () => {});
    readFrame(socket, async line => {
      let msg;
      try { msg = JSON.parse(line); } catch { socket.end(JSON.stringify({ error: 'INVALID_LAYA_SERVER_REQUEST' }) + '\n'); return; }
      if (!isObject(msg) || typeof msg.op !== 'string') { socket.end(JSON.stringify({ error: 'INVALID_LAYA_SERVER_REQUEST' }) + '\n'); return; }
      if (msg.op === 'status') return handleStatus(socket);
      if (msg.op === 'prepare') return handlePrepare(socket);
      if (msg.op === 'infer') return handleInfer(msg, socket);
      socket.end(JSON.stringify({ error: 'INVALID_LAYA_SERVER_REQUEST' }) + '\n');
    });
  });
  server.on('error', () => {});

  const previousUmask = process.umask(0o077);
  try {
    await new Promise((resolve, reject) => {
      server.once('error', reject);
      server.listen(sock, () => resolve());
    });
  } finally { process.umask(previousUmask); }
  fs.chmodSync(sock, 0o600);
  const st = fs.statSync(sock);
  if ((st.mode & 0o777) !== 0o600) fail('UNSAFE_FILE');
  log(home, 'server-start', {});
  scheduleIdleUnload();

  let settingsAtStart; try { settingsAtStart = loadProviderConfig(home); } catch { settingsAtStart = { laya: null }; }
  if (preload && settingsAtStart.laya) {
    layaClient.prepare(settingsAtStart, { resident: true, timeoutMs: settingsAtStart.laya.startupTimeoutMs, env }).then(
      () => { lastUsedAt = new Date().toISOString(); log(home, 'worker-ready', {}); },
      e => log(home, 'worker-start-failed', { reason: errorCode(e) }),
    );
  }

  function close() {
    if (settled) return Promise.resolve();
    settled = true; closing = true;
    clearTimeout(idleTimer);
    layaClient.close();
    return new Promise(resolve => {
      server.close(() => {
        try { if (fs.lstatSync(sock).isSocket()) fs.unlinkSync(sock); } catch { /* already gone */ }
        log(home, 'server-stop', {});
        resolve();
      });
    });
  }
  return Object.freeze({ close, socketPath: sock, layaClient });
}
