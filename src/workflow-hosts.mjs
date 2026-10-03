import { createInterface } from 'node:readline';
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { ensureDir, noSymlinks, readText, atomicWrite } from './storage.mjs';

// These are protocol contracts, not a claim that an installed client passed a native probe.
export const WORKFLOW_HOST_CONTRACTS = Object.freeze({
  codex: Object.freeze({ revision: 'codex-app-server-0.154.0', versions: ['0.154.0'], transport: 'json-rpc-stdio', nativeEvidence: 'UNKNOWN',
    readable: ['command/exec result', 'mcpServer/tool/call result', 'turn notifications'], writable: ['sandboxed command/exec', 'mcpServer/tool/call', 'turn/start', 'turn/interrupt'], source: 'https://learn.chatgpt.com/docs/app-server' }),
  claude: Object.freeze({ revision: 'claude-turn-step-2026-10-03', minimumVersion: '2.1.287', transport: 'mod-async-generator', nativeEvidence: 'UNKNOWN',
    readable: ['turnId', 'index', 'model', 'agentId'], writable: ['text chunk', 'stop chunk'], source: 'https://github.com/anthropics/claude-code/blob/main/mods/types/claude-code.d.ts' }),
  gemini: Object.freeze({ revision: 'gemini-before-model-0.42.0', versions: ['0.42.0'], transport: 'BeforeModel-json', nativeEvidence: 'UNKNOWN',
    readable: ['llm_request'], writable: ['BeforeModel decision deny with synthetic llm_response text'], source: 'https://geminicli.com/docs/hooks/reference/' }),
});

function versionParts(version) { return typeof version === 'string' && /^\d+\.\d+\.\d+$/.test(version) ? version.split('.').map(Number) : null; }
export function matchesWorkflowHost(host, runtime, expectedVersion) {
  const c = WORKFLOW_HOST_CONTRACTS[host], v = versionParts(runtime?.version);
  if (!c || !v || runtime.contractRevision !== c.revision || (expectedVersion && expectedVersion !== runtime.version)) return false;
  if (c.versions) return c.versions.includes(runtime.version);
  const min = versionParts(c.minimumVersion);
  const different = v.findIndex((n, i) => n !== min[i]);
  return different < 0 || v[different] > min[different];
}
function activeMode(getMode) {
  const p = getMode?.();
  return p?.globalMode === 'on' && ['off', 'shadow', 'on'].includes(p.mode) ? p.mode : 'off';
}
function abort(signal) { if (signal?.aborted) throw signal.reason ?? new DOMException('Aborted', 'AbortError'); }

export function isCompletedWorkflow(request, result) {
  const snapshot = result?.snapshot, expected = request?.snapshot;
  if (!snapshot || (snapshot.revision !== null && !/^[a-f0-9]{40,64}$/.test(snapshot.revision)) ||
      !snapshot.files || Array.isArray(snapshot.files) || typeof snapshot.files !== 'object' ||
      Object.values(snapshot.files).some(hash => hash !== null && !/^[a-f0-9]{64}$/.test(hash))) return false;
  if (expected && ((Object.hasOwn(expected, 'revision') && expected.revision !== snapshot.revision) ||
      Object.entries(expected.files ?? {}).some(([file, hash]) => snapshot.files[file] !== hash))) return false;
  return result?.version === 1 && result.recipeRevision === 'workflow-v1' && result.workflow === request?.workflow &&
    result.status === 'done' && result.mode === 'on' && result.needsParent === null && result.authorizesExecution === false &&
    result.acceptance && !Array.isArray(result.acceptance) && Object.values(result.acceptance).includes('met') && Object.values(result.acceptance).every(v => ['met', 'not_applicable'].includes(v)) &&
    result.coverage?.complete === true && Array.isArray(result.coverage.omissions) && result.coverage.omissions.length === 0 &&
    Array.isArray(result.evidence) && result.evidence.length > 0 && result.evidence.every(e => typeof e.ref === 'string' && typeof e.path === 'string' && typeof e.text === 'string' && typeof e.role === 'string' && typeof e.hash === 'string' && /^[a-f0-9]{64}$/.test(e.hash) && snapshot.files[e.path] === e.hash) &&
    Number.isInteger(result.stats?.inferenceCalls) && result.stats.inferenceCalls >= 0 && Number.isInteger(result.stats?.networkCalls) && result.stats.networkCalls >= 0;
}
export function workflowResultText(result) { return JSON.stringify(result); }

// runner is trusted/injected and performs a fresh snapshot recheck; never accept a caller's cached result here.
export async function prepareNativeWorkflow({ host, runtime, expectedVersion, getMode, runner }, envelope, { signal } = {}) {
  const mode = activeMode(getMode);
  if (mode !== 'on' || envelope?.scope !== 'workflow' || !matchesWorkflowHost(host, runtime, expectedVersion)) return { apply: false, reason: mode !== 'on' ? mode : 'unsupported_host_or_scope' };
  try {
    abort(signal);
    const result = await runner.run(envelope.request, { signal });
    abort(signal);
    if (activeMode(getMode) !== 'on' || !isCompletedWorkflow(envelope.request, result)) return { apply: false, reason: 'unresolved', result };
    return { apply: true, text: workflowResultText(result), hostVersion: runtime.version, contractRevision: runtime.contractRevision, result };
  } catch (error) { return { apply: false, reason: signal?.aborted ? 'cancelled' : 'workflow_failed' }; }
}

export async function* claudeWorkflowStep(event, next, options, envelope, { signal } = {}) {
  abort(signal);
  if (event.agentId || event.index !== 0) return yield* next(event);
  const out = await prepareNativeWorkflow({ ...options, host: 'claude' }, envelope, { signal });
  abort(signal);
  if (!out.apply) return yield* next(event);
  yield { kind: 'text', index: 0, text: out.text };
  yield { kind: 'stop', stopReason: 'end_turn', usage: null };
  return { turnId: event.turnId, index: event.index, answer: out.text, toolUses: [], stopReason: 'end_turn', usage: null };
}
export async function geminiWorkflowResponse(options, envelope, { signal } = {}) {
  const out = await prepareNativeWorkflow({ ...options, host: 'gemini' }, envelope, { signal });
  // Installed 0.42.0 consumes the synthetic response only on its BeforeModel blocking path.
  // This decision suppresses model generation; it is never a tool/permission hook response.
  return out.apply ? { decision: 'deny', reason: 'Completed bounded pointsman workflow.', hookSpecificOutput: { llm_response: { candidates: [{ content: { role: 'model', parts: [out.text] }, finishReason: 'STOP' }] } } } : {};
}

// Stable model-hook input omits function calls/results. BeforeAgent is therefore
// required to establish a fresh explicit entry; BeforeModel consumes it only once.
export function geminiWorkflowEnvelope({ home, root, runtime, getMode }, event, { signal, now = Date.now() } = {}) {
  if (!['BeforeAgent', 'BeforeModel'].includes(event?.hook_event_name) ||
      typeof event.session_id !== 'string' || !event.session_id.length || event.session_id.length > 256) return null;
  let lock, pendingFile, applied = false;
  try {
    const hash = text => createHash('sha256').update(text).digest('hex');
    const dir = path.join(home, 'run', 'gemini-workflows');
    const file = path.join(dir, `${hash(JSON.stringify([event.session_id, fs.realpathSync(root)]))}.json`);
    pendingFile = file;
    if (signal?.aborted) return null;
    if (activeMode(getMode) !== 'on' || !matchesWorkflowHost('gemini', runtime)) {
      if (readText(file, { optional: true, privateFile: true, maxBytes: 256 }) !== null) fs.unlinkSync(file);
      return null; // Disabled hooks only invalidate an existing token; they create no state.
    }
    const capture = event.hook_event_name === 'BeforeAgent';
    const last = Array.isArray(event.llm_request?.messages) ? event.llm_request.messages.at(-1) : null;
    const text = capture ? event.prompt : last?.role === 'user' ? last.content : null;
    let request;
    if (typeof text === 'string' && Buffer.byteLength(text) <= 49152 && text.startsWith('pointsman-workflow ')) {
      try { request = JSON.parse(text.slice('pointsman-workflow '.length)); } catch { /* fail open */ }
    }
    if (!request || typeof request !== 'object' || Array.isArray(request)) request = null;
    if (!fs.existsSync(dir) && (!capture || !request)) return null;
    ensureDir(home, true); ensureDir(dir, true);
    const lockPath = path.join(dir, 'lock'); noSymlinks(lockPath);
    // ponytail: a short synchronous directory lock bounds state and serializes
    // capture/consume; contention or an abandoned lock fails open, never waits.
    fs.mkdirSync(lockPath, { mode: 0o700 }); lock = lockPath;
    const read = target => {
      const value = readText(target, { optional: true, privateFile: true, maxBytes: 256 });
      if (value === null) return null;
      try { return JSON.parse(value); } catch { return null; }
    };
    const pending = read(file);
    if (fs.existsSync(file)) { noSymlinks(file); fs.unlinkSync(file); }
    if (!capture) {
      if (!request || !pending || pending.promptHash !== hash(text) || !Number.isFinite(pending.createdAt) ||
          now < pending.createdAt || now - pending.createdAt > 300000) return null;
      applied = true;
      return { scope: 'workflow', request };
    }
    if (!request) return null; // An ordinary/malformed new prompt invalidates its own pending entry.
    for (const name of fs.readdirSync(dir).filter(name => /^[a-f0-9]{64}\.json$/.test(name))) {
      const target = path.join(dir, name), record = read(target);
      if (!record || !Number.isFinite(record.createdAt) || now < record.createdAt || now - record.createdAt > 300000) {
        noSymlinks(target); fs.unlinkSync(target);
      }
    }
    if (fs.readdirSync(dir).filter(name => name.endsWith('.json')).length >= 64) return null;
    atomicWrite(file, JSON.stringify({ promptHash: hash(text), createdAt: now }) + '\n');
  } catch { /* Missing/unsafe/busy state never suppresses a model request. */ }
  finally {
    // A fall-through may already reach the provider. Never leave its entry for
    // a later tool continuation, including when another session holds the lock.
    if (!applied && pendingFile && event.hook_event_name === 'BeforeModel') {
      try { if (readText(pendingFile, { optional: true, privateFile: true, maxBytes: 256 }) !== null) fs.unlinkSync(pendingFile); }
      catch { /* Unsafe/unreadable state cannot apply; never follow it. */ }
    }
    if (lock) { try { fs.rmdirSync(lock); } catch { /* fail open */ } }
  }
  return null;
}

// The caller owns process startup, initialization, trust UI, and permissions. No host patch or process spawning.
export function createCodexWorkflowClient({ rpc, getMode, runtime, expectedVersion, server = 'pointsman', tool = 'run', sandboxPolicy = { type: 'readOnly' }, cwd, commands = {} }) {
  if (!['readOnly', 'workspaceWrite'].includes(sandboxPolicy.type)) throw new Error('UNSAFE_WORKFLOW_SANDBOX');
  const registry = structuredClone(commands);
  return {
    async run(envelope, { signal } = {}) {
      abort(signal);
      let result;
      const enabled = activeMode(getMode) === 'on' && envelope?.scope === 'workflow' && matchesWorkflowHost('codex', runtime, expectedVersion);
      if (enabled) {
        try {
          const response = await rpc.request('mcpServer/tool/call', { threadId: envelope.threadId, server, tool, arguments: envelope.request }, { signal });
          const payload = response.result ?? response;
          if (payload.isError === true) throw new Error('WORKFLOW_TOOL_ERROR');
          result = payload.structuredContent;
          if (!result && Array.isArray(payload.content)) {
            const text = payload.content.find(c => c.type === 'text')?.text;
            if (text) result = JSON.parse(text);
          }
        } catch (error) { if (signal?.aborted) throw error; }
      }
      abort(signal);
      if (enabled && activeMode(getMode) === 'on' && isCompletedWorkflow(envelope.request, result)) return { status: 'completed', result, text: workflowResultText(result), turnsStarted: 0, inferenceEvidence: 'client_dispatch_only' };
      if (!Array.isArray(envelope.input)) throw new Error('MISSING_CONTINUATION_INPUT');
      const input = [...envelope.input];
      if (result) input.push({ type: 'text', text: `Bounded workflow evidence (unresolved): ${JSON.stringify(result)}` });
      const response = await rpc.request('turn/start', { threadId: envelope.threadId, input }, { signal });
      if (signal?.aborted && response.turn?.id) await rpc.request('turn/interrupt', { threadId: envelope.threadId, turnId: response.turn.id });
      abort(signal);
      return { status: 'continued', response, result, turnsStarted: 1, inferenceEvidence: 'UNKNOWN' };
    },
    async runCommand(name, { signal } = {}) {
      abort(signal);
      if (activeMode(getMode) !== 'on' || !matchesWorkflowHost('codex', runtime, expectedVersion)) throw new Error('WORKFLOW_NATIVE_OFF');
      const entry = registry[name];
      if (!entry || !Array.isArray(entry.command) || entry.command.length === 0 || entry.command.some(v => typeof v !== 'string') || !Number.isInteger(entry.timeoutMs) || entry.timeoutMs < 1 || entry.timeoutMs > 60000) throw new Error('UNREGISTERED_WORKFLOW_COMMAND');
      const response = await rpc.request('command/exec', { command: entry.command, cwd, sandboxPolicy, timeoutMs: entry.timeoutMs }, { signal });
      abort(signal);
      return response;
    },
  };
}

// Minimal stdio transport for an already owned app-server process. Approval requests stay with its normal UI.
export function createWorkflowRpc({ input, output, onRequest, onNotification = () => {}, timeoutMs = 60000 }) {
  let id = 0;
  const pending = new Map(), lines = createInterface({ input });
  const send = value => output.write(`${JSON.stringify(value)}\n`);
  lines.on('line', async line => {
    let msg; try { msg = JSON.parse(line); } catch { return; }
    if (msg.method) {
      if (msg.id === undefined) { onNotification(msg); return; }
      try {
        if (!onRequest) throw new Error('Approval UI is not attached');
        send({ id: msg.id, result: await onRequest(msg) });
      } catch { send({ id: msg.id, error: { code: -32000, message: 'Request requires the normal host permission UI' } }); }
      return;
    }
    const item = pending.get(msg.id); if (!item) return;
    pending.delete(msg.id); clearTimeout(item.timer);
    if (msg.error) item.reject(new Error(msg.error.message ?? 'RPC_ERROR')); else item.resolve(msg.result);
  });
  lines.on('close', () => { for (const p of pending.values()) { clearTimeout(p.timer); p.reject(new Error('RPC_CLOSED')); } pending.clear(); });
  return {
    request(method, params, { signal } = {}) {
      abort(signal);
      return new Promise((resolve, reject) => {
        const requestId = ++id;
        const timer = setTimeout(() => { pending.delete(requestId); reject(new Error('RPC_TIMEOUT_OUTCOME_UNKNOWN')); }, timeoutMs);
        pending.set(requestId, { resolve, reject, timer });
        // Keep reading a dispatched operation's ACK on cancellation; the caller must interrupt a returned turn.
        send({ id: requestId, method, params });
      });
    },
    notify: (method, params) => send({ method, ...(params === undefined ? {} : { params }) }),
    close: () => lines.close(),
  };
}
