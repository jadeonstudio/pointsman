import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { randomUUID, createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { errorCode, isObject } from './constants.mjs';
import { resolveHome, appendEvent, ensureDir, atomicWrite, readText } from './storage.mjs';
import { createDecisionEngine } from './engine.mjs';
import { createControlLayer } from './control-layer.mjs';
import { loadFeaturePolicy, TIERS } from './feature-policy.mjs';
import { discoverHostRoles } from './host-roles.mjs';
import { createLinkIndex, recordSubagentStop } from './training/links.mjs';
import { createTrainingStore } from './training/store.mjs';

// P3a hook executor (`pointsman hook --host <host> --event <event>`). Every path here is fail-open:
// a hook must never deny a tool call, never answer a permission request, and must never let an
// internal error, a timeout or a malformed host payload reach the host as anything but "no output".
// The only output this module ever produces is the ON-mode subagent-spawn rewrite documented in
// AGENTS.md ("Owned host hooks"): {hookSpecificOutput:{hookEventName:'PreToolUse', permissionDecision:'allow', updatedInput}}.
export const HOSTS = Object.freeze(['codex', 'claude']);
export const HOOK_EVENTS = Object.freeze(['pre-spawn', 'post-spawn', 'subagent-start', 'subagent-stop']);
export const HOOK_TIMEOUT_MS = 4000;
export const MAX_HOOK_STDIN_BYTES = 256 * 1024;
const PENDING_RETENTION_MS = 60000;
const MAX_PENDING = 256;
// [route scope=<local|cross-module|repository|unknown> complete=<yes|no> failures=<0-100> (impact=<high|normal>)? (exhaustive=<yes|no>)?]
const CONTEXT_RE = /\[route scope=(local|cross-module|repository|unknown) complete=(yes|no) failures=(\d{1,3})(?: impact=(high|normal))?(?: exhaustive=(yes|no))?\]/;

function withTimeout(promise, ms) {
  return new Promise(resolve => {
    let settled = false;
    const timer = setTimeout(() => { if (!settled) { settled = true; resolve(null); } }, ms);
    if (typeof timer.unref === 'function') timer.unref();
    Promise.resolve(promise).then(
      value => { if (!settled) { settled = true; clearTimeout(timer); resolve(value); } },
      () => { if (!settled) { settled = true; clearTimeout(timer); resolve(null); } },
    );
  });
}
/** UTF-8 byte-boundary safe truncation: never keeps a partial multi-byte sequence at the tail. */
export function truncateUtf8(text, maxBytes) {
  const buf = Buffer.from(text, 'utf8');
  if (buf.length <= maxBytes) return text;
  let end = maxBytes;
  while (end > 0 && (buf[end] & 0xc0) === 0x80) end--;
  return buf.subarray(0, end).toString('utf8');
}
/** Finds the first `[route scope=... complete=... failures=...]` marker; returns null when absent or malformed. */
export function parseContextAnnotation(text) {
  if (typeof text !== 'string') return null;
  const m = CONTEXT_RE.exec(text);
  if (!m) return null;
  const previousFailures = Number(m[3]);
  if (!Number.isInteger(previousFailures) || previousFailures < 0 || previousFailures > 100) return null;
  return { scope: m[1], complete: m[2] === 'yes', previousFailures, highImpact: m[4] === 'high', exhaustive: m[5] === 'yes' };
}
/** No external `git` process: reads `.git/HEAD` and follows one ref file. Any failure falls back to a cwd-derived hash. */
export function resolveSnapshotId(cwd) {
  try {
    const head = fs.readFileSync(path.join(cwd, '.git', 'HEAD'), 'utf8').trim();
    const sha = head.startsWith('ref:') ? fs.readFileSync(path.join(cwd, '.git', head.slice(4).trim()), 'utf8').trim() : head;
    if (/^[0-9a-f]{40}$/i.test(sha)) return createHash('sha256').update(sha).digest('hex');
  } catch { /* fall through to the cwd-derived fallback */ }
  return createHash('sha256').update(`cwd:${String(cwd)}`).digest('hex');
}
function profileRoleSet(policy, host) {
  const profile = policy.router.profiles[host] ?? {};
  const roles = new Set();
  for (const tier of TIERS) if (profile[tier]?.role) roles.add(profile[tier].role);
  for (const target of Object.values(profile.intents ?? {})) if (target?.role) roles.add(target.role);
  return roles;
}
// Only subagent spawn tools are ever routed or rewritten, even if a host matcher is misconfigured.
const SPAWN_TOOLS = { claude: name => name === 'Agent' || name === 'Task', codex: name => /^(?:[a-z_]+\.?)?spawn_agent$/.test(name) || name === 'Agent' }; // 0.154.0 sends `agentsspawn_agent`
function extractPreSpawn(host, input) {
  if (!isObject(input) || !isObject(input.tool_input)) return null;
  if (typeof input.tool_name !== 'string' || !SPAWN_TOOLS[host]?.(input.tool_name)) return null;
  const ti = input.tool_input;
  const toolUseId = typeof input.tool_use_id === 'string' && input.tool_use_id ? input.tool_use_id : null;
  const sessionId = typeof input.session_id === 'string' && input.session_id ? input.session_id : null;
  const cwd = typeof input.cwd === 'string' && input.cwd ? input.cwd : process.cwd();
  if (host === 'claude') {
    if (ti.subagent_type !== undefined && (typeof ti.subagent_type !== 'string' || !ti.subagent_type)) return null;
    return { toolUseId, sessionId, cwd, toolInput: ti,
      originalRole: typeof ti.subagent_type === 'string' && ti.subagent_type ? ti.subagent_type : 'general-purpose',
      promptText: typeof ti.prompt === 'string' ? ti.prompt : '', description: typeof ti.description === 'string' ? ti.description : '',
      modelLocked: typeof ti.model === 'string' && ti.model.length > 0 };
  }
  if (host === 'codex') {
    if (typeof ti.agent_type !== 'string' || !ti.agent_type) return null;
    return { toolUseId, sessionId, cwd, toolInput: ti, originalRole: ti.agent_type,
      promptText: typeof ti.message === 'string' ? ti.message : '', description: typeof ti.task_name === 'string' ? ti.task_name : '',
      modelLocked: typeof ti.model === 'string' && ti.model.length > 0 };
  }
  return null;
}
function findAgentId(value, depth = 0) {
  if (depth > 4 || value == null) return null;
  if (typeof value === 'string') { try { return findAgentId(JSON.parse(value), depth + 1); } catch { return null; } }
  if (Array.isArray(value)) { for (const item of value) { const found = findAgentId(item, depth + 1); if (found) return found; } return null; }
  if (isObject(value)) {
    if (typeof value.agentId === 'string' && value.agentId) return value.agentId;
    for (const v of Object.values(value)) { const found = findAgentId(v, depth + 1); if (found) return found; }
  }
  return null;
}
// Codex has no official tool_use_id<->agent_id field, so a short-lived, content-free pending queue
// (session_id + the role that will actually spawn) links a SubagentStart to its pre-spawn decision.
// This is a heuristic, matched on the oldest live candidate, and is reported as such in telemetry.
function pendingDir(home) { return path.join(home, 'links', 'pending'); }
function readPendingEntry(file) {
  const raw = readText(file, { optional: true, privateFile: true, maxBytes: 2048 });
  if (raw === null) return null;
  try {
    const e = JSON.parse(raw);
    if (!isObject(e) || typeof e.session_id !== 'string' || typeof e.agent_type !== 'string' ||
        typeof e.decision_id !== 'string' || typeof e.created_at !== 'string') return null;
    return e;
  } catch { return null; }
}
function prunePending(dir, nowMs) {
  let names; try { names = fs.readdirSync(dir); } catch { return []; }
  const alive = [];
  for (const name of names) {
    if (!name.endsWith('.json')) continue;
    const file = path.join(dir, name);
    const entry = readPendingEntry(file);
    const at = entry ? Date.parse(entry.created_at) : NaN;
    if (!entry || !Number.isFinite(at) || nowMs - at > PENDING_RETENTION_MS) { try { fs.unlinkSync(file); } catch { /* already gone */ } continue; }
    alive.push({ file, entry, at });
  }
  return alive;
}
function addPendingCodex(home, { sessionId, agentType, decisionId }, nowMs) {
  try {
    const dir = pendingDir(home);
    ensureDir(home, true); ensureDir(path.join(home, 'links'), true); ensureDir(dir, true);
    const alive = prunePending(dir, nowMs).sort((a, b) => a.at - b.at);
    while (alive.length >= MAX_PENDING) { const oldest = alive.shift(); try { fs.unlinkSync(oldest.file); } catch { /* already gone */ } }
    const file = path.join(dir, `${randomUUID()}.json`);
    atomicWrite(file, JSON.stringify({ session_id: sessionId, agent_type: agentType, decision_id: decisionId, created_at: new Date(nowMs).toISOString() }) + '\n', { mode: 0o600, expected: null });
  } catch { /* best-effort: a missed pending write only degrades the Codex link heuristic */ }
}
function consumePendingCodex(home, { sessionId, agentType }, nowMs) {
  try {
    const alive = prunePending(pendingDir(home), nowMs).filter(({ entry }) => entry.session_id === sessionId && entry.agent_type === agentType);
    if (!alive.length) return null;
    alive.sort((a, b) => a.at - b.at);
    const [chosen] = alive;
    try { fs.unlinkSync(chosen.file); } catch { /* already gone */ }
    return chosen.entry.decision_id;
  } catch { return null; }
}

async function preSpawn(host, input, ctx) {
  const info = extractPreSpawn(host, input);
  if (!info) {
    // Tool names are host identifiers, not content; keep only a short safe token to diagnose matcher/name mismatches.
    const name = isObject(input) && typeof input.tool_name === 'string' && /^[A-Za-z0-9_.:-]{1,64}$/.test(input.tool_name) ? input.tool_name : null;
    return { output: null, telemetry: { reason: 'INVALID_HOOK_INPUT', tool_name: name } };
  }
  let policy;
  try { policy = loadFeaturePolicy(ctx.home); }
  catch (error) { return { output: null, telemetry: { reason: errorCode(error), original_role: info.originalRole } }; }
  if (!profileRoleSet(policy, host).has(info.originalRole)) {
    return { output: null, telemetry: { reason: 'ROLE_NOT_ROUTABLE', original_role: info.originalRole } };
  }
  const annotation = parseContextAnnotation(info.promptText);
  if (!annotation) {
    // Content-free form only: codex-cli 0.154.0 sends spawn_agent.message as an opaque Fernet-like token the hook cannot read.
    const promptForm = /^gAAAA[A-Za-z0-9_\-=]+$/.test(info.promptText) ? 'opaque' : 'plain';
    return { output: null, telemetry: { reason: 'NO_CONTEXT_ANNOTATION', original_role: info.originalRole, prompt_form: promptForm } };
  }
  const taskText = truncateUtf8(`${info.description}\n${info.promptText}`, 8000);
  const routeInput = { task: taskText, host, risk: 'routine',
    context: { complete: annotation.complete, scope: annotation.scope, previousFailures: annotation.previousFailures,
      highImpact: annotation.highImpact, modelLocked: info.modelLocked, exhaustive: annotation.exhaustive },
    availableRoles: discoverHostRoles(host, { env: ctx.env, userHome: ctx.env.HOME || os.homedir() }), availableSkills: [] };
  const trace = { task_id: randomUUID(), snapshot_id: resolveSnapshotId(info.cwd) };
  const result = await ctx.layer.route(routeInput, { trace });
  const telemetry = { reason: result.reason, original_role: info.originalRole,
    recommended_role: result.route?.role ?? null, decision_id: result.mode !== 'off' ? result.id : null };
  if (result.mode === 'off') return { output: null, telemetry };
  if (info.toolUseId) ctx.links.put({ host, kind: 'tool_use', id: info.toolUseId, decision_id: result.id });
  if (host === 'codex' && info.sessionId) {
    const finalRole = result.apply && result.route?.role ? result.route.role : info.originalRole;
    addPendingCodex(ctx.home, { sessionId: info.sessionId, agentType: finalRole, decisionId: result.id }, ctx.now());
  }
  let output = null;
  if (result.apply && result.route?.role && result.route.role !== info.originalRole) {
    output = host === 'claude'
      ? { hookSpecificOutput: { hookEventName: 'PreToolUse', permissionDecision: 'allow',
          updatedInput: { ...info.toolInput, subagent_type: result.route.role, ...(result.route.model ? { model: result.route.model } : {}) } } }
      : { hookSpecificOutput: { hookEventName: 'PreToolUse', permissionDecision: 'allow',
          updatedInput: { ...info.toolInput, agent_type: result.route.role } } };
  }
  return { output, telemetry };
}
function postSpawn(host, input, ctx) {
  if (host !== 'claude') return { output: null, telemetry: { reason: 'NOOP' } };
  if (!isObject(input) || typeof input.tool_use_id !== 'string' || !input.tool_use_id) return { output: null, telemetry: { reason: 'INVALID_HOOK_INPUT' } };
  const agentId = findAgentId(input.tool_response);
  if (!agentId) return { output: null, telemetry: { reason: 'NO_AGENT_ID' } };
  const r = ctx.links.alias({ host: 'claude', fromKind: 'tool_use', fromId: input.tool_use_id, toKind: 'agent', toId: agentId });
  return { output: null, telemetry: { reason: r.stored ? 'LINKED' : (r.reason ?? 'NOT_LINKED') } };
}
function subagentStart(host, input, ctx) {
  if (host !== 'codex') return { output: null, telemetry: { reason: 'NOOP' } };
  if (!isObject(input) || typeof input.agent_id !== 'string' || !input.agent_id ||
      typeof input.agent_type !== 'string' || !input.agent_type || typeof input.session_id !== 'string' || !input.session_id) {
    return { output: null, telemetry: { reason: 'INVALID_HOOK_INPUT' } };
  }
  const decisionId = consumePendingCodex(ctx.home, { sessionId: input.session_id, agentType: input.agent_type }, ctx.now());
  if (!decisionId) return { output: null, telemetry: { reason: 'NO_PENDING_MATCH' } };
  ctx.links.put({ host: 'codex', kind: 'agent', id: input.agent_id, decision_id: decisionId });
  return { output: null, telemetry: { reason: 'LINKED', decision_id: decisionId, link: 'heuristic' } };
}
function subagentStop(host, input, ctx) {
  if (!isObject(input) || typeof input.agent_id !== 'string' || !input.agent_id) return { output: null, telemetry: { reason: 'INVALID_HOOK_INPUT' } };
  const r = recordSubagentStop(ctx.store, ctx.links, { host, agent_id: input.agent_id, status: 'completed' });
  return { output: null, telemetry: { reason: r.stored ? 'RECORDED' : (r.reason ?? 'NOT_RECORDED') } };
}
const HANDLERS = { 'pre-spawn': preSpawn, 'post-spawn': postSpawn, 'subagent-start': subagentStart, 'subagent-stop': subagentStop };

function logHookEvent({ home, layer, host, event, telemetry, applied, elapsedMs }) {
  try {
    const status = layer.status();
    const mode = status.features.router.mode;
    if (mode === 'off' || !status.telemetry) return;
    appendEvent(home, { kind: 'hook', at: new Date().toISOString(), host, event, reason: telemetry?.reason ?? null,
      mode, applied: Boolean(applied), original_role: telemetry?.original_role ?? null,
      recommended_role: telemetry?.recommended_role ?? null, decision_id: telemetry?.decision_id ?? null, elapsedMs,
      ...(telemetry?.tool_name !== undefined ? { tool_name: telemetry.tool_name } : {}),
      ...(telemetry?.prompt_form !== undefined ? { prompt_form: telemetry.prompt_form } : {}) });
  } catch { /* observability cannot become an availability dependency */ }
}
/** Pure-ish dispatcher: takes an already-parsed hook payload, never throws, only ever logs a content-free event. */
export async function processHookEvent({ host, event, input, home = resolveHome(), env = process.env, now = () => Date.now(),
  layer, links = createLinkIndex({ home, now }), store = createTrainingStore({ home }) } = {}) {
  const start = performance.now();
  if (!HOSTS.includes(host) || !HOOK_EVENTS.includes(event) || typeof layer?.route !== 'function') return { output: null, telemetry: null };
  // The whole owned hook (every event, not only pre-spawn) passes through immediately when global
  // OFF, router OFF or POINTSMAN_DISABLE=1 apply: no route call, no link/pending write, no outcome record.
  let status;
  try { status = layer.status(); } catch { return { output: null, telemetry: null }; }
  if (status.features.router.mode === 'off') return { output: null, telemetry: null };
  const ctx = { home, env, now, layer, links, store };
  let outcome;
  try { outcome = await HANDLERS[event](host, input, ctx); }
  catch (error) { outcome = { output: null, telemetry: { reason: errorCode(error) } }; }
  const elapsedMs = Math.round((performance.now() - start) * 1000) / 1000;
  logHookEvent({ home, layer, host, event, telemetry: outcome.telemetry, applied: Boolean(outcome.output), elapsedMs });
  return { output: outcome.output ?? null, telemetry: outcome.telemetry ?? null };
}
async function readHookInput({ stdin, maxBytes }) {
  if (!stdin || stdin.isTTY) return null;
  const chunks = []; let size = 0;
  try {
    for await (const chunk of stdin) {
      const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      size += buf.length;
      if (size > maxBytes) return null;
      chunks.push(buf);
    }
  } catch { return null; }
  if (!size) return null;
  try { return JSON.parse(Buffer.concat(chunks).toString('utf8')); } catch { return null; }
}
/**
 * CLI entry for `pointsman hook --host H --event E`. Always fail-open: any invalid argument,
 * missing/oversized/malformed stdin, config error, provider error or timeout ends in exit 0 and no
 * stdout output, matching the host contract that a hook must never be visible as a failure.
 */
export async function runHookCli({ host, event, home: homeOverride, env = process.env, now = () => Date.now(),
  stdin = process.stdin, write = text => process.stdout.write(text), timeoutMs = HOOK_TIMEOUT_MS } = {}) {
  try {
    if (!HOSTS.includes(host) || !HOOK_EVENTS.includes(event)) return;
    const home = resolveHome({ ...env, ...(homeOverride ? { POINTSMAN_HOME: homeOverride } : {}) });
    let engine, layer;
    // L3 (2026-09-23): a hook never spawns its own worker and never waits for a resident server's
    // worker to finish loading; a socket that exists but is still loading returns LAYA_NOT_READY at once.
    try { engine = createDecisionEngine({ home, env, layaSpawn: false, layaWait: false }); layer = createControlLayer({ home, env, engine }); }
    catch { return; }
    try {
      const status = layer.status();
      if (status.features.router.mode === 'off') return;
      const start = performance.now();
      let settledReason = null;
      const result = await withTimeout((async () => {
        const input = await readHookInput({ stdin, maxBytes: MAX_HOOK_STDIN_BYTES });
        if (input === null) { settledReason = 'INVALID_STDIN'; return null; }
        const r = await processHookEvent({ host, event, input, home, env, now, layer });
        settledReason = 'PROCESSED';
        return r;
      })(), timeoutMs);
      // Content-free diagnostics: distinguish "host never called us" from "called but input rejected / timed out".
      if (settledReason !== 'PROCESSED') {
        logHookEvent({ home, layer, host, event, telemetry: { reason: settledReason ?? 'HOOK_TIMEOUT' }, applied: false,
          elapsedMs: Math.round((performance.now() - start) * 1000) / 1000 });
      }
      if (result?.output) write(JSON.stringify(result.output) + '\n');
    } finally { engine.close(); }
  } catch { /* fail-open */ }
}
