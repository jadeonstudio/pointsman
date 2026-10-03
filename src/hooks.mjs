import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { randomUUID, randomInt, createHash } from 'node:crypto';
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
// turn-effort/turn-outcome are Claude-only main-loop effort-mod events (docs/plan/2026-09-27-pointsman-effort-mod.md,
// mods/pointsman-effort): gated by the EFFORT feature mode, not router, and never touch subagents.
export const HOOK_EVENTS = Object.freeze(['pre-spawn', 'post-spawn', 'subagent-start', 'subagent-stop', 'turn-effort', 'turn-outcome']);
const EFFORT_ONLY_EVENTS = new Set(['turn-effort', 'turn-outcome']);
/** Which feature mode gates a given hook event: turn-effort/turn-outcome follow `effort`, everything else follows `router`. */
function gateModeFor(status, event) { return EFFORT_ONLY_EVENTS.has(event) ? status.features.effort.mode : status.features.router.mode; }
export const HOOK_TIMEOUT_MS = 4000;
export const MAX_HOOK_STDIN_BYTES = 256 * 1024;
const PENDING_RETENTION_MS = 60000;
const MAX_PENDING = 256;
// How long a turn-effort decision's {arm, level} stays available for the matching turn-outcome to
// read back (a turn can run for several minutes, unlike the router's short pre-spawn->post-spawn gap).
const EFFORT_PENDING_RETENTION_MS = 30 * 60 * 1000;
const MAX_EFFORT_PENDING = 256;
const BARE_COMMAND_RE = /^\/\S+$/;
const EFFORT_LEVELS = new Set(['low', 'medium', 'high', 'xhigh', 'max']);
const SAFE_TOKEN_RE = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,99}$/;
// How long a subagent-stop's {agent_id -> transcript path} memo (written for the foreground case,
// where SubagentStop fires before PostToolUse) stays available for the matching post-spawn to consume.
const TRANSCRIPT_MEMO_RETENTION_MS = 5 * 60 * 1000;
const MAX_TRANSCRIPT_MEMOS = 256;
// Full-run usage transcripts are read once per outcome and capped well under Node's default string limit.
const MAX_TRANSCRIPT_BYTES = 64 * 1024 * 1024;
// [route scope=<local|cross-module|repository|unknown> complete=<yes|no> failures=<0-100> (impact=<high|normal>)? (exhaustive=<yes|no>)?]
const CONTEXT_RE = /\[route scope=(local|cross-module|repository|unknown) complete=(yes|no) failures=(\d{1,3})(?: impact=(high|normal))?(?: exhaustive=(yes|no))?\]/;

function withTimeout(promise, ms, expire = () => {}) {
  return new Promise(resolve => {
    let settled = false; const deadline = performance.now() + ms;
    const timedOut = () => { if (!settled) { settled = true; clearTimeout(timer); expire(); resolve(null); } };
    // Keep the timer referenced: it is always cleared on settle, and an unref'd timer lets Node 22 end
    // the event loop while a never-ending stdin is still pending, so the timeout would never fire.
    const timer = setTimeout(timedOut, ms);
    Promise.resolve(promise).then(
      value => { if (performance.now() >= deadline) return timedOut(); if (!settled) { settled = true; clearTimeout(timer); resolve(value); } },
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

// --- effort-mod pending decisions (turn-effort -> turn-outcome correlation) ---------------------
// `pointsman hook` is a short-lived CLI process per invocation, so an in-memory map cannot bridge
// turn-effort and the later turn-outcome call for the same turn; this content-free file (decision_id
// -> {arm, level}, nothing else) plays the same role addPendingCodex/consumePendingCodex play above.
function effortPendingDir(home) { return path.join(home, 'links', 'pending-effort'); }
function readEffortPendingEntry(file) {
  const raw = readText(file, { optional: true, privateFile: true, maxBytes: 512 });
  if (raw === null) return null;
  try {
    const e = JSON.parse(raw);
    if (!isObject(e) || typeof e.arm !== 'string' || typeof e.created_at !== 'string') return null;
    return e;
  } catch { return null; }
}
function pruneEffortPending(dir, nowMs) {
  let names; try { names = fs.readdirSync(dir); } catch { return []; }
  const alive = [];
  for (const name of names) {
    if (!name.endsWith('.json')) continue;
    const file = path.join(dir, name);
    const entry = readEffortPendingEntry(file);
    const at = entry ? Date.parse(entry.created_at) : NaN;
    if (!entry || !Number.isFinite(at) || nowMs - at > EFFORT_PENDING_RETENTION_MS) { try { fs.unlinkSync(file); } catch { /* already gone */ } continue; }
    alive.push({ file, entry, at });
  }
  return alive;
}
function effortPendingFile(home, decisionId) { return path.join(effortPendingDir(home), `${decisionId}.json`); }
function rememberEffortDecision(home, { decisionId, arm, level }, nowMs) {
  try {
    const dir = effortPendingDir(home);
    ensureDir(home, true); ensureDir(path.join(home, 'links'), true); ensureDir(dir, true);
    const alive = pruneEffortPending(dir, nowMs).sort((a, b) => a.at - b.at);
    while (alive.length >= MAX_EFFORT_PENDING) { const oldest = alive.shift(); try { fs.unlinkSync(oldest.file); } catch { /* already gone */ } }
    atomicWrite(effortPendingFile(home, decisionId), JSON.stringify({ arm, level: level ?? null, created_at: new Date(nowMs).toISOString() }) + '\n', { mode: 0o600, expected: null });
  } catch { /* best-effort: a missed pending write only degrades turn-outcome's arm to 'none' */ }
}
function recallEffortDecision(home, decisionId, nowMs) {
  try {
    pruneEffortPending(effortPendingDir(home), nowMs); // drop expired siblings opportunistically
    const file = effortPendingFile(home, decisionId);
    const entry = readEffortPendingEntry(file);
    if (!entry || nowMs - Date.parse(entry.created_at) > EFFORT_PENDING_RETENTION_MS) return null;
    try { fs.unlinkSync(file); } catch { /* already gone */ }
    return { arm: entry.arm, level: entry.level ?? null };
  } catch { return null; }
}

// --- agent transcript usage (P5 outcome linking) -----------------------------------------------
// SubagentStop fires BEFORE PostToolUse(Agent) for a foreground spawn (documented Claude Code
// hook order, 2026 late). SubagentStop already carries agent_transcript_path directly, so it
// memoizes {agent_id -> transcript path} here (content-free: a path, nothing from the transcript
// itself) for the matching post-spawn -- which resolves the decision via the tool_use link created
// at pre-spawn -- to read full-run usage from once the outcome is actually recorded.
function transcriptMemoDir(home) { return path.join(home, 'links', 'transcripts'); }
function transcriptMemoKey(host, agentId) { return createHash('sha256').update(`${host}|${agentId}`).digest('hex'); }
function pruneTranscriptMemos(dir, nowMs) {
  let names; try { names = fs.readdirSync(dir); } catch { return []; }
  const alive = [];
  for (const name of names) {
    if (!name.endsWith('.json')) continue;
    const file = path.join(dir, name);
    const raw = readText(file, { optional: true, privateFile: true, maxBytes: 4096 });
    let entry = null;
    try { const e = raw === null ? null : JSON.parse(raw); if (e && typeof e.transcript_path === 'string' && typeof e.created_at === 'string') entry = e; } catch { entry = null; }
    const at = entry ? Date.parse(entry.created_at) : NaN;
    if (!entry || !Number.isFinite(at) || nowMs - at > TRANSCRIPT_MEMO_RETENTION_MS) { try { fs.unlinkSync(file); } catch { /* already gone */ } continue; }
    alive.push({ file, entry, at });
  }
  return alive;
}
function rememberAgentTranscript(home, { host, agentId, transcriptPath }, nowMs) {
  try {
    const dir = transcriptMemoDir(home);
    ensureDir(home, true); ensureDir(path.join(home, 'links'), true); ensureDir(dir, true);
    const alive = pruneTranscriptMemos(dir, nowMs).sort((a, b) => a.at - b.at);
    while (alive.length >= MAX_TRANSCRIPT_MEMOS) { const oldest = alive.shift(); try { fs.unlinkSync(oldest.file); } catch { /* already gone */ } }
    const file = path.join(dir, `${transcriptMemoKey(host, agentId)}.json`);
    atomicWrite(file, JSON.stringify({ transcript_path: transcriptPath, created_at: new Date(nowMs).toISOString() }) + '\n', { mode: 0o600, expected: null });
  } catch { /* best-effort: a missed memo only falls back to deriving the path from PostToolUse's own transcript_path */ }
}
function recallAgentTranscript(home, { host, agentId }, nowMs) {
  try {
    const file = path.join(transcriptMemoDir(home), `${transcriptMemoKey(host, agentId)}.json`);
    const raw = readText(file, { optional: true, privateFile: true, maxBytes: 4096 });
    if (raw === null) return null;
    const e = JSON.parse(raw);
    if (typeof e.transcript_path !== 'string' || !e.transcript_path) return null;
    if (!Number.isFinite(Date.parse(e.created_at)) || nowMs - Date.parse(e.created_at) > TRANSCRIPT_MEMO_RETENTION_MS) return null;
    return e.transcript_path;
  } catch { return null; }
}
/** Documented layout: "<session-transcript-dir>/<session>/subagents/agent-<id>.jsonl". */
export function deriveAgentTranscriptPath(mainTranscriptPath, agentId) {
  if (typeof mainTranscriptPath !== 'string' || !mainTranscriptPath.endsWith('.jsonl') || typeof agentId !== 'string' || !agentId) return null;
  const dir = path.dirname(mainTranscriptPath);
  const session = path.basename(mainTranscriptPath, '.jsonl');
  if (!session) return null;
  return path.join(dir, session, 'subagents', `agent-${agentId}.jsonl`);
}
/**
 * Reads ONLY numeric usage fields and the model id from each transcript line (never message
 * content). Refuses anything outside `<HOME>/.claude/projects`, a symlink anywhere on the path, or
 * a non-regular/hard-linked file. Returns null (never throws) on any missing/unsafe/unreadable
 * input, so a missing transcript degrades the outcome event to "no usage", never a hook failure.
 */
export function readAgentUsageSummary(transcriptPath, env = process.env) {
  if (typeof transcriptPath !== 'string' || !transcriptPath) return null;
  const root = path.resolve(env.HOME || os.homedir(), '.claude', 'projects');
  let resolved;
  try { resolved = path.resolve(transcriptPath); } catch { return null; }
  if (resolved !== root && !resolved.startsWith(root + path.sep)) return null;
  // Walk every path component (equivalent to storage.mjs's noSymlinks): a symlinked transcript file
  // or any symlinked ancestor directory is refused, never followed.
  let current = resolved;
  for (;;) {
    try { if (fs.lstatSync(current).isSymbolicLink()) return null; } catch (e) { if (e.code !== 'ENOENT') return null; }
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  let fd;
  try { fd = fs.openSync(resolved, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW || 0)); }
  catch { return null; }
  try {
    const stat = fs.fstatSync(fd);
    if (!stat.isFile() || stat.nlink !== 1) return null;
    const size = Math.min(stat.size, MAX_TRANSCRIPT_BYTES);
    const buf = Buffer.alloc(size);
    fs.readSync(fd, buf, 0, size, 0);
    const text = buf.toString('utf8');
    const byModel = Object.create(null);
    let turns = 0;
    for (const line of text.split('\n')) {
      if (!line.trim()) continue;
      let row; try { row = JSON.parse(line); } catch { continue; }
      const usage = row?.message?.usage, model = row?.message?.model;
      if (!isObject(usage) || typeof model !== 'string' || !model) continue;
      const num = v => Number.isFinite(v) ? v : 0;
      const bucket = byModel[model] ?? (byModel[model] = { input: 0, output: 0, cache_creation: 0, cache_read: 0, turns: 0 });
      bucket.input += num(usage.input_tokens); bucket.output += num(usage.output_tokens);
      bucket.cache_creation += num(usage.cache_creation_input_tokens); bucket.cache_read += num(usage.cache_read_input_tokens);
      bucket.turns += 1; turns += 1;
    }
    if (!turns) return { byModel: null, truncated: stat.size > MAX_TRANSCRIPT_BYTES };
    return { byModel, truncated: stat.size > MAX_TRANSCRIPT_BYTES };
  } catch { return null; }
  finally { fs.closeSync(fd); }
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
  const result = await ctx.layer.route(routeInput, { trace, signal: ctx.signal });
  if (ctx.signal?.aborted || performance.now() >= ctx.deadline) return { output: null, telemetry: null };
  const telemetry = { reason: result.reason, original_role: info.originalRole,
    recommended_role: result.route?.role ?? null, decision_id: result.mode !== 'off' ? result.id : null };
  if (result.mode === 'off') return { output: null, telemetry };
  // Randomised A/B split (`pointsman router ab`): only ever considered when a rewrite would
  // otherwise happen. 'control' logs the proposal but keeps the host's original choice; 'treatment'
  // applies it as before; 'none' means the route didn't apply or already matched the original.
  const wouldRewrite = Boolean(result.apply && result.route?.role && result.route.role !== info.originalRole);
  let arm = 'none';
  if (wouldRewrite) {
    const share = Number.isFinite(policy.router.abControlShare) ? policy.router.abControlShare : 0;
    arm = share > 0 && randomInt(1_000_000) < Math.round(share * 1_000_000) ? 'control' : 'treatment';
  }
  telemetry.arm = arm;
  const applyRewrite = wouldRewrite && arm !== 'control';
  const finalRole = applyRewrite ? result.route.role : info.originalRole;
  const finalModel = applyRewrite ? (result.route.model ?? null) : null;
  if (info.toolUseId) ctx.links.put({ host, kind: 'tool_use', id: info.toolUseId, decision_id: result.id, arm, final_role: finalRole, final_model: finalModel });
  if (host === 'codex' && info.sessionId) {
    addPendingCodex(ctx.home, { sessionId: info.sessionId, agentType: finalRole, decisionId: result.id }, ctx.now());
  }
  let output = null;
  if (applyRewrite) {
    output = host === 'claude'
      ? { hookSpecificOutput: { hookEventName: 'PreToolUse', permissionDecision: 'allow',
          updatedInput: { ...info.toolInput, subagent_type: result.route.role, ...(result.route.model ? { model: result.route.model } : {}) } } }
      : { hookSpecificOutput: { hookEventName: 'PreToolUse', permissionDecision: 'allow',
          updatedInput: { ...info.toolInput, agent_type: result.route.role } } };
  }
  return { output, telemetry };
}
/**
 * PostToolUse(Agent). Claude fires SubagentStop BEFORE this event for a foreground (synchronous)
 * spawn, so the decision is resolved here via the tool_use link created at pre-spawn -- never by
 * waiting on the agent alias. A background spawn (`tool_response.status === 'async_launched'`)
 * carries no usage yet; only the alias is created here, and subagent-stop records the OUTCOME once
 * the agent actually finishes.
 */
function postSpawn(host, input, ctx) {
  if (host !== 'claude') return { output: null, telemetry: { reason: 'NOOP' } };
  if (!isObject(input) || typeof input.tool_use_id !== 'string' || !input.tool_use_id) return { output: null, telemetry: { reason: 'INVALID_HOOK_INPUT' } };
  const response = isObject(input.tool_response) ? input.tool_response : null;
  const agentId = findAgentId(input.tool_response);
  const source = ctx.links.entry({ host: 'claude', kind: 'tool_use', id: input.tool_use_id });
  if (agentId) ctx.links.alias({ host: 'claude', fromKind: 'tool_use', fromId: input.tool_use_id, toKind: 'agent', toId: agentId });
  if (!agentId && !source) return { output: null, telemetry: { reason: 'NO_AGENT_ID' } };
  if (!source) return { output: null, telemetry: { reason: 'NO_LINKED_DECISION' } };
  if (response?.status === 'async_launched') return { output: null, telemetry: { reason: 'ASYNC_LAUNCHED', decision_id: source.decision_id } };
  // Foreground/synchronous completion: prefer the transcript path subagent-stop already memoized
  // (the real, documented order); otherwise derive it from this event's own transcript_path.
  let transcriptPath = agentId ? recallAgentTranscript(ctx.home, { host: 'claude', agentId }, ctx.now()) : null;
  if (!transcriptPath && agentId) transcriptPath = deriveAgentTranscriptPath(input.transcript_path, agentId);
  const usage = transcriptPath ? readAgentUsageSummary(transcriptPath, ctx.env) : null;
  logOutcomeEvent(ctx, { host, event: 'post-spawn', decisionId: source.decision_id, agentId,
    arm: source.arm, finalRole: source.final_role, finalModel: source.final_model,
    durationMs: response?.totalDurationMs, toolUses: response?.totalToolUseCount, usage: usage?.byModel ?? null });
  // Note: this per-invocation telemetry reason feeds the standard hook-event log line (logHookEvent
  // below); the separate content-free OUTCOME line was already written above via logOutcomeEvent.
  // Reusing 'OUTCOME' here would double-log the same outcome under two events.
  return { output: null, telemetry: { reason: 'LINKED', decision_id: source.decision_id } };
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
/**
 * SubagentStop. For the common foreground order (this fires before PostToolUse) the decision is not
 * linked to the agent id yet, so this only records what it can: the {agent_id -> transcript path}
 * memo the matching post-spawn will consume, and the length (never the content) of the last
 * assistant message. When the agent is already linked -- a background spawn whose post-spawn already
 * aliased agent_id to a decision -- this is instead the sole place the OUTCOME gets recorded, reading
 * usage straight from this event's own agent_transcript_path.
 */
function subagentStop(host, input, ctx) {
  if (!isObject(input) || typeof input.agent_id !== 'string' || !input.agent_id) return { output: null, telemetry: { reason: 'INVALID_HOOK_INPUT' } };
  const transcriptPath = typeof input.agent_transcript_path === 'string' && input.agent_transcript_path ? input.agent_transcript_path : null;
  if (transcriptPath) rememberAgentTranscript(ctx.home, { host, agentId: input.agent_id, transcriptPath }, ctx.now());
  const lastMessageLength = typeof input.last_assistant_message === 'string' ? input.last_assistant_message.length : null;
  const entry = ctx.links.entry({ host, kind: 'agent', id: input.agent_id });
  const r = recordSubagentStop(ctx.store, ctx.links, { host, agent_id: input.agent_id, status: 'completed' });
  const telemetry = { reason: r.stored ? 'RECORDED' : (r.reason ?? 'NOT_RECORDED'), ...(lastMessageLength !== null ? { last_message_length: lastMessageLength } : {}) };
  if (entry?.decision_id) {
    const usage = transcriptPath ? readAgentUsageSummary(transcriptPath, ctx.env) : null;
    logOutcomeEvent(ctx, { host, event: 'subagent-stop', decisionId: entry.decision_id, agentId: input.agent_id,
      arm: entry.arm, finalRole: entry.final_role, finalModel: entry.final_model, usage: usage?.byModel ?? null });
  }
  return { output: null, telemetry };
}
// --- effort mod: turn-effort / turn-outcome (Claude-only, gated by `effort` not `router`) -------
function safeToken(value) { return typeof value === 'string' && SAFE_TOKEN_RE.test(value) ? value : null; }
function turnEffortInput(input) {
  if (!isObject(input) || typeof input.text !== 'string') return null;
  if (input.context !== undefined && typeof input.context !== 'string') return null;
  if (input.current_effort !== undefined && typeof input.current_effort !== 'string' && typeof input.current_effort !== 'number') return null;
  if (input.model !== undefined && typeof input.model !== 'string') return null;
  if (input.session_id !== undefined && typeof input.session_id !== 'string') return null;
  if (input.turn_id !== undefined && typeof input.turn_id !== 'string') return null;
  // 'main' (default) is the top-level Claude Code loop; 'subagent' is a spawned agent's own loop
  // (fresh context -- see effort()'s cache-cost gate). since_last_main_ms is the mod's own
  // session-scoped measurement of how long it has been since the previous MAIN-loop request; null/absent
  // means "no earlier main-loop request this session", which counts as cold.
  if (input.loop !== undefined && !['main', 'subagent'].includes(input.loop)) return null;
  if (input.since_last_main_ms !== undefined && input.since_last_main_ms !== null &&
      (typeof input.since_last_main_ms !== 'number' || !Number.isFinite(input.since_last_main_ms) || input.since_last_main_ms < 0)) return null;
  return input;
}
/**
 * `pointsman hook --host claude --event turn-effort` (mods/pointsman-effort's prompt.submit for the
 * main loop, agent.spawn for a subagent). Never changes the prompt; only returns a content-free
 * recommendation for the mod's OWN pending-decision slot to apply on the next turn.step. A bare
 * slash command (`^/\S+$`) is skipped before any inference -- it is not real task text -- exactly
 * like preSpawn's other pre-inference guards above. Cache-cost gating (main-loop cold-only,
 * subagent opt-in) lives in control-layer's effort(), not here.
 */
async function turnEffort(host, input, ctx) {
  if (host !== 'claude') return { output: null, telemetry: { reason: 'NOOP' } };
  const v = turnEffortInput(input);
  if (!v) return { output: null, telemetry: { reason: 'INVALID_HOOK_INPUT' } };
  let status; try { status = ctx.layer.status(); } catch { return { output: null, telemetry: { reason: 'INVALID_CONFIG' } }; }
  const mode = status.features.effort.mode;
  const text = truncateUtf8(v.text, 32 * 1024);
  if (BARE_COMMAND_RE.test(text.trim())) {
    return { output: { decision_id: null, mode, apply: false, level: null, arm: 'none', reason: 'BARE_COMMAND' }, telemetry: { reason: 'BARE_COMMAND' } };
  }
  const context = typeof v.context === 'string' && v.context ? truncateUtf8(v.context, 4 * 1024) : undefined;
  const loop = v.loop === 'subagent' ? 'subagent' : 'main';
  const trace = { task_id: randomUUID(), snapshot_id: resolveSnapshotId(process.cwd()) };
  const result = await ctx.layer.effort({ text, ...(context !== undefined ? { context } : {}), loop,
    ...(loop === 'main' ? { sinceLastMainMs: typeof v.since_last_main_ms === 'number' ? v.since_last_main_ms : null } : {}) }, { trace, signal: ctx.signal });
  if (ctx.signal?.aborted || performance.now() >= ctx.deadline) return { output: null, telemetry: null };
  const telemetry = { reason: result.reason, decision_id: result.mode !== 'off' ? result.id : null, arm: result.arm };
  if (result.mode === 'off') return { output: null, telemetry };
  if (result.mode === 'on' && result.id) rememberEffortDecision(ctx.home, { decisionId: result.id, arm: result.arm, level: result.level }, ctx.now());
  const output = { decision_id: result.id, mode: result.mode, apply: result.apply, level: result.apply ? result.level : null, arm: result.arm, reason: result.reason };
  return { output, telemetry };
}
function turnOutcomeInput(input) {
  if (!isObject(input) || typeof input.turn_id !== 'string' || !input.turn_id) return null;
  if (input.decision_id !== undefined && typeof input.decision_id !== 'string') return null;
  if (input.loop !== undefined && !['main', 'subagent'].includes(input.loop)) return null;
  if (!Number.isInteger(input.steps) || input.steps < 0 || input.steps > 10000) return null;
  if (!Number.isFinite(input.duration_ms) || input.duration_ms < 0 || input.duration_ms > 24 * 3600 * 1000) return null;
  if (input.effort_used !== undefined && typeof input.effort_used !== 'string' && typeof input.effort_used !== 'number') return null;
  if (input.model !== undefined && typeof input.model !== 'string') return null;
  if (input.stop_reason !== undefined && typeof input.stop_reason !== 'string') return null;
  if (!Number.isInteger(input.tool_uses) || input.tool_uses < 0 || input.tool_uses > 100000) return null;
  const u = input.usage;
  if (!isObject(u) || ['input', 'output', 'cache_creation', 'cache_read'].some(k => !Number.isFinite(u[k]) || u[k] < 0 || u[k] > 1e9)) return null;
  return input;
}
/**
 * `pointsman hook --host claude --event turn-outcome` (mods/pointsman-effort's turn.step, on the
 * step that ends the turn). Logs a content-free outcome line for phase 2/3 offline comparison;
 * never text. The arm/level are read back from turn-effort's own pending record (see
 * rememberEffortDecision above), never re-derived from anything the mod sends here.
 */
function turnOutcome(host, input, ctx) {
  if (host !== 'claude') return { output: null, telemetry: { reason: 'NOOP' } };
  const v = turnOutcomeInput(input);
  if (!v) return { output: null, telemetry: { reason: 'INVALID_HOOK_INPUT' } };
  let status; try { status = ctx.layer.status(); } catch { return { output: null, telemetry: { reason: 'INVALID_CONFIG' } }; }
  if (status.features.effort.mode === 'off') return { output: null, telemetry: { reason: 'OFF' } };
  if (!status.telemetry) return { output: null, telemetry: { reason: 'TELEMETRY_DISABLED' } };
  const decisionId = typeof v.decision_id === 'string' && v.decision_id ? v.decision_id : null;
  const pending = decisionId ? recallEffortDecision(ctx.home, decisionId, ctx.now()) : null;
  const arm = pending?.arm ?? 'none';
  try {
    appendEvent(ctx.home, { kind: 'hook', at: new Date().toISOString(), host, event: 'turn-outcome', reason: 'TURN_OUTCOME',
      mode: status.features.effort.mode, decision_id: decisionId, arm, loop: v.loop === 'subagent' ? 'subagent' : 'main',
      steps: v.steps, duration_ms: v.duration_ms, tool_uses: v.tool_uses,
      ...(v.effort_used !== undefined ? { effort_used: EFFORT_LEVELS.has(v.effort_used) ? v.effort_used : (typeof v.effort_used === 'number' ? v.effort_used : null) } : {}),
      ...(v.model !== undefined ? { model: safeToken(v.model) } : {}),
      ...(v.stop_reason !== undefined ? { stop_reason: safeToken(v.stop_reason) } : {}),
      usage: { input: v.usage.input, output: v.usage.output, cache_creation: v.usage.cache_creation, cache_read: v.usage.cache_read } });
  } catch { /* observability cannot become an availability dependency */ }
  return { output: null, telemetry: { reason: 'RECORDED', decision_id: decisionId } };
}
const HANDLERS = { 'pre-spawn': preSpawn, 'post-spawn': postSpawn, 'subagent-start': subagentStart, 'subagent-stop': subagentStop,
  'turn-effort': turnEffort, 'turn-outcome': turnOutcome };

function logHookEvent({ home, layer, host, event, telemetry, applied, elapsedMs }) {
  try {
    const status = layer.status();
    const mode = gateModeFor(status, event);
    if (mode === 'off' || !status.telemetry) return;
    appendEvent(home, { kind: 'hook', at: new Date().toISOString(), host, event, reason: telemetry?.reason ?? null,
      mode, applied: Boolean(applied), original_role: telemetry?.original_role ?? null,
      recommended_role: telemetry?.recommended_role ?? null, decision_id: telemetry?.decision_id ?? null, elapsedMs,
      ...(telemetry?.tool_name !== undefined ? { tool_name: telemetry.tool_name } : {}),
      ...(telemetry?.prompt_form !== undefined ? { prompt_form: telemetry.prompt_form } : {}),
      ...(telemetry?.arm !== undefined ? { arm: telemetry.arm } : {}),
      ...(telemetry?.last_message_length !== undefined ? { last_message_length: telemetry.last_message_length } : {}) });
  } catch { /* observability cannot become an availability dependency */ }
}
/**
 * A distinct content-free log line (kind:'hook', reason:'OUTCOME') for the cost/time A/B report
 * (`pointsman metrics ab`): decision_id, agent_id, the assigned arm, the role/model actually used,
 * host-reported duration/tool-use counts, and per-model token usage summed from the agent transcript.
 * Never the task text, the transcript content, or anything from `state`. Gated the same way as
 * logHookEvent (router mode off, or telemetry disabled, logs nothing).
 */
function logOutcomeEvent(ctx, { host, event, decisionId, agentId, arm, finalRole, finalModel, durationMs, toolUses, usage }) {
  try {
    const status = ctx.layer.status();
    const mode = status.features.router.mode;
    if (mode === 'off' || !status.telemetry) return;
    appendEvent(ctx.home, { kind: 'hook', at: new Date().toISOString(), host, event, reason: 'OUTCOME', mode,
      decision_id: decisionId, agent_id: agentId ?? null, arm: arm ?? 'none',
      final_role: finalRole ?? null, final_model: finalModel ?? null,
      ...(Number.isFinite(durationMs) ? { duration_ms: durationMs } : {}),
      ...(Number.isFinite(toolUses) ? { tool_uses: toolUses } : {}),
      ...(usage ? { usage } : {}) });
  } catch { /* observability cannot become an availability dependency */ }
}
/** Pure-ish dispatcher: takes an already-parsed hook payload, never throws, only ever logs a content-free event. */
export async function processHookEvent({ host, event, input, home = resolveHome(), env = process.env, now = () => Date.now(),
  layer, links = createLinkIndex({ home, now }), store = createTrainingStore({ home }), signal, deadline = Infinity } = {}) {
  const start = performance.now();
  if (signal?.aborted || performance.now() >= deadline) return { output: null, telemetry: null };
  if (!HOSTS.includes(host) || !HOOK_EVENTS.includes(event) || typeof layer?.route !== 'function' || typeof layer?.effort !== 'function') return { output: null, telemetry: null };
  // The whole owned hook passes through immediately when global OFF, POINTSMAN_DISABLE=1, or (for the
  // event's own feature -- router for the original four, effort for turn-effort/turn-outcome) that
  // feature's mode is OFF: no route/effort call, no link/pending write, no outcome record.
  let status;
  try { status = layer.status(); } catch { return { output: null, telemetry: null }; }
  if (gateModeFor(status, event) === 'off') return { output: null, telemetry: null };
  const ctx = { home, env, now, layer, links, store, signal, deadline };
  let outcome;
  try { outcome = await HANDLERS[event](host, input, ctx); }
  catch (error) { outcome = { output: null, telemetry: { reason: errorCode(error) } }; }
  if (signal?.aborted || performance.now() >= deadline) return { output: null, telemetry: null };
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
  stdin = process.stdin, write = text => process.stdout.write(text), timeoutMs = HOOK_TIMEOUT_MS,
  engineFactory = createDecisionEngine } = {}) {
  const start = performance.now(), deadline = start + timeoutMs, controller = new AbortController();
  const expire = () => { controller.abort(); try { stdin?.destroy(); } catch { /* fail-open */ } };
  try {
    if (!HOSTS.includes(host) || !HOOK_EVENTS.includes(event)) return;
    const home = resolveHome({ ...env, ...(homeOverride ? { POINTSMAN_HOME: homeOverride } : {}) });
    let engine, layer;
    // L3 (2026-09-23): a hook never spawns its own worker and never waits for a resident server's
    // worker to finish loading; a socket that exists but is still loading returns LAYA_NOT_READY at once.
    try { engine = engineFactory({ home, env, layaSpawn: false, layaWait: false }); layer = createControlLayer({ home, env, engine }); }
    catch { return; }
    try {
      const status = layer.status();
      if (gateModeFor(status, event) === 'off') return;
      if (performance.now() >= deadline) { expire(); return; }
      let settledReason = null;
      const result = await withTimeout((async () => {
        const input = await readHookInput({ stdin, maxBytes: MAX_HOOK_STDIN_BYTES });
        if (controller.signal.aborted || performance.now() >= deadline) return null;
        if (input === null) { settledReason = 'INVALID_STDIN'; return null; }
        const r = await processHookEvent({ host, event, input, home, env, now, layer, signal: controller.signal, deadline });
        if (controller.signal.aborted || performance.now() >= deadline) return null;
        settledReason = 'PROCESSED';
        return r;
      })(), deadline - performance.now(), expire);
      // Content-free diagnostics: distinguish "host never called us" from "called but input rejected / timed out".
      if (settledReason !== 'PROCESSED') {
        logHookEvent({ home, layer, host, event, telemetry: { reason: settledReason ?? 'HOOK_TIMEOUT' }, applied: false,
          elapsedMs: Math.round((performance.now() - start) * 1000) / 1000 });
      }
      if (!controller.signal.aborted && performance.now() < deadline && result?.output) write(JSON.stringify(result.output) + '\n');
    } finally { engine.close(); }
  } catch { /* fail-open */ }
}
