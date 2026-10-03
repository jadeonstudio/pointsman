import path from 'node:path';
import { promises as fs, constants as fsConstants } from 'node:fs';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash, randomUUID } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { containsSensitiveData } from './contracts.mjs';
import { runRepoEvidence } from './workflow-repo.mjs';
import { runDiagnostics } from './workflow-diagnostics.mjs';

const git = promisify(execFile);
const RECIPES = ['repo-evidence', 'test-diagnose', 'log-triage'];
const DEFAULTS = Object.freeze({ mode: 'off', maxActions: 128, maxMs: 10000, maxDecisionCalls: 2, maxOutputBytes: 24000 });
const CAPS = { maxActions: 1024, maxMs: 120000, maxDecisionCalls: 16, maxOutputBytes: 49152 };
const hash = value => createHash('sha256').update(value).digest('hex');
const object = v => v !== null && typeof v === 'object' && !Array.isArray(v) && [Object.prototype, null].includes(Object.getPrototypeOf(v));
function stop(code) { throw Object.assign(new Error(code), { code }); }
function keys(v, allowed) { if (!object(v) || Object.keys(v).some(k => !allowed.includes(k))) stop('INVALID_WORKFLOW_REQUEST'); }
function policyOf(getPolicy) {
  const raw = getPolicy?.() ?? {};
  keys(raw, [...Object.keys(DEFAULTS), 'revision', 'nativeMode']);
  const p = { ...DEFAULTS, ...raw };
  if (!['off', 'shadow', 'on'].includes(p.mode)) stop('INVALID_WORKFLOW_POLICY');
  for (const [key, cap] of Object.entries(CAPS)) if (!Number.isSafeInteger(p[key]) || p[key] < (key === 'maxDecisionCalls' ? 0 : key === 'maxOutputBytes' ? 1024 : 1) || p[key] > cap) stop('INVALID_WORKFLOW_POLICY');
  return p;
}
function validate(input) {
  keys(input, ['workflow', 'goal', 'inputs', 'acceptance', 'coverage', 'snapshot', 'budget']);
  let serialized;
  try { serialized = JSON.stringify(input); } catch { stop('INVALID_WORKFLOW_REQUEST'); }
  if (Buffer.byteLength(serialized) > 65536) stop('INPUT_TOO_LARGE');
  const r = JSON.parse(serialized);
  if (!RECIPES.includes(r.workflow) || !object(r.inputs) || (r.goal !== undefined && (typeof r.goal !== 'string' || r.goal.length > 4096))) stop('INVALID_WORKFLOW_REQUEST');
  r.coverage ??= 'selective'; r.acceptance ??= [];
  if (!['selective', 'exhaustive'].includes(r.coverage) || !Array.isArray(r.acceptance) || r.acceptance.length > 32 || r.acceptance.some(v => typeof v !== 'string' || !v || v.length > 128)) stop('INVALID_WORKFLOW_REQUEST');
  if (r.budget !== undefined) { keys(r.budget, Object.keys(CAPS)); for (const [k, v] of Object.entries(r.budget)) if (!Number.isSafeInteger(v) || v < (k === 'maxDecisionCalls' ? 0 : k === 'maxOutputBytes' ? 1024 : 1)) stop('INVALID_WORKFLOW_REQUEST'); }
  if (r.snapshot !== undefined) {
    keys(r.snapshot, ['revision', 'files']);
    if (r.snapshot.revision !== undefined && r.snapshot.revision !== null && !/^[a-f0-9]{40,64}$/.test(r.snapshot.revision)) stop('INVALID_SNAPSHOT');
    if (r.snapshot.files !== undefined && (!object(r.snapshot.files) || Object.keys(r.snapshot.files).length > 1024 || Object.values(r.snapshot.files).some(v => v !== null && (typeof v !== 'string' || !/^[a-f0-9]{64}$/.test(v))))) stop('INVALID_SNAPSHOT');
  }
  if (containsSensitiveData(r)) stop('SENSITIVE_INPUT');
  return r;
}

// A path is data, never authority. Refuse private host state and credential stores before opening it.
const OMIT_DIRS = new Set(['.git', 'node_modules', '.ssh', '.aws', '.codex', '.claude', '.config', '.cache', '.venv', '__pycache__', 'coverage', 'dist', 'build', 'docs', 'private', 'captures', 'checkpoints', 'datasets', 'models', 'artifacts', '.pointsman-local']);
function permitted(relative) {
  if (typeof relative !== 'string' || !relative || relative.includes('\0') || relative.includes('\\') || path.isAbsolute(relative)) return false;
  const parts = relative.split('/');
  if (parts.some(p => !p || p === '.' || p === '..' || OMIT_DIRS.has(p))) return false;
  return !parts.some(p => /^(?:\.env(?:\..*)?|credentials?(?:\.(?:env|json|toml|ya?ml))?|auth\.(?:env|json|toml|ya?ml)|secrets?(?:\.(?:env|json|toml|ya?ml))?|.*\.(?:pem|key|p12|pfx)|id_(?:rsa|ed25519)|.*(?:history|backup))$/i.test(p));
}
async function revision(root, signal) {
  try { return (await git('git', ['rev-parse', '--verify', 'HEAD'], { cwd: root, timeout: 1000, maxBuffer: 1024, signal })).stdout.trim(); }
  catch { if (signal.aborted) stop('DEADLINE'); return null; }
}
async function safeBytes(root, relative, maxBytes, signal) {
  if (!permitted(relative)) return { path: relative, reason: 'PATH_REFUSED' };
  let file;
  try {
    let current = root;
    for (const part of relative.split('/')) {
      current = path.join(current, part);
      const entry = await fs.lstat(current);
      if (entry.isSymbolicLink()) return { path: relative, reason: 'PATH_REFUSED' };
      if (entry.isDirectory()) {
        try { await fs.lstat(path.join(current, '.git')); return { path: relative, reason: 'NESTED_REPOSITORY' }; }
        catch (error) { if (error.code !== 'ENOENT') return { path: relative, reason: 'PATH_REFUSED' }; }
      }
    }
    file = await fs.open(current, fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW);
    const stat = await file.stat();
    if (!stat.isFile()) return { path: relative, reason: 'NON_FILE' };
    if (stat.size > maxBytes) return { path: relative, reason: 'FILE_TOO_LARGE' };
    const real = await fs.realpath(current);
    if (!real.startsWith(root + path.sep)) return { path: relative, reason: 'PATH_REFUSED' };
    const bytes = await file.readFile({ signal });
    if (bytes.length > maxBytes) return { path: relative, reason: 'FILE_TOO_LARGE' };
    return { bytes };
  } catch (error) { if (signal.aborted) stop('DEADLINE'); return { path: relative, reason: error.code === 'ENOENT' ? 'FILE_MISSING' : 'PATH_REFUSED' }; }
  finally { await file?.close(); }
}

/** Fixed, bounded evidence segments. Trusted injected capabilities never come from request JSON. */
export function createWorkflowRunner({ engine, root = process.cwd(), getPolicy, capabilities = {} } = {}) {
  return Object.freeze({ async run(input, { signal } = {}) {
    const start = performance.now(), stats = { actions: 0, decisions: 0, networkCalls: 0, inferenceCalls: 0, bytesRead: 0, outputBytes: 0, cacheHits: 0, elapsedMs: 0 };
    let p, request, limits, base, sourceRoot, timer, combined, initialRevision, inventory;
    const cache = new Map(), repeated = new Set(), files = {};
    const result = { version: 1, recipeRevision: 'workflow-v1', id: randomUUID(), workflow: RECIPES.includes(input?.workflow) ? input.workflow : null, mode: 'off', status: 'needs_parent', reason: 'OFF', acceptance: {}, evidence: [], coverage: {}, needsParent: 'Continue the original host workflow.', authorizesExecution: false, stats };
    const clearEvidence = () => { result.evidence = []; result.acceptance = Object.fromEntries((request?.acceptance ?? []).map(item => [item, 'unresolved'])); delete result.details; };
    function check(full = true) {
      if (signal?.aborted) stop('CANCELLED');
      if (combined?.aborted || performance.now() - start >= limits.maxMs) stop('DEADLINE');
      if (!full) return;
      const current = policyOf(getPolicy), now = engine?.status?.() ?? { mode: 'off' };
      if (JSON.stringify(current) !== JSON.stringify(p)) stop('POLICY_CHANGED');
      if (['mode', 'policyRevision', 'provider', 'model', 'checkpoint'].some(k => now[k] !== base[k]) || now.killSwitch) stop('ENGINE_CHANGED');
    }
    function action(identity) {
      check();
      if (repeated.has(identity)) stop('REPEATED_ACTION');
      if (stats.actions >= limits.maxActions) stop('ACTION_BUDGET');
      repeated.add(identity); stats.actions++;
    }
    async function bounded(fn) {
      check();
      let abort;
      try {
        return await Promise.race([Promise.resolve().then(fn), new Promise((_, reject) => {
          abort = () => reject(Object.assign(new Error('DEADLINE'), { code: signal?.aborted ? 'CANCELLED' : 'DEADLINE' }));
          combined.addEventListener('abort', abort, { once: true });
          if (combined.aborted) abort();
        })]);
      } finally { combined.removeEventListener('abort', abort); }
    }
    async function read(relative, _role) {
      check();
      if (cache.has(relative)) { stats.cacheHits++; return structuredClone(cache.get(relative)); }
      if (request.workflow === 'repo-evidence') {
        await listFiles();
        if (!inventory.output.includes(relative)) {
          const refused = { path: relative, reason: 'PATH_NOT_IN_SOURCE_INVENTORY' };
          cache.set(relative, refused); return refused;
        }
      }
      action(`read:${relative}`);
      const raw = await safeBytes(sourceRoot, relative, Math.min(1048576, limits.maxOutputBytes * 8), combined);
      check();
      let value = raw;
      if (raw.bytes) {
        stats.bytesRead += raw.bytes.length;
        const text = raw.bytes.toString('utf8'), digest = hash(raw.bytes);
        files[relative] = digest;
        value = raw.bytes.includes(0) || text.includes('\ufffd') ? { path: relative, reason: 'NON_TEXT' } :
          containsSensitiveData({ text }) ? { path: relative, reason: 'SENSITIVE_SOURCE' } : { path: relative, text, hash: digest, ref: `${relative}@${digest}` };
      } else if (raw.reason === 'FILE_MISSING') files[relative] = null;
      cache.set(relative, value); return structuredClone(value);
    }
    async function scan(relative = '', output = [], omissions = []) {
      const entries = await fs.readdir(path.join(sourceRoot, relative), { withFileTypes: true });
      check();
      for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
        check(false); const name = relative ? `${relative}/${entry.name}` : entry.name;
        if (!permitted(name) || entry.isSymbolicLink()) { omissions.push({ path: name, reason: 'PATH_REFUSED' }); continue; }
        if (output.length + omissions.length >= 10000) stop('INVENTORY_BUDGET');
        if (entry.isDirectory()) {
          try { await fs.lstat(path.join(sourceRoot, name, '.git')); omissions.push({ path: name, reason: 'NESTED_REPOSITORY' }); continue; }
          catch (error) { if (error.code !== 'ENOENT') throw error; }
          await scan(name, output, omissions);
        }
        else if (entry.isFile()) output.push(name);
      }
      return { output, omissions };
    }
    function sourceScopes() {
      const input = request.inputs;
      if (input.paths !== undefined && !Array.isArray(input.paths)) stop('INVALID_REPO_INPUT');
      const scopes = input.paths?.length ? [...input.paths, ...(input.requiredPaths ?? []), ...(input.uncertainPaths ?? []), ...(input.counterevidencePaths ?? [])] : ['.'];
      if (scopes.some(p => p !== '.' && (typeof p !== 'string' || !permitted(p.replace(/\/$/, ''))))) stop('INVALID_REPO_SCOPE');
      return [...new Set(scopes.map(p => p === '.' ? '.' : p.replace(/\/$/, '')))];
    }
    async function inventoryOf() {
      const scopes = sourceScopes(), output = [], omissions = [];
      let listed;
      try {
        listed = (await git('git', ['ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', ...scopes.map(p => `:(literal)${p}`)], { cwd: sourceRoot, timeout: Math.max(1, Math.floor(Math.min(5000, limits.maxMs - (performance.now() - start)))), maxBuffer: 1048576, signal: combined })).stdout.split('\0').filter(Boolean);
      } catch (error) {
        check();
        // An inventory error inside a Git repo is a budget/error boundary, never authority to walk ignored data.
        let gitMetadata = false;
        try { await fs.lstat(path.join(sourceRoot, '.git')); gitMetadata = true; } catch {}
        if (gitMetadata || initialRevision !== null || error.code === 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER') stop(error.code === 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER' ? 'INVENTORY_BUDGET' : 'INVENTORY_FAILED');
      }
      if (listed) {
        if (listed.length > 10000) stop('INVENTORY_BUDGET');
        const directories = new Map();
        for (const name of [...new Set(listed)].sort()) {
          check(false);
          if (!permitted(name)) { omissions.push({ path: name, reason: 'PATH_REFUSED' }); continue; }
          let blocked = false;
          const parts = name.split('/');
          for (let i = 1; i < parts.length; i++) {
            const dir = parts.slice(0, i).join('/');
            if (!directories.has(dir)) {
              let safe = !(await fs.lstat(path.join(sourceRoot, dir))).isSymbolicLink();
              try { await fs.lstat(path.join(sourceRoot, dir, '.git')); safe = false; }
              catch (error) { if (error.code !== 'ENOENT') safe = false; }
              directories.set(dir, safe);
            }
            if (!directories.get(dir)) { blocked = true; break; }
          }
          let symlink = false;
          try { symlink = (await fs.lstat(path.join(sourceRoot, name))).isSymbolicLink(); }
          catch (error) { if (error.code !== 'ENOENT') throw error; }
          if (blocked || symlink) omissions.push({ path: name, reason: 'PATH_REFUSED' }); else output.push(name);
        }
      } else {
        for (const scope of scopes) {
          check();
          if (scope === '.') { await scan('', output, omissions); continue; }
          try {
            let blocked = false, ancestor = sourceRoot;
            for (const part of scope.split('/')) {
              ancestor = path.join(ancestor, part);
              const stat = await fs.lstat(ancestor);
              if (stat.isSymbolicLink()) { blocked = true; break; }
              if (stat.isDirectory()) {
                try { await fs.lstat(path.join(ancestor, '.git')); blocked = true; break; }
                catch (error) { if (error.code !== 'ENOENT') throw error; }
              }
            }
            if (blocked) { omissions.push({ path: scope, reason: 'PATH_REFUSED' }); continue; }
            const entry = await fs.lstat(path.join(sourceRoot, scope));
            if (entry.isSymbolicLink()) { omissions.push({ path: scope, reason: 'PATH_REFUSED' }); continue; }
            if (entry.isDirectory()) {
              try { await fs.lstat(path.join(sourceRoot, scope, '.git')); omissions.push({ path: scope, reason: 'NESTED_REPOSITORY' }); continue; }
              catch (error) { if (error.code !== 'ENOENT') throw error; }
              await scan(scope, output, omissions);
            } else if (entry.isFile()) output.push(scope);
          } catch (error) { if (error.code === 'ENOENT') omissions.push({ path: scope, reason: 'FILE_MISSING' }); else throw error; }
        }
      }
      return { output: [...new Set(output)].sort(), omissions };
    }
    async function listFiles() {
      check(); if (inventory) { stats.cacheHits++; return [...inventory.output]; }
      action('list:source-inventory'); inventory = await inventoryOf(); return [...inventory.output];
    }
    async function fresh() {
      check();
      if (await revision(sourceRoot, combined) !== initialRevision) stop('STALE_SNAPSHOT');
      for (const [relative, digest] of Object.entries(files)) {
        check(false); const raw = await safeBytes(sourceRoot, relative, 1048576, combined);
        if ((raw.bytes ? hash(raw.bytes) : raw.reason === 'FILE_MISSING' ? null : 'refused') !== digest) stop('STALE_SNAPSHOT');
      }
      // New/deleted paths can add or hide a caller. Recheck inventory without charging duplicate actions.
      if (inventory) {
        if (JSON.stringify((await inventoryOf()).output) !== JSON.stringify(inventory.output)) stop('STALE_SNAPSHOT');
      }
      check();
    }
    try {
      p = policyOf(getPolicy); base = structuredClone(engine?.status?.() ?? { mode: 'off' });
      result.mode = base.mode === 'off' || base.killSwitch || p.mode === 'off' ? 'off' : base.mode === 'shadow' || p.mode === 'shadow' ? 'shadow' : 'on';
      if (result.mode === 'off') return result;
      request = validate(input); clearEvidence();
      if (result.mode === 'shadow') { result.reason = 'SHADOW'; return result; }
      limits = Object.fromEntries(Object.keys(CAPS).map(k => [k, Math.min(p[k], request.budget?.[k] ?? p[k])]));
      timer = new AbortController(); const timeout = setTimeout(() => timer.abort(), limits.maxMs);
      timer.timeout = timeout;
      combined = signal ? AbortSignal.any([signal, timer.signal]) : timer.signal;
      check(); sourceRoot = await fs.realpath(root); initialRevision = await revision(sourceRoot, combined);
      if (request.snapshot && Object.hasOwn(request.snapshot, 'revision') && request.snapshot.revision !== initialRevision) stop('STALE_SNAPSHOT');
      for (const [relative, expected] of Object.entries(request.snapshot?.files ?? {})) {
        const value = await read(relative, 'snapshot');
        if ((value.hash ?? (value.reason === 'FILE_MISSING' ? null : 'refused')) !== expected) stop('STALE_SNAPSHOT');
      }
      const ctx = { root: sourceRoot, signal: combined, limits, stats, check: () => check(false), read, listFiles,
        async decide(decision) {
          if (request.coverage === 'exhaustive') return { apply: false, reason: 'EXHAUSTIVE_KEEP_ALL', answers: {} };
          await fresh(); action(`decide:${hash(JSON.stringify(decision))}`);
          if (stats.decisions >= limits.maxDecisionCalls) stop('DECISION_BUDGET');
          stats.decisions++;
          const d = await bounded(() => engine.decide(decision, { signal: combined, modeLimit: p.mode }));
          stats.networkCalls += d.networkCalls ?? 0; stats.inferenceCalls += d.inferenceCalls ?? d.networkCalls ?? 0;
          await fresh();
          return d.apply === true ? structuredClone(d) : { ...d, apply: false, answers: {} };
        },
      };
      if (object(capabilities.registeredTests)) ctx.runRegisteredTest = async name => {
        if (typeof name !== 'string' || !Object.hasOwn(capabilities.registeredTests, name) || typeof capabilities.registeredTests[name] !== 'function') stop('UNREGISTERED_TEST');
        await fresh(); action(`test:${name}`);
        const value = await bounded(() => capabilities.registeredTests[name]({ root: sourceRoot, signal: combined }));
        check();
        if (typeof value?.text === 'string') {
          if (Buffer.byteLength(value.text) > 1048576) return { reason: 'FILE_TOO_LARGE' };
          if (containsSensitiveData({ text: value.text })) return { reason: 'SENSITIVE_SOURCE' };
          const digest = hash(value.text);
          return { path: `registered-tests/${name}.stdout`, text: value.text, hash: digest, ref: `registered-test:${name}@${digest}` };
        }
        return value;
      };
      const packet = await bounded(() => request.workflow === 'repo-evidence' ? runRepoEvidence(request, ctx) : runDiagnostics(request, ctx));
      await fresh();
      if (!object(packet) || !['done', 'needs_parent', 'budget_exhausted', 'cancelled'].includes(packet.status)) stop('INVALID_RECIPE_RESULT');
      Object.assign(result, packet, { snapshot: { revision: initialRevision, files: { ...files } } });
      if (Array.isArray(result.needsParent)) result.needsParent = result.needsParent.length ? result.needsParent.join(' ') : null;
      if (result.status === 'done' && (Object.values(result.acceptance ?? {}).includes('unresolved') || result.needsParent)) {
        result.status = 'needs_parent'; result.reason = 'UNRESOLVED_ACCEPTANCE'; result.needsParent ||= 'Resolve the unmet segment acceptance.';
      }
      result.coverage = { ...result.coverage, excluded: inventory?.omissions ?? [] };
    } catch (error) {
      const code = signal?.aborted ? 'CANCELLED' : error.code ?? 'WORKFLOW_FAILED';
      result.status = code === 'CANCELLED' ? 'cancelled' : ['DEADLINE', 'ACTION_BUDGET', 'DECISION_BUDGET', 'INVENTORY_BUDGET'].includes(code) ? 'budget_exhausted' : 'needs_parent';
      result.reason = code; result.needsParent = `Resolve ${code} before continuing this segment.`;
      clearEvidence(); result.coverage = { requested: request?.coverage ?? input?.coverage ?? 'selective', complete: false, omissions: [{ reason: code }] };
    } finally {
      clearTimeout(timer?.timeout);
      stats.elapsedMs = Math.round((performance.now() - start) * 1000) / 1000;
      if (limits && Buffer.byteLength(JSON.stringify(result)) > limits.maxOutputBytes) {
        result.status = 'budget_exhausted'; result.reason = 'OUTPUT_BUDGET'; result.needsParent = 'Raise the permitted output budget or narrow the delegated segment.';
        clearEvidence(); delete result.snapshot;
        result.coverage = { complete: false, omissions: [{ reason: 'OUTPUT_BUDGET', mandatoryEvidenceRetainedInSource: true }] };
      }
      // Include the counter itself in the byte count.
      stats.outputBytes = Buffer.byteLength(JSON.stringify(result));
      stats.outputBytes = Buffer.byteLength(JSON.stringify(result));
      if (limits && stats.outputBytes > limits.maxOutputBytes) { result.acceptance = {}; stats.outputBytes = Buffer.byteLength(JSON.stringify(result)); }
    }
    return result;
  } });
}

export const workflowSchema = { type: 'object', additionalProperties: false, required: ['workflow', 'inputs'], properties: {
  workflow: { type: 'string', enum: RECIPES }, goal: { type: 'string', maxLength: 4096 }, inputs: { type: 'object' },
  acceptance: { type: 'array', maxItems: 32, items: { type: 'string', maxLength: 128 } }, coverage: { type: 'string', enum: ['selective', 'exhaustive'] },
  snapshot: { type: 'object', additionalProperties: false, properties: { revision: { type: ['string', 'null'] }, files: { type: 'object', additionalProperties: { type: ['string', 'null'] } } } },
  budget: { type: 'object', additionalProperties: false, properties: Object.fromEntries(Object.keys(CAPS).map(k => [k, { type: 'integer', minimum: k === 'maxDecisionCalls' ? 0 : k === 'maxOutputBytes' ? 1024 : 1, maximum: CAPS[k] }])) },
} };
