import { fail } from './constants.mjs';

const bytes = value => Buffer.byteLength(typeof value === 'string' ? value : JSON.stringify(value));
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value) && [Object.prototype, null].includes(Object.getPrototypeOf(value));
function invalid(code = 'INVALID_EVIDENCE_INPUT') { fail(code); }
function keys(value, allowed) { if (!object(value) || Object.keys(value).some(k => !allowed.includes(k))) invalid(); }
const budgetCaps = { maxActions: 1024, maxDecisionCalls: 16, maxOutputBytes: 49152, maxMs: 120000 };
const inherited = {
  budget: { type: 'object', additionalProperties: false, properties: Object.fromEntries(Object.entries(budgetCaps).map(([k, maximum]) => [k, { type: 'integer', minimum: k === 'maxDecisionCalls' ? 0 : k === 'maxOutputBytes' ? 1024 : 1, maximum }])) },
  snapshot: { type: 'object', additionalProperties: false, properties: { revision: { type: ['string', 'null'], pattern: '^[a-f0-9]{40,64}$' }, files: { type: 'object', maxProperties: 1024, additionalProperties: { type: ['string', 'null'], pattern: '^[a-f0-9]{64}$' } } } },
};
const pathsSchema = { type: 'array', maxItems: 32, items: { type: 'string', minLength: 1, maxLength: 512 } };
export const evidenceSchema = { type: 'object', additionalProperties: false, required: ['query', 'terms'], properties: {
  query: { type: 'string', minLength: 1, maxLength: 2048 }, terms: { type: 'array', minItems: 1, maxItems: 16, items: { type: 'string', minLength: 1, maxLength: 128 } },
  ...Object.fromEntries(['paths', 'requiredPaths', 'uncertainPaths', 'counterevidencePaths'].map(k => [k, pathsSchema])),
  semantic: { type: 'boolean', default: false }, risk: { type: 'string', enum: ['routine', 'sensitive'], default: 'routine' },
  coverage: { type: 'string', enum: ['selective', 'exhaustive'], default: 'selective' },
  maxFiles: { type: 'integer', minimum: 1, maximum: 96, default: 48 }, maxSnippets: { type: 'integer', minimum: 1, maximum: 64, default: 32 }, contextLines: { type: 'integer', minimum: 0, maximum: 12, default: 3 }, ...inherited,
} };
export const evidenceReadSchema = { type: 'object', additionalProperties: false, required: ['refs'], properties: {
  refs: { type: 'array', minItems: 1, maxItems: 16, items: { type: 'object', additionalProperties: false, required: ['path', 'hash', 'startLine', 'endLine'], properties: {
    path: { type: 'string', minLength: 1, maxLength: 512 }, hash: { type: 'string', pattern: '^[a-f0-9]{64}$' }, startLine: { type: 'integer', minimum: 1 }, endLine: { type: 'integer', minimum: 1 },
  } } }, ...inherited,
} };
function relative(p, scope = false) {
  return typeof p === 'string' && p.length > 0 && p.length <= 512 && !p.includes('\0') && !p.includes('\\') && !p.startsWith('/') && !/^[A-Za-z]:/.test(p) &&
    (scope && p === '.' || p.replace(/\/$/, '').split('/').every(part => part && part !== '.' && part !== '..'));
}
function common(args) {
  if (args.budget !== undefined) {
    keys(args.budget, Object.keys(budgetCaps));
    for (const [k, v] of Object.entries(args.budget)) if (!Number.isSafeInteger(v) || v < (k === 'maxDecisionCalls' ? 0 : k === 'maxOutputBytes' ? 1024 : 1) || v > budgetCaps[k]) invalid();
  }
  if (args.snapshot !== undefined) {
    keys(args.snapshot, ['revision', 'files']);
    const s = args.snapshot;
    if (s.revision !== undefined && s.revision !== null && (typeof s.revision !== 'string' || !/^[a-f0-9]{40,64}$/.test(s.revision))) invalid();
    if (s.files !== undefined && (!object(s.files) || Object.keys(s.files).length > 1024 || Object.entries(s.files).some(([p, h]) => !relative(p) || h !== null && (typeof h !== 'string' || !/^[a-f0-9]{64}$/.test(h))))) invalid();
  }
}
function collectArgs(args) {
  keys(args, Object.keys(evidenceSchema.properties)); common(args);
  if (typeof args.query !== 'string' || !args.query.trim() || bytes(args.query) > 2048 || !Array.isArray(args.terms) || args.terms.length < 1 || args.terms.length > 16 || args.terms.some(t => typeof t !== 'string' || !t || t.length > 128)) invalid();
  const input = { query: args.query, terms: [...new Set(args.terms)], semantic: false, risk: 'routine', maxFiles: 48, maxSnippets: 32, contextLines: 3 };
  for (const k of ['paths', 'requiredPaths', 'uncertainPaths', 'counterevidencePaths']) {
    if (args[k] !== undefined && (!Array.isArray(args[k]) || args[k].length > 32 || args[k].some(p => !relative(p, k === 'paths')))) invalid();
    input[k] = [...new Set(args[k] ?? [])];
  }
  if (args.semantic !== undefined && typeof args.semantic !== 'boolean' || args.risk !== undefined && !['routine', 'sensitive'].includes(args.risk) || args.coverage !== undefined && !['selective', 'exhaustive'].includes(args.coverage)) invalid();
  for (const [k, max, min] of [['maxFiles', 96, 1], ['maxSnippets', 64, 1], ['contextLines', 12, 0]]) {
    if (args[k] !== undefined && (!Number.isSafeInteger(args[k]) || args[k] < min || args[k] > max)) invalid();
    input[k] = args[k] ?? input[k];
  }
  input.semantic = args.semantic ?? false; input.risk = args.risk ?? 'routine'; return input;
}
function readArgs(args) {
  keys(args, Object.keys(evidenceReadSchema.properties)); common(args);
  if (!Array.isArray(args.refs) || !args.refs.length || args.refs.length > 16) invalid();
  for (const ref of args.refs) {
    keys(ref, ['path', 'hash', 'startLine', 'endLine']);
    if (!relative(ref.path) || typeof ref.hash !== 'string' || !/^[a-f0-9]{64}$/.test(ref.hash) || !Number.isSafeInteger(ref.startLine) || !Number.isSafeInteger(ref.endLine) || ref.startLine < 1 || ref.endLine < ref.startLine) invalid();
  }
  return { refs: args.refs.map(ref => ({ ...ref })) };
}
export function evidenceRequest(args, operation = 'collect-evidence') {
  const inputs = operation === 'collect-evidence' ? collectArgs(args) : operation === 'read-evidence' ? readArgs(args) : invalid();
  return { workflow: operation, inputs, coverage: args.coverage ?? 'selective', acceptance: [], ...(args.budget ? { budget: args.budget } : {}), ...(args.snapshot ? { snapshot: args.snapshot } : {}) };
}
const sourceRef = e => ({ path: e.path, hash: e.hash, startLine: e.startLine, endLine: e.endLine });
function span(file, lines, startLine, endLine, required) {
  const item = { path: file.path, hash: file.hash, startLine, endLine, text: lines.slice(startLine - 1, endLine).join('\n'), ...(required ? { required: true } : {}) };
  return { ref: 'e0', ...item };
}
function packet(coverage, selection) {
  return { status: 'done', reason: 'EVIDENCE_COLLECTED', acceptance: {}, evidence: [], coverage: { requested: coverage, complete: true, filesConsidered: 0, filesRead: 0, omissions: [], unscannedCount: 0, unscannedPaths: [] }, details: { selection, omitted: [], candidateBytes: 0, returnedSourceBytes: 0, omittedMetadataCount: 0 }, needsParent: null };
}
function partial(p, reason) { p.status = 'needs_parent'; p.reason = reason; p.coverage.complete = false; p.needsParent = 'Recover deferred refs or narrow the scope before claiming complete evidence.'; }
// Leave room for the runner's snapshot and status counters; source lines are never cut to fit.
function allowance(ctx, files) { return Math.max(256, (ctx.limits?.maxOutputBytes ?? 24000) - 1400 - [...files].reduce((n, p) => n + bytes(p) + 72, 0)); }
function omit(p, entry, limit) {
  p.details.omitted.push(entry);
  if (bytes(p) > limit) { p.details.omitted.pop(); p.details.omittedMetadataCount++; partial(p, 'METADATA_BUDGET'); }
}
function admit(p, e, maxSnippets, limit) {
  e.ref = `e${p.evidence.length}`; p.evidence.push(e);
  if (p.evidence.length > maxSnippets || bytes(p) > limit) {
    p.evidence.pop(); omit(p, { ...sourceRef(e), reason: 'OUTPUT_BUDGET' }, limit);
    for (const alias of e.aliases ?? []) omit(p, { ...alias, reason: 'OUTPUT_BUDGET' }, limit);
    partial(p, 'OUTPUT_BUDGET'); return false;
  }
  p.details.returnedSourceBytes += bytes(e.text); return true;
}
export async function runCollectEvidence(request, ctx) {
  keys(request.inputs, Object.keys(evidenceSchema.properties).filter(k => !['budget', 'snapshot', 'coverage'].includes(k)));
  const input = collectArgs({ ...request.inputs, coverage: request.coverage ?? 'selective' });
  const coverage = request.coverage ?? 'selective', p = packet(coverage, { requested: input.semantic, applied: false, reason: 'LEXICAL_ONLY', rejected: 0, review: 0 });
  const mandatory = new Set([...input.requiredPaths, ...input.uncertainPaths, ...input.counterevidencePaths]), files = new Set(), all = await ctx.listFiles();
  const scoped = name => !input.paths.length || input.paths.some(s => s === '.' || name === s || name.startsWith(s.endsWith('/') ? s : s + '/'));
  const terms = input.terms.map(t => t.toLowerCase()), relevance = name => terms.reduce((n, term) => n + Number(name.toLowerCase().includes(term)), 0);
  const candidates = [...new Set([...mandatory, ...all.filter(scoped)])].sort((a, b) => Number(mandatory.has(b)) - Number(mandatory.has(a)) || relevance(b) - relevance(a) || a.localeCompare(b));
  p.coverage.filesConsidered = candidates.length;
  p.coverage.limits = ['Only admitted inventory files are considered; lexical matches are evidence, not a complete audit.'];
  const admitted = candidates.slice(0, input.maxFiles), unscanned = candidates.slice(input.maxFiles);
  p.coverage.unscannedCount = unscanned.length; p.coverage.unscannedPaths = unscanned.slice(0, 16);
  if (unscanned.length) partial(p, 'FILE_BUDGET');
  const unique = new Map(), deferred = new Map();
  const defer = entry => {
    const key = `${entry.path}:${entry.hash}:${entry.reason}`, prior = deferred.get(key);
    if (prior) { prior.startLine = Math.min(prior.startLine, entry.startLine); prior.endLine = Math.max(prior.endLine, entry.endLine); }
    else deferred.set(key, entry);
  };
  for (const name of admitted) {
    ctx.check(); const file = await ctx.read(name, mandatory.has(name) ? 'required' : 'search');
    if (typeof file.text !== 'string') { p.coverage.omissions.push({ path: name, reason: file.reason }); partial(p, 'UNREADABLE_SOURCE'); continue; }
    files.add(file.path); p.coverage.filesRead++; const lines = file.text.split('\n'), spans = [];
    if (mandatory.has(name) || coverage === 'exhaustive') spans.push([1, lines.length]);
    else for (let i = 0; i < lines.length; i++) {
      ctx.check(); if (!terms.some(t => lines[i].toLowerCase().includes(t))) continue;
      const start = Math.max(1, i + 1 - input.contextLines), end = Math.min(lines.length, i + 1 + input.contextLines), prior = spans.at(-1);
      if (prior && start <= prior[1] + 1) prior[1] = Math.max(prior[1], end); else spans.push([start, end]);
    }
    if (!spans.length) { defer({ path: file.path, hash: file.hash, startLine: 1, endLine: lines.length, reason: 'NO_LITERAL_MATCH' }); continue; }
    for (const [start, end] of spans) {
      const e = span(file, lines, start, end, mandatory.has(name)); p.details.candidateBytes += bytes(e.text);
      const previous = unique.get(e.text);
      if (previous) { (previous.aliases ??= []).push({ ...sourceRef(e), ...(e.required ? { required: true } : {}) }); if (e.required) previous.required = true; }
      else if (unique.size < 128) unique.set(e.text, e);
      else { defer({ ...sourceRef(e), reason: 'CANDIDATE_BUDGET' }); partial(p, 'CANDIDATE_BUDGET'); }
    }
  }
  const snippets = [...unique.values()].sort((a, b) => Number(Boolean(b.required)) - Number(Boolean(a.required))), eligible = snippets.filter(e => !e.required && e.text.trim() && bytes(e.text) <= 4096);
  let rejected = new Set();
  if (input.semantic) {
    const selection = p.details.selection;
    selection.reason = coverage === 'exhaustive' ? 'EXHAUSTIVE_KEEP_ALL' : input.risk === 'sensitive' ? 'SENSITIVE_SCOPE' : bytes(snippets) <= 4096 ? 'SMALL_PACKET' : !ctx.filter || ctx.limits?.maxDecisionCalls === 0 ? 'FILTER_UNAVAILABLE' : !eligible.length ? 'NO_ELIGIBLE_CANDIDATES' : 'KEEP_ALL_FALLBACK';
    if (selection.reason === 'KEEP_ALL_FALLBACK') {
      try {
        const items = eligible.map((e, i) => ({ id: `e${i}`, text: e.text, required: false }));
        const result = await ctx.filter({ query: input.query, items, coverage, risk: input.risk }); ctx.check();
        selection.reason = result.reason ?? 'KEEP_ALL_FALLBACK';
        if (result.valid === true && result.apply === true && Array.isArray(result.rejectIds) && Array.isArray(result.keepIds) && Array.isArray(result.reviewIds)) {
          const keep = new Set([...result.keepIds, ...result.reviewIds]), ids = new Set(items.map(i => i.id));
          rejected = new Set(result.rejectIds.filter(id => ids.has(id) && !keep.has(id)).map(id => eligible[Number(id.slice(1))]));
          selection.applied = true; selection.rejected = rejected.size; selection.review = result.reviewIds.filter(id => ids.has(id)).length;
        }
      } catch (error) {
        if (['CANCELLED', 'DEADLINE', 'POLICY_CHANGED', 'ENGINE_CHANGED'].includes(error.code)) throw error;
        selection.reason = 'FILTER_FAILED_KEEP_ALL';
      }
    }
  }
  p.details.selection.candidates = snippets.length;
  p.details.selection.protected = snippets.filter(e => e.required).length;
  const limit = allowance(ctx, files);
  for (const e of snippets) {
    if (rejected.has(e)) {
      defer({ ...sourceRef(e), reason: 'SEMANTIC_REJECT' });
      for (const alias of e.aliases ?? []) defer({ ...alias, reason: 'SEMANTIC_REJECT' });
    } else admit(p, e, input.maxSnippets, limit);
  }
  for (const entry of deferred.values()) omit(p, entry, limit);
  // Inventory omissions are bounded too; their exact count remains visible.
  while (bytes(p) > limit && p.coverage.unscannedPaths.length) p.coverage.unscannedPaths.pop();
  while (bytes(p) > limit && p.coverage.omissions.length) { p.coverage.omissions.pop(); p.details.omittedMetadataCount++; partial(p, 'METADATA_BUDGET'); }
  if (p.details.omittedMetadataCount) partial(p, 'METADATA_BUDGET');
  return p;
}
export async function runReadEvidence(request, ctx) {
  keys(request.inputs, ['refs']);
  const { refs } = readArgs(request.inputs), p = packet('selective', { requested: false, applied: false, reason: 'EXACT_SOURCE_RECOVERY' }), files = new Set();
  p.coverage.filesConsidered = new Set(refs.map(ref => ref.path)).size;
  for (const ref of refs) {
    ctx.check(); const file = await ctx.read(ref.path, 'recovery');
    if (typeof file.text !== 'string') { p.coverage.omissions.push({ ...ref, reason: file.reason }); partial(p, 'UNREADABLE_SOURCE'); continue; }
    if (file.hash !== ref.hash) invalid('STALE_SOURCE_REF');
    files.add(file.path); const lines = file.text.split('\n'); if (ref.endLine > lines.length) invalid('INVALID_SOURCE_REF');
    const e = span(file, lines, ref.startLine, ref.endLine, true); p.details.candidateBytes += bytes(e.text);
    admit(p, e, 16, allowance(ctx, files));
  }
  p.coverage.filesRead = files.size; return p;
}
