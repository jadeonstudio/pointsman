import path from 'node:path';
import { createHash } from 'node:crypto';

const excerpt = text => String(text ?? '').slice(0, 700);
const signature = text => createHash('sha256').update(String(text).replace(/\b[0-9a-f]{8}-[0-9a-f-]{27,}\b/gi, '<id>').replace(/\b\d{4}-\d\d-\d\dT[^\s]+/g, '<time>').trim()).digest('hex');
const sourceRef = (source, lineStart, lineEnd = lineStart, byteStart = 0, byteEnd = byteStart) => ({ source: source.ref, path: source.path, hash: source.hash, lineStart, lineEnd, byteStart, byteEnd });
function lines(source) {
  let byteStart = 0;
  return source.text.split('\n').map((text, i) => {
    const byteEnd = byteStart + Buffer.byteLength(text);
    const row = { text, ref: sourceRef(source, i + 1, i + 1, byteStart, byteEnd) };
    byteStart = byteEnd + 1;
    return row;
  });
}
function timestamp(value) {
  if (value == null) return { value: null, uncertainty: 'missing_timestamp' };
  if (typeof value !== 'string' || !/(Z|[+-]\d\d:?\d\d)$/.test(value)) return { value: null, uncertainty: 'timestamp_without_explicit_timezone_or_units' };
  const ms = Date.parse(value);
  return Number.isFinite(ms) ? { value, ms } : { value: null, uncertainty: 'invalid_timestamp' };
}
function logEvent(raw, ref, original) {
  const object = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
  const message = String(object.message ?? object.msg ?? object.error?.message ?? object.error ?? (typeof raw === 'string' ? raw : original));
  const level = String(object.level ?? object.severity ?? '').toLowerCase();
  const state = String(object.status ?? object.event ?? '').toLowerCase();
  const error = /^(error|fatal|critical|warn|warning)$/.test(level) || /\b(error|failed|failure|exception|fatal)\b/i.test(`${state} ${message}`);
  const success = !error && (/\b(success|succeeded|recovered|passed|completed|healthy)\b/i.test(`${state} ${message}`) || state === 'ok');
  const correlations = Object.fromEntries(['correlationId', 'correlation_id', 'traceId', 'trace_id', 'requestId', 'request_id', 'jobId'].filter(k => ['string', 'number'].includes(typeof object[k])).map(k => [k, String(object[k])]));
  const clock = timestamp(object.timestamp ?? object.time ?? object.ts ?? original.match(/\b\d{4}-\d\d-\d\dT[^\s]+/)?.[0]);
  return { kind: error ? 'error' : success ? 'success' : 'other', level: level || 'unknown', code: object.code ?? null, message: excerpt(message), signature: signature(`${level}|${object.code ?? ''}|${message}`), correlations, timestamp: clock.value, clock, ref, snippet: excerpt(original), referenceText: original, excerptTruncated: original.length > 700 };
}
function parseLogs(source, check) {
  const events = [], parseFailures = [], trimmed = source.text.trim();
  if (trimmed.startsWith('[') || (/^\{/.test(trimmed) && source.text.includes('\n') && /\n\s*"/.test(source.text))) {
    let records;
    try {
      const parsed = JSON.parse(source.text);
      records = Array.isArray(parsed) ? parsed : parsed.events ?? parsed.logs ?? parsed.records ?? (parsed.message || parsed.msg ? [parsed] : null);
      if (!Array.isArray(records)) throw new Error('unsupported_json_container');
    } catch { parseFailures.push({ reason: 'malformed_or_unsupported_json_container', ref: sourceRef(source, 1), snippet: excerpt(source.text) }); return { events, parseFailures }; }
    records.forEach((raw, i) => {
      check(); const ref = { ...sourceRef(source, 1, source.text.split('\n').length, 0, Buffer.byteLength(source.text)), recordIndex: i, locator: 'json_record_index_in_source' };
      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) parseFailures.push({ reason: 'unsupported_json_record', ref, snippet: excerpt(JSON.stringify(raw)) });
      events.push({ ...logEvent(raw, ref, JSON.stringify(raw)), recordIndex: i });
    });
    return { events, parseFailures };
  }
  for (const row of lines(source)) {
    check(); if (!row.text.trim()) continue;
    let raw = row.text;
    if (/^\s*[\[{]/.test(row.text)) {
      try { raw = JSON.parse(row.text); if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error(); }
      catch { parseFailures.push({ reason: 'malformed_or_unsupported_json_record', ref: row.ref, snippet: excerpt(row.text) }); events.push({ ...logEvent(row.text, row.ref, row.text), kind: 'parse_failure' }); continue; }
    }
    events.push(logEvent(raw, row.ref, row.text));
  }
  return { events, parseFailures };
}
const xmlText = text => text.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&amp;/g, '&');
function parseTests(source, check) {
  const results = [], parseFailures = [], rows = lines(source);
  function add(name, status, details, ref, file = null) { results.push({ name: String(name || 'unnamed test'), status, details: excerpt(details), referenceText: String(details), signature: signature(details || name), ref, file }); }
  const trimmed = source.text.trim();
  if (/^[\[{]/.test(trimmed)) {
    let parsed;
    try { parsed = JSON.parse(source.text); }
    catch { // Node's JSON reporter emits one event per line.
      try { parsed = rows.filter(r => r.text.trim()).map(r => JSON.parse(r.text)); }
      catch { parseFailures.push({ reason: 'malformed_test_json', ref: sourceRef(source, 1), snippet: excerpt(source.text) }); return { results, parseFailures }; }
    }
    const entries = Array.isArray(parsed) ? parsed : parsed.testResults ?? parsed.tests ?? [parsed];
    if (!Array.isArray(entries)) return { results, parseFailures: [{ reason: 'unsupported_test_json_schema', ref: sourceRef(source, 1), snippet: excerpt(source.text) }] };
    let recordIndex = 0;
    for (const entry of entries) {
      const ref = { ...sourceRef(source, 1, rows.length, 0, Buffer.byteLength(source.text)), recordIndex: recordIndex++, locator: 'json_record_index_in_source' };
      check();
      if (!entry || typeof entry !== 'object' || Array.isArray(entry)) { parseFailures.push({ reason: 'unsupported_test_json_record', ref, snippet: excerpt(JSON.stringify(entry)) }); continue; }
      if (entry.type === 'test:pass' || entry.type === 'test:fail') {
        const d = entry.data ?? {}; add(d.name, entry.type === 'test:pass' ? 'passed' : 'failed', d.details?.error?.message ?? d.details?.error?.stack ?? JSON.stringify(d.details ?? {}), ref, d.file); continue;
      }
      if (['test:enqueue', 'test:dequeue', 'test:start', 'test:plan', 'test:diagnostic', 'test:summary', 'test:stdout', 'test:stderr', 'test:complete', 'test:coverage'].includes(entry.type)) continue;
      if (entry.assertionResults !== undefined && !Array.isArray(entry.assertionResults)) { parseFailures.push({ reason: 'unsupported_assertion_results', ref }); continue; }
      for (const value of entry.assertionResults ?? [entry]) {
        const status = value?.status;
        if (!['passed', 'failed', 'skipped', 'pending', 'todo'].includes(status)) { parseFailures.push({ reason: 'unsupported_test_status', ref, snippet: excerpt(JSON.stringify(value)) }); continue; }
        const messages = value.failureMessages ?? [value.message ?? value.error?.stack ?? value.error?.message ?? ''];
        if (!Array.isArray(messages)) { parseFailures.push({ reason: 'unsupported_failure_messages', ref }); continue; }
        add(value.fullName ?? value.title ?? value.name, status, messages.join('\n'), ref, entry.name ?? entry.testFilePath ?? value.file);
      }
    }
  } else if (/^\s*</.test(trimmed)) {
    // ponytail: a bounded JUnit subset, unsupported/malformed XML remains parent work.
    if (!/<testsuites?\b/.test(trimmed) || !/<\/testsuites?>\s*$/.test(trimmed)) parseFailures.push({ reason: 'malformed_or_unsupported_junit', ref: sourceRef(source, 1), snippet: excerpt(source.text) });
    for (const match of source.text.matchAll(/<testcase\b([^>]*?)(?:\/>|>([\s\S]*?)<\/testcase>)/g)) {
      check(); const attrs = Object.fromEntries([...match[1].matchAll(/(\w+)\s*=\s*(["'])(.*?)\2/g)].map(m => [m[1], xmlText(m[3])]));
      const body = match[2] ?? '', failed = body.match(/<(failure|error)\b([^>]*)(?:\/>|>([\s\S]*?)<\/(?:failure|error)>)/);
      const line = source.text.slice(0, match.index).split('\n').length;
      add(attrs.name, failed ? 'failed' : /<skipped\b/.test(body) ? 'skipped' : 'passed', failed ? xmlText(`${failed[2]} ${failed[3] ?? ''}`) : '', sourceRef(source, line, line + match[0].split('\n').length - 1, Buffer.byteLength(source.text.slice(0, match.index)), Buffer.byteLength(source.text.slice(0, match.index + match[0].length))), attrs.file);
    }
  } else {
    for (let i = 0; i < rows.length; i++) {
      check(); const match = rows[i].text.match(/^\s*(not ok|ok)\s+\d+\s*(?:-\s*)?(.*)$/) ?? rows[i].text.match(/^\s*([✖×✔✓])\s+(.+?)(?:\s+\([\d.]+m?s\))?$/);
      if (!match) continue;
      const name = match[2], status = /^(not ok|✖|×)$/.test(match[1]) ? 'failed' : /#\s*(SKIP|TODO)\b/i.test(name) ? 'skipped' : 'passed';
      let end = i + 1;
      while (end < rows.length && !/^\s*(?:not ok|ok)\s+\d+\b|^\s*[✖×✔✓]\s|^# (?:tests|pass|fail)\b/.test(rows[end].text)) end++;
      const details = status === 'failed' ? rows.slice(i + 1, end).map(r => r.text).join('\n') : '';
      add(name, status, details, { ...rows[i].ref, lineEnd: end, byteEnd: rows[Math.max(i, end - 1)].ref.byteEnd });
    }
  }
  if (!results.length && !parseFailures.length) parseFailures.push({ reason: 'unsupported_or_empty_test_reporter', ref: sourceRef(source, 1), snippet: excerpt(source.text) });
  return { results, parseFailures };
}
async function sourceEvidence(items, ctx) {
  const found = new Map(), omissions = [];
  for (const item of items) {
    ctx.check(); const raw = `${item.file ?? ''}\n${item.referenceText ?? item.details ?? item.snippet ?? ''}`;
    const refs = [...raw.matchAll(/(?:file:\/\/)?((?:\/?[\w.@-]+\/)*[\w.@-]+\.(?:[cm]?[jt]sx?|py|go|rs|java|rb))(?::(\d+)(?::\d+)?)?/g)];
    for (const match of refs) {
      const relative = path.isAbsolute(match[1]) ? path.relative(ctx.root, match[1]) : match[1];
      if (relative.startsWith('..') || path.isAbsolute(relative)) { omissions.push({ path: match[1], reason: 'outside_root_reference' }); continue; }
      const line = Number(match[2] ?? 1), key = `${relative}:${line}`;
      if (found.has(key)) continue;
      const source = await ctx.read(relative, 'diagnostic_source');
      if (typeof source.text !== 'string') { omissions.push({ path: relative, reason: source.reason }); found.set(key, null); continue; }
      const content = lines(source), start = Math.max(0, line - 4), end = Math.min(content.length, line + 3);
      if (line > content.length) omissions.push({ path: relative, line, reason: 'referenced_line_out_of_range' });
      found.set(key, { path: source.path, hash: source.hash, ref: { ...sourceRef(source, start + 1, end), byteStart: content[start]?.ref.byteStart ?? 0, byteEnd: content[end - 1]?.ref.byteEnd ?? 0 }, snippet: excerpt(content.slice(start, end).map(r => r.text).join('\n')), requestedLine: line, lineExists: line <= content.length });
    }
  }
  return { spans: [...found.values()].filter(Boolean), omissions };
}

/** Fixed, read-only recipes: filesystem and optional test authority belong to ctx. */
export async function runDiagnostics(request, ctx) {
  const workflow = request.workflow, inputs = request.inputs ?? {}, omissions = [], sources = [];
  const supported = ['test-diagnose', 'log-triage'].includes(workflow);
  const paths = workflow === 'log-triage' ? inputs.paths : inputs.resultPaths;
  const invalid = !supported || (paths !== undefined && (!Array.isArray(paths) || paths.some(p => typeof p !== 'string'))) || (!paths?.length && !inputs.registeredTest);
  const evidence = { sources, parseFailures: [], sourceEvidence: [] };
  const packet = (status, reason, acceptance, needsParent) => {
    const refs = [...(evidence.groups ?? []).flatMap(g => [g.first, ...(g.last !== g.first ? [g.last] : [])]), ...(evidence.failures ?? []).flatMap(g => [g.first, g.last]), ...(evidence.passed ?? []), ...evidence.parseFailures, ...evidence.sourceEvidence];
    const excerpts = new Map();
    for (const record of refs) {
      const ref = record.ref; if (!ref) continue;
      const key = `${ref.path}:${ref.lineStart}:${ref.byteStart}:${record.recordIndex ?? ref.recordIndex ?? ''}`;
      excerpts.set(key, { ref: ref.source ?? ref, path: ref.path, hash: ref.hash, text: record.snippet ?? record.details ?? record.message ?? '', role: record.requestedLine ? 'diagnostic_source' : 'diagnostic_observation', startLine: ref.lineStart, endLine: ref.lineEnd, byteStart: ref.byteStart, byteEnd: ref.byteEnd, ...(ref.recordIndex !== undefined ? { recordIndex: ref.recordIndex, locator: ref.locator } : {}) });
    }
    return { status, reason, acceptance, evidence: [...sources.map(s => ({ ref: s.ref, path: s.path, hash: s.hash, text: `Input ${s.path}`, role: 'diagnostic_input' })), ...excerpts.values()], details: { ...evidence, sources: sources.map(({ text, ...s }) => s) }, coverage, needsParent };
  };
  const coverage = { requested: paths ?? [], read: [], omissions, complete: false, exhaustive: false, scope: 'provided_inputs_only' };
  if (invalid) return packet('needs_parent', 'INVALID_DIAGNOSTIC_INPUT', {}, ['Provide permitted input paths for a supported diagnostic recipe.']);
  ctx.check();
  if (inputs.registeredTest) {
    if (workflow !== 'test-diagnose' || typeof inputs.registeredTest !== 'string' || !ctx.runRegisteredTest) return packet('needs_parent', 'REGISTERED_TEST_UNAVAILABLE', {}, ['Supply an existing result or a trusted registered test capability.']);
    const result = await ctx.runRegisteredTest(inputs.registeredTest);
    if (result && typeof result.text === 'string') { sources.push(result); coverage.nativeUnsupported = 'inline_registered_test_not_file_snapshot'; }
    else if (Array.isArray(result?.resultPaths)) coverage.requested = [...new Set([...(paths ?? []), ...result.resultPaths])];
    else omissions.push({ test: inputs.registeredTest, reason: result?.reason ?? 'registered_test_returned_no_report' });
  }
  for (const file of [...new Set(coverage.requested)]) {
    ctx.check(); const source = await ctx.read(file, 'diagnostic_input');
    if (typeof source.text === 'string') sources.push(source);
    else omissions.push({ path: file, reason: source.reason });
  }
  coverage.read = sources.map(s => s.path);
  let retained;
  if (workflow === 'log-triage') {
    const window = inputs.timeWindow;
    if (window && ((!window.from && !window.to) || (window.from && !Number.isFinite(Date.parse(window.from))) || (window.to && !Number.isFinite(Date.parse(window.to))) || (window.from && window.to && Date.parse(window.from) > Date.parse(window.to)))) return packet('needs_parent', 'INVALID_TIME_WINDOW', {}, ['Provide a valid time window.']);
    const groups = new Map(), timeline = [], clockUncertainty = [], correlations = new Map();
    let excluded = 0, parsedCount = 0;
    for (const source of sources) {
      const parsed = parseLogs(source, () => ctx.check()); evidence.parseFailures.push(...parsed.parseFailures);
      let previous;
      for (const event of parsed.events) {
        ctx.check(); parsedCount++;
        if (event.clock.uncertainty) clockUncertainty.push({ ref: event.ref, reason: event.clock.uncertainty });
        if (previous !== undefined && event.clock.ms < previous) clockUncertainty.push({ ref: event.ref, reason: 'source_clock_moved_backwards' });
        if (event.clock.ms !== undefined) previous = event.clock.ms;
        if (event.clock.ms !== undefined && ((window?.from && event.clock.ms < Date.parse(window.from)) || (window?.to && event.clock.ms > Date.parse(window.to)))) { excluded++; continue; }
        const { clock, ...record } = event;
        let group = groups.get(event.signature);
        if (!group) { group = { signature: event.signature, kind: event.kind, count: 0, first: record, last: record, correlationIds: [] }; groups.set(event.signature, group); }
        group.count++; group.last = record;
        for (const [key, value] of Object.entries(event.correlations)) {
          const id = `${key}:${value}`; if (!group.correlationIds.includes(id)) group.correlationIds.push(id);
          if (!correlations.has(id)) correlations.set(id, { id, errors: [], successes: [] });
          if (event.kind === 'error') correlations.get(id).errors.push(event.ref);
          if (event.kind === 'success') correlations.get(id).successes.push(event.ref);
        }
        // Preserve incident transitions in original file order; wall clocks are unverified.
        if (event.kind !== 'other') timeline.push(record);
      }
    }
    evidence.groups = [...groups.values()]; evidence.timeline = timeline; evidence.parsedCount = parsedCount;
    evidence.clockUncertainty = clockUncertainty; evidence.clockOrdering = 'source_order_only_cross_source_clock_unverified';
    evidence.correlations = [...correlations.values()]; evidence.contrary = timeline.filter(e => e.kind === 'success');
    coverage.excludedByWindow = excluded; coverage.unknownTimestampsRetained = clockUncertainty.filter(x => x.reason !== 'source_clock_moved_backwards').length;
    retained = evidence.groups.flatMap(g => [g.first, ...(g.last !== g.first ? [g.last] : [])]);
  } else {
    const tests = [], groups = new Map();
    for (const source of sources) { const parsed = parseTests(source, () => ctx.check()); tests.push(...parsed.results); evidence.parseFailures.push(...parsed.parseFailures); }
    for (const item of tests.filter(t => t.status === 'failed')) {
      const key = item.signature; if (!groups.has(key)) groups.set(key, { signature: key, count: 0, tests: [], first: item, last: item });
      const group = groups.get(key); group.count++; group.tests.push(item.name); group.last = item;
    }
    evidence.failures = [...groups.values()]; evidence.passed = tests.filter(t => t.status === 'passed'); evidence.skipped = tests.filter(t => !['failed', 'passed'].includes(t.status));
    evidence.possibleFlakes = [...new Set(tests.filter(t => t.status === 'failed' && evidence.passed.some(p => p.name === t.name)).map(t => t.name))];
    evidence.rerunSet = [...new Set(tests.filter(t => t.status === 'failed').map(t => t.name))];
    evidence.cause = evidence.failures.length ? 'unresolved_from_reporter_evidence' : 'no_failure_observed';
    evidence.contrary = evidence.passed; retained = tests.filter(t => t.status === 'failed');
  }
  const fetched = await sourceEvidence(retained, ctx); evidence.sourceEvidence = fetched.spans; omissions.push(...fetched.omissions);
  for (const record of [...retained, ...(evidence.timeline ?? []), ...(evidence.passed ?? []), ...(evidence.skipped ?? [])]) delete record.referenceText;
  evidence.sources = sources.map(({ text, ...s }) => s);
  const missing = omissions.length > 0 || evidence.parseFailures.length > 0;
  coverage.complete = !missing;
  const unresolvedCause = workflow === 'test-diagnose' && evidence.failures.length > 0;
  const available = new Set(['sources', 'parse_failures', 'contrary_evidence', ...(workflow === 'log-triage' ? ['timeline', 'groups', 'counts', 'correlations', 'clock_uncertainty', 'singleton_errors', 'first_errors'] : ['failure_groups', 'passing_evidence', 'affected_rerun_set', 'unresolved_cause'])]);
  const acceptance = Object.fromEntries((request.acceptance?.length ? request.acceptance : [...available]).map(item => [item, available.has(item) && !missing ? 'met' : 'unresolved']));
  const needsParent = [ ...(missing ? ['Resolve omitted sources or unsupported/malformed records before claiming full input coverage.'] : []), ...(unresolvedCause ? ['Determine the cause from the failure and source evidence; fixture, product and environment causes remain unproven. Select a rerun only with a new hypothesis.'] : []), ...(Object.values(acceptance).includes('unresolved') && !missing ? ['Resolve acceptance items unsupported by this recipe.'] : []) ];
  ctx.check();
  return packet(needsParent.length ? 'needs_parent' : 'done', missing ? 'INCOMPLETE_DIAGNOSTIC_EVIDENCE' : unresolvedCause ? 'FAILURE_CAUSE_UNRESOLVED' : needsParent.length ? 'ACCEPTANCE_UNRESOLVED' : 'DIAGNOSTIC_SEGMENT_COMPLETE', acceptance, needsParent);
}
