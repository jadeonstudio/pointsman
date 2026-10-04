import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { evidenceRequest, evidenceSchema, evidenceReadSchema, runCollectEvidence, runReadEvidence } from '../src/evidence.mjs';
const hash = text => createHash('sha256').update(text).digest('hex');
function context(sources, extra = {}) {
  let reads = 0;
  return { listFiles: async () => Object.keys(sources), read: async path => { reads++; const text = sources[path]; return text === undefined ? { path, reason: 'FILE_MISSING' } : { path, text, hash: hash(text) }; },
    check() {}, limits: { maxOutputBytes: 24000, maxDecisionCalls: 1 }, stats: {}, get reads() { return reads; }, ...extra };
}
const req = extra => evidenceRequest({ query: 'Find transaction evidence', terms: ['transaction'], contextLines: 0, ...extra });

test('literal matches merge overlapping context and deduplicate all aliases', async () => {
  const text = 'first\ntransaction one\ntransaction two\nlast\n';
  const p = await runCollectEvidence(req({ contextLines: 1 }), context({ 'a.js': text, 'b.log': text }));
  assert.equal(p.status, 'done'); assert.equal(p.evidence.length, 1);
  assert.equal(p.evidence[0].text, text.slice(0, -1)); assert.equal(p.evidence[0].startLine, 1); assert.equal(p.evidence[0].endLine, 4);
  assert.equal(p.evidence[0].aliases[0].path, 'b.log');
  const literal = await runCollectEvidence(evidenceRequest({ query: 'literal', terms: ['[x]'] }), context({ 'a.log': 'x\n[x]' }));
  assert.equal(literal.evidence.length, 1);
});

test('nonASCII is exact, case insensitive and required aliases protect entire candidate', async () => {
  const text = '트랜잭션 SUCCESS\n';
  const p = await runCollectEvidence(evidenceRequest({ query: '확인', terms: ['success'], requiredPaths: ['b.log'], contextLines: 1 }), context({ 'a.log': text, 'b.log': text }));
  assert.equal(p.evidence.length, 1); assert.equal(p.evidence[0].text, text); assert.equal(p.evidence[0].required, true);
  assert.equal(p.details.candidateBytes, Buffer.byteLength(text) * 2);
});

test('semantic rejects only ordinary valid applied items and supports exact recovery', async () => {
  const sources = { 'required.log': 'required', 'uncertain.log': 'uncertain', 'counter.log': 'counter', 'keep.log': 'transaction keep ' + 'k'.repeat(1600), 'drop.log': 'transaction drop ' + 'd'.repeat(1600), 'review.log': 'transaction review ' + 'r'.repeat(1600) };
  let calls = 0;
  const ctx = context(sources, { filter: async input => {
    calls++; assert.equal(input.items.length, 3); assert.ok(input.items.every(i => !i.required));
    const drop = input.items.find(i => i.text.includes('drop')).id, review = input.items.find(i => i.text.includes('review')).id;
    return { valid: true, apply: true, keepIds: [], rejectIds: [drop, review, 'unknown'], reviewIds: [review], reason: 'PARTIAL_KEEP' };
  } });
  const p = await runCollectEvidence(req({ semantic: true, requiredPaths: ['required.log'], uncertainPaths: ['uncertain.log'], counterevidencePaths: ['counter.log'] }), ctx);
  assert.equal(calls, 1); assert.equal(p.evidence.length, 5); assert.equal(p.details.selection.rejected, 1);
  const dropped = p.details.omitted.find(e => e.reason === 'SEMANTIC_REJECT'); assert.equal(dropped.path, 'drop.log');
  const recovered = await runReadEvidence(evidenceRequest({ refs: [{ path: dropped.path, hash: dropped.hash, startLine: dropped.startLine, endLine: dropped.endLine }] }, 'read-evidence'), ctx);
  assert.equal(recovered.evidence[0].text, sources['drop.log']);
});

test('failed/unapplied provider keeps candidates; small/exhaustive/sensitive/zero calls bypass', async () => {
  const sources = { 'a.log': 'transaction ' + 'a'.repeat(2500), 'b.log': 'transaction ' + 'b'.repeat(2500) };
  for (const filter of [async () => { throw new Error('down'); }, async () => ({ valid: true, apply: false, rejectIds: ['e0'], keepIds: [], reviewIds: [] }), async () => ({ valid: false, apply: true, rejectIds: ['e0'], keepIds: [], reviewIds: [] })]) {
    const p = await runCollectEvidence(req({ semantic: true }), context(sources, { filter })); assert.equal(p.evidence.length, 2); assert.equal(p.details.selection.applied, false);
  }
  for (const extra of [{ coverage: 'exhaustive' }, { risk: 'sensitive' }, {}]) {
    const p = await runCollectEvidence(req({ semantic: true, ...extra }), context(extra.coverage || extra.risk ? sources : { 'a.log': 'transaction' }, { filter: async () => assert.fail('bypass') }));
    assert.equal(p.details.selection.applied, false);
  }
  await runCollectEvidence(req({ semantic: true }), context(sources, { limits: { maxDecisionCalls: 0, maxOutputBytes: 24000 }, filter: async () => assert.fail('zero calls') }));
});

test('read refuses stale hash and out of range refs, never substitutes newer text', async () => {
  const reference = { path: 'a.log', hash: hash('old'), startLine: 1, endLine: 1 };
  await assert.rejects(runReadEvidence(evidenceRequest({ refs: [reference] }, 'read-evidence'), context({ 'a.log': 'new' })), { code: 'STALE_SOURCE_REF' });
  await assert.rejects(runReadEvidence(evidenceRequest({ refs: [{ ...reference, hash: hash('new'), endLine: 2 }] }, 'read-evidence'), context({ 'a.log': 'new' })), { code: 'INVALID_SOURCE_REF' });
});

test('caps retain exact oversized refs and report unscanned paths without truncation', async () => {
  const sources = { 'b.log': 'transaction small', 'required.log': 'transaction ' + 'r'.repeat(12000), 'a.log': 'transaction small' };
  const p = await runCollectEvidence(req({ requiredPaths: ['required.log'], maxFiles: 2, maxSnippets: 1 }), context(sources, { limits: { maxOutputBytes: 4096, maxDecisionCalls: 0 } }));
  assert.equal(p.status, 'needs_parent'); assert.equal(p.coverage.unscannedCount, 1); assert.equal(p.coverage.filesRead, 2);
  assert.ok(p.details.omitted.some(e => e.path === 'required.log' && e.hash === hash(sources['required.log']) && e.startLine === 1 && e.endLine === 1));
  assert.ok(p.evidence.every(e => e.text === sources[e.path])); assert.ok(Buffer.byteLength(JSON.stringify(p)) <= 4096);
});

test('exhaustive keeps admitted whole files with no provider and reports file cap', async () => {
  const p = await runCollectEvidence(req({ coverage: 'exhaustive', semantic: true, maxFiles: 1 }), context({ 'a.log': 'no match\nfull source', 'b.log': 'transaction' }, { filter: async () => assert.fail('exhaustive') }));
  assert.equal(p.evidence[0].text, 'no match\nfull source'); assert.equal(p.coverage.complete, false); assert.equal(p.reason, 'FILE_BUDGET');
});

test('strict flat validation, defaults and schemas', () => {
  assert.equal(evidenceRequest({ query: 'q', terms: ['t'] }).inputs.semantic, false);
  assert.equal(evidenceSchema.additionalProperties, false); assert.equal(evidenceReadSchema.properties.refs.items.additionalProperties, false);
  for (const extra of [{ root: '/tmp' }, { terms: [] }, { terms: ['x'.repeat(129)] }, { query: '가'.repeat(700) }, { paths: ['/tmp'] }, { requiredPaths: ['../x'] }, { semantic: 1 }, { maxFiles: 97 }, { contextLines: -1 }, { budget: { maxMs: 0 } }, { snapshot: { files: { 'x': 'invalid' } } }]) assert.throws(() => req(extra), { code: 'INVALID_EVIDENCE_INPUT' });
  for (const refs of [[], [{ path: '../x', hash: hash(''), startLine: 1, endLine: 1 }], [{ path: 'a.log', hash: hash(''), startLine: 2, endLine: 1 }]]) assert.throws(() => evidenceRequest({ refs }, 'read-evidence'), { code: 'INVALID_EVIDENCE_INPUT' });
});
