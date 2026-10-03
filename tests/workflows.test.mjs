import test from 'node:test';
import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { createWorkflowRunner, workflowSchema } from '../src/workflows.mjs';

const sha = text => createHash('sha256').update(text).digest('hex');
const request = (extra = {}) => ({ workflow: 'repo-evidence', inputs: { symbols: ['work'] }, acceptance: ['definition', 'direct_callers', 'tests'], ...extra });
async function fixture(t, extra = {}) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'pointsman-workflow-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  await fs.mkdir(path.join(root, 'tests'));
  await fs.writeFile(path.join(root, 'main.mjs'), 'export function work(value) {\n  return value + 1;\n}\n');
  await fs.writeFile(path.join(root, 'caller.mjs'), "import { work } from './main.mjs';\nexport const answer = work(1);\n");
  await fs.writeFile(path.join(root, 'tests/main.test.mjs'), "import { work } from '../main.mjs';\nassert.equal(work(1), 2);\n");
  for (const [name, text] of Object.entries(extra)) await fs.writeFile(path.join(root, name), text);
  let policy = { mode: 'on', maxActions: 128, maxMs: 5000, maxDecisionCalls: 2, maxOutputBytes: 24000 };
  let state = { mode: 'on', policyRevision: 'engine1', provider: 'laya' }, calls = 0;
  const engine = { status: () => state, decide: async () => { calls++; return { apply: true, id: 'decision1', answers: { branch: { value: 'candidate_0' } }, inferenceCalls: 1, networkCalls: 0 }; } };
  const runner = options => createWorkflowRunner({ root, engine, getPolicy: () => policy, ...options });
  return { root, engine, runner, calls: () => calls, policy: () => policy, setPolicy: value => { policy = { ...policy, ...value }; }, setState: value => { state = { ...state, ...value }; } };
}

test('OFF and SHADOW preserve host path without any repository or model action', async () => {
  for (const [globalMode, featureMode, reason] of [['off', 'on', 'OFF'], ['on', 'off', 'OFF'], ['shadow', 'on', 'SHADOW'], ['on', 'shadow', 'SHADOW']]) {
    const runner = createWorkflowRunner({ root: '/nonexistent-root', engine: { status: () => ({ mode: globalMode }), decide: () => assert.fail('no inference') }, getPolicy: () => ({ mode: featureMode }) });
    const result = await runner.run(request());
    assert.equal(result.reason, reason); assert.equal(result.status, 'needs_parent'); assert.equal(result.stats.actions, 0); assert.equal(result.authorizesExecution, false);
  }
});

test('collects source-linked definitions, callers, tests and actual hashes in one segment', async t => {
  const f = await fixture(t), result = await f.runner().run(request());
  assert.equal(result.status, 'done'); assert.equal(result.needsParent, null);
  assert.deepEqual(result.acceptance, { definition: 'met', direct_callers: 'met', tests: 'met' });
  for (const role of ['definition', 'direct_caller', 'test']) assert(result.evidence.some(e => e.role === role));
  const source = result.evidence.find(e => e.role === 'definition');
  assert.equal(source.hash, sha(await fs.readFile(path.join(f.root, source.path))));
  assert(source.ref.includes(source.hash)); assert.equal(result.details.dynamicEdges, 'UNKNOWN');
  assert.equal(result.snapshot.files['main.mjs'], source.hash); assert.equal(f.calls(), 0);
  assert.equal(result.stats.outputBytes, Buffer.byteLength(JSON.stringify(result)));
  assert.deepEqual(workflowSchema.properties.workflow.enum, ['repo-evidence', 'test-diagnose', 'log-triage']);
});

test('missing definition/test and unknown acceptance remain unresolved', async t => {
  const f = await fixture(t), result = await f.runner().run(request({ inputs: { symbols: ['unknown'] }, acceptance: ['definition', 'tests', 'runtime_graph'] }));
  assert.equal(result.status, 'needs_parent'); assert.equal(result.acceptance.runtime_graph, 'unresolved');
  assert.equal(result.acceptance.definition, 'unresolved'); assert.equal(result.evidence.length, 0);
});

test('required, uncertain, contrary sources survive selective and exhaustive coverage', async t => {
  const f = await fixture(t, { 'contrary.md': 'work cannot handle null.', 'uncertain.md': 'work behavior is unclear.' });
  for (const coverage of ['selective', 'exhaustive']) {
    const result = await f.runner().run(request({ coverage, inputs: { symbols: ['work'], semantic: true, requiredPaths: ['main.mjs'], uncertainPaths: ['uncertain.md'], counterevidencePaths: ['contrary.md'] } }));
    for (const role of ['required', 'uncertain', 'counterevidence']) assert(result.evidence.some(e => e.role === role), role);
    assert.equal(result.coverage.complete, true);
  }
  assert.equal(f.calls(), 0);
});

test('path traversal, symlink roots/ancestors, credentials and env files are never ingested', async t => {
  const f = await fixture(t), outside = await fs.mkdtemp(path.join(os.tmpdir(), 'pointsman-outside-'));
  t.after(() => fs.rm(outside, { recursive: true, force: true }));
  await fs.writeFile(path.join(outside, 'outside.mjs'), 'function work() { return 9; }');
  await fs.symlink(outside, path.join(f.root, 'linked'));
  await fs.symlink(path.join(outside, 'outside.mjs'), path.join(f.root, 'leak.mjs'));
  await fs.writeFile(path.join(f.root, '.env'), 'private=envcontents');
  await fs.writeFile(path.join(f.root, 'credentials.env'), 'private=credentialcontents');
  const result = await f.runner().run(request({ inputs: { symbols: ['work'], requiredPaths: ['../outside.mjs', 'linked/outside.mjs', 'leak.mjs', '.env', 'credentials.env', path.join(outside, 'outside.mjs')] } }));
  assert.equal(result.status, 'needs_parent');
  assert(!JSON.stringify(result).includes('return 9')); assert(!JSON.stringify(result).includes('envcontents')); assert(!JSON.stringify(result).includes('credentialcontents'));
  assert.equal(result.coverage.omissions.length, 6);
});

test('source content secret detection refuses excerpts and leaves coverage incomplete', async t => {
  const f = await fixture(t, { 'unsafe.mjs': ['const ', 'password', '=abcdefghi;\nwork();'].join('') });
  const result = await f.runner().run(request());
  assert(result.coverage.omissions.some(o => o.reason === 'SENSITIVE_SOURCE'));
  assert(!JSON.stringify(result).includes('abcdefghi')); assert.equal(result.status, 'needs_parent');
});

test('invalid request cannot inject capabilities, shell or roots; request is detached', async t => {
  const f = await fixture(t);
  for (const extra of [{ root: '/tmp' }, { command: 'pwd' }, { budget: { maxActions: -1 } }, { inputs: { symbols: ['work'], code: 'execute' } }]) {
    const result = await f.runner().run(request(extra)); assert.equal(result.status, 'needs_parent'); assert.match(result.reason, /^INVALID_/);
  }
  const input = request({ inputs: { symbols: ['work'], semantic: true } });
  await fs.writeFile(path.join(f.root, 'other.mjs'), 'export function work() { return 5; }');
  const pending = f.runner().run(input); input.inputs.symbols[0] = 'changed';
  const result = await pending; assert(result.details.symbols.work); assert(!result.details.symbols.changed);
});

test('budget only lowers policy limits; actions and outputs cannot hide missing evidence', async t => {
  const f = await fixture(t);
  const actions = await f.runner().run(request({ budget: { maxActions: 1 } }));
  assert.equal(actions.status, 'budget_exhausted'); assert.equal(actions.reason, 'ACTION_BUDGET'); assert.equal(actions.stats.actions, 1); assert.equal(actions.evidence.length, 0);
  f.setPolicy({ maxActions: 2 });
  const raised = await f.runner().run(request({ budget: { maxActions: 1000 } })); assert.equal(raised.stats.actions, 2); assert.equal(raised.reason, 'ACTION_BUDGET');
  f.setPolicy({ maxActions: 128 });
  await fs.writeFile(path.join(f.root, 'large.md'), 'counterevidence '.repeat(300));
  const output = await f.runner().run(request({ inputs: { symbols: ['work'], counterevidencePaths: ['large.md'] }, budget: { maxOutputBytes: 1024 } }));
  assert.equal(output.reason, 'OUTPUT_BUDGET'); assert.equal(output.status, 'budget_exhausted'); assert.equal(output.evidence.length, 0);
  assert(Buffer.byteLength(JSON.stringify(output)) <= 1024);
});

test('actual supplied dirty-file and Git revision snapshots must match', async t => {
  const f = await fixture(t);
  execFileSync('git', ['init', '-q', f.root]);
  execFileSync('git', ['-C', f.root, 'add', '.']);
  execFileSync('git', ['-C', f.root, '-c', 'user.name=Workflow Test', '-c', 'user.email=workflow@example.invalid', 'commit', '-qm', 'fixture']);
  const revision = execFileSync('git', ['-C', f.root, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  const snapshot = { revision, files: { 'main.mjs': sha(await fs.readFile(path.join(f.root, 'main.mjs'))) } };
  const good = await f.runner().run(request({ snapshot })); assert.equal(good.status, 'done', JSON.stringify(good)); assert.equal(good.snapshot.revision, revision); assert(good.stats.cacheHits > 0);
  await fs.appendFile(path.join(f.root, 'main.mjs'), '// dirty\n');
  const dirty = await f.runner().run(request({ snapshot })); assert.equal(dirty.reason, 'STALE_SNAPSHOT'); assert.equal(dirty.evidence.length, 0);
  const stale = await f.runner().run(request({ snapshot: { revision: '0'.repeat(40) } })); assert.equal(stale.reason, 'STALE_SNAPSHOT');
});

test('ambiguous definitions retain all originals without an inert semantic call', async t => {
  const f = await fixture(t, { 'other.mjs': 'export function work() { return 9; }' });
  f.engine.decide = () => assert.fail('no decision without an actionable branch');
  for (const coverage of ['selective', 'exhaustive']) {
    const r = await f.runner().run(request({ coverage, budget: { maxDecisionCalls: 0 }, inputs: { symbols: ['work'], semantic: true } }));
    assert.equal(r.details.selectedBranch, null); assert.equal(r.status, 'needs_parent'); assert.equal(r.details.symbols.work.definitions.length, 2);
    assert.equal(r.details.semanticSelection, 'NOT_IMPLEMENTED'); assert.equal(r.stats.decisions, 0);
  }
});

test('late policy, engine and frozen file changes invalidate test results before consumption', async t => {
  const f = await fixture(t);
  const req = { workflow: 'test-diagnose', inputs: { registeredTest: 'safe' }, acceptance: ['passing_evidence'], snapshot: { files: { 'main.mjs': sha(await fs.readFile(path.join(f.root, 'main.mjs'))) } } };
  const run = callback => f.runner({ capabilities: { registeredTests: { safe: async () => { await callback(); return { text: 'ok 1 - passing\n1..1\n' }; } } } }).run(req);
  assert.equal((await run(() => f.setPolicy({ mode: 'off' }))).reason, 'POLICY_CHANGED'); f.setPolicy({ mode: 'on' });
  assert.equal((await run(() => f.setState({ policyRevision: 'engine2' }))).reason, 'ENGINE_CHANGED');
  const changed = await run(() => fs.appendFile(path.join(f.root, 'main.mjs'), '// changed\n'));
  assert.equal(changed.reason, 'STALE_SNAPSHOT'); assert.equal(changed.evidence.length, 0);
});

test('cancelled and expired ignored-signal capabilities return promptly without late evidence', async t => {
  const f = await fixture(t);
  const req = { workflow: 'test-diagnose', inputs: { registeredTest: 'safe' }, acceptance: ['passing_evidence'] };
  const controller = new AbortController();
  const cancelled = await f.runner({ capabilities: { registeredTests: { safe: async () => { controller.abort(); return { text: 'ok 1 - passing\n' }; } } } }).run(req, { signal: controller.signal });
  assert.equal(cancelled.status, 'cancelled'); assert.equal(cancelled.evidence.length, 0);
  const expired = await f.runner({ capabilities: { registeredTests: { safe: () => new Promise(() => {}) } } }).run({ ...req, budget: { maxMs: 100 } });
  assert.equal(expired.status, 'budget_exhausted'); assert.equal(expired.reason, 'DEADLINE'); assert.equal(expired.evidence.length, 0);
});

test('trusted registered test capability runs exactly once with bounded signal and hashed reporter', async t => {
  const f = await fixture(t); let runs = 0;
  const registeredTests = { safe: async ({ root, signal }) => { runs++; assert.equal(root, await fs.realpath(f.root)); assert(signal instanceof AbortSignal); return { text: 'ok 1 - passing\n1..1\n' }; } };
  const req = { workflow: 'test-diagnose', inputs: { registeredTest: 'safe' }, acceptance: ['passing_evidence'], coverage: 'selective' };
  const result = await f.runner({ capabilities: { registeredTests } }).run(req);
  assert.equal(runs, 1); assert.equal(result.status, 'done'); assert.equal(result.needsParent, null);
  const reporter = result.evidence.find(e => e.role === 'diagnostic_input');
  assert.equal(reporter.hash, sha('ok 1 - passing\n1..1\n')); assert.match(reporter.ref, /^registered-test:safe@/);
  const absent = await f.runner({ capabilities: { registeredTests } }).run({ ...req, inputs: { registeredTest: 'unregistered' } });
  assert.equal(absent.reason, 'UNREGISTERED_TEST'); assert.equal(runs, 1);
  f.setPolicy({ mode: 'shadow' }); await f.runner({ capabilities: { registeredTests } }).run(req); assert.equal(runs, 1);
});

test('registered test reporter cannot expose sensitive output and a timeout never becomes completion', async t => {
  const f = await fixture(t);
  const req = { workflow: 'test-diagnose', inputs: { registeredTest: 'safe' }, acceptance: ['passing_evidence'] };
  const sensitive = await f.runner({ capabilities: { registeredTests: { safe: async () => ({ text: ['password', '=abcdefghi'].join('') }) } } }).run(req);
  assert.equal(sensitive.status, 'needs_parent'); assert(!JSON.stringify(sensitive).includes('abcdefghi'));
  const timeout = await f.runner({ capabilities: { registeredTests: { safe: () => new Promise(() => {}) } } }).run({ ...req, budget: { maxMs: 100 } });
  assert.equal(timeout.status, 'budget_exhausted'); assert.equal(timeout.reason, 'DEADLINE');
});

test('Git source inventory excludes ignored, private, artifact and nested repository contents', async t => {
  const f = await fixture(t);
  await fs.writeFile(path.join(f.root, '.gitignore'), 'ignored/\n');
  for (const name of ['ignored', 'docs', 'captures', 'models', 'nested']) {
    await fs.mkdir(path.join(f.root, name));
    await fs.writeFile(path.join(f.root, name, 'hidden.mjs'), `export function work() { return '${name}-private-sentinel'; }`);
  }
  execFileSync('git', ['init', '-q', f.root]); execFileSync('git', ['init', '-q', path.join(f.root, 'nested')]);
  const result = await f.runner().run(request());
  assert.equal(result.status, 'done'); assert.equal(result.details.symbols.work.definitions.length, 1);
  assert(!JSON.stringify(result).includes('private-sentinel'));
  for (const name of ['ignored', 'docs', 'captures', 'models', 'nested']) assert(!Object.keys(result.snapshot.files).some(file => file.startsWith(`${name}/`)));
  const explicit = await f.runner().run(request({ inputs: { symbols: ['work'], requiredPaths: ['ignored/hidden.mjs'] } }));
  assert.equal(explicit.status, 'needs_parent'); assert(!JSON.stringify(explicit).includes('ignored-private-sentinel'));
  assert(explicit.coverage.omissions.some(o => o.path === 'ignored/hidden.mjs' && o.reason === 'PATH_NOT_IN_SOURCE_INVENTORY'));
});

test('non-Git scoped inventory skips unrelated trees and nested ancestors within a four-action budget', async t => {
  const f = await fixture(t);
  for (let i = 0; i < 24; i++) {
    const dir = path.join(f.root, `unrelated-${i}`); await fs.mkdir(dir);
    await fs.writeFile(path.join(dir, 'other.mjs'), 'export function work() { return 99; }');
  }
  await fs.mkdir(path.join(f.root, 'nested/sub'), { recursive: true }); await fs.mkdir(path.join(f.root, 'nested/.git'));
  await fs.writeFile(path.join(f.root, 'nested/sub/hidden.mjs'), 'export function work() { return 77; }');
  const result = await f.runner().run(request({ budget: { maxActions: 4 }, inputs: { symbols: ['work'], paths: ['main.mjs', 'caller.mjs', 'tests'] } }));
  assert.equal(result.status, 'done'); assert.equal(result.stats.actions, 4); assert.equal(result.coverage.filesRead, 3);
  assert(Object.keys(result.snapshot.files).every(file => !file.startsWith('unrelated-')));
  const nested = await f.runner().run(request({ inputs: { symbols: ['work'], paths: ['nested/sub'] } }));
  assert.equal(nested.status, 'needs_parent'); assert.equal(nested.stats.bytesRead, 0); assert(!JSON.stringify(nested).includes('return 77'));
});
