import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import { createWorkflowRunner } from '../src/workflows.mjs';

const exec = promisify(execFile), hash = text => createHash('sha256').update(text).digest('hex');
const TASKS = Object.freeze([
  { id: 'repository', request: { workflow: 'repo-evidence', inputs: { symbols: ['sum'], paths: ['src', 'tests'] } },
    expected: { status: 'done', definitions: ['src/math.mjs'], callers: ['src/use.mjs', 'tests/math.test.mjs'], tests: ['tests/math.test.mjs'] } },
  { id: 'logs', request: { workflow: 'log-triage', inputs: { paths: ['run.ndjson'] }, acceptance: ['counts', 'singleton_errors', 'first_errors', 'contrary_evidence', 'clock_uncertainty'] },
    expected: { status: 'done', groups: [['mismatch', 2, 1, 2], ['singleton failure', 1, 3, 3], ['recovered', 1, 4, 4]], contraryLines: [4], parseFailures: 0, clockUncertainty: 0 } },
  { id: 'tests', request: { workflow: 'test-diagnose', inputs: { resultPaths: ['test.tap'] }, acceptance: ['failure_groups', 'passing_evidence', 'affected_rerun_set', 'unresolved_cause'] },
    expected: { status: 'needs_parent', failures: ['sum mismatch'], passed: ['sum passed'], rerunSet: ['sum mismatch'], cause: 'unresolved_from_reporter_evidence', sources: ['tests/math.test.mjs'] } },
]);
const unique = values => [...new Set(values)].sort();
function canonical(value) { return JSON.stringify(value, Object.keys(value).sort()); }
function shuffled(items, random) { const out = [...items]; for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; } return out; }
function seeded(seed) { let n = seed >>> 0; return () => { n = (Math.imul(n, 1664525) + 1013904223) >>> 0; return n / 4294967296; }; }

async function fixture(root) {
  const files = {
    'src/math.mjs': 'export function sum(a, b) { return a + b; }\n',
    'src/use.mjs': "import { sum } from './math.mjs';\nexport const total = sum(1, 2);\n",
    'tests/math.test.mjs': "import test from 'node:test';\nimport assert from 'node:assert/strict';\nimport { sum } from '../src/math.mjs';\ntest('sum passed', () => assert.equal(sum(1, 2), 3));\ntest('sum mismatch', () => assert.equal(sum(1, 2), 4));\n",
    'log-fixture.mjs': "import { sum } from './src/math.mjs';\nconst emit=(message,level,n)=>console.log(JSON.stringify({timestamp:`2026-10-03T00:00:0${n}Z`,message,level,traceId:'fixture-run'}));\nfor(let i=0;i<2;i++) if(sum(1,2)!==4) emit('mismatch','error',i);\ntry { throw Error('singleton failure'); } catch(e) { emit(e.message,'error',2); }\nif(sum(1,2)===3) emit('recovered','info',3);\n",
  };
  for (const [name, text] of Object.entries(files)) { await fs.mkdir(path.dirname(path.join(root, name)), { recursive: true }); await fs.writeFile(path.join(root, name), text); }
  const logs = await exec(process.execPath, ['log-fixture.mjs'], { cwd: root, timeout: 5000 });
  await fs.writeFile(path.join(root, 'run.ndjson'), logs.stdout);
  let report;
  // Fixture has no environment inputs. Isolate NODE_TEST_CONTEXT inherited from an outer test host.
  try { report = await exec(process.execPath, ['--test', '--test-reporter=tap', 'tests/math.test.mjs'], { cwd: root, timeout: 5000, env: {} }); }
  catch (error) { if (error.code !== 1 || !error.stdout?.includes('not ok')) throw error; report = error; }
  await fs.writeFile(path.join(root, 'test.tap'), report.stdout);
  const names = [...Object.keys(files), 'run.ndjson', 'test.tap'];
  return Object.fromEntries(await Promise.all(names.map(async name => [name, hash(await fs.readFile(path.join(root, name)))])));
}
async function unchanged(root, manifest) {
  for (const [name, expected] of Object.entries(manifest)) if (hash(await fs.readFile(path.join(root, name))) !== expected) throw new Error(`FROZEN_FIXTURE_CHANGED:${name}`);
}

// A competent purpose-built stdlib batch. No executor calls, simulated pauses, or per-tool model turns.
async function codeBatch({ root, task }) {
  let actions = 0;
  const read = async name => { actions++; return fs.readFile(path.join(root, name), 'utf8'); };
  const evidence = [];
  let facts;
  if (task.id === 'repository') {
    const entries = await Promise.all(['src', 'tests'].map(async dir => { actions++; return (await fs.readdir(path.join(root, dir))).filter(name => name.endsWith('.mjs')).map(name => `${dir}/${name}`); }));
    const definitions = [], callers = [], tests = [];
    await Promise.all(entries.flat().map(async name => {
      const text = await read(name); evidence.push({ path: name, hash: hash(text), text });
      for (const line of text.split('\n')) {
        if (/\bfunction\s+sum\b/.test(line)) definitions.push(name);
        else if (/\bsum\s*\(/.test(line)) callers.push(name);
        if (name.startsWith('tests/') && /\bsum\b/.test(line)) tests.push(name);
      }
    }));
    facts = { status: 'done', definitions: unique(definitions), callers: unique(callers), tests: unique(tests) };
  } else if (task.id === 'logs') {
    const text = await read('run.ndjson'), groups = new Map(), contraryLines = [];
    let clockUncertainty = 0, previous;
    text.trim().split('\n').forEach((line, i) => {
      const row = JSON.parse(line), key = `${row.level}:${row.message}`;
      if (!groups.has(key)) groups.set(key, [row.message, 0, i + 1, i + 1]);
      const group = groups.get(key); group[1]++; group[3] = i + 1;
      if (row.message === 'recovered') contraryLines.push(i + 1);
      const ms = Date.parse(row.timestamp);
      if (!/Z$|[+-]\d\d:?\d\d$/.test(row.timestamp) || !Number.isFinite(ms)) clockUncertainty++;
      else { if (previous !== undefined && ms < previous) clockUncertainty++; previous = ms; }
    });
    evidence.push({ path: 'run.ndjson', hash: hash(text), text });
    facts = { status: 'done', groups: [...groups.values()], contraryLines, parseFailures: 0, clockUncertainty };
  } else {
    const text = await read('test.tap'), failures = [], passed = [];
    for (const line of text.split('\n')) {
      const m = line.match(/^(not ok|ok) \d+ - (.+)$/);
      if (m) (m[1] === 'not ok' ? failures : passed).push(m[2]);
    }
    const referenced = unique([...text.matchAll(/((?:\/?[\w.@-]+\/)*[\w.@-]+\.mjs):\d+(?::\d+)?/g)].map(m => path.isAbsolute(m[1]) ? path.relative(root, m[1]) : m[1]).filter(name => !name.startsWith('..') && !path.isAbsolute(name)));
    evidence.push({ path: 'test.tap', hash: hash(text), text });
    await Promise.all(referenced.map(async name => { const text = await read(name); evidence.push({ path: name, hash: hash(text), text }); }));
    facts = { status: failures.length ? 'needs_parent' : 'done', failures: unique(failures), passed: unique(passed), rerunSet: unique(failures), cause: failures.length ? 'unresolved_from_reporter_evidence' : 'no_failure_observed', sources: referenced };
  }
  return { result: { facts, evidence }, actions, decisionCalls: 0, inferenceCalls: 0, networkCalls: 0 };
}
function factsOf(task, result) {
  if (result.facts) return result.facts;
  const d = result.details;
  if (!d) throw new Error(`NO_TASK_EVIDENCE:${result.reason}`);
  const record = value => {
    if (typeof value !== 'string') return value;
    if (d.recordTableVersion !== 1 || !Object.hasOwn(d.records ?? {}, value)) throw new Error(`MISSING_RECORD_ID:${value}`);
    return d.records[value];
  };
  if (task.id === 'repository') {
    const paths = refs => unique(refs.map(ref => result.evidence.find(e => e.ref === ref)?.path));
    return { status: result.status, definitions: paths(d.symbols.sum.definitions), callers: paths(d.symbols.sum.directCallers), tests: paths(d.symbols.sum.tests) };
  }
  if (task.id === 'logs') return { status: result.status, groups: d.groups.map(g => [record(g.first).message, g.count, record(g.first).ref.lineStart, record(g.last).ref.lineStart]), contraryLines: d.contrary.map(x => record(x).ref.lineStart), parseFailures: d.parseFailures.length, clockUncertainty: d.clockUncertainty.length };
  return { status: result.status, failures: unique(d.failures.flatMap(g => g.tests)), passed: unique(d.passed.map(t => record(t).name)), rerunSet: [...d.rerunSet].sort(), cause: d.cause, sources: unique(d.sourceEvidence.map(e => record(e).path)) };
}
function check(task, result, manifest) {
  const facts = factsOf(task, result);
  if (canonical(facts) !== canonical(task.expected)) throw new Error(`TASK_CHECK_FAILED:${task.id}:${JSON.stringify(facts)}`);
  if (!result.evidence?.length || result.evidence.some(e => manifest[e.path] !== e.hash)) throw new Error('SOURCE_HASH_CHECK_FAILED');
  return facts;
}
function quantile(values, p) { const ordered = [...values].sort((a, b) => a - b); return ordered[Math.ceil(p * ordered.length) - 1]; }

/** Trusted injected modelArm is for separately authorized measured consumers, never enabled by CLI request data. */
export async function runWorkflowBenchmark({ rounds = 5, seed = 1, modelArm, hostTrace, signal } = {}) {
  if (!Number.isInteger(rounds) || rounds < 1 || rounds > 50 || !Number.isInteger(seed)) throw new Error('INVALID_BENCHMARK_OPTIONS');
  if (hostTrace && (hostTrace.measured !== true || typeof hostTrace.evidenceRef !== 'string' || !hostTrace.evidenceRef)) throw new Error('UNVERIFIED_HOST_TRACE');
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'pointsman-workflow-benchmark-')));
  try {
    const sourceUrls = ['scripts/workflow-benchmark.mjs', 'src/workflows.mjs', 'src/workflow-repo.mjs', 'src/workflow-diagnostics.mjs', 'src/contracts.mjs'].map(name => [name, new URL(`../${name}`, import.meta.url)]);
    const runtimeSourceHashes = Object.fromEntries(await Promise.all(sourceUrls.map(async ([name, url]) => [name, hash(await fs.readFile(url))])));
    const manifest = await fixture(root), observations = [], random = seeded(seed);
    const runner = createWorkflowRunner({ root, getPolicy: () => ({ mode: 'on' }), engine: { status: () => ({ mode: 'on', policyRevision: 'benchmark-frozen' }), decide: () => { throw new Error('UNCONNECTED_MODEL_DECISION'); } } });
    const arms = [{ id: 'code_batch', run: codeBatch }, { id: 'deterministic_executor', run: async ({ request, signal }) => {
      const result = await runner.run(request, { signal });
      return { result, actions: result.stats.actions, decisionCalls: result.stats.decisions, inferenceCalls: result.stats.inferenceCalls, networkCalls: result.stats.networkCalls };
    } }];
    const q = modelArm?.qualification;
    const qualified = typeof modelArm?.run === 'function' && q?.status === 'qualified' && ['consumer', 'revision', 'evidenceRef'].every(k => typeof q[k] === 'string' && q[k].length > 0);
    if (qualified) arms.push({ id: 'executor_with_model', run: modelArm.run });
    for (let round = 0; round < rounds; round++) {
      const jobs = shuffled(TASKS.flatMap(task => arms.map(arm => ({ task, arm }))), random);
      for (const { task, arm } of jobs) {
        if (signal?.aborted) throw signal.reason ?? new Error('BENCHMARK_CANCELLED');
        await unchanged(root, manifest);
        const started = performance.now();
        let measured, failure = null, facts;
        try {
          measured = await arm.run({ root, request: structuredClone(task.request), task: { id: task.id }, signal });
          if (['actions', 'decisionCalls', 'inferenceCalls', 'networkCalls'].some(k => !Number.isSafeInteger(measured[k]) || measured[k] < 0)) throw new Error('MISSING_MEASURED_COUNTERS');
          if (arm.id === 'executor_with_model' && (measured.decisionConsumerUsed !== true || measured.decisionCalls < 1 || measured.inferenceCalls < 1)) throw new Error('MODEL_CONSUMER_NOT_EXECUTED');
        } catch (error) { failure = error.message; }
        const elapsedMs = performance.now() - started;
        try { await unchanged(root, manifest); if (!failure) facts = check(task, measured.result, manifest); }
        catch (error) { failure = error.message; }
        observations.push({ round, task: task.id, arm: arm.id, order: observations.length, elapsedMs,
          actions: measured?.actions ?? 'UNKNOWN', outputBytes: measured?.result ? Buffer.byteLength(JSON.stringify(measured.result)) : 'UNKNOWN',
          decisionCalls: measured?.decisionCalls ?? 'UNKNOWN', inferenceCalls: measured?.inferenceCalls ?? 'UNKNOWN', networkCalls: measured?.networkCalls ?? 'UNKNOWN',
          decisionConsumerUsed: measured?.decisionConsumerUsed === true,
          independentCheck: failure ? 'FAIL' : 'PASS', failure, facts });
        if (failure?.startsWith('FROZEN_FIXTURE_CHANGED')) throw new Error(failure);
      }
    }
    const summary = TASKS.flatMap(task => arms.map(arm => {
      const rows = observations.filter(row => row.task === task.id && row.arm === arm.id);
      const bytes = rows.map(row => row.outputBytes).filter(value => typeof value === 'number');
      return { task: task.id, arm: arm.id, runs: rows.length, passed: rows.filter(r => r.independentCheck === 'PASS').length,
        elapsedMs: { p50: quantile(rows.map(r => r.elapsedMs), 0.5), p95: quantile(rows.map(r => r.elapsedMs), 0.95) },
        actions: [...new Set(rows.map(r => r.actions))], outputBytes: bytes.length ? { min: Math.min(...bytes), max: Math.max(...bytes) } : 'UNKNOWN' };
    }));
    for (const [name, url] of sourceUrls) if (hash(await fs.readFile(url)) !== runtimeSourceHashes[name]) throw new Error(`RUNTIME_SOURCE_CHANGED:${name}`);
    return { version: 1, scope: 'offline controlled local segment mechanism comparison', node: process.version, seed, rounds, fixtureHashes: manifest, runtimeSourceHashes,
      fixtureProvenance: 'Frozen local repository plus real Node test/TAP and log-fixture execution; controlled fixtures, not natural workloads.',
      modelArm: qualified ? { status: observations.some(row => row.arm === 'executor_with_model' && row.decisionConsumerUsed && row.decisionCalls > 0 && row.inferenceCalls > 0) ? 'EXECUTED' : 'NONEXECUTABLE', qualification: q } : { status: 'NONEXECUTABLE', reason: 'No qualified meaningful model decision consumer is connected.' },
      metricDefinitions: { elapsedMs: 'Actual arm execution including returned packet, excluding fixture/hash/oracle checks.', actions: 'Observed arm filesystem/executor action counters; implementation units differ.', outputBytes: 'Actual serialized returned evidence packet bytes; packet schemas differ.' },
      parentModelRequests: hostTrace?.parentModelRequests ?? 'UNKNOWN', promptCache: hostTrace?.promptCache ?? 'UNKNOWN', endUserE2E: hostTrace?.endUserE2E ?? 'UNKNOWN', cost: hostTrace?.cost ?? 'UNKNOWN',
      hostTrace: hostTrace ?? null, observations, summary, conclusion: 'Prerequisite mechanism evidence only; no task-quality or A-win claim.' };
  } finally { await fs.rm(root, { recursive: true, force: true }); }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const rounds = process.argv[2] === undefined ? 5 : Number(process.argv[2]);
  const seed = process.argv[3] === undefined ? 1 : Number(process.argv[3]);
  const report = await runWorkflowBenchmark({ rounds, seed });
  console.log(JSON.stringify(report, null, 2));
  if (report.observations.some(row => row.independentCheck !== 'PASS')) process.exitCode = 1;
}
