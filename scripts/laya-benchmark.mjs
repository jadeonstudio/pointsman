#!/usr/bin/env node
// Reproducible, content-free benchmark comparing several registered Laya checkpoints against the
// SAME evaluation split of a built distill dataset, for a model-card table. Never prints task
// text, only counts/ids/rates.
//
// Usage:
//   node scripts/laya-benchmark.mjs --dataset <version> --split test \
//     --model <label>=<candidateHash> [--model <label2>=<hash2> ...] \
//     [--input-fit task-head|lossless] [--limit N] [--latency-runs N] --out <file.json>
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';
import { parseArgs } from 'node:util';
import { resolveHome } from '../src/storage.mjs';
import { DEFAULTS } from '../src/constants.mjs';
import { HASH } from '../src/training/schema.mjs';
import { createTrainingStore } from '../src/training/store.mjs';
import { readDataset } from '../src/training/dataset.mjs';
import { loadCandidate, loadQualification, inferOne, labelSourceSummary } from '../src/training/laya-lifecycle.mjs';
import { createLayaClient } from '../src/inference.mjs';
import { validateRequest, wireRequest } from '../src/contracts.mjs';
import { ROUTE_QUESTIONS } from '../src/routing.mjs';
import { detectLang } from '../src/training/laya-distill.mjs';

const QUESTION_IDS = Object.keys(ROUTE_QUESTIONS); // ['intent', 'difficulty', 'risk']
const INFER_TIMEOUT_MS = 30000;

// --- pure aggregation/formatting (unit-tested with synthetic records; no model needed) --------

/** records: [{correct, refused}] for ONE question_id. */
export function questionAgreement(records) {
  const n = records.length;
  const refused = records.filter(r => r.refused).length;
  const correct = records.filter(r => r.correct).length;
  return { n, agreement: n ? correct / n : null, refused };
}
/** Difficulty-only: |predicted - target| <= 1, among answered (non-refused) records. */
export function withinOneRate(records) {
  const answered = records.filter(r => !r.refused);
  const within = answered.filter(r => Math.abs(Number(r.predicted) - Number(r.target)) <= 1).length;
  return { n: answered.length, rate: answered.length ? within / answered.length : null };
}
/** Selective agreement/coverage at a fixed metric threshold (the candidate's own qualified threshold). */
export function selectiveAgreement(records, threshold) {
  const covered = records.filter(r => !r.refused && r.metric >= threshold);
  const correct = covered.filter(r => r.correct).length;
  return { n: covered.length, coverage: records.length ? covered.length / records.length : 0, agreement: covered.length ? correct / covered.length : null };
}
export function summarizeLatencies(msValues) {
  const sorted = [...msValues].sort((a, b) => a - b);
  const n = sorted.length;
  const pct = p => n ? sorted[Math.min(n - 1, Math.floor(p * n))] : null;
  return { n, p50: pct(0.5), p95: pct(0.95) };
}
/** records: normalized per-question-per-sample rows, one model's worth: {question_id, lang, correct, refused, metric, predicted, target}. */
export function aggregateModel(records, { qualifiedThreshold = null } = {}) {
  const byQuestionOf = subset => {
    const out = {};
    for (const qid of QUESTION_IDS) {
      const qs = subset.filter(r => r.question_id === qid);
      out[qid] = questionAgreement(qs);
      if (qid === 'difficulty') out[qid].within_one = withinOneRate(qs);
    }
    return out;
  };
  const byLang = {};
  for (const lang of ['ko', 'en']) byLang[lang] = byQuestionOf(records.filter(r => r.lang === lang));
  const riskRecords = records.filter(r => r.question_id === 'risk');
  const highToSafe = riskRecords.filter(r => r.target === 'high' && r.predicted === 'safe').length;
  const selective = qualifiedThreshold == null ? null : selectiveAgreement(records, qualifiedThreshold);
  return { n: records.length, by_question: byQuestionOf(records), by_lang: byLang, high_to_safe: highToSafe, selective };
}
/** Keep every question row for the first `limit` distinct tasks (by request_hash), in dataset order. */
export function limitByTask(samples, limit) {
  if (!Number.isInteger(limit)) return samples;
  const order = [];
  const seen = new Set();
  for (const s of samples) if (!seen.has(s.request_hash)) { seen.add(s.request_hash); order.push(s.request_hash); }
  const allowed = new Set(order.slice(0, Math.max(0, limit)));
  return samples.filter(s => allowed.has(s.request_hash));
}
/** Up to `count` distinct request states (deduplicated), for a fixed, reused warm-latency workload. */
export function fixedStates(samples, count = 3) {
  const seen = new Set(), out = [];
  for (const s of samples) {
    const key = JSON.stringify(s.state);
    if (seen.has(key)) continue;
    seen.add(key); out.push(s.state);
    if (out.length >= count) break;
  }
  return out;
}
function pct(v) { return v == null ? 'n/a' : `${(v * 100).toFixed(1)}%`; }
function ms(v) { return v == null ? 'n/a' : `${Math.round(v)}ms`; }
export function formatMarkdownTable(modelsReport) {
  const header = '| model | intent | difficulty | difficulty ±1 | risk | high→safe | refused | selective agreement@coverage | cold start | warm p50/p95 |';
  const sep = '|---|---|---|---|---|---|---|---|---|---|';
  const rows = Object.entries(modelsReport).map(([label, m]) => {
    const refused = QUESTION_IDS.reduce((s, qid) => s + m.by_question[qid].refused, 0);
    const selective = m.selective ? `${pct(m.selective.agreement)}@${pct(m.selective.coverage)}` : 'n/a';
    return `| ${label} | ${pct(m.by_question.intent.agreement)} | ${pct(m.by_question.difficulty.agreement)} | ${pct(m.by_question.difficulty.within_one.rate)} | ${pct(m.by_question.risk.agreement)} | ${m.high_to_safe} | ${refused} | ${selective} | ${ms(m.latency.cold_start_ms)} | ${ms(m.latency.warm_p50_ms)}/${ms(m.latency.warm_p95_ms)} |`;
  });
  return ['Claude reference-label agreement, not accuracy', '', header, sep, ...rows].join('\n');
}

// --- worker peak-RSS sampling (best-effort; cheap `ps`, never blocks the run) ------------------
class RssSampler {
  constructor(getPid, { intervalMs = 200, psImpl = spawnSync } = {}) { this.getPid = getPid; this.intervalMs = intervalMs; this.psImpl = psImpl; this.peak = null; this.timer = null; }
  sampleOnce() {
    const pid = this.getPid();
    if (!pid) return;
    let r; try { r = this.psImpl('ps', ['-o', 'rss=', '-p', String(pid)], { encoding: 'utf8' }); } catch { return; }
    if (!r || r.error || r.status !== 0) return;
    const kb = parseInt(String(r.stdout).trim(), 10);
    if (Number.isFinite(kb)) this.peak = this.peak == null ? kb : Math.max(this.peak, kb);
  }
  start() { this.sampleOnce(); this.timer = setInterval(() => this.sampleOnce(), this.intervalMs); this.timer.unref?.(); }
  stop() { if (this.timer) clearInterval(this.timer); return this.peak; }
}

// --- CLI orchestration (not unit-tested; exercised by the real smoke run) ----------------------
function parseModelSpec(spec) {
  const at = spec.indexOf('=');
  if (at <= 0) { process.stderr.write(`--model must be label=candidateHash, got "${spec}"\n`); process.exitCode = 2; return null; }
  const label = spec.slice(0, at), hash = spec.slice(at + 1);
  if (!HASH.test(hash)) { process.stderr.write(`--model "${label}" candidate hash is not a valid 64-hex checkpoint id\n`); process.exitCode = 2; return null; }
  return { label, hash };
}
async function benchmarkModel(home, { hash, evalSamples, latencyRuns, inputFit }) {
  const laya = { ...loadCandidate(home, hash), inputFit };
  const pids = [];
  const layaClient = createLayaClient({ spawnImpl: (...args) => { const child = spawn(...args); pids.push(child.pid); return child; } });
  const rss = new RssSampler(() => pids[pids.length - 1]);
  try {
    const t0 = performance.now();
    await layaClient.prepare({ laya }, { timeoutMs: laya.startupTimeoutMs });
    const coldStartMs = performance.now() - t0;
    rss.start();
    const records = [];
    for (const sample of evalSamples) {
      const r = await inferOne(layaClient, laya, sample, { timeoutMs: INFER_TIMEOUT_MS });
      records.push({ question_id: r.question_id, lang: detectLang(sample.state.task), correct: r.correct, refused: r.refused, metric: r.metric, predicted: r.predicted, target: sample.target.value });
    }
    const states = fixedStates(evalSamples);
    const warmTimes = [];
    if (states.length) {
      for (let i = 0; i < latencyRuns; i++) {
        const state = states[i % states.length];
        const request = validateRequest({ purpose: 'route', risk: 'routine', state, questions: ROUTE_QUESTIONS }, DEFAULTS);
        const payload = wireRequest(request, laya.model);
        const t1 = performance.now();
        await layaClient.infer(payload, { laya }, { timeoutMs: INFER_TIMEOUT_MS });
        warmTimes.push(performance.now() - t1);
      }
    }
    const peakRssKb = rss.stop();
    const qualification = loadQualification(home, hash);
    const agg = aggregateModel(records, { qualifiedThreshold: qualification?.minConfidence ?? null });
    const latency = summarizeLatencies(warmTimes);
    return { checkpoint: hash, input_fit: laya.inputFit, ...agg,
      latency: { cold_start_ms: coldStartMs, warm_p50_ms: latency.p50, warm_p95_ms: latency.p95, peak_rss_kb: peakRssKb } };
  } finally {
    rss.stop();
    layaClient.close();
  }
}
async function main() {
  const { values } = parseArgs({ options: {
    dataset: { type: 'string' }, split: { type: 'string', default: 'test' },
    model: { type: 'string', multiple: true }, 'input-fit': { type: 'string', default: 'task-head' },
    limit: { type: 'string' }, 'latency-runs': { type: 'string', default: '30' }, out: { type: 'string' },
  } });
  const usage = 'usage: node scripts/laya-benchmark.mjs --dataset HASH --split train|calibration|test --model LABEL=HASH [--model LABEL2=HASH2 ...] [--input-fit task-head|lossless] [--limit N] [--latency-runs N] --out FILE.json';
  if (!values.dataset || !HASH.test(values.dataset)) { process.stderr.write(`${usage}\n--dataset is required and must be a 64-hex dataset version\n`); process.exitCode = 2; return; }
  if (!['train', 'calibration', 'test'].includes(values.split)) { process.stderr.write(`${usage}\n--split must be train, calibration or test\n`); process.exitCode = 2; return; }
  if (!Array.isArray(values.model) || !values.model.length) { process.stderr.write(`${usage}\nat least one --model is required\n`); process.exitCode = 2; return; }
  if (!['task-head', 'lossless'].includes(values['input-fit'])) { process.stderr.write(`${usage}\n--input-fit must be task-head or lossless\n`); process.exitCode = 2; return; }
  if (!values.out) { process.stderr.write(`${usage}\n--out is required\n`); process.exitCode = 2; return; }
  const modelSpecs = values.model.map(parseModelSpec);
  if (modelSpecs.some(m => m === null)) return; // parseModelSpec already set exitCode and printed a usage error
  const limit = values.limit !== undefined ? Number(values.limit) : undefined;
  if (limit !== undefined && (!Number.isInteger(limit) || limit < 1)) { process.stderr.write('--limit must be a positive integer\n'); process.exitCode = 2; return; }
  const latencyRuns = Number(values['latency-runs']);
  if (!Number.isInteger(latencyRuns) || latencyRuns < 0) { process.stderr.write('--latency-runs must be a non-negative integer\n'); process.exitCode = 2; return; }

  const home = resolveHome(process.env);
  const store = createTrainingStore({ home });
  const { samples } = readDataset(store, values.dataset);
  const splitSamples = samples.filter(s => s.split === values.split);
  const evalSamples = limitByTask(splitSamples, limit);

  const modelsReport = {};
  for (const { label, hash } of modelSpecs) {
    modelsReport[label] = await benchmarkModel(home, { hash, evalSamples, latencyRuns, inputFit: values['input-fit'] });
  }
  const output = {
    dataset_version: values.dataset, split: values.split,
    sample_counts: { total_split_rows: splitSamples.length, evaluated_rows: evalSamples.length, evaluated_tasks: new Set(evalSamples.map(s => s.request_hash)).size },
    models: modelsReport,
    ...labelSourceSummary(evalSamples),
    hardware: { cpu_model: os.cpus()[0]?.model ?? null, total_memory_bytes: os.totalmem() },
    generated_at: new Date().toISOString(),
  };
  fs.writeFileSync(path.resolve(values.out), JSON.stringify(output, null, 2) + '\n');
  process.stdout.write(formatMarkdownTable(modelsReport) + '\n');
}
if (import.meta.url === `file://${process.argv[1]}`) await main();
