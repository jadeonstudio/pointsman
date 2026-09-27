import path from 'node:path';
import { readText } from '../storage.mjs';
import { DEFAULTS, PURPOSES, ID, fail } from '../constants.mjs';
import { validateRequest, wireRequest } from '../contracts.mjs';
// Canonical validation shares the inference input contract.
import { POLICY_VERSION, HASH, MAX_DERIVED_BYTES, encode, digest, safeContent, targetDistribution, validateTarget, validateProvenance, only, id } from './schema.mjs';
import { EVALUATION_POLICY, indexEvents, evaluateDecision, pairedPreferences, summarizeComparisons } from './evaluate.mjs';

export function validateDatasetSource(store) {
  const s = store.scan(), i = indexEvents(s);
  return { ok: Object.values(i.report).every(n => n === 0), ...i.report, decisions: i.decisions.size,
    outcomes: [...i.outcomes.values()].reduce((n, x) => n + x.length, 0), onlineLearning: false };
}
function groupedSplits(samples) {
  const parent = new Map();
  const find = key => { if (!parent.has(key)) parent.set(key, key); if (parent.get(key) !== key) parent.set(key, find(parent.get(key))); return parent.get(key); };
  const union = (a, b) => { a = find(a); b = find(b); if (a !== b) parent.set(a > b ? a : b, a > b ? b : a); };
  for (const s of samples) {
    for (const task of s.task_ids) union(`task:${task}`, `input:${s.request_hash}`);
    union(`input:${s.request_hash}`, `state:${digest(s.state)}`);
  }
  for (const s of samples) {
    s.group_id = digest(find(`task:${s.task_id}`));
    const n = parseInt(s.group_id.slice(0, 8), 16) % 100;
    s.split = n < 80 ? 'train' : n < 90 ? 'calibration' : 'test';
  }
}
export function buildDataset(store, { allowSmall = false } = {}) {
  return store.lock(() => {
    const snapshot = store.scanUnlocked(), index = indexEvents(snapshot);
    const rows = [], candidates = [], excluded = { missingTaskSnapshot: 0, noSupportedLabel: 0, conflicts: 0, duplicates: 0, unknownProvenance: 0 };
    for (const [key, event] of [...index.decisions].sort(([a], [b]) => a.localeCompare(b))) {
      const d = event.data, outcomes = index.outcomes.get(key) || [], evaluation = evaluateDecision(event, outcomes);
      rows.push({ decision: event, evaluation, outcome: outcomes.find(e => e.data.final && e.data.executed)?.data });
      if (!d.trace.task_id || !d.trace.snapshot_id) { excluded.missingTaskSnapshot++; continue; }
      if (d.provenance.model_version === 'unknown' || d.provenance.checkpoint === 'unknown' ||
          (d.provenance.provider === 'jev' && /(?:latest|preview)$/.test(d.provenance.model_version))) { excluded.unknownProvenance++; continue; }
      if (!evaluation.labels.length) { excluded.noSupportedLabel++; continue; }
      for (const a of evaluation.labels) {
        const question = d.request.questions[a.question_id];
        const target = { value: a.value, probabilities: targetDistribution(question, a.value) };
        const sample = { schema_version: 1, sample_id: digest({ request_hash: d.request_hash, question_id: a.question_id, target }),
          task_id: d.trace.task_id, snapshot_id: d.trace.snapshot_id, task_ids: [d.trace.task_id], snapshot_ids: [d.trace.snapshot_id], request_hash: d.request_hash,
          purpose: d.request.purpose, state: d.request.state, question_id: a.question_id, question,
          target, label_source: a.source, label_confidence: a.label_confidence,
          label_confidence_is_calibrated: false, evaluation_policy_version: POLICY_VERSION,
          provenance: [d.provenance], raw_refs: { decisions: [event.event_id], outcomes: [a.outcome_id] } };
        safeContent(sample); candidates.push(sample);
      }
    }
    const labelsByInput = new Map();
    for (const s of candidates) {
      const key = `${s.request_hash}:${s.question_id}`;
      if (!labelsByInput.has(key)) labelsByInput.set(key, new Set()); labelsByInput.get(key).add(encode(s.target));
    }
    const unique = new Map();
    for (const s of candidates) {
      if (labelsByInput.get(`${s.request_hash}:${s.question_id}`).size !== 1) { excluded.conflicts++; continue; }
      if (!unique.has(s.sample_id)) unique.set(s.sample_id, s);
      else {
        excluded.duplicates++; const first = unique.get(s.sample_id);
        first.task_ids = [...new Set([...first.task_ids, ...s.task_ids])].sort();
        first.snapshot_ids = [...new Set([...first.snapshot_ids, ...s.snapshot_ids])].sort();
        first.provenance = [...new Map([...first.provenance, ...s.provenance].map(p => [encode(p), p])).values()].sort((a, b) => encode(a).localeCompare(encode(b)));
        for (const k of ['decisions', 'outcomes']) first.raw_refs[k] = [...new Set([...first.raw_refs[k], ...s.raw_refs[k]])].sort();
        first.label_confidence = Math.min(first.label_confidence, s.label_confidence);
        // Do not silently relabel human supervision as objective supervision when deduplicating.
        if (first.label_source !== s.label_source) first.label_source = 'mixed_independent';
      }
    }
    const samples = [...unique.values()].sort((a, b) => a.sample_id.localeCompare(b.sample_id));
    groupedSplits(samples);
    const minStrongLabelsPerPurpose = store.config().minStrongLabelsPerPurpose;
    const purposeCounts = samples.reduce((out, s) => { out[s.purpose] = (out[s.purpose] || 0) + 1; return out; }, Object.create(null));
    const shortfall = Object.fromEntries(Object.entries(purposeCounts).filter(([, n]) => n < minStrongLabelsPerPurpose));
    const smallSampleOverride = Object.keys(shortfall).length > 0;
    if (smallSampleOverride && !allowSmall) {
      return { built: false, reason: 'DATASET_TOO_SMALL', sample_count: samples.length, purpose_counts: purposeCounts,
        shortfall, min_strong_labels_per_purpose: minStrongLabelsPerPurpose };
    }
    const preferences = pairedPreferences(rows);
    const data = samples.map(encode).join('\n') + (samples.length ? '\n' : '');
    const prefs = preferences.map(encode).join('\n') + (preferences.length ? '\n' : '');
    if (Buffer.byteLength(data) + Buffer.byteLength(prefs) > MAX_DERIVED_BYTES) fail('DATASET_TOO_LARGE');
    const source_digest = digest({ events: snapshot.events.filter(e => e.kind !== 'evaluations').map(e => e.checksum).sort(), invalid: snapshot.invalid });
    const version = digest({ source_digest, policy: EVALUATION_POLICY, data: digest(data), preferences: digest(prefs) });
    const distribution = field => samples.reduce((out, s) => { out[s[field]] = (out[s[field]] || 0) + 1; return out; }, Object.create(null));
    const providers = Object.create(null);
    for (const s of samples) for (const p of new Set(s.provenance.map(p => p.provider))) providers[p] = (providers[p] || 0) + 1;
    const manifest = { schema_version: 1, dataset_version: version, generated_at: new Date().toISOString(), source_digest,
      sample_count: samples.length, preference_count: preferences.length, purpose_distribution: distribution('purpose'),
      provider_distribution: providers, label_source_distribution: distribution('label_source'), split_distribution: distribution('split'),
      evaluation_policy_version: POLICY_VERSION, filter_rules: EVALUATION_POLICY, exclusions: { ...index.report, ...excluded },
      data_sha256: digest(data), preferences_sha256: digest(prefs),
      split_rule: 'connected ALL task IDs OR identical request/state hash; deterministic 80/10/10 group split', online_learning: false,
      min_strong_labels_per_purpose: minStrongLabelsPerPurpose, small_sample_override: smallSampleOverride && allowSmall };
    const manifestPath = path.join(store.root, 'manifests', `${version}.json`);
    const previous = readText(manifestPath, { optional: true, privateFile: true, maxBytes: 49152 });
    store.writeDerived(`datasets/${version}/canonical.jsonl`, data);
    store.writeDerived(`datasets/${version}/preferences.jsonl`, prefs);
    if (previous === null) store.writeDerived(`manifests/${version}.json`, encode(manifest) + '\n');
    else { const old = JSON.parse(previous); if (old.dataset_version !== version || old.data_sha256 !== digest(data) || old.preferences_sha256 !== digest(prefs)) fail('DATASET_MANIFEST_MISMATCH'); }
    return { dataset_version: version, sample_count: samples.length, preference_count: preferences.length,
      manifest: manifestPath, exclusions: manifest.exclusions, reproducible: true, trained: false };
  });
}
export function readDataset(store, version) {
  if (typeof version !== 'string' || !HASH.test(version)) fail('INVALID_DATASET_VERSION');
  const manifest = JSON.parse(readText(path.join(store.root, 'manifests', `${version}.json`), { privateFile: true, maxBytes: 49152 }));
  const contents = readText(path.join(store.root, 'datasets', version, 'canonical.jsonl'), { privateFile: true, maxBytes: MAX_DERIVED_BYTES });
  if (manifest.dataset_version !== version || manifest.data_sha256 !== digest(contents) || manifest.evaluation_policy_version !== POLICY_VERSION) fail('DATASET_MANIFEST_MISMATCH');
  const samples = contents.trim() ? contents.trim().split('\n').map(JSON.parse) : [];
  if (samples.length !== manifest.sample_count) fail('DATASET_MANIFEST_MISMATCH');
  for (const s of samples) {
    safeContent(s);
    only(s, ['schema_version','sample_id','task_id','snapshot_id','task_ids','snapshot_ids','request_hash','purpose','state','question_id','question','target','label_source','label_confidence','label_confidence_is_calibrated','evaluation_policy_version','provenance','raw_refs','group_id','split'], ['sample_id','task_ids','snapshot_ids','request_hash','provenance','raw_refs','question','target']);
    if (!Array.isArray(s.provenance) || !s.provenance.length || !Array.isArray(s.task_ids) || !s.task_ids.length || !Array.isArray(s.snapshot_ids) || !s.snapshot_ids.length || !PURPOSES.includes(s.purpose)) fail('INVALID_CANONICAL_DATASET');
    s.provenance.forEach(validateProvenance); s.task_ids.forEach(id); id(s.task_id);
    if (!s.task_ids.includes(s.task_id) || !s.snapshot_ids.includes(s.snapshot_id) || s.snapshot_ids.some(x => !HASH.test(x)) || !HASH.test(s.request_hash) || !HASH.test(s.group_id)) fail('INVALID_CANONICAL_DATASET');
    validateRequest({purpose:s.purpose,risk:'routine',state:s.state,questions:{[s.question_id]:s.question}}, DEFAULTS);
    if (s.schema_version !== 1 || s.evaluation_policy_version !== POLICY_VERSION || !['train', 'calibration', 'test'].includes(s.split)) fail('INVALID_CANONICAL_DATASET');
    // A distilled teacher label is soft, self-reported supervision from an LLM call, not an independently
    // verified assertion (see laya-distill.mjs): it may only ground TRAIN samples.
    if (s.label_source === 'teacher' && s.split !== 'train') fail('TEACHER_LABEL_IN_EVAL_SPLIT');
    if (!Number.isFinite(s.label_confidence) || s.label_confidence < EVALUATION_POLICY.minLabelConfidence || s.label_confidence > 1 ||
        // 'ai_reference' (owner decision 2026-09-23): an AI-reference-model
        // label (e.g. Claude), recorded with the labeling model in provenance, never as 'human'; it
        // may ground ANY split (unlike 'teacher', which is restricted to train just below).
        !['objective', 'human', 'mixed_independent', 'teacher', 'ai_reference'].includes(s.label_source)) fail('INVALID_CANONICAL_DATASET');
    if (s.raw_refs.distill !== undefined) {
      only(s.raw_refs, ['distill'], ['distill']);
      only(s.raw_refs.distill, ['run', 'task_id'], ['run', 'task_id']);
      if (typeof s.raw_refs.distill.run !== 'string' || !ID.test(s.raw_refs.distill.run) || !HASH.test(s.raw_refs.distill.task_id)) fail('INVALID_CANONICAL_DATASET');
    } else {
      for (const key of ['decisions','outcomes']) { if (!Array.isArray(s.raw_refs[key]) || !s.raw_refs[key].length) fail('INVALID_CANONICAL_DATASET'); s.raw_refs[key].forEach(id); }
    }
    validateTarget(s.question, s.target.value);
    if (s.label_source === 'teacher') {
      // Soft target: sum to 1 within tolerance and cover exactly the question's option keys; not forced one-hot.
      const keys = s.question.type === 'choice' ? Object.keys(s.question.criteria) : s.question.type === 'noul' ? ['false', 'true'] : s.question.criteria.map((_, i) => String(i));
      only(s.target.probabilities, keys, keys);
      const values = Object.values(s.target.probabilities);
      if (values.some(v => typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 1) || Math.abs(values.reduce((a, b) => a + b, 0) - 1) > 0.02) fail('INVALID_CANONICAL_DATASET');
    } else if (encode(targetDistribution(s.question, s.target.value)) !== encode(s.target.probabilities)) fail('INVALID_CANONICAL_DATASET');
    if (s.sample_id !== digest({ request_hash: s.request_hash, question_id: s.question_id, target: s.target })) fail('INVALID_CANONICAL_DATASET');
  }
  return { manifest, samples, contents };
}
export const LAYA_EXPORT_VERSION = 'laya-typed-decisions-json-v2';
export const LAYA_UPSTREAM = 'NandhaKishorM/laya@42626c348753fbb17572a813127df2278a1ec527:notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb';
// Match official Agent._to_internal json.dumps(list): ensure_ascii=True, comma-space.
const pythonString = s => JSON.stringify(s).replace(/[-￿]/g, c => String.fromCharCode(92, 117) + c.charCodeAt(0).toString(16).padStart(4, '0'));
export function layaQuestion(sample) {
  const q = wireRequest({state:sample.state,questions:{[sample.question_id]:sample.question}}, 'export').questions[sample.question_id];
  return {...q, instructions: '[' + q.instructions.map(pythonString).join(', ') + ']'};
}
export function exportDataset(store, version, format = 'laya') {
  if (!['laya', 'canonical'].includes(format)) fail('INVALID_EXPORT_FORMAT');
  return store.lock(() => {
    const { manifest, samples, contents } = readDataset(store, version);
    const folder = `exports/${version}/${format}`;
    const files = [];
    if (format === 'canonical') files.push(store.writeDerived(`${folder}/canonical.jsonl`, contents));
    else {
      if (['train', 'calibration', 'test'].some(split => !samples.some(s => s.split === split))) fail('EMPTY_SPLIT');
      for (const split of ['train', 'calibration', 'test']) {
        const rows = samples.filter(s => s.split === split).map(s => ({
          // These THREE values are JSON strings because the official notebook calls json.loads on each.
          state: JSON.stringify(s.state), questions: JSON.stringify({ [s.question_id]: layaQuestion(s) }),
          gold: JSON.stringify({ [s.question_id]: { probabilities: s.target.probabilities } }),
        }));
        files.push(store.writeDerived(`${folder}/${split}.jsonl`, rows.map(encode).join('\n') + (rows.length ? '\n' : '')));
      }
    }
    files.push(store.writeDerived(`${folder}/manifest.json`, encode({ dataset_version: version,
      sample_count: manifest.sample_count, format, exporter_version: LAYA_EXPORT_VERSION,
      upstream_contract: LAYA_UPSTREAM, input_transform: 'wire-request-v1 plus official json.dumps(instructions)', evaluation_policy_version: POLICY_VERSION,
      source_data_sha256: manifest.data_sha256, training_executed: false,
      loader: 'datasets.load_dataset("json", data_files={"train": "train.jsonl", "validation": "calibration.jsonl", "test": "test.jsonl"})',
      note: 'Do not merge the holdout into training. Run tokenizer admission and inspect label quality before offline training.' }) + '\n'));
    return { dataset_version: version, format, files, samples: samples.length, trained: false };
  });
}
export function datasetStats(store) {
  return { ...summarizeComparisons(store.scan()), capture: store.status().trainingCapture, rawDataInOutput: false };
}
