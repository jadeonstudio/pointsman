import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { resolveHome, ensureDir, noSymlinks, readText, atomicWrite } from '../storage.mjs';
import { errorCode, fail } from '../constants.mjs';
import { SCHEMA_VERSION, KINDS, UUID, MAX_DERIVED_BYTES, encode, digest, only, safeContent, validateDecision, validateOutcome, validateEvent } from './schema.mjs';

export function outsideGit(target) {
  let dir = path.resolve(target);
  noSymlinks(dir);
  for (;;) {
    const marker = path.join(dir, '.git');
    const s = fs.lstatSync(marker, { throwIfNoEntry: false });
    if (s && (!s.isDirectory() || fs.readdirSync(marker).some(k => ['HEAD', 'objects', 'refs', 'config', 'index', 'commondir'].includes(k)))) fail('TRAINING_IN_REPOSITORY_REFUSED');
    if (fs.existsSync(path.join(dir, 'HEAD')) && fs.existsSync(path.join(dir, 'objects')) && fs.existsSync(path.join(dir, 'refs'))) fail('TRAINING_IN_REPOSITORY_REFUSED');
    const parent = path.dirname(dir); if (parent === dir) return; dir = parent;
  }
}
export const DEFAULT_MIN_STRONG_LABELS_PER_PURPOSE = 100;
export function createTrainingStore({ home = resolveHome(), now = () => new Date().toISOString() } = {}) {
  const root = path.join(home, 'training'), settings = path.join(home, 'training.json');
  function config() {
    const raw = readText(settings, { optional: true, privateFile: true, maxBytes: 2048 });
    if (raw === null) return { version: 1, trainingCapture: false, minStrongLabelsPerPurpose: DEFAULT_MIN_STRONG_LABELS_PER_PURPOSE };
    let c; try { c = JSON.parse(raw); } catch { fail('INVALID_TRAINING_CONFIG'); }
    only(c, ['version', 'trainingCapture', 'generation', 'minStrongLabelsPerPurpose'], ['version', 'trainingCapture']);
    if (c.generation !== undefined && !UUID.test(c.generation)) fail('INVALID_TRAINING_CONFIG');
    if (c.version !== 1 || typeof c.trainingCapture !== 'boolean') fail('INVALID_TRAINING_CONFIG');
    if (c.minStrongLabelsPerPurpose !== undefined && (!Number.isInteger(c.minStrongLabelsPerPurpose) || c.minStrongLabelsPerPurpose < 1 || c.minStrongLabelsPerPurpose > 100000)) fail('INVALID_TRAINING_CONFIG');
    return { minStrongLabelsPerPurpose: DEFAULT_MIN_STRONG_LABELS_PER_PURPOSE, ...c };
  }
  function ticket() { try { const c = config(); return c.trainingCapture ? (c.generation ?? 'legacy-enabled') : null; } catch { return null; } }
  function status() {
    try { return { ...config(), root, onlineLearning: false, contentInStatus: false }; }
    catch (e) { return { version: 1, trainingCapture: false, error: errorCode(e), root, onlineLearning: false, contentInStatus: false }; }
  }
  function lock(fn) {
    outsideGit(root); ensureDir(home, true); ensureDir(root, true);
    const dir = path.join(root, '.lock');
    noSymlinks(dir);
    try { fs.mkdirSync(dir, { mode: 0o700 }); }
    catch (e) { if (e.code === 'EEXIST') fail('TRAINING_LOCKED'); throw e; }
    try { return fn(); } finally { fs.rmdirSync(dir); }
  }
  function setCapture(enabled) {
    if (typeof enabled !== 'boolean') fail('INVALID_TRAINING_CONFIG');
    return lock(() => {
      const old = readText(settings, { optional: true, privateFile: true, maxBytes: 2048 });
      let base; try { base = old === null ? {} : JSON.parse(old); } catch { fail('INVALID_TRAINING_CONFIG'); }
      only(base, ['version', 'trainingCapture', 'generation', 'minStrongLabelsPerPurpose'], []);
      atomicWrite(settings, encode({ version: 1, ...base, trainingCapture: enabled, generation: randomUUID() }) + '\n', { expected: old });
      return status();
    });
  }
  function setMinLabels(count) {
    if (!Number.isInteger(count) || count < 1 || count > 100000) fail('INVALID_TRAINING_CONFIG');
    return lock(() => {
      const old = readText(settings, { optional: true, privateFile: true, maxBytes: 2048 });
      let base; try { base = old === null ? { version: 1, trainingCapture: false } : JSON.parse(old); } catch { fail('INVALID_TRAINING_CONFIG'); }
      only(base, ['version', 'trainingCapture', 'generation', 'minStrongLabelsPerPurpose'], ['version', 'trainingCapture']);
      atomicWrite(settings, encode({ ...base, minStrongLabelsPerPurpose: count }) + '\n', { expected: old });
      return status();
    });
  }
  function appendUnlocked(kind, data, eventId = randomUUID()) {
    if (!config().trainingCapture) return { stored: false, reason: 'CAPTURE_OFF' };
    if (!KINDS.includes(kind) || !UUID.test(eventId)) fail('INVALID_TRAINING_SCHEMA');
    const dir = path.join(root, 'raw', kind); ensureDir(dir, true);
    if (fs.readdirSync(dir).length >= 100000) fail('TRAINING_STORAGE_LIMIT');
    const file = path.join(dir, `${eventId}.json`);
    const previous = readText(file, { optional: true, privateFile: true, maxBytes: 49152 });
    if (previous !== null) {
      const existing = validateEvent(JSON.parse(previous));
      if (existing.kind !== kind || encode(existing.data) !== encode(data)) fail('TRAINING_EVENT_CONFLICT');
      return { stored: true, event_id: eventId, duplicate: true };
    }
    const body = { schema_version: SCHEMA_VERSION, event_id: eventId, kind, created_at: now(), data };
    const e = { ...body, checksum: digest(body) }; validateEvent(e);
    const bytes = encode(e) + '\n';
    if (Buffer.byteLength(bytes) > 49152) fail('TRAINING_SENSITIVE_OR_OVERSIZED');
    // Immutable per-event files avoid interleaved JSONL writes and require no database runtime.
    // A crash can leave only this one incomplete event; readers reject it by schema/checksum.
    const fd = fs.openSync(file, fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL | (fs.constants.O_NOFOLLOW || 0), 0o600);
    try { fs.writeFileSync(fd, bytes); fs.fsyncSync(fd); } finally { fs.closeSync(fd); }
    const parent = fs.openSync(dir, 'r'); try { fs.fsyncSync(parent); } finally { fs.closeSync(parent); }
    return { stored: true, event_id: eventId, duplicate: false };
  }
  function append(kind, data, eventId) {
    if (!config().trainingCapture) return { stored: false, reason: 'CAPTURE_OFF' };
    return lock(() => appendUnlocked(kind, data, eventId));
  }
  function decision(data, { secret = '', expectedTicket } = {}) {
    try {
      if (!config().trainingCapture) return { stored: false, reason: 'CAPTURE_OFF' };
      if (expectedTicket !== undefined && (expectedTicket === null || ticket() !== expectedTicket)) return { stored: false, reason: 'CAPTURE_CONSENT_CHANGED' };
      safeContent(data, secret); validateDecision(data);
      return lock(() => {
        if (expectedTicket !== undefined && (expectedTicket === null || ticket() !== expectedTicket)) return { stored: false, reason: 'CAPTURE_CONSENT_CHANGED' };
        return appendUnlocked('decisions', structuredClone(data), data.decision_id);
      });
    } catch (e) { return { stored: false, reason: errorCode(e) }; }
  }
  function outcome(data, { eventId, trust = 'operator' } = {}) {
    if (!config().trainingCapture) return { stored: false, reason: 'CAPTURE_OFF' };
    const value = structuredClone(data);
    if (trust === 'host') {
      value.source = 'host_review';
      if (Array.isArray(value.labels)) value.labels = value.labels.map(a => ({ ...a, source: 'host_review' }));
    }
    validateOutcome(value);
    return append('outcomes', value, eventId);
  }
  function scanUnlocked() {
    outsideGit(root);
    const events = [], invalid = [], duplicates = new Map(); let totalBytes = 0;
    for (const kind of KINDS) {
      const dir = path.join(root, 'raw', kind); noSymlinks(dir);
      if (!fs.existsSync(dir)) continue;
      const names = fs.readdirSync(dir).sort();
      if (names.length > 100000) fail('TRAINING_STORAGE_LIMIT');
      for (const name of names) {
        if (!name.endsWith('.json')) continue;
        try {
          const source = readText(path.join(dir, name), { privateFile: true, maxBytes: 49152 });
          totalBytes += Buffer.byteLength(source);
          if (totalBytes > 64 * 1024 * 1024) fail('TRAINING_SNAPSHOT_TOO_LARGE');
          const e = validateEvent(JSON.parse(source));
          if (e.kind !== kind || name !== `${e.event_id}.json`) fail('TRAINING_EVENT_PATH_MISMATCH');
          if (duplicates.has(e.event_id)) fail('TRAINING_DUPLICATE_EVENT');
          duplicates.set(e.event_id, e); events.push(e);
        } catch (error) { if (errorCode(error) === 'TRAINING_SNAPSHOT_TOO_LARGE') throw error; invalid.push({ kind, code: errorCode(error) }); }
      }
    }
    return { events, invalid };
  }
  function scan() {
    if (!fs.existsSync(root)) return { events: [], invalid: [] };
    return lock(scanUnlocked);
  }
  function writeDerived(relative, contents) {
    if (typeof relative !== 'string' || path.isAbsolute(relative) || relative.split(/[\\/]/).includes('..') || !/^(datasets|manifests|exports)\//.test(relative)) fail('INVALID_DATASET_PATH');
    outsideGit(root);
    if (typeof contents !== 'string' || Buffer.byteLength(contents) > MAX_DERIVED_BYTES) fail('DATASET_TOO_LARGE');
    for (const line of contents.split('\n').filter(x => x.trim())) safeContent(JSON.parse(line));
    const file = path.join(root, relative); noSymlinks(file); ensureDir(path.dirname(file), true);
    const old = readText(file, { optional: true, privateFile: true, maxBytes: MAX_DERIVED_BYTES });
    if (old !== null) { if (old !== contents) fail('IMMUTABLE_DATASET_CONFLICT'); return file; }
    atomicWrite(file, contents, { expected: null }); return file;
  }
  return Object.freeze({ root, home, status, config, ticket, setCapture, setMinLabels, lock, scan, scanUnlocked, appendUnlocked, decision, outcome, writeDerived });
}
