import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { randomUUID } from 'node:crypto';
import { DEFAULTS, MODES, MODEL_ID, fail, isObject } from './constants.mjs';

export function resolveHome(env = process.env) {
  const home = env.POINTSMAN_HOME || path.join(env.HOME || os.homedir(), '.local/share/pointsman');
  if (!path.isAbsolute(home)) fail('HOME_MUST_BE_ABSOLUTE');
  return path.resolve(home);
}
export function noSymlinks(target) {
  let current = path.resolve(target);
  for (;;) {
    try { if (fs.lstatSync(current).isSymbolicLink()) fail('UNSAFE_SYMLINK'); }
    catch (e) { if (e.code !== 'ENOENT') throw e; }
    const parent = path.dirname(current);
    if (parent === current) return;
    current = parent;
  }
}
export function ensureDir(dir, privateDir = false) {
  noSymlinks(dir);
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  const stat = fs.statSync(dir);
  if (!stat.isDirectory()) fail('UNSAFE_DIRECTORY');
  if (typeof process.getuid === 'function' && stat.uid !== process.getuid()) fail('WRONG_OWNER');
  if (privateDir && (stat.mode & 0o077)) fail('PRIVATE_DIRECTORY_REQUIRED');
}
export function readText(file, { optional = false, privateFile = false, maxBytes = 1048576 } = {}) {
  noSymlinks(file);
  let fd;
  try { fd = fs.openSync(file, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW || 0)); }
  catch (e) { if (optional && e.code === 'ENOENT') return null; throw e; }
  try {
    const s = fs.fstatSync(fd);
    if (!s.isFile() || s.nlink !== 1) fail('UNSAFE_FILE');
    if (s.size > maxBytes) fail('FILE_TOO_LARGE');
    if (privateFile) {
      if (typeof process.getuid === 'function' && s.uid !== process.getuid()) fail('WRONG_OWNER');
      if (s.mode & 0o077) fail('PRIVATE_FILE_REQUIRED');
    }
    const bytes = Buffer.alloc(Math.min(s.size + 1, maxBytes + 1));
    const count = fs.readSync(fd, bytes, 0, bytes.length, 0);
    if (count > maxBytes || fs.fstatSync(fd).size > s.size) fail('FILE_CHANGED_OR_TOO_LARGE');
    return bytes.subarray(0, count).toString('utf8');
  } finally { fs.closeSync(fd); }
}
export function atomicWrite(file, text, { expected, mode = 0o600 } = {}) {
  noSymlinks(file);
  ensureDir(path.dirname(file));
  const temporary = `${file}.${randomUUID()}.tmp`;
  let fd;
  try {
    if (expected !== undefined && readText(file, { optional: true }) !== expected) fail('WRITE_CONFLICT');
    fd = fs.openSync(temporary, 'wx', mode);
    fs.writeFileSync(fd, text, 'utf8'); fs.fsyncSync(fd); fs.closeSync(fd); fd = undefined;
    if (expected !== undefined && readText(file, { optional: true }) !== expected) fail('WRITE_CONFLICT');
    noSymlinks(file);
    fs.renameSync(temporary, file);
  } finally {
    if (fd !== undefined) fs.closeSync(fd);
    try { fs.unlinkSync(temporary); } catch (e) { if (e.code !== 'ENOENT') throw e; }
  }
}
export function validateConfig(raw) {
  if (!isObject(raw) || Object.keys(raw).some(k => !Object.hasOwn(DEFAULTS, k))) fail('INVALID_CONFIG');
  const c = { ...DEFAULTS, ...raw };
  if (c.version !== 1 || !MODES.includes(c.mode) || typeof c.model !== 'string' || !MODEL_ID.test(c.model) || typeof c.telemetry !== 'boolean') fail('INVALID_CONFIG');
  const ranges = {
    timeoutMs: [100, 10000], maxInputBytes: [512, 48000], maxQuestions: [1, 8],
    maxCallsPerMinute: [1, 600], maxInFlight: [1, 8], circuitFailureThreshold: [1, 10], circuitCooldownMs: [1000, 300000],
  };
  for (const [k, [min, max]] of Object.entries(ranges)) if (!Number.isInteger(c[k]) || c[k] < min || c[k] > max) fail('INVALID_CONFIG');
  for (const k of ['minConfidence', 'minChoiceProbability', 'noulCertainty']) {
    if (!Number.isFinite(c[k]) || c[k] < 0.5 || c[k] > 1) fail('INVALID_CONFIG');
  }
  return c;
}
export function loadConfig(home, env = process.env) {
  const text = readText(path.join(home, 'config.json'), { optional: true, privateFile: true, maxBytes: 16384 });
  let raw;
  try { raw = text === null ? {} : JSON.parse(text); } catch { fail('INVALID_CONFIG'); }
  const config = validateConfig(raw);
  if (env.POINTSMAN_DISABLE && !['0', '1'].includes(env.POINTSMAN_DISABLE)) fail('INVALID_KILL_SWITCH');
  return env.POINTSMAN_DISABLE === '1' ? { ...config, mode: 'off' } : config;
}
export function setMode(home, mode, env = process.env) {
  if (!MODES.includes(mode)) fail('INVALID_MODE');
  ensureDir(home, true);
  const file = path.join(home, 'config.json');
  const previous = readText(file, { optional: true, privateFile: true });
  // Emergency off must remain possible even when the config was damaged.
  let config;
  try { config = loadConfig(home, { ...env, POINTSMAN_DISABLE: '0' }); }
  catch (e) { if (mode !== 'off') throw e; config = { ...DEFAULTS }; }
  config.mode = mode;
  atomicWrite(file, JSON.stringify(config, null, 2) + '\n', { expected: previous });
  return config;
}
function validKey(value) {
  if (typeof value !== 'string' || !/^[\x21-\x7e]{8,512}$/.test(value) || /["'`\\]/.test(value)) fail('INVALID_API_KEY');
  return value;
}
export function getCredential(home, env = process.env) {
  if (env.TYPESAFE_API_KEY?.trim()) return { key: validKey(env.TYPESAFE_API_KEY.trim()), source: 'environment' };
  const file = path.join(home, 'credentials.env');
  const text = readText(file, { optional: true, privateFile: true, maxBytes: 1024 });
  if (text === null) return { key: null, source: 'missing' };
  const stat = fs.statSync(home);
  if (stat.mode & 0o077) fail('PRIVATE_DIRECTORY_REQUIRED');
  const match = /^TYPESAFE_API_KEY=([^\r\n]+)\r?\n?$/.exec(text);
  if (!match) fail('INVALID_CREDENTIAL_FILE');
  return { key: validKey(match[1]), source: 'managed-file' };
}
export function saveCredential(home, key) {
  validKey(key);
  // Do not create a credential file anywhere beneath a Git worktree.
  let current = path.resolve(home);
  for (;;) {
    const marker = path.join(current, '.git');
    const stat = fs.lstatSync(marker, { throwIfNoEntry: false });
    if (stat) {
      // A tool cache named .git is not a repository; retain protection for partial Git metadata and worktree links.
      if (!stat.isDirectory() || fs.readdirSync(marker).some(name =>
        ['HEAD', 'objects', 'refs', 'config', 'index', 'commondir'].includes(name))) fail('KEY_IN_REPOSITORY_REFUSED');
    }
    const parent = path.dirname(current); if (parent === current) break; current = parent;
  }
  ensureDir(home, true);
  atomicWrite(path.join(home, 'credentials.env'), `TYPESAFE_API_KEY=${key}\n`);
}
export function removeCredential(home) {
  const file = path.join(home, 'credentials.env');
  if (readText(file, { optional: true, privateFile: true }) !== null) fs.unlinkSync(file);
}
export function appendEvent(home, event) {
  try {
    ensureDir(home, true);
    const dir = path.join(home, 'logs'); ensureDir(dir, true);
    const file = path.join(dir, `events-${new Date().toISOString().slice(0, 10)}.jsonl`);
    noSymlinks(file);
    const fd = fs.openSync(file, fs.constants.O_CREAT | fs.constants.O_APPEND | fs.constants.O_WRONLY | (fs.constants.O_NOFOLLOW || 0), 0o600);
    try {
      const stat = fs.fstatSync(fd);
      if (!stat.isFile() || stat.nlink !== 1 || (stat.mode & 0o077) || stat.size > 10 * 1024 * 1024) return false;
      if (typeof process.getuid === 'function' && stat.uid !== process.getuid()) return false;
      fs.writeSync(fd, JSON.stringify(event) + '\n');
      return true;
    } finally { fs.closeSync(fd); }
  } catch { return false; } // Observability cannot become an availability dependency.
}
