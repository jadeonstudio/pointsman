import fs from 'node:fs';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { resolveHome, ensureDir, noSymlinks, readText, atomicWrite } from '../storage.mjs';
import { fail } from '../constants.mjs';
import { outsideGit } from './store.mjs';
import { only, id as validId } from './schema.mjs';

// Correlates a host's tool_use_id/agent_id to a decision_id so a later SubagentStop can record a
// weak host_review outcome. No prompt, transcript, or message content is ever stored here.
const HOSTS = ['codex', 'claude'];
const KINDS = ['tool_use', 'agent'];
export const LINK_RETENTION_MS = 30 * 24 * 60 * 60 * 1000;
export const MAX_LINKS = 10000;
const keyDigest = (host, kind, refId) => createHash('sha256').update(`${host}|${kind}|${refId}`).digest('hex');

export function createLinkIndex({ home = resolveHome(), now = () => Date.now(), maxLinks = MAX_LINKS } = {}) {
  const root = path.join(home, 'links');
  function file(host, kind, refId) {
    if (!HOSTS.includes(host) || !KINDS.includes(kind) || typeof refId !== 'string' || !refId.trim() || refId.length > 256) fail('INVALID_LINK_KEY');
    return path.join(root, `${keyDigest(host, kind, refId)}.json`);
  }
  function readEntry(target) {
    const raw = readText(target, { optional: true, privateFile: true, maxBytes: 4096 });
    if (raw === null) return null;
    let e; try { e = JSON.parse(raw); } catch { return null; }
    try { only(e, ['decision_id', 'host', 'kind', 'created_at'], ['decision_id', 'host', 'kind', 'created_at']); validId(e.decision_id); } catch { return null; }
    if (!HOSTS.includes(e.host) || !KINDS.includes(e.kind)) return null;
    const at = Date.parse(e.created_at);
    if (!Number.isFinite(at) || now() - at > LINK_RETENTION_MS) { try { fs.unlinkSync(target); } catch { /* already gone */ } return null; }
    return e;
  }
  function prune() {
    if (!fs.existsSync(root)) return;
    const names = fs.readdirSync(root).filter(n => n.endsWith('.json'));
    // Cheap fast path: only pay for full validation+sort once actually over the cap.
    // Individual expired entries are still reclaimed lazily by get()/readEntry() on access.
    if (names.length <= maxLinks) return;
    outsideGit(root);
    const live = [];
    for (const name of names) { const target = path.join(root, name); const e = readEntry(target); if (e) live.push({ target, at: Date.parse(e.created_at) }); }
    if (live.length > maxLinks) {
      live.sort((a, b) => a.at - b.at);
      for (const { target } of live.slice(0, live.length - maxLinks)) { try { fs.unlinkSync(target); } catch { /* already gone */ } }
    }
  }
  function put({ host, kind, id, decision_id }) {
    validId(decision_id);
    ensureDir(home, true); ensureDir(root, true); outsideGit(root);
    const target = file(host, kind, id); noSymlinks(target);
    const body = { decision_id, host, kind, created_at: new Date(now()).toISOString() };
    atomicWrite(target, JSON.stringify(body) + '\n', { mode: 0o600 });
    prune();
    return { stored: true };
  }
  function get({ host, kind, id }) {
    if (!fs.existsSync(root)) return null;
    return readEntry(file(host, kind, id))?.decision_id ?? null;
  }
  function alias({ host, fromKind, fromId, toKind, toId }) {
    const decision_id = get({ host, kind: fromKind, id: fromId });
    if (!decision_id) return { stored: false, reason: 'NO_LINKED_DECISION' };
    return put({ host, kind: toKind, id: toId, decision_id });
  }
  return Object.freeze({ put, get, alias, root });
}
// completed is NOT success; only an executed, checked runner/human outcome may claim task_succeeded.
export function recordSubagentStop(store, links, { host, agent_id, status }) {
  if (!['completed', 'failed', 'interrupted'].includes(status)) fail('INVALID_SUBAGENT_STATUS');
  const decision_id = links.get({ host, kind: 'agent', id: agent_id });
  if (!decision_id) return { stored: false, reason: 'NO_LINKED_DECISION' };
  const outcome = { decision_id, execution_id: randomUUID(), executed: false, final: true, source: 'host_review',
    executed_answers: {}, metrics: { task_succeeded: null }, checks: [], labels: [], host_review: status === 'completed' ? 'uncertain' : 'fail' };
  return store.outcome(outcome, { trust: 'host' });
}
