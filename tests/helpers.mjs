import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createDecisionEngine } from '../src/engine.mjs';
import { SMOKE_REQUEST, SMOKE_RESPONSE } from '../src/cli.mjs';
import { atomicWrite, setMode, loadConfig } from '../src/storage.mjs';
export const KEY = 'test-only-not-a-real-typesafe-key';
export const request = () => structuredClone(SMOKE_REQUEST);
export const response = () => structuredClone(SMOKE_RESPONSE);
export function fixture(t, overrides = {}) {
  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-test-')));
  t.after(() => fs.rmSync(home, { recursive: true, force: true }));
  setMode(home, 'on', {});
  const env = { TYPESAFE_API_KEY: KEY };
  let calls = 0;
  const engine = createDecisionEngine({ home, env, provider: async () => { calls++; return response(); }, ...overrides });
  const config = patch => atomicWrite(path.join(home, 'config.json'), JSON.stringify({ ...loadConfig(home, {}), ...patch }));
  return { home, env, engine, config, calls: () => calls };
}
export function tempHome(t) {
  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pm-host-test-')));
  t.after(() => fs.rmSync(home, { recursive: true, force: true })); return home;
}
export const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
export function deferred() { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; }
