import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createDecisionEngine, decideOrDelegate } from '../src/engine.mjs';
import { setMode } from '../src/storage.mjs';
import { SMOKE_REQUEST, SMOKE_RESPONSE } from '../src/cli.mjs';

// Entirely synthetic. Proves control flow, not real cost/latency/accuracy.
const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-demo-')));
try {
  let hostCalls = 0, jevCalls = 0;
  const engine = createDecisionEngine({ home, env: { TYPESAFE_API_KEY: 'offline-demo-not-a-real-key' },
    provider: async () => { jevCalls++; return structuredClone(SMOKE_RESPONSE); } });
  const delegate = () => { hostCalls++; return 'host-route'; };
  const use = answers => answers.category.value;
  setMode(home, 'off');
  await decideOrDelegate(engine, SMOKE_REQUEST, { use, delegate });
  setMode(home, 'on');
  const route = await decideOrDelegate(engine, SMOKE_REQUEST, { use, delegate });
  console.log(JSON.stringify({ synthetic: true, externalApiCalls: 0, hostCalls, fixtureJevCalls: jevCalls, route, note: 'The ON branch skipped the delegate. This is not an end-to-end coding benchmark.' }, null, 2));
} finally { fs.rmSync(home, { recursive: true, force: true }); }
