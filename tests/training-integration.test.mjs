import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { createDecisionEngine } from '../src/engine.mjs';
import { normalizeInference, createLayaClient, validateProviderConfig, decisionScope } from '../src/inference.mjs';
import { setMode } from '../src/storage.mjs';
import { DEFAULTS } from '../src/constants.mjs';
import { startMcp, TOOLS } from '../src/mcp.mjs';
import { readMetrics } from '../src/metrics.mjs';
import { digest } from '../src/training/schema.mjs';
import { buildDataset, readDataset, exportDataset } from '../src/training/dataset.mjs';
import { evaluateDecision, pairedPreferences } from '../src/training/evaluate.mjs';
import { fixture, request, response, trace, decision, outcome, layaConfig, KEY, splitFor, stateForSplit, fillerRequest } from './training-helpers.mjs';
const ROOT = fileURLToPath(new URL('../', import.meta.url));
const cli = (f, args, input) => spawnSync(process.execPath, [path.join(ROOT, 'bin/pointsman.mjs'), ...args], {
  input, encoding: 'utf8', timeout: 8000, env: { ...process.env, HOME: f.home, POINTSMAN_HOME: f.home, TYPESAFE_API_KEY: '', POINTSMAN_DISABLE: '0' },
});
const identity = l => ({ model: l.model, checkpoint: l.checkpoint, runtime_version: l.runtimeVersion, device: l.device, precision: 'torch.float32' });

test('all provider, capture and dataset CLI commands are wired, offline and strict', t => {
  const f = fixture(t, false);
  for (const args of [['training','capture','status'], ['training','capture','on'], ['training','capture','off'], ['dataset','stats'], ['dataset','validate'], ['provider','status']]) {
    const r = cli(f,args); assert.equal(r.status,0,r.stderr); assert.equal(r.stdout.includes(KEY),false);
  }
  assert.equal(cli(f,['compare']).status,2); assert.equal(cli(f,['training','capture','on','extra']).status,2);
  assert.equal(cli(f,['dataset','train']).status,2); assert.equal(cli(f,['provider','other']).status,2);
  assert.ok(cli(f,['help']).stdout.includes('dataset export'));
});
test('Laya readiness and ON do not require a TypeSafe credential; OFF does no inference', async t => {
  const f = fixture(t,false); const p = layaConfig(f.home); p.laya.python = process.execPath;
  fs.writeFileSync(path.join(f.home,'model.safetensors'),'synthetic, not a model'); f.writeProviders(p);
  const r = cli(f,['on']); assert.equal(r.status,0,r.stderr); assert.equal(JSON.parse(r.stdout).credential,'not-required');
  assert.equal(cli(f,['off']).status,0);
  let calls=0; const e=createDecisionEngine({home:f.home,env:{},provider:async()=>{calls++;throw new Error('unexpected');}});t.after(()=>e.close());
  assert.equal((await e.decide(request())).reason,'OFF');assert.equal(calls,0);
});
test('Laya normalization preserves provenance and requires checkpoint and question-specific qualification', t => {
  const p=layaConfig('/tmp');const raw={...response(),identity:identity(p.laya)};
  const n=normalizeInference('laya',raw,request(),DEFAULTS,p);assert.equal(n.eligible,false);assert.equal(n.provenance.provider,'laya');
  p.laya.qualification={checkpoint:p.laya.checkpoint,calibrationVersion:'fixture-v1',purposes:['route'],minConfidence:.9,minChoiceProbability:.9,noulCertainty:.95};
  assert.equal(normalizeInference('laya',raw,request(),DEFAULTS,p).eligible,false);
  p.laya.qualification.decisionIdentity={version:1,checkpoint:p.laya.checkpoint,runtimeVersion:p.laya.runtimeVersion,
    precision:p.laya.precision??'fp32',inputFit:p.laya.inputFit??'lossless',calibrationVersion:'fixture-v1',
    scopes:Object.keys(request().questions).map(id=>{const actual=decisionScope(request(),id);return {...actual,
      familyId:'fixture-decision',familyRevision:'fixture-v1',stateBuilderRevision:`payload-state-schema-v1:${actual.stateSchemaHash}`,threshold:.9};})};
  assert.equal(normalizeInference('laya',raw,request(),DEFAULTS,p).eligible,true);
  assert.throws(()=>normalizeInference('laya',{...raw,identity:{...raw.identity,checkpoint:'b'.repeat(64)}},request(),DEFAULTS,p),/LAYA_IDENTITY_MISMATCH/);
});
test('laya.precision defaults to fp32, only accepts fp32|fp16, and binds qualification to it', t => {
  const base = layaConfig('/tmp');
  assert.equal(validateProviderConfig(structuredClone(base)).laya.precision, 'fp32');
  assert.equal(validateProviderConfig({ ...structuredClone(base), laya: { ...base.laya, precision: 'fp16' } }).laya.precision, 'fp16');
  for (const bad of ['bf16', 'int8', 'FP32', '']) {
    assert.throws(() => validateProviderConfig({ ...structuredClone(base), laya: { ...base.laya, precision: bad } }), /INVALID_PROVIDER_CONFIG/);
  }
  const qualified = { ...structuredClone(base), laya: { ...base.laya, precision: 'fp16',
    qualification: { checkpoint: base.laya.checkpoint, calibrationVersion: 'v1', purposes: ['route'], minConfidence: .9, minChoiceProbability: .9, noulCertainty: .9 } } };
  assert.equal(validateProviderConfig(structuredClone(qualified)).laya.precision, 'fp16');
  const matchingQualification = { ...qualified, laya: { ...qualified.laya, qualification: { ...qualified.laya.qualification, precision: 'fp16' } } };
  assert.equal(validateProviderConfig(structuredClone(matchingQualification)).laya.qualification.precision, 'fp16');
  const mismatchedQualification = { ...qualified, laya: { ...qualified.laya, qualification: { ...qualified.laya.qualification, precision: 'fp32' } } };
  assert.throws(() => validateProviderConfig(structuredClone(mismatchedQualification)), /INVALID_PROVIDER_CONFIG/);
});
test('qualification.routeGate validates strictly: exact keys, decision-v1 method, and bounded tierCoverage/maxHostProbability', t => {
  const base = layaConfig('/tmp');
  const qualified = gate => ({ ...structuredClone(base), laya: { ...base.laya,
    qualification: { checkpoint: base.laya.checkpoint, calibrationVersion: 'v1', purposes: ['route'], minConfidence: .9, minChoiceProbability: .9, noulCertainty: .9,
      ...(gate !== undefined ? { routeGate: gate } : {}) } } });
  assert.equal(validateProviderConfig(structuredClone(qualified(undefined))).laya.qualification.routeGate, undefined);
  const good = { method: 'decision-v1', tierCoverage: .9, maxHostProbability: .05 };
  assert.deepEqual(validateProviderConfig(structuredClone(qualified(good))).laya.qualification.routeGate, good);
  // Shape violations (extra/missing keys) go through the shared training-schema `only()` helper and
  // throw its own INVALID_TRAINING_SCHEMA code; value/range violations throw INVALID_PROVIDER_CONFIG.
  for (const [bad, code] of [
    [{ method: 'decision-v1', tierCoverage: .9, maxHostProbability: .05, extra: 1 }, 'INVALID_TRAINING_SCHEMA'], // unexpected key
    [{ tierCoverage: .9, maxHostProbability: .05 }, 'INVALID_TRAINING_SCHEMA'], // missing method
    [{ method: 'decision-v2', tierCoverage: .9, maxHostProbability: .05 }, 'INVALID_PROVIDER_CONFIG'], // wrong method literal
    [{ method: 'decision-v1', tierCoverage: .4, maxHostProbability: .05 }, 'INVALID_PROVIDER_CONFIG'], // tierCoverage below 0.5
    [{ method: 'decision-v1', tierCoverage: 1, maxHostProbability: .05 }, 'INVALID_PROVIDER_CONFIG'], // tierCoverage above 0.999
    [{ method: 'decision-v1', tierCoverage: .9, maxHostProbability: -.01 }, 'INVALID_PROVIDER_CONFIG'], // maxHostProbability below 0
    [{ method: 'decision-v1', tierCoverage: .9, maxHostProbability: .6 }, 'INVALID_PROVIDER_CONFIG'], // maxHostProbability above 0.5
    [{ method: 'decision-v1', tierCoverage: 'high', maxHostProbability: .05 }, 'INVALID_PROVIDER_CONFIG'], // non-numeric
  ]) assert.throws(() => validateProviderConfig(structuredClone(qualified(bad))), new RegExp(code), JSON.stringify(bad));
});
test('laya identity precision must match the configured precision (fp32<->fp16), independent of checkpoint/device checks', t => {
  const p = layaConfig('/tmp', { precision: 'fp16' });
  const raw = { ...response(), identity: { ...identity(p.laya), precision: 'torch.float16' } };
  assert.equal(normalizeInference('laya', raw, request(), DEFAULTS, p).provenance.precision, 'torch.float16');
  assert.throws(() => normalizeInference('laya', { ...raw, identity: { ...raw.identity, precision: 'torch.float32' } }, request(), DEFAULTS, p), /LAYA_IDENTITY_MISMATCH/);
  const fp32 = layaConfig('/tmp');
  assert.throws(() => normalizeInference('laya', { ...response(), identity: { ...identity(fp32.laya), precision: 'torch.float16' } }, request(), DEFAULTS, fp32), /LAYA_IDENTITY_MISMATCH/);
  assert.throws(() => normalizeInference('laya', { ...response(), identity: { ...identity(fp32.laya), precision: 'torch.bfloat16' } }, request(), DEFAULTS, fp32), /LAYA_IDENTITY_MISMATCH/);
});
test('local inference counts never masquerade as Jev network calls or Jev token usage', async t => {
  const f=fixture(t);const p=layaConfig(f.home);f.writeProviders(p);
  const e=createDecisionEngine({home:f.home,env:{},provider:async()=>({...response(),identity:identity(p.laya)})});t.after(()=>e.close());
  const r=await e.decide({...request(),trace:trace()});assert.equal(r.networkCalls,0);assert.equal(r.inferenceCalls,1);assert.equal(r.apply,false);
  const m=readMetrics(f.home);assert.equal(m.inferenceCalls,1);assert.equal(m.jevReportedTokens.input,0);assert.equal(m.providerActivity.laya.inputTokens,100);
});
test('Jev/Laya compare is explicit, independent, and invalidates stale active results', async t => {
  const f=fixture(t);const p=layaConfig(f.home);p.provider='jev';f.writeProviders(p);let calls=0;
  const e=createDecisionEngine({home:f.home,env:f.env,provider:async payload=>{
    calls++;if(payload.model==='laya/base'){setMode(f.home,'off',{});return {...response(),identity:identity(p.laya)};}return response();
  }});t.after(()=>e.close());
  await assert.rejects(e.compare(request()),/EXPLICIT_REMOTE/);assert.equal(calls,0);
  const r=await e.compare({...request(),trace:trace()},{remoteConsent:true});assert.equal(calls,2);
  assert.equal(r.active.apply,false);assert.deepEqual(r.active.answers,{});assert.equal(r.observers[0].applied,false);
  assert.equal('answers' in r.observers[0],false);
});
test('warm worker uses one child, preserves split UTF-8, and receives no inherited API secrets', async t => {
  const f=fixture(t,false);const p=layaConfig(f.home,{model:'laya/\uD55C\uAE00'});p.laya.python=process.execPath;
  fs.writeFileSync(path.join(f.home,'model.safetensors'),'fake');let starts=0,kills=0;
  const client=createLayaClient({spawnImpl:(command,args,options)=>{
    starts++;assert.equal(options.shell,false);assert.equal(options.env.TYPESAFE_API_KEY,undefined);assert.equal(options.env.HF_TOKEN,undefined);
    const c=new EventEmitter();c.stdin=new PassThrough();c.stdout=new PassThrough();c.stderr=new PassThrough();c.kill=()=>{kills++;queueMicrotask(()=>c.emit('exit',0));return true;};
    c.stdin.on('data',b=>{const msg=JSON.parse(b.toString());const out=Buffer.from(JSON.stringify(msg.init?{ready:true,identity:identity(p.laya)}:{id:msg.id,result:{...response(),identity:identity(p.laya)}})+'\n');
      queueMicrotask(()=>{for(let i=0;i<out.length;i++)c.stdout.write(out.subarray(i,i+1));});});return c;
  }});t.after(()=>client.close());
  const r=await client.infer(request(),p,{timeoutMs:500,env:{TYPESAFE_API_KEY:KEY,HF_TOKEN:'do-not-inherit'}});
  assert.equal(r.identity.model,'laya/\uD55C\uAE00');await client.infer(request(),p,{timeoutMs:500,env:{}});assert.equal(starts,1);
  client.close();assert.ok(kills>=1);assert.equal(client.status().running,false);
});
test('deadline includes startup, while a timed out request preserves the warm worker', async t => {
  const f=fixture(t,false),p=layaConfig(f.home);p.laya.python=process.execPath;fs.writeFileSync(path.join(f.home,'model.safetensors'),'fake');let starts=0,requests=0;
  const client=createLayaClient({spawnImpl:()=>{starts++;const c=new EventEmitter();c.stdin=new PassThrough();c.stdout=new PassThrough();c.stderr=new PassThrough();c.kill=()=>true;
    c.stdin.on('data',b=>{const m=JSON.parse(b.toString());if(m.init)setTimeout(()=>c.stdout.write(JSON.stringify({ready:true,identity:identity(p.laya)})+'\n'),30);else{requests++;c.stdout.write(JSON.stringify({id:m.id,result:{...response(),identity:identity(p.laya)}})+'\n');}});return c;}});t.after(()=>client.close());
  await assert.rejects(client.infer(request(),p,{timeoutMs:10,env:{}}),/TIMEOUT/);await new Promise(r=>setTimeout(r,35));
  await client.infer(request(),p,{timeoutMs:100,env:{}});assert.equal(starts,1);assert.equal(requests,1);assert.equal(client.status().running,true);
});
test('pre-aborted preparation never spawns, and an expired startup handles its later failure', async t => {
  const f=fixture(t,false),p=layaConfig(f.home);p.laya.python=process.execPath;fs.writeFileSync(path.join(f.home,'model.safetensors'),'fake');let starts=0;
  const client=createLayaClient({spawnImpl:()=>{starts++;const c=new EventEmitter();c.stdin=new PassThrough();c.stdout=new PassThrough();c.stderr=new PassThrough();c.kill=()=>true;setTimeout(()=>c.emit('error',new Error('late startup failure')),20);return c;}});t.after(()=>client.close());
  const abort=new AbortController();abort.abort();await assert.rejects(client.prepare(p,{signal:abort.signal,env:{}}),/CANCELLED/);assert.equal(starts,0);
  await assert.rejects(client.infer(request(),p,{timeoutMs:1,env:{}}),/TIMEOUT/);await new Promise(r=>setTimeout(r,25));
});
test('cancelled request tombstones its late response without stopping other requests', async t => {
  const f=fixture(t,false),p=layaConfig(f.home);p.laya.python=process.execPath;fs.writeFileSync(path.join(f.home,'model.safetensors'),'fake');let child;
  const client=createLayaClient({spawnImpl:()=>{const c=child=new EventEmitter();c.stdin=new PassThrough();c.stdout=new PassThrough();c.stderr=new PassThrough();c.kill=()=>true;
    c.stdin.on('data',b=>{const m=JSON.parse(b.toString());if(m.init)queueMicrotask(()=>c.stdout.write(JSON.stringify({ready:true,identity:identity(p.laya)})+'\n'));else if(m.state.name!=='A')queueMicrotask(()=>c.stdout.write(JSON.stringify({id:m.id,result:{...response(),identity:identity(p.laya)}})+'\n'));else setTimeout(()=>c.stdout.write(JSON.stringify({id:m.id,result:{...response(),identity:identity(p.laya)}})+'\n'),30);});return c;}});t.after(()=>client.close());
  const a={...request(),state:{name:'A'}},b={...request(),state:{name:'B'}},c={...request(),state:{name:'C'}};
  const timed=client.infer(a,p,{timeoutMs:10,env:{}});await new Promise(r=>setTimeout(r,2));
  const [br,cr]=await Promise.all([client.infer(b,p,{timeoutMs:100,env:{}}),client.infer(c,p,{timeoutMs:100,env:{}})]);
  await assert.rejects(timed,/TIMEOUT/);assert.equal(br.identity.model,p.laya.model);assert.equal(cr.identity.model,p.laya.model);await new Promise(r=>setTimeout(r,35));
  assert.deepEqual(client.status(),{running:true,ready:true,resident:false,inFlight:0,generation:1,identity:identity(p.laya)});child.stdout.write(JSON.stringify({id:'unknown',result:{}})+'\n');assert.equal(client.status().running,false);
});
test('a short timed-out request does not consume a longer pending request liveness budget', async t => {
  const f=fixture(t,false),p=layaConfig(f.home);p.laya.python=process.execPath;fs.writeFileSync(path.join(f.home,'model.safetensors'),'fake');
  const client=createLayaClient({spawnImpl:()=>{const c=new EventEmitter();c.stdin=new PassThrough();c.stdout=new PassThrough();c.stderr=new PassThrough();c.kill=()=>true;
    c.stdin.on('data',b=>{const m=JSON.parse(b.toString());if(m.init)queueMicrotask(()=>c.stdout.write(JSON.stringify({ready:true,identity:identity(p.laya)})+'\n'));else if(m.state.name==='B')setTimeout(()=>c.stdout.write(JSON.stringify({id:m.id,result:{...response(),identity:identity(p.laya)}})+'\n'),1050);});return c;}});t.after(()=>client.close());
  const a=client.infer({...request(),state:{name:'A'}},p,{timeoutMs:20,env:{}});await new Promise(r=>setTimeout(r,2));
  const b=client.infer({...request(),state:{name:'B'}},p,{timeoutMs:1500,env:{}});await assert.rejects(a,/TIMEOUT/);assert.equal((await b).identity.model,p.laya.model);
});
test('MCP exposes one bounded weak-evidence recorder and closes provider runtime', async t => {
  const f=fixture(t);const d=f.save(decision());const input=new PassThrough(),output=new PassThrough();const messages=[];
  output.on('data',b=>messages.push(...b.toString().trim().split('\n').map(JSON.parse)));
  let closed=0;const engine={...f.engine,close:()=>{closed++;f.engine.close();}};const server=startMcp(engine,{input,output});t.after(()=>{server.close();input.destroy();output.destroy();});
  const send=(id,method,params)=>input.write(JSON.stringify({jsonrpc:'2.0',...(id?{id}:{}),method,params})+'\n');
  send(1,'initialize',{protocolVersion:'2025-06-18',clientInfo:{name:'training-test',version:'1'},capabilities:{}});send(null,'notifications/initialized');
  send(2,'tools/call',{name:'record',arguments:{kind:'outcome',data:outcome(d)}});
  await new Promise(r=>setTimeout(r,20));const r=messages.find(x=>x.id===2);assert.equal(r.result.isError,false);
  const s=f.store.scan();const o=s.events.find(e=>e.kind==='outcomes');assert.equal(o.data.source,'host_review');assert.equal(o.data.labels[0].source,'host_review');
  assert.equal(evaluateDecision(s.events.find(e=>e.kind==='decisions'),[o]).labels.length,0);
  assert.ok(TOOLS.find(t=>t.name==='decide').inputSchema.properties.trace);
  server.close();assert.ok(closed>0);
});
test('Score and Noul supervised exports preserve label order without invented fractional targets', t => {
  const f=fixture(t);let d=decision();d.request.questions={flag:{type:'noul',instructions:'Is additional inspection needed?'},depth:{type:'score',instructions:'Rate scope.',criteria:['small','medium','large']}};
  d.request_hash=digest(d.request);d.answers={flag:{type:'noul',value:false,confidence:null,probabilityTrue:.01},depth:{type:'score',value:.2,confidence:.9,probabilities:{0:.8,1:.2,2:0}}};f.save(d);
  const o=outcome(d,{executed:false,executed_answers:{},metrics:{},source:'human',checks:[],labels:[
    {question_id:'flag',value:false,source:'human',label_confidence:1,evidence_ref:'sha256:'+'a'.repeat(64)},
    {question_id:'depth',value:1,source:'human',label_confidence:1,evidence_ref:'sha256:'+'a'.repeat(64)},]});f.store.outcome(o);
  // Both samples share one request_hash, so they always land in the same split; fill the
  // other two so the laya export (which refuses an empty split) can be exercised too.
  let cursor=0;
  for (const split of ['train','calibration','test'].filter(s=>s!==splitFor(d.request_hash))) {
    const i=stateForSplit(fillerRequest,split,cursor);cursor=i+1;
    const filler=decision();filler.request=fillerRequest(i);filler.request_hash=digest(filler.request);
    f.save(filler);f.store.outcome(outcome(filler));
  }
  const b=buildDataset(f.store);assert.equal(b.sample_count,4);const samples=readDataset(f.store,b.dataset_version).samples;
  assert.deepEqual(samples.find(s=>s.question_id==='flag').target.probabilities,{false:1,true:0});
  assert.deepEqual(samples.find(s=>s.question_id==='depth').target.probabilities,{0:0,1:1,2:0});assert.equal(exportDataset(f.store,b.dataset_version).samples,4);
});
test('routing preference requires separately executed comparable alternatives, not shadow guesses', t => {
  const d1=decision(),d2=decision();d2.trace=d1.trace;d2.answers.worker={type:'choice',value:'strong',confidence:.99,selectedProbability:.99,probabilities:{light:.01,strong:.99}};
  const o1=outcome(d1),o2=outcome(d2);o2.metrics.latency_ms=9999;o2.metrics.token_usage=9000;
  const rows=[d1,d2].map((d,i)=>{const o=[o1,o2][i],de={event_id:d.decision_id,data:d},oe={event_id:randomUUID(),data:o};return{decision:de,outcome:o,evaluation:evaluateDecision(de,[oe])};});
  assert.equal(pairedPreferences(rows).length,1);assert.equal(pairedPreferences(rows)[0].is_ground_truth,false);
  rows[1].outcome.execution_id=rows[0].outcome.execution_id;assert.equal(pairedPreferences(rows).length,0);
});
