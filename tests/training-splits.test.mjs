import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { ControlError } from '../src/constants.mjs';
import { randomUUID } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fixture, decision } from './training-helpers.mjs';
import { digest, encode, POLICY_VERSION } from '../src/training/schema.mjs';
import { groupedSplits, writeIndependentDataset, readDataset, exportDataset, splitIdentity, assertSplitIsolation } from '../src/training/dataset.mjs';

function row(family, variant = 'en') {
  const d = decision('host'), state = { family, variant, facts: 'bounded independent facts' };
  const question = { type: 'choice', instructions: 'Choose the option grounded in the supplied facts.', criteria: { yes: 'Grounded', no: 'Unsupported' } };
  const target = { value: 'yes', probabilities: { yes: 1, no: 0 } }, request_hash = digest({ purpose: 'select', risk: 'routine', state, questions: { answer: question } });
  const task_id = randomUUID(), snapshot_id = digest(state);
  return { schema_version: 2, sample_id: digest({ request_hash, question_id: 'answer', target }), task_id, task_ids: [task_id], snapshot_id, snapshot_ids: [snapshot_id], request_hash,
    purpose: 'select', state, question_id: 'answer', question, target, label_source: 'objective', label_confidence: 1, label_confidence_is_calibrated: false, evaluation_policy_version: POLICY_VERSION,
    provenance: [d.provenance], lineage: { source_id: `source-${family}`, template_id: `template-${family}`, semantic_family_id: family, sibling_ids: [`siblings-${family}`] },
    data_rights: { source: 'independently authored fixture', license: 'CC0-1.0', revision: 'fixture-v1', permitted_use: ['learning','evaluation'], redistribution: 'allowed', transformations: ['translation'] },
    oracle: { id: 'fixture-oracle', revision: 'v1', evidence_sha256: digest({ family, target }) }, raw_refs: { independent: { source_id: `source-${family}`, case_id: `${family}-${variant}` } } };
}
const corpus = () => Array.from({length:20},(_,i) => [row(`family-${i}`),row(`family-${i}`,'ko')]).flat();
test('four-way groups keep authored siblings together and remain stable as family samples scale', () => {
  const small = corpus(), large = [...structuredClone(small), ...Array.from({length:20},(_,i) => row(`family-${i}`,'counterfactual'))];
  groupedSplits(small,{version:2}); groupedSplits(large,{version:2});
  assert.deepEqual(new Set(small.map(s => s.split)),new Set(['train','dev','calibration','test']));
  for (const s of small) assert.equal(s.split,large.find(x=>x.sample_id===s.sample_id).split);
  assertSplitIsolation(small);
  const tampered = structuredClone(small); tampered[1].split = tampered[0].split === 'test' ? 'train' : 'test';
  assert.throws(() => assertSplitIsolation(tampered),/DATASET_FAMILY_LEAKAGE/);
});
test('independent canonical/export hashes are consumed by the Python checker and trainer before any model load', t => {
  const f = fixture(t), built = writeIndependentDataset(f.store,corpus()), read = readDataset(f.store,built.dataset_version);
  assert.equal(read.manifest.split_version,2); assert.deepEqual(read.manifest.split_hashes,splitIdentity(read.samples).split_hashes);
  const exported = exportDataset(f.store,built.dataset_version), dir=path.dirname(exported.files.find(x=>x.endsWith('manifest.json')));
  const checker = spawnSync('python3',['training/laya-kit/check_export.py','--export-dir',dir,'--dataset-manifest',built.manifest],{encoding:'utf8'});
  assert.equal(checker.status,0,checker.stdout+checker.stderr);
  const trainer = spawnSync('python3',['-c',"import importlib.util, pathlib, sys; p=pathlib.Path('training/laya-kit/train_from_export.py'); s=importlib.util.spec_from_file_location('kit',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); x=m.verify_export_manifest(pathlib.Path(sys.argv[1])); assert m.selection_split(x,True)=='dev'",dir],{encoding:'utf8'});
  assert.equal(trainer.status,0,trainer.stdout+trainer.stderr);
  const meta=path.join(dir,'metadata.jsonl'); const records=fs.readFileSync(meta,'utf8').trim().split('\n').map(JSON.parse); records[0].split='test'; fs.writeFileSync(meta,records.map(encode).join('\n')+'\n');
  const bad = spawnSync('python3',['training/laya-kit/check_export.py','--export-dir',dir],{encoding:'utf8'}); assert.notEqual(bad.status,0);
});
test('independent objective rows require documented evaluation permission and oracle identity',t => {
  const f=fixture(t), rows=corpus(); delete rows[0].oracle; assert.throws(()=>writeIndependentDataset(f.store,rows),/INVALID_CANONICAL/);
  const denied=corpus(); denied.forEach(s=>s.data_rights.permitted_use=['learning']); assert.throws(()=>writeIndependentDataset(f.store,denied),/DATASET_USE_NOT_PERMITTED/);
});

test('regression holdout is labeled as a test copy; prospective evidence must have disjoint families',async t=>{
  const {freezeHoldout} = await import('../src/training/laya-lifecycle.mjs');
  const f=fixture(t), a=writeIndependentDataset(f.store,corpus());
  const copy=freezeHoldout(f.home,{datasetVersion:a.dataset_version,name:'copy',store:f.store});
  assert.equal(copy.role,'regression_copy'); assert.equal(copy.independent_confirmation,false);
  assert.throws(()=>freezeHoldout(f.home,{datasetVersion:a.dataset_version,name:'wrong',role:'prospective',trainingDatasetVersion:a.dataset_version,store:f.store}),/PROSPECTIVE_DATASET_REQUIRED/);
  const overlapping=corpus().map(s=>({...s,state:{...s.state,new_time:'later'}, request_hash:digest({...s.state,new_time:'later'})}));
  for(const s of overlapping)s.sample_id=digest({request_hash:s.request_hash,question_id:s.question_id,target:s.target});
  const b=writeIndependentDataset(f.store,overlapping);
  assert.throws(()=>freezeHoldout(f.home,{datasetVersion:b.dataset_version,name:'overlap',role:'prospective',trainingDatasetVersion:a.dataset_version,store:f.store}),/PROSPECTIVE_FAMILY_LEAKAGE/);
  const fresh=corpus().map(s=>{ const family='fresh-'+s.lineage.semantic_family_id; return row(family,s.state.variant); });
  const c=writeIndependentDataset(f.store,fresh);
  const prospective=freezeHoldout(f.home,{datasetVersion:c.dataset_version,name:'fresh',role:'prospective',trainingDatasetVersion:a.dataset_version,store:f.store});
  assert.equal(prospective.independent_confirmation,true);
});
test('prediction collector defaults to dev and requires sealed identity for final test',async t=>{
  const {registerCheckpoint,collectCandidatePredictions} = await import('../src/training/laya-lifecycle.mjs');
  const f=fixture(t), built=writeIndependentDataset(f.store,corpus()), dir=path.join(f.home,'base');fs.mkdirSync(dir);
  fs.writeFileSync(path.join(dir,'model.safetensors'),'fixture');fs.writeFileSync(path.join(dir,'rl_agent_config.json'),'{}');fs.mkdirSync(path.join(dir,'tokenizer'));fs.writeFileSync(path.join(dir,'tokenizer/tokenizer.json'),'{}');
  const registered=registerCheckpoint(f.home,{checkpointDir:dir,python:'/usr/bin/python3',device:'cpu',model:'laya/fixture',fingerprintImpl:()=>digest('fixture')});
  let calls=0;
  const layaClient={async infer(payload,{laya}){calls++;const qid=Object.keys(payload.questions)[0];return {identity:{model:laya.model,checkpoint:laya.checkpoint,runtime_version:laya.runtimeVersion,device:laya.device,precision:'torch.float32'},answers:{[qid]:{type:'choice',choice:'yes',confidence:.99,probabilities:{yes:.99,no:.01}}},usage:{input_tokens:1,output_tokens:0}};}};
  const result=await collectCandidatePredictions(f.home,{candidateHash:registered.checkpoint,datasetVersion:built.dataset_version,store:f.store,layaClient});
  assert.equal(result.manifest.split,'dev');assert.equal(result.rows.length,calls);assert.ok(result.rows.every(x=>x.status==='ok'&&x.sample_id&&x.checkpoint));
  await assert.rejects(collectCandidatePredictions(f.home,{candidateHash:registered.checkpoint,datasetVersion:built.dataset_version,split:'test',store:f.store,layaClient}),/SEALED_SPEC_REQUIRED/);
  assert.ok(result.rows.every(x=>Number.isFinite(x.elapsedMs)&&x.elapsedMs>=0));
  const sealed=await collectCandidatePredictions(f.home,{candidateHash:registered.checkpoint,datasetVersion:built.dataset_version,split:'test',sealedSpec:digest('frozen specification'),store:f.store,layaClient});
  assert.equal(sealed.manifest.sealed_spec_sha256,digest('frozen specification'));
  let closed=0,index=0;
  const failures=['LAYA_TIMEOUT','INFERENCE_REJECTED','INVALID_RESPONSE','INPUT_TRUNCATED'];
  const owned={close(){closed++;},async infer(){throw new ControlError(failures[index++]);}};
  const failurePacket=await collectCandidatePredictions(f.home,{candidateHash:registered.checkpoint,datasetVersion:built.dataset_version,store:f.store,clientFactory:()=>owned});
  assert.deepEqual(failurePacket.rows.map(x=>x.status),['timeout','rejected','error','unsupported_input']);
  assert.equal(closed,1);assert.ok(failurePacket.rows.every(x=>x.error&&x.elapsedMs>=0));
  const controller=new AbortController();
  const cancelling={close(){closed++;},async infer(){controller.abort();throw new ControlError('CANCELLED');}};
  await assert.rejects(collectCandidatePredictions(f.home,{candidateHash:registered.checkpoint,datasetVersion:built.dataset_version,store:f.store,clientFactory:()=>cancelling,signal:controller.signal}),/CANCELLED/);
  assert.equal(closed,2);
  const next=new AbortController();
  await assert.rejects(collectCandidatePredictions(f.home,{candidateHash:registered.checkpoint,datasetVersion:built.dataset_version,store:f.store,layaClient:{close(){closed++;},async infer(){next.abort();throw new ControlError('CANCELLED');}},signal:next.signal}),/CANCELLED/);
  assert.equal(closed,2,'injected clients remain owned by the caller');
  await assert.rejects(collectCandidatePredictions(f.home,{candidateHash:registered.checkpoint,datasetVersion:built.dataset_version,store:f.store,clientFactory:()=>{throw Error('must not construct an aborted run');},signal:controller.signal}),/CANCELLED/);
  fs.writeFileSync(path.join(registered.modelPath,'training_metadata.json'),JSON.stringify({exporter_version:'laya-typed-decisions-json-v3',export_dataset_version:digest('wrong dataset')}));
  await assert.rejects(collectCandidatePredictions(f.home,{candidateHash:registered.checkpoint,datasetVersion:built.dataset_version,store:f.store,clientFactory:()=>{throw Error('must validate identity before constructing client');}}),/TRAINING_IDENTITY_MISMATCH/);
});
