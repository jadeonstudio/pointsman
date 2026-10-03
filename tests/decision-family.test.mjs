import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { DEFAULTS } from '../src/constants.mjs';
import { decisionScope, validateProviderConfig, normalizeInference, MAX_PROVIDER_BYTES } from '../src/inference.mjs';
import { familyQualification, promoteCandidate, adoptCandidate, rollbackLaya } from '../src/training/laya-lifecycle.mjs';
import { digest } from '../src/training/schema.mjs';
import { ROUTE_QUESTIONS } from '../src/routing.mjs';
import { atomicWrite } from '../src/storage.mjs';

const laya = () => ({ python: '/usr/bin/python3', modelPath: '/private/tmp', model: 'laya/fixture', checkpoint: 'a'.repeat(64), runtimeVersion: '0.3.4', device: 'cpu', precision: 'fp32', inputFit: 'lossless' });
const request = () => ({ purpose: 'select', risk: 'routine', state: { policy: 'Use supported evidence.', facts: { observed: true } }, questions: { decision: { type: 'choice', instructions: 'Apply state.policy.', criteria: { yes: 'supported', no: 'unsupported' } } } });
const scope = r => { const actual=decisionScope(r,'decision'); return { ...actual, familyId: 'evidence-relevance', familyRevision: 'rule-constructor-v1', stateBuilderRevision: `payload-state-schema-v1:${actual.stateSchemaHash}`, threshold: .9 }; };
const qualification = (l,r) => ({ checkpoint: l.checkpoint, calibrationVersion: 'cal-v1', purposes: [r.purpose], minConfidence: .9, minChoiceProbability: .9, noulCertainty: .9,
  decisionIdentity: { version: 1, checkpoint: l.checkpoint, runtimeVersion: l.runtimeVersion, precision: l.precision, inputFit: l.inputFit, calibrationVersion: 'cal-v1', scopes: [scope(r)] } });
function raw(l,r) { return { identity: { model:l.model, checkpoint:l.checkpoint, runtime_version:l.runtimeVersion, device:l.device, precision:'torch.float32' },
  answers:Object.fromEntries(Object.entries(r.questions).map(([name,q]) => [name,q.type === 'choice' ? {type:'choice',choice:Object.keys(q.criteria)[0],confidence:.99,probabilities:Object.fromEntries(Object.keys(q.criteria).map((k,i)=>[k,i?0:1]))} : {type:'score',score:0,confidence:.99,probabilities:Object.fromEntries(q.criteria.map((_,i)=>[i,i?0:1]))}])) }; }
function temp(t) { const dir=fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(),'pointsman-family-'))); t.after(()=>fs.rmSync(dir,{recursive:true,force:true})); return dir; }

test('runtime binds actual question/order/state policy/schema; a family label is not sufficient', () => {
  const r=request(), l=laya(); l.qualification=qualification(l,r);
  const normalize=q=>normalizeInference('laya',raw(l,q),q,DEFAULTS,{laya:l});
  assert.equal(normalize(r).qualified,true); assert.equal(normalize(r).eligible,true);
  for(const mutate of [q=>{q.questions.decision.instructions+=' Other task.';},q=>{q.questions.decision.criteria={no:'unsupported',yes:'supported'};},q=>{q.state.policy='Different rule';},q=>{q.state.facts.observed='yes';}]) {
    const changed=structuredClone(r); mutate(changed); assert.equal(normalize(changed).qualified,false);
  }
  const nullable=structuredClone(r); nullable.state.facts.observed=null; assert.equal(normalize(nullable).qualified,true);
  const unrelated=request(); unrelated.questions.second=structuredClone(unrelated.questions.decision); assert.equal(normalize(unrelated).qualified,false);
});
test('legacy canonical route/effort retains eligibility; unrelated legacy same-purpose is unqualified', () => {
  const l=laya(); l.qualification={checkpoint:l.checkpoint,calibrationVersion:'legacy',purposes:['route','select'],minConfidence:.9,minChoiceProbability:.9,noulCertainty:.9};
  const route={purpose:'route',risk:'routine',state:{task:'Explain this code.',context:{complete:true,exhaustive:false,highImpact:false,modelLocked:false,previousFailures:0,scope:'local'}},questions:structuredClone(ROUTE_QUESTIONS)};
  assert.equal(normalizeInference('laya',raw(l,route),route,DEFAULTS,{laya:l}).qualified,true);
  // A future shared prompt edit must not silently enlarge an already-issued legacy qualification.
  const original = ROUTE_QUESTIONS.intent.instructions;
  try {
    ROUTE_QUESTIONS.intent.instructions += ' Changed future prompt.';
    const changed = { ...route, questions: structuredClone(ROUTE_QUESTIONS) };
    assert.equal(normalizeInference('laya',raw(l,changed),changed,DEFAULTS,{laya:l}).qualified,false);
    assert.equal(normalizeInference('laya',raw(l,route),route,DEFAULTS,{laya:l}).qualified,true);
  } finally { ROUTE_QUESTIONS.intent.instructions = original; }
  const generic=request(); generic.purpose='route'; assert.equal(normalizeInference('laya',raw(l,generic),generic,DEFAULTS,{laya:l}).qualified,false);
  generic.purpose='select'; assert.equal(normalizeInference('laya',raw(l,generic),generic,DEFAULTS,{laya:l}).qualified,false);
});
test('provider rejects changed proof identities, duplicated or oversized scopes and oversized serialization', () => {
  const r=request(), base=laya(); base.qualification=qualification(base,r);
  const valid=()=>({version:1,provider:'laya',laya:structuredClone(base)});
  validateProviderConfig(valid());
  for(const key of ['checkpoint','runtimeVersion','precision','inputFit','calibrationVersion']) { const c=valid(); c.laya.qualification.decisionIdentity[key]='changed'; assert.throws(()=>validateProviderConfig(c),/INVALID_PROVIDER_CONFIG/); }
  const duplicate=valid(); duplicate.laya.qualification.decisionIdentity.scopes.push(scope(r)); assert.throws(()=>validateProviderConfig(duplicate));
  const tooMany=valid(); tooMany.laya.qualification.decisionIdentity.scopes=Array.from({length:129},()=>scope(r)); assert.throws(()=>validateProviderConfig(tooMany));
  const tooLarge=valid(); tooLarge.laya.modelPath='/'+ 'x'.repeat(MAX_PROVIDER_BYTES); assert.throws(()=>validateProviderConfig(tooLarge),/INVALID_PROVIDER_CONFIG/);
});
test('deployment family gate spans independent semantic groups and question type; spec revisions must match', t => {
  const root=temp(t), version=digest('dataset'); fs.mkdirSync(path.join(root,'datasets',version),{recursive:true});
  const source='f'.repeat(64), spec={revision:'rules-v1',generator_sha256:source,rules:[{id:'rule-a',task:'evidence-relevance'},{id:'rule-b',task:'evidence-relevance'},{id:'rule-c',task:'next-branch'}]};
  atomicWrite(path.join(root,'datasets',version,'oracle_specs.json'),JSON.stringify(spec));
  const row=(id,split)=>({purpose:'select',split,question_id:'decision',question:request().questions.decision,state:{...request().state,policy:id},
    lineage:{semantic_family_id:id},oracle:{id,revision:'rules-v1'},data_rights:{revision:source},provenance:[{provider:'host',model:'independent-rule-oracle',checkpoint:source,preprocessing_version:'facts-v1'}]});
  const samples=[row('rule-a','calibration'),row('rule-b','test'),row('rule-c','test')], holdout=[row('rule-b','test')];
  const good={metric:.99,correct:true,refused:false,label_source:'objective'};
  const records={calibration:[good],test:[good,good],holdout:[good]}, params={targetAccuracy:.8,minCoverage:.5,minCalibration:1,minTest:1,minLowerBound:0,minThreshold:.5};
  const result=familyQualification({root},version,samples,holdout,records,params);
  assert.equal(result.scopes.length,2); assert(result.scopes.every(s=>s.familyId==='evidence-relevance'));
  assert.deepEqual(result.evidence.families['evidence-relevance:choice'].independentSemanticGroups,{calibration:['rule-a'],test:['rule-b'],holdout:['rule-b']});
  assert.equal(result.evidence.families['next-branch:choice'].passed,false);
  samples[0].oracle.revision='changed'; assert.throws(()=>familyQualification({root},version,samples,holdout,records,params),/DECISION_CONSTRUCTOR_IDENTITY_MISMATCH/);
  assert.equal(familyQualification({root},digest('absent'),[],[],records,params).evidence.status,'UNQUALIFIED_GENERIC');
});
test('promote/adopt carry proof and rollback restores prior metadata; empty operational envelope cannot promote', t => {
  const home=temp(t), candidate=laya(); candidate.modelPath=home;
  const r=request(), q={...qualification(candidate,r),qualified:true,operationalEligible:true,
    familyQualification:{status:'BOUNDED_ENVELOPE_QUALIFIED',families:{'evidence-relevance:choice':{passed:true}}},holdout:'h',holdout_sha256:digest('holdout')};
  atomicWrite(path.join(home,'laya/candidates',`${candidate.checkpoint}.json`),JSON.stringify(candidate));
  atomicWrite(path.join(home,'laya/qualifications',`${candidate.checkpoint}.json`),JSON.stringify(q));
  atomicWrite(path.join(home,'laya/comparisons',`none__${candidate.checkpoint}__h.json`),JSON.stringify({candidate:candidate.checkpoint,active:null,holdout:'h',holdout_sha256:q.holdout_sha256,
    decision_identity_sha256:digest(q.decisionIdentity),purposes:{select:{candidate:{raw:{n:10,accuracy:1}}}}}));
  const promoted=promoteCandidate(home,{candidateHash:candidate.checkpoint,holdoutName:'h'}); assert.equal(promoted.promoted,true);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(home,'providers.json'))).laya.qualification.decisionIdentity,q.decisionIdentity);
  rollbackLaya(home); assert.equal(JSON.parse(fs.readFileSync(path.join(home,'providers.json'))).laya,null);
  atomicWrite(path.join(home,'laya/published',`${candidate.checkpoint}.json`),JSON.stringify({repo:'fixture/model',revision:'b'.repeat(40),qualification:q}));
  assert.equal(adoptCandidate(home,{candidateHash:candidate.checkpoint}).adopted,true);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(home,'providers.json'))).laya.qualification.decisionIdentity,q.decisionIdentity);
  rollbackLaya(home);
  q.decisionIdentity.scopes=[];
  atomicWrite(path.join(home,'laya/qualifications',`${candidate.checkpoint}.json`),JSON.stringify(q));
  assert.deepEqual(promoteCandidate(home,{candidateHash:candidate.checkpoint,holdoutName:'h'}).violations,['NO_QUALIFIED_DECISION_FAMILY']);
  atomicWrite(path.join(home,'laya/published',`${candidate.checkpoint}.json`),JSON.stringify({qualification:q}));
  assert.throws(()=>adoptCandidate(home,{candidateHash:candidate.checkpoint}),/NO_QUALIFIED_DECISION_FAMILY/);
});
