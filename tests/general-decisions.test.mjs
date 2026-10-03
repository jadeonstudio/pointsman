import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { RULES, oracle, generateCorpus, corpusReport, writeCorpus } from '../training/general-decisions/corpus.mjs';
import { freezeEvaluation, evaluatePredictions, comparePredictions } from '../training/general-decisions/evaluate.mjs';
import { createTrainingStore } from '../src/training/store.mjs';
import { readDataset } from '../src/training/dataset.mjs';
import { digest } from '../src/training/schema.mjs';

test('independent manually specified facts exercise priority, missing facts and counterfactual policies',()=>{
  const cases=[
    ['current-evidence',{same_scope:true,same_revision:true,fresh:true,retracted:false,contradicted:false,direct:true},'relevant'],
    ['current-evidence',{same_scope:true,same_revision:true,fresh:false,retracted:false,contradicted:true,direct:true},'irrelevant'],
    ['source-resolution',{primary_present:true,primary_current:true,secondary_present:true,secondary_current:true,disagree:true,primary_retracted:false},'use'],
    ['quoted-negative',{quoted_command:true,observation:false,counterexample:true,withdrawn:false,matches_claim:true,current:true},'relevant'],
    ['evidence-completeness',{required_check:true,check_passed:false,same_input:true,executed:true,source_independent:true,failure_reported:false},'refutes'],
    ['revision-branch',{input_changed:true,snapshot_exists:true,dependencies_ready:true,writer_idle:false,contract_known:true,output_current:true},'resnapshot'],
    ['lost-ack-branch',{mutation_sent:true,ack_received:false,readback_done:false,effect_exists:null,idempotent:true,retry_available:true},'readback'],
    ['candidate-recall',{matching_candidate:false,candidate_current:false,candidate_valid:false,missing_candidate:true,evidence_complete:true,candidates_conflict:false},'other'],
    ['dependency-barrier',{dependency_failed:true,dependency_pending:true,input_accepted:true,resource_free:true,budget_available:true,contract_valid:true},'repair'],
    ['failure-origin',{fixture_valid:false,runtime_started:false,same_contract:true,runtime_error:true,reproduced:true,dependency_available:true},'preparation'],
    ['failure-timeout',{deadline_expired:true,request_sent:true,readback_available:true,effect_confirmed:true,transport_failed:true,input_valid:true},'late_ack'],
    ['failure-contract',{input_admitted:true,question_matches:false,distribution_valid:true,result_received:true,provider_failed:true,schema_current:true},'output_contract'],
    ['failure-conflict',{sources_disagree:true,same_revision:true,same_scope:true,trusted_primary:false,tool_error:true,reproduced:true},'tool'],
    ['bounded-retry',{attempt_available:false,new_evidence:true,new_hypothesis:true,new_capability:true,same_failure:true,required_boundary:true},'escalate'],
    ['scope-boundary',{required_for_goal:true,within_scope:false,shared_contract_changed:false,existing_rule_sufficient:true,facts_complete:true,active_writer:false},'escalate'],
    ['acceptance-stop',{required_checks_passed:true,goal_met:true,new_failure:false,input_changed:false,writer_active:false,critical_risk:false},'stop'],
    ['writer-ownership',{writer_active:true,same_resource:true,snapshot_frozen:true,owner_known:true,conflict_detected:false,independent_work_ready:true},'wait'],
  ];
  for(const [id,facts,expected] of cases) assert.equal(RULES.find(r=>r.id===id).labels[oracle(id,facts)],expected,id);
  const facts={same_scope:true,same_revision:true,fresh:false,retracted:false,contradicted:null,direct:true};
  assert.equal(oracle('current-evidence',facts),0);
  assert.equal(oracle('current-evidence',facts,true),1,'same-state strict metadata policy must change gold');
  assert.throws(()=>oracle('current-evidence',{...facts,fresh:'unknown'}),/INVALID_ORACLE/);
});

test('corpus has honest variant counts and immutable family assignment when scaling',()=>{
  const small=generateCorpus(),large=generateCorpus({count:20000}),report=corpusReport(small);
  assert.deepEqual(small,generateCorpus());
  assert.equal(report.rows,2000); assert.equal(report.semantic_families,16);
  assert.equal(report.group_leakage,0); assert.equal(report.unreproducible_labels,0);
  for(const row of small) {
    const rule=RULES.find(r=>r.id===row.oracle.id);
    assert.equal(row.state.policy,row.state.language==='en'?rule.en:rule.ko,'full authored policy must survive question-head compaction');
    assert.deepEqual(Object.keys(row.state.facts),rule.fields,'all oracle facts must remain available');
    assert.ok(row.question.instructions.length<180,'policy text belongs in state, not Unicode-escaped question JSON');
  }
  assert.ok(report.distinct_structured_cases<report.rows,'render variants must not masquerade as independent cases');
  assert.deepEqual(new Set(Object.keys(report.split)),new Set(['train','dev','calibration','test']));
  assert.deepEqual(report.family_split,corpusReport(large).family_split);
  const changed=structuredClone(small), noul=changed.find(r=>r.question.type==='noul'); noul.target.value=!noul.target.value;
  assert.equal(corpusReport(changed).unreproducible_labels,1);
  const leaked=structuredClone(small); leaked[0].split=leaked[0].split==='train'?'test':'train';
  assert.ok(corpusReport(leaked).group_leakage>0);
  assert.throws(()=>freezeEvaluation(leaked),/DATASET_FAMILY_LEAKAGE/);
  const siblings=small.filter(r=>r.oracle.id==='current-evidence'&&digest(r.state)===digest(small[0].state));
  assert.equal(siblings.length,4);
  assert.notEqual(siblings[0].question.instructions,siblings[1].question.instructions);
  assert.notDeepEqual(siblings[0].question.criteria,siblings[2].question.criteria);
  assert.equal(siblings[0].question.criteria[siblings[0].target.value],siblings[2].question.criteria[siblings[2].target.value]);
});

test('canonical writer exports v3 metadata and hashes without fabricated runner events',t=>{
  const home=fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(),'pointsman-independent-test-'))); t.after(()=>fs.rmSync(home,{recursive:true,force:true}));
  const store=createTrainingStore({home}),result=writeCorpus(store,{count:2000});
  const {manifest,samples}=readDataset(store,result.dataset_version);
  assert.equal(manifest.source_kind,'independently_authored_oracle'); assert.equal(samples.length,2000);
  const exportManifest=JSON.parse(fs.readFileSync(result.export.files.find(x=>x.endsWith('manifest.json')),'utf8'));
  assert.equal(exportManifest.schema_version,3);
  const metadata=fs.readFileSync(result.export.files.find(x=>x.endsWith('metadata.jsonl')),'utf8');
  assert.equal(digest(metadata),exportManifest.metadata_sha256);
  assert.equal(store.scan().events.length,0,'objective authored rules are not trusted host executions');
  for(const row of samples) assert.equal(row.raw_refs.independent.source_id,row.lineage.source_id);
});

test('frozen model-neutral prediction join reports missing/rejected/malformed and semantic metrics',()=>{
  const rows=generateCorpus(), selected=rows.filter(r=>r.split==='test');
  const predictions=selected.map(r=>({sample_id:r.sample_id,model_id:'fixture-contract-only',checkpoint:'fixed-fixture',status:'ok',probabilities:r.target.probabilities}));
  const spec=freezeEvaluation(rows), perfect=evaluatePredictions(rows,predictions,{spec});
  assert.equal(perfect.metrics.full_envelope_accuracy,1); assert.equal(perfect.metrics.macro_f1,1);
  assert.equal(perfect.metrics.nll,0); assert.equal(perfect.metrics.brier,0);
  const missing=predictions.slice(1), partial=evaluatePredictions(rows,missing,{spec});
  assert.equal(partial.metrics.status.missing,1); assert.equal(partial.metrics.served_accuracy,1);
  assert.equal(partial.metrics.full_envelope_accuracy,(selected.length-1)/selected.length);
  const rejected=structuredClone(predictions); rejected[0].status='rejected'; rejected[1].probabilities={bad:1}; rejected[2].status='unsupported_input';
  const bad=evaluatePredictions(rows,rejected,{spec}); assert.equal(bad.metrics.status.rejected,1); assert.equal(bad.metrics.status.malformed,1); assert.equal(bad.metrics.status.unsupported_input,1);
  assert.throws(()=>evaluatePredictions(rows,[predictions[0],predictions[0]],{spec}),/INVALID_PREDICTION_JOIN/);
  assert.throws(()=>evaluatePredictions(rows,[{...predictions[0],sample_id:'foreign'}],{spec}),/INVALID_PREDICTION_JOIN/);
  assert.throws(()=>evaluatePredictions(rows,predictions,{spec:{...spec,expected_samples:1}}),/EVALUATION_SPEC_MISMATCH/);
  const comparison=comparePredictions(rows,predictions,predictions,{bootstrap:100});
  assert.deepEqual(comparison.bootstrap_95,[0,0]); assert.equal(comparison.status,'INCONCLUSIVE');
});
