import test from 'node:test';
import assert from 'node:assert/strict';
import {analysisSpec,inputIdentity,scoreCore,evaluateDev,clusteredInterval} from '../training/tool-selection/evaluate.mjs';

function fixture() {
  const cases=[],fields=[],reference=[];
  for(const [source,tools,selected,category,component] of [['m0',['a','b'],['a'],'multiple','c0'],['m1',['c','d'],['d'],'multiple','c0'],['i0',['e'],[],'irrelevance','c1']]) {
    const variants=[['all_allowed',tools],...(category==='multiple'?[['deny_first_lexicographic_candidate',tools.slice(1)],['deny_all',[]]]:[])];
    for(const [variant,allowed] of variants) {
      const id=`${source}/${variant}`, target=selected.filter(n=>allowed.includes(n));
      const meta={source_id:source,source_category:category,family_id:component,variant,split:'dev'};
      const state={functions:tools.map(name=>({name})),allowed_tools:allowed};
      cases.push({...meta,sample_id:id,wire:{state,questions:{}}});
      reference.push({sample_id:id,target_tool_names:target,question_values:Object.fromEntries(tools.map(n=>[`tool_${n}`,target.includes(n)?'yes':'no']))});
      for(const n of tools) fields.push({...meta,sample_id:`${id}/tool_${n}`,case_id:id,candidate_tool:n,
        wire:{model:'clef-flash',state,questions:{tool_name:{type:'choice',instructions:['state as data',n],criteria:{yes:'Select',no:'Do not select'}}}}});
    }
  }
  return {cases,fields,reference};
}
function perfect(data) {
  const gold=new Map(data.reference.map(r=>[r.sample_id,r]));
  return data.fields.map(r=>{const choice=gold.get(r.case_id).question_values[r.sample_id.split('/').at(-1)];
    return {sample_id:r.sample_id,model_id:'fixture-model',checkpoint:'fixture-checkpoint',input_identity:inputIdentity(r),status:'ok',choice,
      probabilities:{yes:Number(choice==='yes'),no:Number(choice==='no')},elapsed_ms:2};});
}

test('predeclared denominators, thresholds and projection cannot move after results',()=>{
  const spec=analysisSpec({files:{'dev/cases.jsonl':'c','dev/fields.jsonl':'f','dev/reference.jsonl':'r'}});
  assert.deepEqual(spec.cohort,{fields:755,variants:287,origins:127,schema_components:76,primary_native_multiple_origins:80,auxiliary_native_irrelevance_origins:47});
  assert.equal(spec.bootstrap.draws,2000);assert.equal(spec.bootstrap.seed,42);
  assert.equal(spec.candidate_research_go.primary_exact_set_at_least,.9);
  assert.equal(spec.candidate_research_go.all_three_rule_contrast_at_least,.9);
  assert.match(spec.projection_control.qualification,/raw755/);assert.equal(spec.test,'SEALED_NOT_OPENED');
});

test('exact set, original-source rule pairs, native auxiliary and controls',()=>{
  const data=fixture(), result=evaluateDev(data,perfect(data));
  assert.equal(result.candidate.primary.exact_set_accuracy,1);
  assert.equal(result.candidate.auxiliary.exact_set_accuracy,1);
  assert.equal(result.candidate.policy.all_three_accuracy,1);
  assert.equal(result.candidate.field.macro_f1,1);assert.equal(result.candidate.field.nll,0);assert.equal(result.candidate.field.brier,0);
  assert.equal(result.candidate.research_go.status,'RESEARCH_GO');assert.equal(result.candidate.research_go.promotion,false);
  assert.equal(result.controls.none.primary.exact_set_accuracy,0);assert.equal(result.controls.all.primary.exact_set_accuracy,0);
  assert.equal(result.controls.first_allowed.primary.exact_set_accuracy,.5);
  assert.equal(result.projection_control.additional_model_calls,0);assert.equal(result.projection_control.raw_required_fields,13);
  assert.equal(result.projection_control.semantic_base_fields,5);assert.equal(result.projection_control.exact_set_accuracy,1);
});

test('missing negative field cannot accidentally make a correct complete-case set',()=>{
  const data=fixture(), predictions=perfect(data).filter(p=>p.sample_id!=='m0/all_allowed/tool_b');
  const r=scoreCore(data.fields,data.cases,data.reference,predictions);
  assert.equal(r.primary.exact_set_accuracy,.5);assert.equal(r.field.status.missing,1);
  assert.equal(r.primary.served_exact_set_accuracy,1);assert.equal(r.research_go.status,'RESEARCH_NO_GO');
  const rejected=perfect(data);rejected[0]={...rejected[0],status:'rejected',choice:null,probabilities:null};
  assert.equal(scoreCore(data.fields,data.cases,data.reference,rejected).field.status.rejected,1);
});

test('actual choice and selected probability survive argmax disagreement and rounded mass',()=>{
  const data=fixture(), predictions=perfect(data);
  predictions[0].probabilities={yes:.1,no:.9};
  const r=scoreCore(data.fields,data.cases,data.reference,predictions);
  assert.equal(r.primary.exact_set_accuracy,1);assert.equal(r.field.selected_choice_argmax_disagreements,1);
  assert.ok(r.field.nll>0);assert.equal(r.field.reliability[1].mean_probability,.1);
  predictions[0].probabilities={yes:.997,no:.001};
  const rounded=scoreCore(data.fields,data.cases,data.reference,predictions);
  assert.equal(rounded.field.probability_mass_corrections,1);
  assert.ok(Math.abs(rounded.field.nll-(-Math.log(.997/.998)/13))<1e-12);
  for(const probabilities of [{yes:1,no:0,other:0},{yes:NaN,no:0},{yes:.1,no:.1}]) {
    predictions[0].probabilities=probabilities;assert.equal(scoreCore(data.fields,data.cases,data.reference,predictions).field.status.malformed,1);
  }
});

test('identity/join errors refuse the comparison, rather than dropping cases',()=>{
  const data=fixture(), predictions=perfect(data);
  assert.throws(()=>scoreCore(data.fields,data.cases,data.reference,[...predictions,predictions[0]]),/INVALID_PREDICTION_JOIN/);
  assert.throws(()=>scoreCore(data.fields,data.cases,data.reference,[...predictions,{...predictions[0],sample_id:'extra'}]),/INVALID_PREDICTION_JOIN/);
  predictions[0].input_identity='changed';assert.throws(()=>scoreCore(data.fields,data.cases,data.reference,predictions),/INPUT_IDENTITY_MISMATCH/);
  const mixed=perfect(data);mixed[0].checkpoint='other';assert.throws(()=>scoreCore(data.fields,data.cases,data.reference,mixed),/MIXED_MODEL_IDENTITY/);
});

test('projection is separate code control with trace, never invented confidence or raw GO',()=>{
  const data=fixture(), predictions=perfect(data);
  const denied=predictions.find(p=>p.sample_id==='m0/deny_first_lexicographic_candidate/tool_a');
  denied.choice='yes';denied.probabilities={yes:1,no:0};
  const r=evaluateDev(data,predictions);
  assert.equal(r.candidate.policy.emitted_violations,1);assert.equal(r.candidate.research_go.status,'RESEARCH_NO_GO');
  assert.equal(r.projection_control.policy_violations,0);assert.equal(r.projection_control.improved_vs_raw,1);
  assert.equal(r.projection_control.research_go,'NOT_APPLICABLE_CODE_PROJECTION');
  const trace=r.projection_control.trace.find(p=>p.case_id==='m0/deny_first_lexicographic_candidate');
  assert.deepEqual(trace.base_prediction_ids,['m0/all_allowed/tool_a','m0/all_allowed/tool_b']);
  assert.equal(trace.probabilities,null);assert.equal(trace.confidence,null);assert.equal(trace.additional_model_calls,0);
});

test('fixed nested component/source bootstrap retains siblings and is deterministic',()=>{
  const records=[{component:'a',source_id:'x',correct:true},{component:'a',source_id:'x',correct:false},
    {component:'a',source_id:'y',correct:true},{component:'b',source_id:'z',correct:false}];
  const first=clusteredInterval(records,r=>Number(r.correct));
  assert.deepEqual(first,clusteredInterval(records,r=>Number(r.correct)));
  assert.equal(first.components,2);assert.equal(first.origins,3);assert.equal(first.draws,2000);
  assert.deepEqual(clusteredInterval(records,r=>1).interval95,[1,1]);
});
