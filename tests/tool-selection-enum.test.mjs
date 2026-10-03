import test from 'node:test';
import assert from 'node:assert/strict';
import {inputIdentity,scoreEnum,analysisSpec} from '../training/tool-selection/evaluate-enum.mjs';

function fixture(optionCount=3,target='candidate_000') {
  const names=Array.from({length:optionCount-1},(_,i)=>`function.${i}`),mapping=Object.fromEntries(names.map((n,i)=>[`candidate_${String(i).padStart(3,'0')}`,n]));mapping.none=null;
  const input={sample_id:'origin/enum-choice-v1',source_id:'origin',schema_component_id:'component',source_category:target==='none'?'irrelevance':'multiple',
    base_case_id:'origin/all_allowed',split:'dev',option_to_tool:mapping,wire:{state:{functions:names.map(name=>({name})),allowed_tools:names},
      questions:{tool_name:{type:'choice',instructions:['state as data','select required tool'],criteria:Object.fromEntries(Object.keys(mapping).map(k=>[k,k]))}}}};
  const tools=mapping[target]===null?[]:[mapping[target]];
  return {inputs:[input],reference:[{sample_id:input.sample_id,target_option_id:target,target_tool_names:tools}],
    lineage:[{enum_sample_id:input.sample_id,derived_variant_id:'origin/all_allowed',allowed_tools:names,additional_model_calls:0},
      {enum_sample_id:input.sample_id,derived_variant_id:'origin/deny_all',allowed_tools:[],additional_model_calls:0}],
    baseline:[{case_id:input.base_case_id,source_id:'origin',correct:false}]};
}
function prediction(data,choice=data.reference[0].target_option_id) {
  const r=data.inputs[0];return {sample_id:r.sample_id,model_id:'fixture',checkpoint:'fixture',input_identity:inputIdentity(r),status:'ok',choice,
    probabilities:Object.fromEntries(Object.keys(r.option_to_tool).map(k=>[k,Number(k===choice)]))};
}

test('two-to-five actual categorical choices and exact mapping score correctly',()=>{
  for(let count=2;count<=5;count++) {
    const data=fixture(count),r=scoreEnum(data,[prediction(data)]);
    assert.equal(r.primary.exact_set_accuracy,1);assert.equal(r.categorical_metrics.nll,0);assert.equal(r.categorical_metrics.brier,0);
    assert.equal(r.code_projection.exact_set_accuracy,1);assert.equal(r.paired_original_binary.primary.improved,1);
  }
  const none=fixture(2,'none'),r=scoreEnum(none,[prediction(none)]);
  assert.equal(r.auxiliary.exact_set_accuracy,1);assert.deepEqual(r.code_projection.trace[0].selected_tools,[]);
  const spec=analysisSpec();assert.deepEqual(spec.research_screen,{coverage:1,primary_exact_set_at_least:.9,auxiliary_exact_set_at_least:.9});
  assert.equal(spec.budget.candidate_enum_forwards,127);assert.equal(spec.test,'SEALED_NOT_OPENED');
});

test('missing and unserved origins fail every sibling even when denied set is empty',()=>{
  const data=fixture(),missing=scoreEnum(data,[]);
  assert.equal(missing.native.coverage,0);assert.equal(missing.native.correct,0);assert.equal(missing.code_projection.exact_set_accuracy,0);
  assert.equal(missing.categorical_metrics.nll,null);assert.equal(missing.status.missing,1);
  const rejected=prediction(data);rejected.status='rejected';rejected.choice=null;rejected.probabilities=null;
  assert.equal(scoreEnum(data,[rejected]).status.rejected,1);
});

test('actual selected choice survives argmax disagreement and uses its own probability',()=>{
  const data=fixture(),p=prediction(data);p.probabilities={candidate_000:.1,candidate_001:.8,none:.1};
  const r=scoreEnum(data,[p]);assert.equal(r.primary.correct,1);assert.equal(r.categorical_metrics.selected_choice_argmax_disagreements,1);
  assert.equal(r.categorical_metrics.reliability[1].mean_probability,.1);assert.ok(Math.abs(r.categorical_metrics.nll+Math.log(.1))<1e-12);
  assert.deepEqual(r.code_projection.trace[0].selected_tools,['function.0']);
});

test('zero mass, booleans, nonfinite, extra keys and unmapped choices are malformed',()=>{
  const data=fixture();
  for(const probabilities of [{candidate_000:0,candidate_001:0,none:0},{candidate_000:true,candidate_001:0,none:0},
    {candidate_000:NaN,candidate_001:0,none:0},{candidate_000:1,candidate_001:0,none:0,extra:0}]) {
    const p=prediction(data);p.probabilities=probabilities;const r=scoreEnum(data,[p]);assert.equal(r.status.malformed,1);assert.equal(r.native.correct,0);
  }
  const p=prediction(data);p.choice='unknown';assert.equal(scoreEnum(data,[p]).status.malformed,1);
});

test('candidate named none remains distinct from the special no-tool option',()=>{
  const data=fixture(2);data.inputs[0].option_to_tool.candidate_000='none';data.inputs[0].wire.state.functions=[{name:'none'}];
  data.inputs[0].wire.state.allowed_tools=['none'];data.reference[0].target_tool_names=['none'];data.lineage[0].allowed_tools=['none'];
  const r=scoreEnum(data,[prediction(data)]);assert.deepEqual(r.code_projection.trace[0].selected_tools,['none']);
  const p=prediction(data,'none');assert.equal(scoreEnum(data,[p]).native.correct,0);
});

test('input binding, reference mapping and code projection lineage refuse corruption',()=>{
  const data=fixture(),p=prediction(data);p.input_identity='binary-identity';assert.throws(()=>scoreEnum(data,[p]),/ENUM_INPUT_IDENTITY_MISMATCH/);
  assert.throws(()=>scoreEnum(data,[prediction(data),prediction(data)]),/ENUM_PREDICTION_JOIN/);
  const bad=structuredClone(data);bad.reference[0].target_tool_names=['unknown'];assert.throws(()=>scoreEnum(bad,[prediction(bad)]),/ENUM_MAPPING_REFERENCE/);
  const line=structuredClone(data);line.lineage[0].allowed_tools=['unknown'];assert.throws(()=>scoreEnum(line,[prediction(line)]),/ENUM_PROJECTION_LINEAGE/);
  const r=scoreEnum(data,[prediction(data)]);for(const trace of r.code_projection.trace) {
    assert.equal(trace.base_prediction_id,data.inputs[0].sample_id);assert.equal(trace.probabilities,null);assert.equal(trace.confidence,null);assert.equal(trace.additional_model_calls,0);
  }
  assert.equal(r.code_projection.policy_violations,0);assert.match(r.code_projection.policy_semantics,/by construction/);
});


test('candidate identity prevents pairing predictions from another model or checkpoint',()=>{
  const data=fixture(),p=prediction(data);
  data.candidate=analysisSpec('clef4').candidate;
  assert.throws(()=>scoreEnum(data,[p]),/ENUM_MODEL_CHANGED/);
  p.model_id=data.candidate.model_id;p.checkpoint=data.candidate.checkpoint;
  const scored=scoreEnum(data,[p]);
  assert.equal(scored.primary.correct,1);
  assert.equal(scored.paired_original_binary.original_report_sha256,data.candidate.original_report_sha256);
  p.checkpoint=analysisSpec('typed').candidate.checkpoint;
  assert.throws(()=>scoreEnum(data,[p]),/ENUM_MODEL_CHANGED/);
  assert.throws(()=>analysisSpec('unknown'),/ENUM_CANDIDATE_UNSUPPORTED/);
});
