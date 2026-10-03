import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {digest,encode} from '../../src/training/schema.mjs';
import {clusteredInterval} from './evaluate.mjs';

export const REVISION='bfcl-dev-enum-evaluation-v2';
const FROZEN={'blind-inputs.jsonl':'b755a9364dd91fccc4206a8dbe0ff6789ee1c5e9fbb2fc75594ac163f6fd2977',
  'reference.jsonl':'796938193d4d0632f2aa47d721089812a0aaed996068cbc2af9f808e034df457',
  'projection-lineage.jsonl':'0a47bc81df276b7e1cbc761db259af51139c0db944e8c04e40ebbaadba019217',
  'specification.json':'ea825ab3e9043bfc559013855692b1cf8a71c55c987b1228eeffd67af209067d'};
const CANDIDATES={
  typed:{model_id:'tool-selection/typed',checkpoint:'56f6474957ea3e5660562efd7e588ee631045a3c6ff557934787ac252fd5350c',
    original_report_sha256:'55f64b16b904a26aabeeb419c4a07bd9632ff2b9d943c51ab6230c58eb4708a8'},
  clef4:{model_id:'tool-selection/clef4',checkpoint:'ef744cdfbba595a68a082368f047d6472a6e5ef576c97c074fb15d3291001b97',
    original_report_sha256:'451ecba34cefe67a2bb1d629ec4c4d121f42910aac8346ad4690e8824a53595f'}
};
function candidate(name) {if(!Object.hasOwn(CANDIDATES,name)) throw new Error('ENUM_CANDIDATE_UNSUPPORTED');return CANDIDATES[name];}
const BINARY_SOURCE='a645a289198e94788add25e187c29ec92e39159dd8b603e8d0db165518e25dd2';
const STATUSES=['ok','timeout','rejected','unsupported_input','error'];
const mean=xs=>xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:null;
const fraction=(n,d)=>d?n/d:null;
const same=(a,b)=>encode([...a].sort())===encode([...b].sort());
const rows=p=>fs.readFileSync(p,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
export const inputIdentity=r=>digest({state:r.wire.state,questions:r.wire.questions});

export function loadPrototype(root,original,name='typed') {
  const selected=candidate(name);
  for(const [name,hash] of Object.entries(FROZEN)) if(digest(fs.readFileSync(path.join(root,name),'utf8'))!==hash) throw new Error('ENUM_ARTIFACT_CHANGED');
  if(digest(fs.readFileSync(original,'utf8'))!==selected.original_report_sha256) throw new Error('ORIGINAL_MODEL_REPORT_CHANGED');
  if(digest(fs.readFileSync(new URL('./evaluate.mjs',import.meta.url),'utf8'))!==BINARY_SOURCE) throw new Error('BINARY_SCORER_CHANGED');
  const inputs=rows(path.join(root,'blind-inputs.jsonl')),reference=rows(path.join(root,'reference.jsonl')),lineage=rows(path.join(root,'projection-lineage.jsonl'));
  if(inputs.length!==127||reference.length!==127||lineage.length!==287||new Set(inputs.map(r=>r.source_id)).size!==127
    ||new Set(inputs.map(r=>r.schema_component_id)).size!==76||inputs.some(r=>r.split!=='dev')
    ||inputs.filter(r=>r.source_category==='multiple').length!==80) throw new Error('ENUM_DEV_COHORT_CHANGED');
  const baseline=JSON.parse(fs.readFileSync(original,'utf8')).projection_control.trace.filter(r=>r.case_id===r.base_case_id);
  if(baseline.length!==127) throw new Error('ORIGINAL_NATIVE_OUTCOMES_MISSING');
  return {inputs,reference,lineage,baseline,candidate:selected};
}

export function analysisSpec(name='typed') {
  const selected=candidate(name);
  return {revision:REVISION,scope:'DEV enum127 ONLY:80primary multiple/47auxiliary irrelevance; code projection287',
    scorer_sha256:digest(fs.readFileSync(fileURLToPath(import.meta.url),'utf8')),prototype_hashes:FROZEN,
    frozen_binary_scorer_sha256:BINARY_SOURCE,candidate:selected,
    prediction:{required:['sample_id','model_id','checkpoint','input_identity','status'],statuses:STATUSES,
      input_identity:'SHA256 encode({state:wire.state,questions:wire.questions}); never binary identity',
      served:'actual choice must belong to per-row criteria; probabilities exact same keys, finite numbers[0,1],positive mass,abs(sum-1)<=0.02; normalize mass only',
      confidence:'normalized actual selected enum-choice probability; preserve choice/argmax disagreement; no provider-confidence substitution'},
    denominators:{native:'all127; every missing/malformed/unserved case is wrong',primary:80,auxiliary:47,
      probability_metrics:'valid served enum cases only, expose count; categorical NLLfloor1e-15/Brier sum over actual options/ECE10equal bins; separate from binary loss space',
      projection:'all287; valid enum singleton/empty intersect each allowed_tools with no substitution; unserved origin fails every sibling; no derived probabilities/confidence'},
    paired:'Same-origin native correctness against the same model\'s original binary fixed outcomes; original all_allowed code-projection trace is unchanged from raw binary; no original report rewrite',
    bootstrap:{draws:2000,seed:42,reuse:'frozen clusteredInterval component→source nested resampling; origin weight1; all derived siblings remain in source/component cluster'},
    research_screen:{coverage:1,primary_exact_set_at_least:.9,auxiliary_exact_set_at_least:.9},
    screen_scope:'ENUM_RESEARCH_SCREEN only; code policy0 by construction is not raw learned-rule gate, A/B/C, task completion or promotion; no fitted thresholds/calibration/test selection',
    controls:['none','first_lexicographic_candidate'],
    budget:{candidate_enum_forwards:127,other_candidate_forwards:0,projection_additional_calls:0,
      execution:'Separate root acceptance required; prior closed 882-call experiment is not repeated'},
    temperature:'Existing candidate configuration unchanged; no tuning or calibration fitting',
    test:'SEALED_NOT_OPENED',pretraining_contamination:'UNKNOWN',model_execution:'CLOSED until root accepts fresh spec/producer manifest'};
}

export function scoreEnum(data,predictions) {
  const byInput=new Map(data.inputs.map(r=>[r.sample_id,r])),byReference=new Map(data.reference.map(r=>[r.sample_id,r])),byPrediction=new Map();
  if(byInput.size!==data.inputs.length||byReference.size!==data.reference.length||byInput.size!==byReference.size||[...byReference.keys()].some(k=>!byInput.has(k))) throw new Error('ENUM_REFERENCE_JOIN');
  for(const p of predictions) {
    if(!p||!byInput.has(p.sample_id)||byPrediction.has(p.sample_id)||!STATUSES.includes(p.status)
      ||![p.model_id,p.checkpoint,p.input_identity].every(v=>typeof v==='string'&&v.length)) throw new Error('ENUM_PREDICTION_JOIN');
    if(p.input_identity!==inputIdentity(byInput.get(p.sample_id))) throw new Error('ENUM_INPUT_IDENTITY_MISMATCH');
    if(data.candidate&&(p.model_id!==data.candidate.model_id||p.checkpoint!==data.candidate.checkpoint)) throw new Error('ENUM_MODEL_CHANGED');
    byPrediction.set(p.sample_id,p);
  }
  if(new Set(predictions.map(p=>encode([p.model_id,p.checkpoint]))).size>1) throw new Error('MIXED_ENUM_MODEL');
  const records=data.inputs.map(row=>{
    const ref=byReference.get(row.sample_id),q=row.wire.questions.tool_name,keys=Object.keys(q.criteria),mapping=row.option_to_tool,p=byPrediction.get(row.sample_id);
    if(q.type!=='choice'||keys.length<2||keys.length>5||!same(keys,Object.keys(mapping))||mapping.none!==null||!keys.includes(ref.target_option_id)
      ||ref.target_tool_names.length>1||!same(ref.target_tool_names,mapping[ref.target_option_id]===null?[]:[mapping[ref.target_option_id]])) throw new Error('ENUM_MAPPING_REFERENCE');
    const offered=row.wire.state.functions.map(f=>f.name),named=Object.values(mapping).filter(v=>v!==null);
    if(new Set(named).size!==named.length||!same(named,offered)) throw new Error('ENUM_CANDIDATE_MAPPING');
    const base={sample_id:row.sample_id,base_case_id:row.base_case_id,source_id:row.source_id,component:row.schema_component_id,category:row.source_category,
      status:p?.status??'missing',correct:false,selected_tools:null,target_tools:ref.target_tool_names};
    if(base.status!=='ok') return base;
    const raw=p.probabilities;
    if(!keys.includes(p.choice)||!raw||!same(Object.keys(raw),keys)||keys.some(k=>typeof raw[k]!=='number'||!Number.isFinite(raw[k])||raw[k]<0||raw[k]>1)) return {...base,status:'malformed'};
    const mass=keys.reduce((n,k)=>n+raw[k],0),mass_error=Math.abs(mass-1);
    if(mass<=0||mass_error>.02) return {...base,status:'malformed'};
    const probs=Object.fromEntries(keys.map(k=>[k,raw[k]/mass])),selected=mapping[p.choice];
    return {...base,choice:p.choice,correct:p.choice===ref.target_option_id,selected_tools:selected===null?[]:[selected],confidence:probs[p.choice],mass_error,
      argmax_disagreement:probs[p.choice]<Math.max(...Object.values(probs)),nll:-Math.log(Math.max(1e-15,probs[ref.target_option_id])),
      brier:keys.reduce((n,k)=>n+(probs[k]-Number(k===ref.target_option_id))**2,0)};
  });
  const served=records.filter(r=>r.status==='ok'),summary=rs=>({count:rs.length,served:rs.filter(r=>r.status==='ok').length,
    coverage:fraction(rs.filter(r=>r.status==='ok').length,rs.length),correct:rs.filter(r=>r.correct).length,
    exact_set_accuracy:mean(rs.map(r=>Number(r.correct))),conditional_served_accuracy:mean(rs.filter(r=>r.status==='ok').map(r=>Number(r.correct)))});
  const primary=records.filter(r=>r.category==='multiple'),auxiliary=records.filter(r=>r.category==='irrelevance');
  const reliability=Array.from({length:10},(_,bin)=>{const rs=served.filter(r=>Math.min(9,Math.floor(r.confidence*10))===bin);
    return {count:rs.length,mean_probability:mean(rs.map(r=>r.confidence)),accuracy:mean(rs.map(r=>Number(r.correct)))};});
  const byRecord=new Map(records.map(r=>[r.sample_id,r])),projected=data.lineage.map(line=>{
    const base=byRecord.get(line.enum_sample_id),row=byInput.get(line.enum_sample_id);
    if(!base||!Array.isArray(line.allowed_tools)||line.allowed_tools.some(n=>!Object.values(row.option_to_tool).includes(n))||line.additional_model_calls!==0) throw new Error('ENUM_PROJECTION_LINEAGE');
    const selected=base.selected_tools?.filter(n=>line.allowed_tools.includes(n)),target=base.target_tools.filter(n=>line.allowed_tools.includes(n));
    return {source_id:base.source_id,component:base.component,case_id:line.derived_variant_id,base_prediction_id:base.sample_id,status:base.status,
      correct:base.status==='ok'&&same(selected,target),selected_tools:selected??null,probabilities:null,confidence:null,additional_model_calls:0};
  });
  if(new Set(projected.map(r=>r.case_id)).size!==projected.length) throw new Error('DUPLICATE_PROJECTION_VARIANT');
  const baseline=new Map(data.baseline.map(r=>[r.case_id,r]));
  const pair=rs=>{const paired=rs.map(r=>{const old=baseline.get(r.base_case_id);if(!old||old.source_id!==r.source_id) throw new Error('ORIGINAL_MODEL_NATIVE_JOIN');
    return {...r,previous_correct:old.correct,difference:Number(r.correct)-Number(old.correct)};});
    return {count:paired.length,improved:paired.filter(r=>r.difference===1).length,regressed:paired.filter(r=>r.difference===-1).length,
      accuracy_difference:mean(paired.map(r=>r.difference)),interval95:clusteredInterval(paired,r=>r.difference)};};
  const coverage=summary(records).coverage,conditions={coverage:coverage===1,primary:primary.length>0&&summary(primary).exact_set_accuracy>=.9,
    auxiliary:auxiliary.length>0&&summary(auxiliary).exact_set_accuracy>=.9};
  return {revision:REVISION,scope:'DEV ENUM ONLY',model_identity:predictions.length?{model_id:predictions[0].model_id,checkpoint:predictions[0].checkpoint}:null,
    native:summary(records),primary:summary(primary),auxiliary:summary(auxiliary),
    status:Object.fromEntries([...new Set(records.map(r=>r.status))].sort().map(k=>[k,records.filter(r=>r.status===k).length])),
    categorical_metrics:{served_cases:served.length,nll:mean(served.map(r=>r.nll)),brier:mean(served.map(r=>r.brier)),reliability,
      ece:served.length?reliability.reduce((n,b)=>n+(b.count?b.count*Math.abs(b.accuracy-b.mean_probability):0),0)/served.length:null,
      selected_choice_argmax_disagreements:served.filter(r=>r.argmax_disagreement).length,probability_mass_corrections:served.filter(r=>r.mass_error>1e-12).length,
      comparison:'Categorical option-space losses; not binary loss-space equivalents'},
    intervals:{native:clusteredInterval(records,r=>Number(r.correct)),primary:clusteredInterval(primary,r=>Number(r.correct)),auxiliary:clusteredInterval(auxiliary,r=>Number(r.correct))},
    paired_original_binary:{native:pair(records),primary:pair(primary),auxiliary:pair(auxiliary),original_report_sha256:data.candidate?.original_report_sha256??null},
    code_projection:{cases:projected.length,served:projected.filter(r=>r.status==='ok').length,exact_set_accuracy:mean(projected.map(r=>Number(r.correct))),
      interval95:clusteredInterval(projected,r=>Number(r.correct)),policy_violations:0,policy_semantics:'by construction; no raw learned-rule gate',additional_model_calls:0,trace:projected},
    controls:Object.fromEntries(['none','first_lexicographic_candidate'].map(control=>{const correct=data.inputs.map(row=>{const key=control==='none'?'none':Object.keys(row.option_to_tool).filter(k=>k!=='none').sort()[0];return {category:row.source_category,correct:key===byReference.get(row.sample_id).target_option_id};});
      return [control,{native_accuracy:mean(correct.map(r=>Number(r.correct))),primary_accuracy:mean(correct.filter(r=>r.category==='multiple').map(r=>Number(r.correct))),auxiliary_accuracy:mean(correct.filter(r=>r.category==='irrelevance').map(r=>Number(r.correct)))}];})),
    research_screen:{conditions,status:Object.values(conditions).every(Boolean)?'ENUM_RESEARCH_SCREEN_GO':'ENUM_RESEARCH_SCREEN_NO_GO',promotion:false,A_B_C:'NOT_ESTABLISHED'},
    claim:'New representation/compute hypothesis only; existing binary gates/reports unchanged; no derived confidence, learned-policy, AST/task or test claim'};
}

if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
  const {values}=parseArgs({options:{model:{type:'string',default:'typed'},root:{type:'string'},original:{type:'string'},freeze:{type:'boolean'},out:{type:'string'},spec:{type:'string'},predictions:{type:'string'},'prediction-sha':{type:'string'}}});
  if(!values.root||!values.original||!values.out) throw new Error('ENUM_ROOT_ORIGINAL_FRESH_OUTPUT_REQUIRED');
  const data=loadPrototype(values.root,values.original,values.model),spec=analysisSpec(values.model);let report=spec;
  if(!values.freeze) {
    if(!values.spec||!values.predictions||!values['prediction-sha']) throw new Error('FROZEN_ENUM_SPEC_PREDICTIONS_REQUIRED');
    if(encode(JSON.parse(fs.readFileSync(values.spec,'utf8')))!==encode(spec)) throw new Error('ENUM_SPEC_CHANGED');
    if(digest(fs.readFileSync(values.predictions,'utf8'))!==values['prediction-sha']) throw new Error('ENUM_PREDICTIONS_CHANGED');
    const predictions=rows(values.predictions);
    report={...scoreEnum(data,predictions),spec_sha256:digest(fs.readFileSync(values.spec,'utf8')),predictions_sha256:values['prediction-sha']};
  }
  fs.writeFileSync(values.out,JSON.stringify(report,null,2)+'\n',{flag:'wx',mode:0o600});
  console.log(JSON.stringify({revision:REVISION,scope:'DEV ENUM ONLY',output_sha256:digest(fs.readFileSync(values.out,'utf8')),frozen:!!values.freeze,
    ...(values.freeze?{}:{primary:report.primary,auxiliary:report.auxiliary,research_screen:report.research_screen})}));
}
