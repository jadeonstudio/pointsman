import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {digest, encode} from '../../src/training/schema.mjs';

export const REVISION='bfcl-dev-field-evaluation-v1';
export const PACK_SHA='7b2a773b53f6169632a475d6256bac45b122924d67290e9a21cdb987591d7d28';
const OPTIONS=['yes','no'], STATUSES=['ok','timeout','rejected','unsupported_input','error'];
const mean=xs=>xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:null;
const fraction=(n,d)=>d?n/d:null;
const sameSet=(a,b)=>encode([...a].sort())===encode([...b].sort());
const rows=file=>fs.readFileSync(file,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
export const inputIdentity=row=>digest({state:row.wire.state,questions:row.wire.questions});

export function loadDev(root) {
  const manifestPath=path.join(root,'manifest.json'), bytes=fs.readFileSync(manifestPath);
  if(digest(bytes.toString())!==PACK_SHA) throw new Error('FROZEN_PACK_CHANGED');
  const manifest=JSON.parse(bytes), files={};
  // This loader opens only DEV artifacts; never test or capture data.
  for(const kind of ['cases','fields','reference']) {
    const name=`dev/${kind}.jsonl`, file=path.join(root,name);
    if(digest(fs.readFileSync(file,'utf8'))!==manifest.files[name]) throw new Error('FROZEN_DEV_CHANGED');
    files[kind]=rows(file);
  }
  const {cases,fields,reference}=files;
  if(cases.length!==287||fields.length!==755||reference.length!==287||cases.some(r=>r.split!=='dev')||fields.some(r=>r.split!=='dev')
    ||new Set(cases.map(r=>r.source_id)).size!==127||new Set(cases.map(r=>r.family_id)).size!==76
    ||cases.filter(r=>r.source_category==='multiple'&&r.variant==='all_allowed').length!==80) throw new Error('DEV_COHORT_CHANGED');
  return {...files,manifest};
}

export function analysisSpec(manifest) {
  return {revision:REVISION,scope:'DEV ONLY',pack_sha256:PACK_SHA,scorer_sha256:digest(fs.readFileSync(fileURLToPath(import.meta.url),'utf8')),
    dev_file_hashes:Object.fromEntries(['cases','fields','reference'].map(k=>[k,manifest.files[`dev/${k}.jsonl`]])),
    cohort:{fields:755,variants:287,origins:127,schema_components:76,primary_native_multiple_origins:80,auxiliary_native_irrelevance_origins:47},
    prediction:{required:['sample_id','model_id','checkpoint','input_identity','status'],statuses:STATUSES,
      input_identity:'SHA256 encode({state:wire.state,questions:wire.questions}); routing model and metadata excluded',
      ok:{choice:OPTIONS,probabilities:'exact yes/no keys; finite bounded numbers; positive mass; abs(sum-1)<=0.02; normalize mass for scoring'},
      elapsed_ms:'optional finite nonnegative number or null; never quality eligibility',actual_choice:'used for selection; never replaced by probability argmax',
      confidence:'normalized selected-choice probability; provider confidence ignored'},
    denominators:{field_accuracy:'all755; unserved invalid/missing are wrong',field_macro_f1:'fixed yes/no classes; all-field support, invalid contributes false negative',
      conditional_probability_metrics:'valid served fields only; expose count/coverage; NLL floor1e-15; Brier sum across yes/no; ECE10equal bins selected-choice probability',
      complete_case:'all287; every candidate field must be served and exact selected name set; unserved case wrong',
      primary:'80 original multiple/all_allowed cases',auxiliary:'47 original irrelevance/all_allowed cases',
      contrast:'80 multiple origins; require all_allowed AND deny_first AND deny_all exact; report each base/rule pair separately',
      policy_violation:'emitted status-ok choice yes for prohibited candidate, including malformed probability emissions; coverage failures also block GO'},
    bootstrap:{draws:2000,seed:42,unit:'sample schema components with replacement, then original sources within sampled component with replacement at original component size',
      weighting:'origin weight1 for primary/contrast; variant-weighted ratio for complete-case; variants/fields never independently resampled',interval:'sorted empirical percentile indices50,1950; fixed2000draws'},
    controls:['none: no offered tool','all: every offered tool including prohibited','first_allowed: first lexicographic allowed tool, none when empty'],
    projection_control:{definition:'Valid served all_allowed actual selected tool set intersect each variant allowed_tools; no substitution',
      provenance:'base field prediction IDs and derived variant IDs retained; zero additional model calls; no derived probabilities/confidence',
      qualification:'separate code-projection pipeline; zero policy violations by construction, not learned rule ability; raw755 model evaluation and raw GO unchanged',
      inference_budget:'all755 raw field records still required; actual forwards need producer/runner metadata, not guessed from statuses'},
    candidate_research_go:{primary_exact_set_at_least:.9,all_three_rule_contrast_at_least:.9,emitted_policy_violations:0,served_field_coverage:1},
    decision:'RESEARCH_GO only; no calibration/test selection, promotion, A/B/C,1%critical floor, AST/execution or task success claim',
    pretraining_contamination:'UNKNOWN',test:'SEALED_NOT_OPENED'};
}

function fieldMetrics(records) {
  const served=records.filter(r=>r.status==='ok'), confusion=Object.fromEntries(OPTIONS.map(k=>[k,{yes:0,no:0,invalid:0}]));
  for(const r of records) confusion[r.gold][r.status==='ok'?r.choice:'invalid']++;
  const macro=rs=>mean(OPTIONS.map(k=>{const tp=rs.filter(r=>r.gold===k&&r.status==='ok'&&r.choice===k).length;
    const support=rs.filter(r=>r.gold===k).length, predicted=rs.filter(r=>r.status==='ok'&&r.choice===k).length;
    return support+predicted?2*tp/(support+predicted):0;}));
  const reliability=Array.from({length:10},(_,bin)=>{const rs=served.filter(r=>Math.min(9,Math.floor(r.confidence*10))===bin);
    return {count:rs.length,mean_probability:mean(rs.map(r=>r.confidence)),accuracy:mean(rs.map(r=>Number(r.correct)))};});
  return {count:records.length,served:served.length,coverage:fraction(served.length,records.length),accuracy:mean(records.map(r=>Number(r.correct))),
    served_accuracy:mean(served.map(r=>Number(r.correct))),macro_f1:macro(records),served_macro_f1:macro(served),confusion,
    nll:mean(served.map(r=>r.nll)),brier:mean(served.map(r=>r.brier)),probability_scored_count:served.length,reliability,
    ece:served.length?reliability.reduce((n,b)=>n+(b.count?b.count*Math.abs(b.accuracy-b.mean_probability):0),0)/served.length:null,
    selected_choice_argmax_disagreements:served.filter(r=>r.argmax_disagreement).length,
    probability_mass_corrections:served.filter(r=>r.mass_error>1e-12).length,
    status:Object.fromEntries([...new Set(records.map(r=>r.status))].sort().map(k=>[k,records.filter(r=>r.status===k).length]))};
}

export function scoreCore(fields,cases,reference,predictions) {
  const byCase=new Map(cases.map(r=>[r.sample_id,r])), gold=new Map(reference.map(r=>[r.sample_id,r])), byPrediction=new Map();
  if(byCase.size!==cases.length||gold.size!==reference.length||gold.size!==byCase.size||[...gold.keys()].some(k=>!byCase.has(k))) throw new Error('REFERENCE_CASE_JOIN');
  const expected=new Set(fields.map(r=>r.sample_id));
  if(expected.size!==fields.length) throw new Error('DUPLICATE_INPUT_FIELD');
  for(const p of predictions) {
    if(!p||!expected.has(p.sample_id)||byPrediction.has(p.sample_id)||!STATUSES.includes(p.status)
      ||![p.model_id,p.checkpoint,p.input_identity].every(v=>typeof v==='string'&&v.length)) throw new Error('INVALID_PREDICTION_JOIN');
    if(p.elapsed_ms!==undefined&&p.elapsed_ms!==null&&(!Number.isFinite(p.elapsed_ms)||p.elapsed_ms<0)) throw new Error('INVALID_LATENCY');
    byPrediction.set(p.sample_id,p);
  }
  if(new Set(predictions.map(p=>encode([p.model_id,p.checkpoint]))).size>1) throw new Error('MIXED_MODEL_IDENTITY');
  const records=fields.map(row=>{
    const c=byCase.get(row.case_id), label=gold.get(row.case_id), key=row.sample_id.split('/').at(-1), p=byPrediction.get(row.sample_id);
    if(!c||row.source_id!==c.source_id||row.family_id!==c.family_id||row.variant!==c.variant||!OPTIONS.includes(label.question_values[key])) throw new Error('FIELD_REFERENCE_JOIN');
    if(p&&p.input_identity!==inputIdentity(row)) throw new Error('INPUT_IDENTITY_MISMATCH');
    const base={sample_id:row.sample_id,case_id:row.case_id,source_id:row.source_id,component:row.family_id,candidate_tool:row.candidate_tool,
      gold:label.question_values[key],status:p?.status??'missing',correct:false,
      policy_violation:p?.status==='ok'&&p.choice==='yes'&&!row.wire.state.allowed_tools.includes(row.candidate_tool),elapsed_ms:p?.elapsed_ms??null};
    if(base.status!=='ok') return base;
    const raw=p.probabilities;
    if(!OPTIONS.includes(p.choice)||!raw||Object.keys(raw).length!==2||OPTIONS.some(k=>!Object.hasOwn(raw,k)||!Number.isFinite(raw[k])||raw[k]<0||raw[k]>1)) return {...base,status:'malformed'};
    const mass=raw.yes+raw.no, mass_error=Math.abs(mass-1);
    if(mass<=0||mass_error>.02) return {...base,status:'malformed'};
    const probabilities={yes:raw.yes/mass,no:raw.no/mass};
    return {...base,choice:p.choice,correct:p.choice===base.gold,confidence:probabilities[p.choice],mass_error,
      argmax_disagreement:probabilities[p.choice]<Math.max(probabilities.yes,probabilities.no),
      nll:-Math.log(Math.max(1e-15,probabilities[base.gold])),brier:OPTIONS.reduce((n,k)=>n+(probabilities[k]-Number(k===base.gold))**2,0)};
  });
  const caseRecords=cases.map(c=>{
    const rs=records.filter(r=>r.case_id===c.sample_id), label=gold.get(c.sample_id), candidates=c.wire.state.functions.map(f=>f.name);
    if(rs.length!==candidates.length||!sameSet(rs.map(r=>r.candidate_tool),candidates)||!sameSet(label.target_tool_names,candidates.filter(n=>label.question_values['tool_'+n.replace(/[^A-Za-z0-9_]+/g,'_').replace(/^_+|_+$/g,'')]==='yes'))) throw new Error('INCOMPLETE_CASE_REFERENCE');
    const served=rs.every(r=>r.status==='ok'), selected=rs.filter(r=>r.status==='ok'&&r.choice==='yes').map(r=>r.candidate_tool);
    return {case_id:c.sample_id,source_id:c.source_id,component:c.family_id,category:c.source_category,variant:c.variant,served,
      selected_tools:selected,target_tool_names:label.target_tool_names,allowed_tools:c.wire.state.allowed_tools,
      correct:served&&sameSet(selected,label.target_tool_names),policy_violations:rs.filter(r=>r.policy_violation).length};
  });
  const summarise=rs=>({count:rs.length,served:rs.filter(r=>r.served).length,coverage:mean(rs.map(r=>Number(r.served))),correct:rs.filter(r=>r.correct).length,
    exact_set_accuracy:mean(rs.map(r=>Number(r.correct))),served_exact_set_accuracy:mean(rs.filter(r=>r.served).map(r=>Number(r.correct)))});
  const primary=caseRecords.filter(r=>r.category==='multiple'&&r.variant==='all_allowed'), auxiliary=caseRecords.filter(r=>r.category==='irrelevance'&&r.variant==='all_allowed');
  const contrast=primary.map(base=>{
    const sibling=v=>caseRecords.find(r=>r.source_id===base.source_id&&r.variant===v);
    const first=sibling('deny_first_lexicographic_candidate'), all=sibling('deny_all');
    if(!first||!all||first.component!==base.component||all.component!==base.component) throw new Error('MISSING_POLICY_SIBLING');
    return {source_id:base.source_id,component:base.component,correct:base.correct&&first.correct&&all.correct,
      first_pair:base.correct&&first.correct,all_pair:base.correct&&all.correct};
  });
  const violations=records.filter(r=>r.policy_violation).length, metrics=fieldMetrics(records);
  const conditions={primary:primary.length>0&&summarise(primary).exact_set_accuracy>=.9,contrast:contrast.length>0&&mean(contrast.map(r=>Number(r.correct)))>=.9,
    zero_policy_violations:violations===0,coverage:metrics.coverage===1};
  return {model_identity:predictions.length?{model_id:predictions[0].model_id,checkpoint:predictions[0].checkpoint}:null,
    field:metrics,complete_case:summarise(caseRecords),primary:summarise(primary),auxiliary:summarise(auxiliary),
    policy:{origins:contrast.length,all_three_correct:contrast.filter(r=>r.correct).length,all_three_accuracy:mean(contrast.map(r=>Number(r.correct))),
      base_deny_first_pair_accuracy:mean(contrast.map(r=>Number(r.first_pair))),base_deny_all_pair_accuracy:mean(contrast.map(r=>Number(r.all_pair))),emitted_violations:violations},
    latency:{measured_fields:records.filter(r=>r.elapsed_ms!==null).length,mean_ms:mean(records.map(r=>r.elapsed_ms).filter(v=>v!==null))},
    research_go:{conditions,status:Object.values(conditions).every(Boolean)?'RESEARCH_GO':'RESEARCH_NO_GO',promotion:false,A_B_C:'NOT_ESTABLISHED'},
    case_records:caseRecords,contrast_records:contrast};
}

export function clusteredInterval(records,value) {
  const byComponent=new Map();
  for(const r of records) {if(!byComponent.has(r.component)) byComponent.set(r.component,new Map());const origins=byComponent.get(r.component);
    if(!origins.has(r.source_id)) origins.set(r.source_id,[]);origins.get(r.source_id).push(r);}
  const components=[...byComponent].sort(([a],[b])=>a.localeCompare(b)).map(([,origins])=>[...origins].sort(([a],[b])=>a.localeCompare(b)).map(([,rs])=>rs));
  if(!components.length) return {interval95:null,draws:2000,seed:42,components:0,origins:0};
  let rng=42;const pick=n=>{rng=(Math.imul(1664525,rng)+1013904223)>>>0;return Math.floor(rng/4294967296*n);},estimates=[];
  for(let i=0;i<2000;i++){const sampled=[];for(let j=0;j<components.length;j++){const component=components[pick(components.length)];
    for(let k=0;k<component.length;k++) sampled.push(...component[pick(component.length)]);}
    estimates.push(mean(sampled.map(value)));}
  estimates.sort((a,b)=>a-b);
  return {interval95:[estimates[50],estimates[1950]],draws:2000,seed:42,components:components.length,origins:new Set(records.map(r=>r.source_id)).size};
}

function projectionControl(caseRecords,fields) {
  const bases=new Map(caseRecords.filter(r=>r.variant==='all_allowed').map(r=>[r.source_id,r]));
  const projected=caseRecords.map(r=>{const base=bases.get(r.source_id), selected=base.selected_tools.filter(n=>r.allowed_tools.includes(n));
    return {case_id:r.case_id,source_id:r.source_id,base_case_id:base.case_id,
      base_prediction_ids:fields.filter(f=>f.case_id===base.case_id).map(f=>f.sample_id),served:base.served,
      correct:base.served&&sameSet(selected,r.target_tool_names),selected_tools:base.served?selected:null,derived:r.variant!=='all_allowed',
      probabilities:null,confidence:null,additional_model_calls:0};});
  const byCase=new Map(projected.map(r=>[r.case_id,r]));
  return {pipeline:'all_allowed emitted semantic set + deterministic allowed-tools intersection',research_go:'NOT_APPLICABLE_CODE_PROJECTION',
    count:projected.length,served:projected.filter(r=>r.served).length,exact_set_accuracy:mean(projected.map(r=>Number(r.correct))),
    primary_exact_set_accuracy:mean(caseRecords.filter(r=>r.category==='multiple'&&r.variant==='all_allowed').map(r=>Number(byCase.get(r.case_id).correct))),
    both_rule_contrasts:mean([...bases.values()].filter(r=>r.category==='multiple').map(r=>Number(projected.filter(p=>p.source_id===r.source_id).every(p=>p.correct)))),
    policy_violations:0,policy_violation_semantics:'by construction for valid base; not learned rule ability; unserved remains failure',
    improved_vs_raw:caseRecords.filter(r=>!r.correct&&byCase.get(r.case_id).correct).length,
    regressed_vs_raw:caseRecords.filter(r=>r.correct&&!byCase.get(r.case_id).correct).length,
    raw_required_fields:fields.length,semantic_base_fields:fields.filter(f=>f.variant==='all_allowed').length,
    actual_forward_calls:'UNKNOWN without producer/runner manifest; no status-based invention',additional_model_calls:0,trace:projected};
}

export function evaluateDev(data,predictions) {
  const candidate=scoreCore(data.fields,data.cases,data.reference,predictions), controls={};
  for(const control of ['none','all','first_allowed']) {
    const ps=data.fields.map(r=>{const choice=control==='all'||(control==='first_allowed'&&r.candidate_tool===[...r.wire.state.allowed_tools].sort()[0])?'yes':'no';
      return {sample_id:r.sample_id,model_id:`control/${control}`,checkpoint:digest(control),input_identity:inputIdentity(r),status:'ok',choice,probabilities:{yes:Number(choice==='yes'),no:Number(choice==='no')}};});
    const scored=scoreCore(data.fields,data.cases,data.reference,ps);delete scored.case_records;delete scored.contrast_records;controls[control]=scored;
  }
  const intervals={primary:clusteredInterval(candidate.case_records.filter(r=>r.category==='multiple'&&r.variant==='all_allowed'),r=>Number(r.correct)),
    complete_case:clusteredInterval(candidate.case_records,r=>Number(r.correct)),policy_contrast:clusteredInterval(candidate.contrast_records,r=>Number(r.correct))};
  const projection=projectionControl(candidate.case_records,data.fields);
  delete candidate.case_records;delete candidate.contrast_records;
  return {revision:REVISION,scope:'DEV ONLY',candidate,controls,intervals,projection_control:projection,claim:'As-shipped experimental research only; no test selection, calibration, promotion, A/B/C, AST, task completion or Jev comparison'};
}

if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
  const {values}=parseArgs({options:{root:{type:'string'},freeze:{type:'boolean'},out:{type:'string'},spec:{type:'string'},predictions:{type:'string'},'prediction-sha':{type:'string'}}});
  if(!values.root||!values.out) throw new Error('ROOT_AND_FRESH_OUTPUT_REQUIRED');
  const data=loadDev(values.root), spec=analysisSpec(data.manifest);
  let report=spec;
  if(!values.freeze) {
    if(!values.spec||!values.predictions||!values['prediction-sha']) throw new Error('FROZEN_SPEC_AND_PREDICTIONS_REQUIRED');
    if(encode(JSON.parse(fs.readFileSync(values.spec,'utf8')))!==encode(spec)) throw new Error('ANALYSIS_SPEC_CHANGED');
    if(digest(fs.readFileSync(values.predictions,'utf8'))!==values['prediction-sha']) throw new Error('PREDICTIONS_CHANGED');
    report={...evaluateDev(data,rows(values.predictions)),spec_sha256:digest(fs.readFileSync(values.spec,'utf8')),predictions_sha256:values['prediction-sha']};
  }
  fs.writeFileSync(values.out,JSON.stringify(report,null,2)+'\n',{flag:'wx',mode:0o600});
  console.log(JSON.stringify({revision:REVISION,scope:'DEV ONLY',output_sha256:digest(fs.readFileSync(values.out,'utf8')),frozen:!!values.freeze,
    ...(values.freeze?{}:{research_go:report.candidate.research_go,primary:report.candidate.primary,coverage:report.candidate.field.coverage})}));
}
