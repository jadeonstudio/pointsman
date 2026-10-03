import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { digest, encode } from '../../src/training/schema.mjs';
import { assertSplitIsolation } from '../../src/training/dataset.mjs';
import { RULES, corpusReport } from './corpus.mjs';

export const EVALUATION_REVISION='independent-workflow-eval-v1';
const mean=xs=>xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:null;
const keys=row=>Object.keys(row.target.probabilities);
function semantic(row,key) {
  if(row.question.type==='noul') return key;
  const description=row.question.type==='score'?row.question.criteria[Number(key)]:row.question.criteria[key];
  const rule=RULES.find(r=>r.id===row.oracle.id), index=rule.labels.indexOf(description);
  return rule.labels[index<0?rule.koLabels.indexOf(description):index];
}
export function freezeEvaluation(rows,{split='test'}={}) {
  if(!['train','dev','calibration','test'].includes(split)) throw new Error('INVALID_EVAL_SPLIT');
  assertSplitIsolation(rows);
  const selected=rows.filter(r=>r.split===split);
  if(!selected.length) throw new Error('EMPTY_EVAL_SPLIT');
  return {revision:EVALUATION_REVISION,split,sample_ids_sha256:digest(selected.map(r=>r.sample_id).sort()),canonical_sha256:digest(rows.map(encode).join('\n')+'\n'),
    expected_samples:selected.length,semantic_families:[...new Set(selected.map(r=>r.lineage.semantic_family_id))].sort(),
    metrics:['full_envelope_accuracy','served_accuracy','macro_f1','nll','brier','reliability','ordinal_mae','severe_ordinal_error','risk_coverage'],
    thresholds:[.5,.7,.8,.9,.95,.99],selection:'dev only; calibration only fits temperatures/gates; test is sealed and is never used to choose a threshold',
    missing_prediction:'counted as full-envelope failure; excluded from distribution scoring with missing count exposed',
    confidence_semantics:'maximum normalized class probability; never provider confidence field',
    clustering:'semantic family; generated rows are correlated and are not independent task evidence',
    candidate_recall:'generator represents other/insufficient cases; operational candidate-set recall remains UNKNOWN',
    requirements:'B/C PASS criteria and critical floors must come from the approved PLAN; absent provider comparisons remain UNKNOWN'};
}
function summaries(records) {
  const served=records.filter(r=>r.status==='ok'), labels=new Set(served.flatMap(r=>[r.gold,r.predicted])), f1=[];
  for(const label of labels) { const tp=served.filter(r=>r.gold===label&&r.predicted===label).length, fp=served.filter(r=>r.gold!==label&&r.predicted===label).length, fn=served.filter(r=>r.gold===label&&r.predicted!==label).length; f1.push(2*tp+fp+fn?2*tp/(2*tp+fp+fn):0); }
  const score=served.filter(r=>r.type==='score');
  const reliability=Array.from({length:10},(_,bin)=>{ const sample=served.filter(r=>Math.min(9,Math.floor(r.confidence*10))===bin); return {lower:bin/10,upper:(bin+1)/10,count:sample.length,mean_probability:mean(sample.map(r=>r.confidence)),accuracy:mean(sample.map(r=>Number(r.correct)))}; });
  return {count:records.length,served:served.length,coverage:records.length?served.length/records.length:null,
    full_envelope_accuracy:mean(records.map(r=>Number(r.correct===true))),served_accuracy:mean(served.map(r=>Number(r.correct))),macro_f1:mean(f1),
    nll:mean(served.map(r=>r.nll)),brier:mean(served.map(r=>r.brier)),reliability,
    expected_calibration_error:served.length?reliability.reduce((n,b)=>n+(b.count?b.count/served.length*Math.abs(b.accuracy-b.mean_probability):0),0):null,
    ordinal_mae:mean(score.map(r=>r.ordinal_error)),severe_ordinal_error:mean(score.map(r=>Number(r.ordinal_error>=2))),
    effective_family_count:new Set(records.map(r=>r.family)).size,
    status:Object.fromEntries([...new Set(records.map(r=>r.status))].sort().map(k=>[k,records.filter(r=>r.status===k).length]))};
}
export function evaluatePredictions(rows,predictions,{split='test',spec=freezeEvaluation(rows,{split})}={}) {
  const frozen=freezeEvaluation(rows,{split});
  if(encode(spec)!==encode(frozen)) throw new Error('EVALUATION_SPEC_MISMATCH');
  const selected=rows.filter(r=>r.split===split), byId=new Map(), allIds=new Set(rows.map(r=>r.sample_id));
  for(const p of predictions) {
    if(!p||typeof p.sample_id!=='string'||byId.has(p.sample_id)||!allIds.has(p.sample_id)) throw new Error('INVALID_PREDICTION_JOIN');
    if(typeof p.model_id!=='string'||!p.model_id||typeof p.checkpoint!=='string'||!p.checkpoint) throw new Error('MISSING_MODEL_IDENTITY');
    if(!['ok','timeout','rejected','unsupported_input','error'].includes(p.status)) throw new Error('INVALID_PREDICTION_STATUS');
    byId.set(p.sample_id,p);
  }
  const identities=new Set(predictions.map(p=>`${p.model_id}:${p.checkpoint}`)); if(identities.size>1) throw new Error('MIXED_CANDIDATE_IDENTITY');
  const records=selected.map(row=>{
    const p=byId.get(row.sample_id), base={sample_id:row.sample_id,family:row.lineage.semantic_family_id,language:row.state.language,domain:row.state.domain,type:row.question.type,task:RULES.find(r=>r.id===row.oracle.id).task,
      status:p?.status??'missing',gold:`${row.lineage.semantic_family_id}/${row.question.type}/${semantic(row,String(row.target.value))}`,correct:false};
    if(base.status!=='ok') return base;
    const k=keys(row), probs=p.probabilities;
    if(!probs||Object.keys(probs).length!==k.length||k.some(x=>!Object.hasOwn(probs,x)||typeof probs[x]!=='number'||!Number.isFinite(probs[x])||probs[x]<0||probs[x]>1)||Math.abs(k.reduce((n,x)=>n+probs[x],0)-1)>1e-6) return {...base,status:'malformed'};
    const predicted=k.reduce((a,b)=>probs[b]>probs[a]?b:a), gold=String(row.target.value);
    const expected=row.question.type==='score'?k.reduce((n,x)=>n+Number(x)*probs[x],0):null;
    return {...base,predicted:`${row.lineage.semantic_family_id}/${row.question.type}/${semantic(row,predicted)}`,correct:predicted===gold,confidence:probs[predicted],
      nll:-Math.log(Math.max(1e-15,probs[gold])),brier:k.reduce((n,x)=>n+(probs[x]-(x===gold?1:0))**2,0),ordinal_error:expected===null?null:Math.abs(expected-Number(gold))};
  });
  const slices={};
  for(const field of ['family','language','domain','type','task']) slices[field]=Object.fromEntries([...new Set(records.map(r=>r[field]))].sort().map(value=>[value,summaries(records.filter(r=>r[field]===value))]));
  const risk_coverage=spec.thresholds.map(threshold=>{ const applied=records.filter(r=>r.status==='ok'&&r.confidence>=threshold); return {threshold,applied:applied.length,coverage:applied.length/records.length,error_rate:mean(applied.map(r=>Number(!r.correct))),effective_family_count:new Set(applied.map(r=>r.family)).size}; });
  const absentFamilies=RULES.map(r=>r.id).filter(id=>!records.some(r=>r.family===id));
  return {revision:EVALUATION_REVISION,spec_sha256:digest(spec),predictions_sha256:digest(predictions.map(encode).join('\n')+'\n'),model_identity:[...identities][0]??null,metrics:summaries(records),slices,risk_coverage,
    absent_families:absentFamilies,absent_family_status:'UNKNOWN',operational_utility:'UNKNOWN: corpus predictions are not executor task outcomes',jev_comparison:'UNKNOWN: no authorized Jev competitor comparison',claim:'Independent synthetic rule correctness for this frozen split only; no provider superiority or automatic adoption.'};
}
export function comparePredictions(rows,left,right,{split='test',bootstrap=1000,seed=42}={}) {
  if(!Number.isInteger(bootstrap)||bootstrap<100||bootstrap>10000||!Number.isSafeInteger(seed)||seed<0) throw new Error('INVALID_BOOTSTRAP');
  const a=evaluatePredictions(rows,left,{split}),b=evaluatePredictions(rows,right,{split});
  const families=Object.keys(a.slices.family), differences=families.map(id=>a.slices.family[id].full_envelope_accuracy-b.slices.family[id].full_envelope_accuracy), estimates=[];
  let rng=seed>>>0;
  for(let i=0;i<bootstrap;i++) { const sample=[]; for(let j=0;j<families.length;j++) { rng=(Math.imul(1664525,rng)+1013904223)>>>0; sample.push(differences[Math.floor(rng/4294967296*families.length)]); } estimates.push(mean(sample)); }
  estimates.sort((x,y)=>x-y);
  return {revision:EVALUATION_REVISION,left_identity:a.model_identity,right_identity:b.model_identity,paired_family_count:families.length,family_macro_accuracy_difference:mean(differences),
    bootstrap_95:[estimates[Math.floor(bootstrap*.025)],estimates[Math.min(bootstrap-1,Math.floor(bootstrap*.975))]],bootstrap,seed,
    status:families.length<5?'INCONCLUSIVE':'MEASURED_SYNTHETIC_ONLY',superiority:'UNKNOWN: approved critical floors, sufficient independent families and permitted comparator are separate requirements'};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
  const {values}=parseArgs({options:{dataset:{type:'string'},manifest:{type:'string'},predictions:{type:'string'},spec:{type:'string'},freeze:{type:'boolean'},split:{type:'string',default:'test'}}});
  if(!values.dataset||!values.manifest) throw new Error('DATASET_AND_MANIFEST_REQUIRED');
  const contents=fs.readFileSync(values.dataset,'utf8'), manifest=JSON.parse(fs.readFileSync(values.manifest,'utf8'));
  if(digest(contents)!==manifest.data_sha256) throw new Error('DATASET_MANIFEST_MISMATCH');
  const rows=contents.trim().split('\n').map(JSON.parse);
  if(rows.length!==manifest.sample_count) throw new Error('DATASET_MANIFEST_MISMATCH');
  const report=corpusReport(rows); if(report.group_leakage||report.unreproducible_labels) throw new Error('CORPUS_INTEGRITY_FAILURE');
  const spec=values.spec?JSON.parse(fs.readFileSync(values.spec,'utf8')):freezeEvaluation(rows,{split:values.split});
  if(values.freeze) console.log(encode(spec));
  else { if(!values.predictions||!values.spec) throw new Error('FROZEN_SPEC_AND_PREDICTIONS_REQUIRED'); console.log(encode(evaluatePredictions(rows,fs.readFileSync(values.predictions,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse),{split:values.split,spec}))); }
}
