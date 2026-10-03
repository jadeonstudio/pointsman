import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { DEFAULTS } from '../../src/constants.mjs';
import { validateRequest } from '../../src/contracts.mjs';
import { digest, encode, targetDistribution } from '../../src/training/schema.mjs';
import { createTrainingStore } from '../../src/training/store.mjs';
import { readDataset, writeIndependentDataset, exportDataset, assertSplitIsolation } from '../../src/training/dataset.mjs';
import { RULES, oracle, corpusReport } from './corpus.mjs';
import { freezeEvaluation } from './evaluate.mjs';

export const CURRICULUM_REVISION='train-contrast-rare-anchors-v1';
const transformHash=digest(fs.readFileSync(new URL(import.meta.url),'utf8'));
const changed=row=>row.question.instructions.includes('Exception:')||row.question.instructions.includes('예외:');
const withoutException=text=>text.replace(/ Exception: any null overrides policy: [^.]+\./,'').replace(/ 예외: null이면 [^.]+\./,'');
const caseKey=row=>`${row.oracle.id}:${digest(row.state)}`;
const semantic=row=>{const rule=RULES.find(r=>r.id===row.oracle.id);return rule.labels[oracle(rule.id,row.state.facts,changed(row))];};
const group=rows=>{const groups=new Map();for(const row of rows){const key=caseKey(row);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(row);}return groups;};

export function buildCurriculum(samples) {
  assertSplitIsolation(samples);
  const train=samples.filter(r=>r.split==='train'),initial=corpusReport(train);
  if(initial.group_leakage||initial.unreproducible_labels)throw new Error('CURRICULUM_SOURCE_ORACLE_INVALID');
  const cases=group(train),classCounts=new Map();
  for(const row of train){const key=`${row.oracle.id}:${semantic(row)}`;classCounts.set(key,(classCounts.get(key)||0)+1);}
  const rare=new Set([...classCounts].filter(([,n])=>n<10).map(([key])=>key)),informative=new Set(),anchors=new Set();
  for(const [key,rows] of cases){
    const row=rows[0],hasPair=rows.some(changed)&&rows.some(r=>!changed(r));
    const changes=row.question.type==='noul'?rows.some(a=>rows.some(b=>changed(a)!==changed(b)&&withoutException(a.question.instructions)===withoutException(b.question.instructions)&&a.target.value!==b.target.value)):
      oracle(row.oracle.id,row.state.facts)!==oracle(row.oracle.id,row.state.facts,true);
    if(hasPair&&changes)informative.add(key);
    if(rows.some(r=>rare.has(`${r.oracle.id}:${semantic(r)}`)))anchors.add(key);
  }
  const selected=new Set([...informative,...anchors]),rows=[];
  for(const source of samples){
    if(source.split!=='train'){rows.push(structuredClone(source));continue;}
    if(!selected.has(caseKey(source)))continue;
    const row=structuredClone(source),siblings=cases.get(caseKey(source));
    if(row.question.type==='choice'&&changed(row)){
      // Original pair 0/1 and 2/3 share description order; inherit that pair's BASE keys/order.
      const base=siblings.find(r=>!changed(r)&&JSON.stringify(Object.values(r.question.criteria))===JSON.stringify(Object.values(row.question.criteria)));
      if(!base)throw new Error('MISSING_CONTROLLED_BASE_PAIR');
      const meaning=row.question.criteria[row.target.value];row.question.criteria=structuredClone(base.question.criteria);
      const value=Object.keys(row.question.criteria).find(k=>row.question.criteria[k]===meaning);
      if(value===undefined)throw new Error('CURRICULUM_MEANING_CHANGED');
      row.target={value,probabilities:targetDistribution(row.question,value)};
    }
    const request=validateRequest({purpose:row.purpose,risk:'routine',state:row.state,questions:{[row.question_id]:row.question}},DEFAULTS);
    row.request_hash=digest(request);row.sample_id=digest({request_hash:row.request_hash,question_id:row.question_id,target:row.target});
    row.data_rights.transformations=[...row.data_rights.transformations,`${CURRICULUM_REVISION}:${transformHash}`];
    row.provenance=row.provenance.map(p=>({...p,preprocessing_version:CURRICULUM_REVISION}));
    // Facts, source/oracle revision, objective semantics, task and family lineage stay unchanged.
    rows.push(row);
  }
  const trainRows=rows.filter(r=>r.split==='train'),newCases=group(trainRows),pairCounts={choice:{pure_pairs:0,gold_changes:0},noul:{pure_pairs:0,gold_changes:0},score:{pure_pairs:0,gold_changes:0}};
  for(const siblings of newCases.values())for(let i=0;i<siblings.length;i++)for(let j=i+1;j<siblings.length;j++){
    const a=siblings[i],b=siblings[j];if(changed(a)===changed(b)||JSON.stringify(a.question.criteria)!==JSON.stringify(b.question.criteria)||withoutException(a.question.instructions)!==withoutException(b.question.instructions))continue;
    pairCounts[a.question.type].pure_pairs++;pairCounts[a.question.type].gold_changes+=Number(a.target.value!==b.target.value);
  }
  assertSplitIsolation(rows);
  const report=corpusReport(trainRows);if(report.group_leakage||report.unreproducible_labels)throw new Error('CURRICULUM_ORACLE_CHANGED');
  return {rows,report:{revision:CURRICULUM_REVISION,transform_sha256:transformHash,source_rows:samples.length,source_train_rows:train.length,source_train_cases:cases.size,
    train_rows:trainRows.length,train_unique_cases:newCases.size,informative_cases:informative.size,rare_anchor_cases:anchors.size,
    additional_rare_anchor_cases:[...anchors].filter(k=>!informative.has(k)).length,
    informative_train_rows:trainRows.filter(r=>informative.has(caseKey(r))).length,rare_family_classes:Object.fromEntries([...classCounts].filter(([key])=>rare.has(key))),pure_rule_pairs:pairCounts,
    original_oracle_revisions:[...new Set(samples.map(r=>r.oracle.revision))].sort(),held_source_revisions:[...new Set(samples.filter(r=>r.split!=='train').map(r=>r.data_rights.revision))].sort(),
    oracle_semantics_changed:false,rows_oversampled:false,held_semantic_evaluation:false,held_handling:'byte/hash preservation and canonical integrity only; no held rule-oracle evaluation or model decisions',coverage_claim:'Measured TRAIN curriculum follow-up, not an equal-update causal comparison or broad generalization claim',corpus:report}};
}

export function writeCurriculum(store,sourceVersion) {
  const original=readDataset(store,sourceVersion),built=buildCurriculum(original.samples),written=writeIndependentDataset(store,built.rows),next=readDataset(store,written.dataset_version);
  const oldLines=new Map(original.contents.trim().split('\n').map(line=>[JSON.parse(line).sample_id,line])),newLines=new Map(next.contents.trim().split('\n').map(line=>[JSON.parse(line).sample_id,line]));
  const heldHashes={};
  for(const split of ['dev','calibration','test']){
    const old=original.samples.filter(r=>r.split===split),fresh=next.samples.filter(r=>r.split===split);
    if(encode(old.map(r=>r.sample_id))!==encode(fresh.map(r=>r.sample_id))||old.some(r=>oldLines.get(r.sample_id)!==newLines.get(r.sample_id)))throw new Error('HELD_CANONICAL_BYTES_CHANGED');
    heldHashes[split]={rows:old.length,canonical_rows_sha256:digest(old.map(r=>oldLines.get(r.sample_id)).join('\n')+'\n'),sample_ids_sha256:digest(old.map(r=>r.sample_id).sort()),identical_row_bytes:true};
  }
  const exported=exportDataset(store,written.dataset_version,'laya');
  for(const split of ['dev','calibration','test']){
    const old=fs.readFileSync(path.join(store.root,'exports',sourceVersion,'laya',`${split}.jsonl`),'utf8'),fresh=fs.readFileSync(exported.files.find(f=>f.endsWith(`/${split}.jsonl`)),'utf8');
    if(old!==fresh)throw new Error('HELD_EXPORT_BYTES_CHANGED');heldHashes[split].export_sha256=digest(fresh);
  }
  const report={...built.report,source_dataset_version:sourceVersion,dataset_version:written.dataset_version,held:heldHashes,
    split_distribution:next.manifest.split_distribution,source_data_sha256:original.manifest.data_sha256,training_executed:false};
  store.writeDerived(`datasets/${written.dataset_version}/curriculum_report.json`,encode(report)+'\n');
  store.writeDerived(`datasets/${written.dataset_version}/corpus_report.json`,encode(built.report.corpus)+'\n');
  store.writeDerived(`datasets/${written.dataset_version}/dev_spec.json`,encode(freezeEvaluation(next.samples,{split:'dev'}))+'\n');
  return {...written,export:exported,report,dev_spec:path.join(store.root,'datasets',written.dataset_version,'dev_spec.json')};
}

if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href){
  const {values}=parseArgs({options:{home:{type:'string'},'source-version':{type:'string'}}});
  if(!values.home||!values['source-version'])throw new Error('EXPLICIT_HOME_AND_SOURCE_VERSION_REQUIRED');
  console.log(encode(writeCurriculum(createTrainingStore({home:fs.realpathSync(values.home)}),values['source-version'])));
}
