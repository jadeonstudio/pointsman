"""CPU-only, frozen paired policy-placement preparation; never loads model weights."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import time

REVISION = "policy-placement-original-oracle-v1"
ORIGINAL = "ba27f11ee6e5bf296c536db112db6ecafa961b686c05edcc3d02ea13a728c070"
CURRICULUM = "275d73372d3ed2d57320072fa6b6c682cf272832c295a0d9eff261e1e1c4c935"
ORACLE_SOURCE = "96b075b11e896f7207648f36334d8c42d173429c0e1cccc80c4d82062a2183ea"
SELECTED_IDS = "3942c0d66797a652149301bee2f441c9e95b593bd9b13884efc3dbba2634eda7"
PREDICTIONS = "dad2e3b3f7337942598881a7b0658886bb9b7acc3b05284b48927ea111723fd0"
CHECKPOINT = "2d78acadca4a2d3865b6c9efd8402d1b2483c1dc6c43a802f81ecc72dcb45e63"
REPO = Path(__file__).resolve().parents[2]


def helper():
    spec = importlib.util.spec_from_file_location("placement_clef_helpers", Path(__file__).with_name("evaluate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relocate(sample):
    """Move, rather than duplicate, the exact original policy; retain every suffix byte."""
    row = copy.deepcopy(sample)
    old = row["question"]["instructions"]
    prefix = "state.policy 적용." if row["state"]["language"] == "ko" else "Apply state.policy."
    if not old.startswith(prefix) or "state.policy" in old[len(prefix):]:
        raise ValueError("UNEXPECTED_OR_DANGLING_POLICY_REFERENCE")
    policy = row["state"].pop("policy")
    if not isinstance(policy, str) or not policy:
        raise ValueError("EMPTY_SOURCE_POLICY")
    lead = "다음 정책을 적용하세요: " if row["state"]["language"] == "ko" else "Apply the following policy: "
    row["question"]["instructions"] = lead + policy + old[len(prefix):]
    assert row["state"] == {k: v for k, v in sample["state"].items() if k != "policy"}
    assert row["question"]["criteria"] == sample["question"]["criteria"]
    assert row["target"] == sample["target"]
    return row


NODE_SOURCE = r"""
import fs from 'node:fs';
import {readDataset} from './src/training/dataset.mjs';
import {createTrainingStore} from './src/training/store.mjs';
import {RULES,oracle,SOURCE_SHA256} from './training/general-decisions/corpus.mjs';
import {digest,encode,targetDistribution} from './src/training/schema.mjs';
import {freezeEvaluation} from './training/general-decisions/evaluate.mjs';
const home=process.argv[1],old=readDataset(createTrainingStore({home}),process.argv[2]),current=readDataset(createTrainingStore({home}),process.argv[3]);
const originalSource=fs.readFileSync('./training/general-decisions/corpus.mjs','utf8').replace('independent-workflow-rules-v3-explicit-priority','independent-workflow-rules-v2-state-policy').replace('Apply these rules in order; the first matching rule wins. ','').replace('아래 규칙을 순서대로 적용하며 처음 성립한 규칙의 결과를 선택합니다. ','');
if(digest(originalSource)!==process.argv[4])throw Error('IMPORTED_ORACLE_SEMANTICS_CHANGED');
const dev=current.samples.filter(r=>r.split==='dev'),oldDev=old.samples.filter(r=>r.split==='dev');
if(encode(dev)!==encode(oldDev)||dev.length!==125)throw Error('ORIGINAL_DEV_CHANGED');
const changed=r=>r.question.instructions.includes('Exception:')||r.question.instructions.includes('예외:');
const specs=[['current-evidence','score'],['revision-branch','choice'],['acceptance-stop','choice']],supp=[],selection=[];
for(const [family,type]of specs)for(const language of ['en','ko','mixed']){
 const cases=new Map();for(const r of old.samples.filter(r=>r.split==='train'&&r.oracle.id===family&&r.question.type===type&&r.state.language===language)){
  if(oracle(family,r.state.facts,false)===oracle(family,r.state.facts,true))continue;
  const id=r.raw_refs.independent.case_id;if(!cases.has(id))cases.set(id,[]);cases.get(id).push(r);
 }
 const eligible=[...cases].filter(([,r])=>r.length===4).sort(([a],[b])=>a.localeCompare(b));if(!eligible.length)throw Error('MISSING_PREREGISTERED_STRATUM');
 const [case_id,rows]=eligible[0];selection.push({family,type,language,case_id,eligible_cases:eligible.length,original_sample_ids:rows.map(r=>r.sample_id).sort()});
 for(const source of rows){const row=structuredClone(source);
  if(type==='choice'&&changed(row)){const base=rows.find(r=>!changed(r)&&JSON.stringify(Object.values(r.question.criteria))===JSON.stringify(Object.values(row.question.criteria)));if(!base)throw Error('MISSING_BASE_CRITERIA');
   const meaning=row.question.criteria[row.target.value];row.question.criteria=structuredClone(base.question.criteria);row.target.value=Object.keys(row.question.criteria).find(k=>row.question.criteria[k]===meaning);row.target.probabilities=targetDistribution(row.question,row.target.value);}
  supp.push({source_sample_id:source.sample_id,sample:row});
 }
}
const originals=new Map(old.contents.trim().split('\n').map(line=>[JSON.parse(line).sample_id,line]));
function verify(row){const rule=RULES.find(r=>r.id===row.oracle.id),strict=changed(row),meaning=oracle(rule.id,row.state.facts,strict),labels=row.state.language==='ko'?rule.koLabels:rule.labels;
 if(row.oracle.revision!==process.argv[4]||row.oracle.evidence_sha256!==digest({facts:row.state.facts,ruleChange:strict}))throw Error('ORACLE_IDENTITY_CHANGED');
 const policy=row.state.language==='en'?rule.en:rule.ko,oldPolicy=rule.id==='failure-contract'?policy.replace('Apply these rules in order; the first matching rule wins. ','').replace('아래 규칙을 순서대로 적용하며 처음 성립한 규칙의 결과를 선택합니다. ',''):policy;
 if(row.state.policy!==oldPolicy)throw Error('POLICY_PRIORITY_OR_SOURCE_CHANGED');
 if(row.question.type==='noul'){const text=row.question.instructions,query=labels.findIndex(label=>text.endsWith(`Is the outcome ${label}?`)||text.endsWith(`Is the outcome NOT ${label}?`)||text.endsWith(` ${label}임?`)||text.endsWith(` ${label} 아님?`)),negated=text.includes('Is the outcome NOT ')||text.endsWith(' 아님?');if(query<0||row.target.value!==((meaning===query)!==negated))throw Error('NOUL_ORACLE_DISAGREEMENT');}
 else if(row.question.criteria[row.target.value]!==labels[meaning])throw Error('ORACLE_DISAGREEMENT');
}
for(const r of [...dev,...supp.map(r=>r.sample)])verify(r);
const overlap=dev.filter(r=>(r.state.facts.input_admitted===false||r.state.facts.schema_current===false)&&r.state.facts.provider_failed===true&&r.state.facts.result_received===false);
if(overlap.length!==8||overlap.some(r=>r.question.type!=='noul'))throw Error('AMBIGUITY_SCOPE_CHANGED');
console.log(encode({dev,supp,selection,selected_ids_sha256:digest(selection.flatMap(x=>x.original_sample_ids).sort()),source_manifest_sha256:digest(fs.readFileSync(home+'/training/manifests/'+process.argv[2]+'.json','utf8')),source_data_sha256:old.manifest.data_sha256,core_data_sha256:current.manifest.data_sha256,core_canonical_rows_sha256:digest(dev.map(r=>originals.get(r.sample_id)).join('\n')+'\n'),core_spec:freezeEvaluation(current.samples,{split:'dev'}),corpus_import_sha256:SOURCE_SHA256,ambiguous_sample_ids:overlap.map(r=>r.sample_id).sort()}));
"""

NODE_IDENTITIES = r"""
import fs from 'node:fs';import {validateRequest,wireRequest} from './src/contracts.mjs';import {DEFAULTS} from './src/constants.mjs';import {digest,encode} from './src/training/schema.mjs';
const rows=JSON.parse(fs.readFileSync(0,'utf8'));
for(const r of rows){const s=r.sample,request=validateRequest({purpose:s.purpose,risk:'routine',state:s.state,questions:{[s.question_id]:s.question}},DEFAULTS),wire=wireRequest(request,'clef-flash');r.request_sha256=digest(request);r.criteria_sha256=digest(s.question.criteria);r.facts_sha256=digest(s.state.facts);r.shared_wire_sha256=digest({state:wire.state,questions:wire.questions});r.wire=wire;
 s.request_hash=r.request_sha256;if(!r.reuse_frozen_prediction)s.sample_id=digest({revision:r.revision,source_sample_id:r.source_sample_id,arm:r.arm,request_hash:s.request_hash,target:s.target});r.sample_id=s.sample_id;
}
console.log(encode(rows));
"""


def node(script, *args, input_rows=None):
    result = subprocess.run(["node", "--input-type=module", "-e", script, *map(str, args)], cwd=REPO,
                            input=None if input_rows is None else json.dumps(input_rows, ensure_ascii=False),
                            capture_output=True, check=True, text=True)
    return json.loads(result.stdout)


def prepare(root, home, output):
    local = helper()
    if output == REPO or REPO in output.parents:
        raise ValueError("PRIVATE_PREPARATION_IN_REPOSITORY_REFUSED")
    if output.exists():
        raise ValueError("FROZEN_PLACEMENT_OUTPUT_EXISTS")
    baseline = root / "artifacts8/predictions.jsonl"
    if local.file_sha(baseline) != PREDICTIONS:
        raise ValueError("FROZEN_BASELINE_CHANGED")
    frozen_manifest = json.loads((root / "artifacts8/manifest.json").read_text())
    if frozen_manifest["checkpoint"] != CHECKPOINT:
        raise ValueError("FROZEN_CHECKPOINT_CHANGED")
    source = node(NODE_SOURCE, home, ORIGINAL, CURRICULUM, ORACLE_SOURCE)
    if source["selected_ids_sha256"] != SELECTED_IDS:
        raise ValueError("PREREGISTERED_SELECTION_CHANGED")
    old_spec = json.loads((home / "evaluation/clef-flash/score-spec.json").read_text())
    if source["core_spec"] != old_spec["evaluation"]:
        raise ValueError("CORE_SCORING_SPEC_CHANGED")
    records = []
    for cohort, selected in [("original-dev", [{"source_sample_id": s["sample_id"], "sample": s} for s in source["dev"]]), ("train-supplement", source["supp"])]:
        for item in selected:
            for arm in ("A", "B"):
                sample = copy.deepcopy(item["sample"]) if arm == "A" else relocate(item["sample"])
                records.append({"revision": REVISION, "cohort": cohort, "arm": arm, "source_sample_id": item["source_sample_id"],
                                "source_split": item["sample"]["split"], "case_id": item["sample"]["raw_refs"]["independent"]["case_id"],
                                "pair_id": local.sha((REVISION + ':' + item["source_sample_id"]).encode()),
                                "reuse_frozen_prediction": cohort == "original-dev" and arm == "A", "sample": sample})
    records = node(NODE_IDENTITIES, input_rows=records)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root / "upstream", local_files_only=True)
    official = local.official(root / "upstream")
    previous = {r["sample_id"]: r for r in map(json.loads, baseline.read_text().splitlines())}
    for row in records:
        encoded = local.encode_complete(tokenizer, row["wire"], official)
        row.update(input_sha256=local.sha(json.dumps(row["wire"], ensure_ascii=False, separators=(',', ':')).encode()),
                   encoded_input_sha256=local.sha(json.dumps(encoded.input_ids, separators=(',', ':')).encode()),
                   encoded_tokens=len(encoded.input_ids), head_option_order=list(encoded.questions[0].option_ids),
                   question_span=list(encoded.questions[0].question_span), option_spans=[list(s) for s in encoded.questions[0].option_spans])
        if row["reuse_frozen_prediction"]:
            old = previous[row["sample_id"]]
            for field in ("shared_wire_sha256", "input_sha256", "encoded_input_sha256", "encoded_tokens", "head_option_order"):
                if row[field] != old[field]:
                    raise ValueError("REUSED_NATIVE_INPUT_CHANGED:" + field)
    assertions = validate_pairs(records)
    assert len(records) == 322 and sum(not r["reuse_frozen_prediction"] for r in records) == 197
    assert len({r['sample_id'] for r in records}) == 322 and len({r['case_id'] for r in records}) == 41
    output.mkdir(parents=True, mode=0o700)
    corpus = ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in records)
    (output / "paired-corpus.jsonl").write_text(corpus); (output / "paired-corpus.jsonl").chmod(0o600)
    calls = [r["sample_id"] for r in records if not r["reuse_frozen_prediction"]]
    local.write_json(output / "new-call-ids.json", calls)
    prereg = {"revision": REVISION, "status": "FROZEN_CPU_PREPARATION_ONLY", "source_datasets": [ORIGINAL, CURRICULUM],
              "original_oracle_source_sha256": ORACLE_SOURCE, "corpus_import_sha256": source["corpus_import_sha256"],
              "source_data_sha256": source["source_data_sha256"], "core_data_sha256": source["core_data_sha256"],
              "core_canonical_rows_sha256": source["core_canonical_rows_sha256"], "supplement_selection": source["selection"],
              "selected_original_ids_sha256": SELECTED_IDS, "core_evaluation": source["core_spec"], "core_gate": old_spec["gate"],
              "priority_ambiguity_sample_ids": source["ambiguous_sample_ids"], "priority_ambiguity_rows": 8,
              "metric_semantics": "Frozen pre-fix oracle agreement; eight original Noul rows retain source-priority ambiguity. Also report unambiguous subset; never replace full-core gates.",
              "supplement_scope": "Nine original TRAIN cases, previously used in training. Development-only coverage; not held-family or independent confirmation.",
              "supplement_metrics": ["agreement", "macro_f1", "NLL", "Brier", "ECE", "language/type/class slices", "pure-rule both-correct", "prediction-switch", "permutation consistency", "Score ordinal MAE/severe error"],
              "paired_metrics": ["core five unchanged gates", "KO Choice", "improvements/regressions per fact", "arm B minus A metrics"],
              "factor": "Exact original policy moved from state into authorized question text; suffix, facts, candidate meanings/order, wrapper, gold unchanged. Supplemental candidate normalization identical in both arms.",
              "checkpoint": CHECKPOINT, "baseline_prediction_file_sha256": PREDICTIONS, "new_calls": 197, "reused_calls": 125,
              "paired_rows": 322, "unique_cases": 41, "arm_rows": 161, "core_rows_per_arm": 125, "supplement_rows_per_arm": 36,
              "fit": {"min_tokens": min(r["encoded_tokens"] for r in records), "max_tokens": max(r["encoded_tokens"] for r in records), "limit": local.MAX_TOKENS, "rejected": 0, "dropped": 0},
              "assertions": assertions, "inference_executed": False, "training_executed": False, "sealed_test_used": False,
              "source_revisions": {"preparation": local.file_sha(__file__), "helper": local.file_sha(Path(__file__).with_name('evaluate.py')),
                                   "contracts": local.file_sha(REPO/'src/contracts.mjs'), "metric_evaluator": old_spec['source_revisions']['evaluator'],
                                   "official_encoder": local.file_sha(root/'upstream/joint_schema_model.py'), "tokenizer": local.file_sha(root/'upstream/tokenizer.json')},
              "frozen_model_identity": frozen_manifest["identity"]}
    local.write_json(output / "preregistration.json", prereg)
    local.write_json(output / "manifest.json", {"revision": REVISION, "preregistration_sha256": local.file_sha(output/'preregistration.json'),
                     "paired_corpus_sha256": local.file_sha(output/'paired-corpus.jsonl'), "new_call_ids_sha256": local.file_sha(output/'new-call-ids.json'),
                     "frozen": True, "inference_executed": False, "new_calls": len(calls), "paired_rows": len(records)})
    return {"output": str(output), "manifest_sha256": local.file_sha(output/'manifest.json'), "new_calls": len(calls), "paired_rows": len(records), "fit": prereg["fit"], "assertions": assertions}


def validate_pairs(records):
    pairs = {}
    for row in records:
        pair = pairs.setdefault(row["pair_id"], {})
        if row['arm'] in pair:
            raise ValueError('DUPLICATE_PLACEMENT_ARM')
        pair[row["arm"]] = row
    for pair in pairs.values():
        if set(pair) != {"A", "B"}:
            raise ValueError("MISSING_PLACEMENT_ARM")
        a, b = pair["A"], pair["B"]
        if b["sample"] != relocate(a["sample"]):
            # Request/sample identities legitimately differ; compare only governed content.
            expected = relocate(a["sample"])
            for key in ("state", "question", "target", "oracle", "lineage", "data_rights"):
                if b["sample"][key] != expected[key]:
                    raise ValueError("PLACEMENT_CHANGED_MEANING:" + key)
        for key in ("facts_sha256", "criteria_sha256", "head_option_order"):
            if a[key] != b[key]:
                raise ValueError("PLACEMENT_CHANGED_IDENTITY:" + key)
    pure = 0
    supplement = [r for r in records if r["cohort"] == "train-supplement"]
    for i, a in enumerate(supplement):
        for b in supplement[i+1:]:
            qa, qb = a["sample"]["question"], b["sample"]["question"]
            strict = lambda q: 'Exception:' in q['instructions'] or '예외:' in q['instructions']
            if a['arm'] != b['arm'] or a['case_id'] != b['case_id'] or strict(qa) == strict(qb):
                continue
            if a['criteria_sha256'] == b['criteria_sha256'] and a['head_option_order'] == b['head_option_order']:
                if a['sample']['target']['value'] == b['sample']['target']['value']:
                    raise ValueError('SUPPLEMENT_RULE_PAIR_NOT_INFORMATIVE')
                pure += 1
    if supplement and pure != 36:
        raise ValueError('SUPPLEMENT_PURE_PAIR_COUNT_CHANGED')
    return {"placement_pairs": len(pairs), "exact_policy_moves": len(pairs), "unchanged_facts_criteria_targets": len(pairs), "supplement_pure_rule_pairs_both_arms": pure}


def admit_consumer(root, experiment, expected_manifest):
    """CPU identity/token admission completes before any model or GPU allocation."""
    local = helper()
    if local.file_sha(experiment/'manifest.json') != expected_manifest:
        raise ValueError('PLACEMENT_MANIFEST_CHANGED')
    manifest = json.loads((experiment/'manifest.json').read_text())
    for name, key in [('paired-corpus.jsonl', 'paired_corpus_sha256'), ('preregistration.json', 'preregistration_sha256'), ('new-call-ids.json', 'new_call_ids_sha256')]:
        if local.file_sha(experiment/name) != manifest[key]:
            raise ValueError('PLACEMENT_ARTIFACT_CHANGED:' + name)
    prereg = json.loads((experiment/'preregistration.json').read_text())
    baseline = json.loads((root/'artifacts8/manifest.json').read_text())
    identity = prereg['frozen_model_identity']
    if baseline['identity'] != identity or baseline['checkpoint'] != CHECKPOINT or local.runtime() != identity['runtime']:
        raise ValueError('MODEL_OR_RUNTIME_IDENTITY_CHANGED')
    if local.file_sha(root/'artifacts8/predictions.jsonl') != PREDICTIONS:
        raise ValueError('FROZEN_BASELINE_CHANGED')
    if local.file_sha(Path(__file__).with_name('evaluate.py')) != identity['evaluator_source_sha256']:
        raise ValueError('FROZEN_FORWARD_HELPER_CHANGED')
    for name, sha in identity['converted_hashes'].items():
        if local.file_sha(root/'mlx8'/name) != sha:
            raise ValueError('CONVERTED_WEIGHT_IDENTITY_CHANGED')
    for name in ('joint_schema_model.py', 'joint_head.safetensors', 'joint_head_config.json', 'config.json', 'tokenizer.json'):
        if local.file_sha(root/'upstream'/name) != identity['input_hashes'][name]:
            raise ValueError('TOKENIZER_OR_OFFICIAL_HEAD_CHANGED')
    rows = list(map(json.loads, (experiment/'paired-corpus.jsonl').read_text().splitlines()))
    validate_pairs(rows)
    calls = [r for r in rows if not r['reuse_frozen_prediction']]
    if len(calls) != 197 or [r['sample_id'] for r in calls] != json.loads((experiment/'new-call-ids.json').read_text()):
        raise ValueError('CALL_SET_CHANGED')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root/'upstream', local_files_only=True)
    official = local.official(root/'upstream')
    encoded = []
    for row in calls:
        wire = row['wire']
        if wire != local.wire(row['sample']) or local.sha(json.dumps(wire, ensure_ascii=False, separators=(',', ':')).encode()) != row['input_sha256']:
            raise ValueError('PREPARED_WIRE_CHANGED')
        item = local.encode_complete(tokenizer, wire, official)
        if local.sha(json.dumps(item.input_ids, separators=(',', ':')).encode()) != row['encoded_input_sha256'] or len(item.input_ids) != row['encoded_tokens'] or list(item.questions[0].option_ids) != row['head_option_order']:
            raise ValueError('PREPARED_NATIVE_ENCODING_CHANGED')
        encoded.append(item)
    return local, prereg, tokenizer, official, calls, encoded


def predict(root, experiment, output, expected_manifest):
    if output.exists():
        raise ValueError('PREDICTION_OUTPUT_EXISTS')
    local, prereg, tokenizer, official, calls, encoded = admit_consumer(root, experiment, expected_manifest)
    # Exactly the existing evaluator's isolated MLX backbone -> BF16 Torch head forward.
    import mlx.core as mx
    import torch
    from mlx_lm.utils import load_model
    local.bounded_mlx()
    stage_started = local.utc_now()
    started = time.monotonic()
    model, _ = load_model(root/'mlx8', lazy=False)
    model.eval()
    schema_head = local.head(root/'upstream', official)
    lm_weight = model.language_model.lm_head.weight
    mx.eval(lm_weight); mx.synchronize()
    lexical = torch.as_tensor(lm_weight)
    assert lexical.dtype == torch.bfloat16 and lexical.shape == (248320, 4096)
    assert model.language_model.model.embed_tokens.weight.dtype == mx.bfloat16
    load_seconds = time.monotonic() - started
    output.mkdir(parents=True, mode=0o700)
    prediction_file = output/'predictions.jsonl'
    statuses, elapsed, aborted = {}, [], False
    with prediction_file.open('x') as stream:
        prediction_file.chmod(0o600)
        for row, record in zip(calls, encoded):
            started = time.monotonic()
            result = {k: row[k] for k in ('sample_id', 'source_sample_id', 'pair_id', 'cohort', 'arm', 'shared_wire_sha256', 'input_sha256', 'encoded_input_sha256', 'encoded_tokens', 'head_option_order')}
            result.update(model_id='clef-flash-mlx8-official-head', checkpoint=CHECKPOINT)
            if aborted:
                result.update(status='error', probabilities=None, error='RUN_ABORTED_AFTER_INFERENCE_ERROR')
            else:
                try:
                    ids = mx.array([record.input_ids], dtype=mx.int32)
                    hidden = model.model(ids)
                    mx.eval(hidden); mx.synchronize()
                    torch_hidden = torch.as_tensor(hidden)
                    batch = official.collate_records([record], tokenizer.pad_token_id, torch.device('mps'))
                    with torch.inference_mode():
                        logits = schema_head(torch_hidden, batch['input_ids'], batch['attention_mask'], batch['records'], lexical)[0][0]
                        p = dict(zip(record.questions[0].option_ids, logits.float().softmax(-1).cpu().tolist()))
                    answer = official.systemone_answer(row['sample']['question'], p)
                    probabilities = {'false': 1-answer['noul'], 'true': answer['noul']} if row['sample']['question']['type'] == 'noul' else answer['probabilities']
                    assert all(0 <= v <= 1 for v in probabilities.values()) and abs(sum(probabilities.values())-1) <= .02
                    result.update(status='ok', probabilities=probabilities)
                    del hidden, torch_hidden, logits, batch
                except Exception as error:
                    result.update(status='error', probabilities=None, error=type(error).__name__)
                    aborted = True  # Account for every remaining input without an automatic retry.
            torch.mps.synchronize()
            result['elapsedMs'] = (time.monotonic()-started)*1000
            elapsed.append(result['elapsedMs']); statuses[result['status']] = statuses.get(result['status'], 0)+1
            stream.write(json.dumps(result, ensure_ascii=False, separators=(',', ':'))+'\n'); stream.flush()
            mx.clear_cache()
            if len(elapsed)%10 == 0:
                print(json.dumps({'completed': len(elapsed), 'total': 197, 'status': statuses}), flush=True)
    local.write_json(output/'manifest.json', {'revision': REVISION, 'checkpoint': CHECKPOINT, 'identity': prereg['frozen_model_identity'],
                    'placement_manifest_sha256': expected_manifest, 'paired_executor_sha256': local.file_sha(__file__),
                    'predictions_file_sha256': local.file_sha(prediction_file), 'sample_count': len(calls), 'status': statuses,
                    'reused_core_A': 125, 'new_inference_rows': 197, 'inference_generation_steps': 0,
                    'dataset_versions': prereg['source_datasets'], 'original_oracle_source_sha256': ORACLE_SOURCE,
                    'original_priority_ambiguity_rows': 8, 'training_executed': False, 'sealed_splits_inferred': False,
                    'stage_started_at_utc': stage_started, 'stage_finished_at_utc': local.utc_now(), 'model_load_seconds': load_seconds,
                    'elapsed_ms': elapsed, 'memory': local.memory(), 'runtime': local.runtime(),
                    'qualification': 'Development policy-placement experiment only; not independent confirmation or promotion'})
    return {'output': str(output), 'manifest_sha256': local.file_sha(output/'manifest.json'), 'new_calls': len(calls), 'status': statuses}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment', type=Path)
    parser.add_argument('--manifest-sha')
    parser.add_argument('--check-consumer', action='store_true')
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()
    if args.predict or args.check_consumer:
        if args.experiment is None or not args.manifest_sha or (args.predict and args.check_consumer):
            parser.error('consumer requires --experiment and --manifest-sha; choose check or predict')
        if args.predict:
            result = predict(args.root.resolve(), args.experiment.resolve(), args.output.resolve(), args.manifest_sha)
        else:
            local, _, _, _, calls, _ = admit_consumer(args.root.resolve(), args.experiment.resolve(), args.manifest_sha)
            result = {'new_calls': len(calls), 'model_inference': False, 'executor_sha256': local.file_sha(__file__), 'frozen_manifest_sha256': args.manifest_sha, 'status': 'READY'}
            local.write_json(args.output.resolve(), result)
    else:
        result = prepare(args.root.resolve(), args.home.resolve(), args.output.resolve())
    print(json.dumps(result, sort_keys=True))
