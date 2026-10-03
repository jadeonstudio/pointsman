"""Pinned BFCL source -> private typed wire pack; no inference or training."""
import argparse
import collections
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

REVISION = "916260dfc116bf06793a1af79b4ec8195b0453b6"
FROZEN = {
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "eval_runner.py": "62d5fffe6caef974cf1e9cff70ba5b0f5f0a1ea9e8dcbf22953193b01b10dd97",
    "multiple.json": "8fb6b043cc80a1a0d5b5a578d5dd0593ed454d922d47034dcfdadfc4e36f7e5d",
    "irrelevance.json": "d5a6da2d26e4f40f535120b75c2b61a5887ce4cf1dc04078c9e33a59901733b7",
    "multiple-reference.json": "a17c86584546eb92899217f55751cecd95dacd2077f30366e19d3a6fa93d5fdd",
    "admission-preregistration.json": "6d94a86b4aea34a28450f4d981de456cff3827b8989ee1b67d63d9c502be9f3d",
    "policy-contrast-preregistration.json": "531dcfcbea6b5a18a7620fcce80fc551b58b83f7a4e75762e1554db3d9bd1c6c",
    "policy-wording-clarification.json": "16aea71aef1654474df5764f8d957e19b18e00119ceda58004e1235d18a7b311",
}
ORIGINS_SHA = "cb9f05f9c81f0cf8ba530a151af2fa124052980003737a5fa63b9c319671e579"
PREPARED_INPUT_SHA = "8eda4839fb57cfc8c9c3e25e774d26fc576c71511c9de710baa7be9422ccf9b9"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value).encode()).hexdigest()


def lines(rows):
    return ("".join(canonical(r) + "\n" for r in rows)).encode()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def question_id(name):
    return "tool_" + re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def origins(data):
    rows, seen = [], set()
    for category in ("multiple", "irrelevance"):
        for index, item in enumerate(data[category]):
            fs = item["function"]
            names = [f.get("name") for f in fs if isinstance(f, dict)]
            if (not isinstance(item["question"], list) or not fs or len(names) != len(fs)
                    or any(not isinstance(n, str) or not n for n in names)
                    or len(set(names)) != len(names)
                    or len({question_id(n) for n in names}) != len(names) or item["id"] in seen):
                raise ValueError("INVALID_OR_COLLIDING_SOURCE_INPUT")
            seen.add(item["id"])
            rows.append({"category": category, "source_index": index, "id": item["id"],
                         "input_identity": digest({"question": item["question"], "function": fs}),
                         "function_ids": sorted(digest(f) for f in fs),
                         "function_set": digest(sorted(fs, key=canonical)), "functions_count": len(fs),
                         "input_characters": len(canonical({"question": item["question"], "function": fs}))})
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    shared = {}
    for i, row in enumerate(rows):
        for tool in row["function_ids"]:
            if tool in shared:
                parent[find(i)] = find(shared[tool])
            else:
                shared[tool] = i
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    for group in groups.values():
        family = digest(sorted({f for r in group for f in r["function_ids"]}))
        bucket = int(hashlib.sha256(("bfcl-tool-selection-r1:" + family).encode()).hexdigest(), 16) % 100
        for row in group:
            row.update(family_id=family, split="train" if bucket < 60 else "dev" if bucket < 80 else "test")
    return rows


def reference_names(value):
    if not isinstance(value, list) or not value or any(not isinstance(x, dict) or len(x) != 1 for x in value):
        raise ValueError("INVALID_BENCHMARK_REFERENCE")
    return {name for call in value for name in call}


def prepare(data, metadata, reference, rule):
    raw = {r["id"]: r for values in data.values() for r in values}
    cases, labels = [], []
    for meta in metadata:
        item = raw[meta["id"]]
        offered = sorted(f["name"] for f in item["function"])
        # Irrelevance's no-call oracle is an explicit official category definition.
        gold = reference_names(reference[item["id"]]) if meta["category"] == "multiple" else set()
        if not gold <= set(offered):
            raise ValueError("REFERENCE_OUTSIDE_CANDIDATES")
        variants = [("all_allowed", offered)]
        if meta["category"] == "multiple":
            variants += [("deny_first_lexicographic_candidate", offered[1:]), ("deny_all", [])]
        for variant, allowed in variants:
            sid = item["id"] + "/" + variant
            questions = {question_id(f["name"]): {"type": "choice", "instructions": rule + " Choose yes only for this candidate: " + f["name"],
                         "criteria": {"yes": "Select this tool.", "no": "Do not select this tool."}} for f in item["function"]}
            request = {"questions": questions, "state": {"conversation": item["question"], "functions": item["function"],
                       "allowed_tools": allowed, "task": "BFCL policy-authorized tool-name selection; no execution"}}
            cases.append({"sample_id": sid, "source_id": item["id"], "source_category": meta["category"], "variant": variant,
                          "family_id": meta["family_id"], "split": meta["split"], "request_identity": digest(request), "request": request})
            labels.append({"sample_id": sid, "target_tool_names": sorted(gold & set(allowed)),
                           "question_values": {question_id(n): "yes" if n in gold and n in allowed else "no" for n in offered},
                           "label_source": "benchmark_reference" if variant == "all_allowed" else "derived_policy_oracle",
                           "provenance": {"source_revision": REVISION, "source_category": meta["category"],
                               "benchmark_gold": "possible_answer function-name set" if meta["category"] == "multiple" else "official no-call category",
                               "evidence_sha256": FROZEN["multiple-reference.json" if meta["category"] == "multiple" else "eval_runner.py"],
                               "derivation": None if variant == "all_allowed" else "published selected names intersect allowed_tools; no substitution"}})
    return cases, labels


def common_wires(cases):
    # Reuse the real detached-request validator, runtime wrapper and isolation checker.
    script = """import fs from 'node:fs';
import {validateRequest,wireRequest} from './src/contracts.mjs';
import {DEFAULTS} from './src/constants.mjs';
import {assertSplitIsolation} from './src/training/dataset.mjs';
const rows=JSON.parse(fs.readFileSync(0,'utf8'));
assertSplitIsolation(rows.map(r=>({group_id:r.family_id,request_hash:r.request_identity,state:r.request.state,split:r.split,lineage:{source_id:r.source_id,template_id:r.family_id}})));
for(const r of rows){const checked=validateRequest({purpose:'select',risk:'routine',...r.request},DEFAULTS);
 r.wire=wireRequest(checked,'clef-flash');delete r.request;}
console.log(JSON.stringify(rows));"""
    result = subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps(cases),
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, check=True, text=True)
    return json.loads(result.stdout)


def token_lengths(rows, upstream):
    sys.dont_write_bytecode = True
    expected = {"joint_schema_model.py": "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3",
                "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
                "tokenizer_config.json": "91a08f825d370d085d692e04cf117cdd7faad7bf18e996f1e6031b6dab03db72",
                "config.json": "66f87f6fb2616b46604daf2a9c67ddc87938296d07156efa34d59b5be49e3238",
                "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"}
    for name, value in expected.items():
        if digest((upstream / name).read_bytes()) != value:
            raise ValueError("TOKENIZER_SOURCE_CHANGED")
    from transformers import AutoTokenizer
    spec = importlib.util.spec_from_file_location("bfcl_clef_encoder", upstream / "joint_schema_model.py")
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    tokenizer = AutoTokenizer.from_pretrained(upstream, local_files_only=True)
    result = []
    for row in rows:
        record = official.encode_record(tokenizer, row["wire"], max_length=1_000_000)
        if len(record.questions) != len(row["wire"]["questions"]) or any(set(q.option_ids) != {"yes", "no"} for q in record.questions):
            raise ValueError("ENCODED_SCHEMA_CHANGED")
        result.append({"sample_id": row["sample_id"], "split": row["split"], "tokens": len(record.input_ids),
                       "token_ids_sha256": digest(list(record.input_ids))})
    return result, {"upstream_revision": "17f0b0ad64efb65d273590632833508766b2aae6", "assets": expected,
                    "transformers": importlib.metadata.version("transformers"), "tokenizers": importlib.metadata.version("tokenizers"),
                    "max_length": 1_000_000, "truncation": False, "weights_loaded": False, "gpu_calls": 0}


def build(archive, output, upstream=None):
    if output.exists():
        raise ValueError("OUTPUT_EXISTS_USE_FRESH_DIRECTORY")
    for name, expected in FROZEN.items():
        if digest((archive / name).read_bytes()) != expected:
            raise ValueError("FROZEN_SOURCE_CHANGED: " + name)
    data = {c: read_rows(archive / (c + ".json")) for c in ("multiple", "irrelevance")}
    metadata = origins(data)
    if digest(lines(metadata)) != ORIGINS_SHA:
        raise ValueError("INPUT_ADMISSION_OR_SPLIT_CHANGED")
    reference_rows = read_rows(archive / "multiple-reference.json")
    reference = {r["id"]: r["ground_truth"] for r in reference_rows}
    if len(reference) != len(reference_rows) or set(reference) != {r["id"] for r in data["multiple"]}:
        raise ValueError("REFERENCE_JOIN_CHANGED")
    rule = json.loads((archive / "policy-wording-clarification.json").read_text())["rule"]
    cases, labels = prepare(data, metadata, reference, rule)
    if digest(lines(cases)) != PREPARED_INPUT_SHA:
        raise ValueError("FROZEN_CANDIDATE_INPUT_CHANGED")
    fields = []
    for row in cases:
        for name in row["request"]["state"]["functions"]:
            tool = name["name"]
            q = row["request"]["questions"][question_id(tool)]
            request = {"state": row["request"]["state"], "questions": {"tool_name": q}}
            fields.append({**{k: v for k, v in row.items() if k not in ("request", "request_identity", "sample_id")},
                           "sample_id": row["sample_id"] + "/" + question_id(tool), "case_id": row["sample_id"],
                           "candidate_tool": tool, "question_id": "tool_name", "request_identity": digest(request), "request": request})
    case_wires, field_wires = common_wires(cases), common_wires(fields)
    lengths, tokenizer = token_lengths(case_wires + field_wires, upstream) if upstream else ([], {"status": "NOT_RUN"})
    output.mkdir(parents=True, mode=0o700)
    files = {}
    split_by_case = {c["sample_id"]: c["split"] for c in cases}
    for split in ("train", "dev", "test"):
        for kind, rows in (("cases", case_wires), ("fields", field_wires), ("reference", labels)):
            path = output / split / (kind + ".jsonl")
            path.parent.mkdir(exist_ok=True, mode=0o700)
            selected = [r for r in rows if r.get("split", split_by_case.get(r["sample_id"])) == split]
            path.write_bytes(lines(selected)); path.chmod(0o600)
            files[str(path.relative_to(output))] = digest(path.read_bytes())
    (output / "LICENSE").write_bytes((archive / "LICENSE").read_bytes())
    files["LICENSE"] = FROZEN["LICENSE"]
    if lengths:
        (output / "token-lengths.jsonl").write_bytes(lines(lengths)); files["token-lengths.jsonl"] = digest(lines(lengths))
    manifest = {"schema": 1, "contract": "bfcl-tool-name-choice-wire-v1", "builder_sha256": digest(Path(__file__).read_bytes()),
        "source_revision": REVISION, "source_hashes": FROZEN, "source_input_freeze_sha256": ORIGINS_SHA,
        "original_candidate_input_sha256": PREPARED_INPUT_SHA, "source_cases": len(metadata),
        "schema_components": len({r["family_id"] for r in metadata}), "case_variants": len(cases), "binary_fields": len(fields),
        "split_cases": dict(collections.Counter(r["split"] for r in metadata)),
        "split_variants": dict(collections.Counter(r["split"] for r in cases)), "split_fields": dict(collections.Counter(r["split"] for r in fields)),
        "split_schema_components": {s: len({r["family_id"] for r in metadata if r["split"] == s}) for s in ("train", "dev", "test")},
        "references": dict(collections.Counter(r["label_source"] for r in labels)), "files": files, "request_validation": "PASS",
        "split_isolation": "PASS; fixed source/function-schema components; no re-splitting", "test_status": "SEALED_NOT_EVALUATED",
        "wire_semantics": "real portable validateRequest+wireRequest; data wrapper added; case multi-field and field common tool_name Choice yes/no are distinct explicit envelopes",
        "tokenizer": tokenizer, "token_summary": {"minimum": min((r["tokens"] for r in lengths), default=None),
            "maximum": max((r["tokens"] for r in lengths), default=None), "encoded_requests": len(lengths),
            "over_2048": sum(r["tokens"] > 2048 for r in lengths), "over_16384": sum(r["tokens"] > 16384 for r in lengths)},
        "training_executed": False, "model_calls": 0, "jev_calls": 0, "pretraining_contamination": "UNKNOWN",
        "claim": "Independent published benchmark name projection plus local policy derivative; no AST/execution/task success/C qualification"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clef-upstream", type=Path)
    args = parser.parse_args()
    result = build(args.archive, args.output, args.clef_upstream)
    print(json.dumps({k: result[k] for k in ("source_cases", "schema_components", "case_variants", "binary_fields", "split_cases", "split_variants", "split_fields", "token_summary", "test_status")}, indent=2))
