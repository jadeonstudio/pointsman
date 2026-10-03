"""Read-only official-tokenizer budget check; no weights, inference, download or training."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_export(export_dir, model_dir, diagnose_only=False):
    from transformers import AutoTokenizer
    from laya.agent import Agent
    from laya.common import serialize_state, render_options, build_sequence, QTYPES
    trainer = load_module("corpus_admission_trainer", "training/laya-kit/train_from_export.py")
    worker = load_module("corpus_admission_worker", "workers/laya_worker.py")
    tok = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"), local_files_only=True)
    cfg = trainer.resolve_effective_cfg(json.loads((model_dir / "rl_agent_config.json").read_text()))
    agent = types.SimpleNamespace(tok=tok, cfg=cfg, _to_internal=Agent._to_internal)
    manifest = json.loads((export_dir / "manifest.json").read_text())
    report = {"revision": "official-lossless-corpus-admission-v1", "model_config_sha256": sha(model_dir / "rl_agent_config.json"),
              "admission_sha256": sha(__file__),
              "tokenizer_sha256": trainer.directory_sha256(model_dir / "tokenizer"), "trainer_sha256": sha(ROOT / "training/laya-kit/train_from_export.py"),
              "worker_sha256": sha(ROOT / "workers/laya_worker.py"), "export_manifest_sha256": sha(export_dir / "manifest.json"),
              "max_len": cfg["max_len"], "head_max_len": cfg["head_max_len"], "splits": {}, "diagnose_only": diagnose_only,
              "weights_loaded": False, "inference_executed": False, "training_executed": False}
    total_failed = 0
    for split in ("train", "dev", "calibration", "test"):
        file = export_dir / f"{split}.jsonl"
        if sha(file) != manifest["split_files"][split]["sha256"]:
            raise ValueError("EXPORT_MANIFEST_MISMATCH")
        rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        if len(rows) != manifest["split_files"][split]["count"]:
            raise ValueError("EXPORT_MANIFEST_COUNT_MISMATCH")
        causes, language = Counter(), Counter()
        maximum = dict(head=0, state=0, option=0, sequence=0)
        failed = 0
        for row in rows:
            state, questions = json.loads(row["state"]), json.loads(row["questions"])
            for question in questions.values():
                q = Agent._to_internal(question)
                head = tok(f'{q["t"]} question: {q["ins"]}', add_special_tokens=False)["input_ids"]
                options = [tok(" " + option, add_special_tokens=False)["input_ids"] for option in render_options(q)]
                st = tok(serialize_state(state), add_special_tokens=False)["input_ids"]
                option_cost = sum(len(o) + 1 for o in options)
                expected = len(head) + option_cost + len(st) + 4
                maximum = {"head": max(maximum["head"], len(head)), "state": max(maximum["state"], len(st)),
                           "option": max(maximum["option"], max(map(len, options))), "sequence": max(maximum["sequence"], expected)}
                if len(head) > max(8, cfg["head_max_len"] - option_cost):
                    causes["question_head"] += 1
                if any(len(o) > 48 for o in options):
                    causes["option"] += 1
                if expected > cfg["max_len"]:
                    causes["state_or_total"] += 1
            try:
                worker.assert_lossless(agent, state, questions)
            except ValueError:
                failed += len(questions)
                language[state.get("language", "unknown")] += len(questions)
        info = {"rows": len(rows), "worker_failed": failed, "max_tokens": maximum, "causes": dict(causes), "failure_language": dict(language), "data_sha256": sha(file)}
        if not diagnose_only:
            items, dropped, total = trainer.preprocess_split(rows, tok, cfg, render_options, build_sequence, QTYPES, agent, input_fit="lossless")
            info.update(trainer_admitted=len(items), trainer_dropped=dropped, trainer_total=total)
            if dropped or len(items) != total:
                failed += dropped or total - len(items)
        report["splits"][split] = info
        total_failed += failed
    report["status"] = "DIAGNOSED" if diagnose_only else "PASS" if total_failed == 0 else "FAIL"
    report["failed_rows"] = sum(v["worker_failed"] for v in report["splits"].values())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = inspect_export(args.export_dir, args.model_dir, args.diagnose_only)
    text = json.dumps(result, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if result["status"] in ("PASS", "DIAGNOSED") else 1)
