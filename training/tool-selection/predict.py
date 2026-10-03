"""Blind DEV-only producers; prepare is CPU-only, run requires a frozen manifest."""
import argparse
from collections import Counter
import contextlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
MANIFEST = "7b2a773b53f6169632a475d6256bac45b122924d67290e9a21cdb987591d7d28"
FIELDS = "03cf28ef0229d26692ea034638f9b216b3df457645c29c9a2e556d16320e94b8"
CHECKPOINTS = {"base": "2d115cbafc7a79194d6958794408e727b933887dcd03241d590824c58de67aed",
               "d6": "d145e848f7827fdd0a643b90727e65a8fe67d5637bf8c6c07ab3d26511edd434",
               "typed": "56f6474957ea3e5660562efd7e588ee631045a3c6ff557934787ac252fd5350c",
               "clef8": "2d78acadca4a2d3865b6c9efd8402d1b2483c1dc6c43a802f81ecc72dcb45e63"}
LIMIT = 14 * 1024**3


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


local = module("tool_clef_helpers", REPO / "training/clef-local/evaluate.py")
worker = module("tool_laya_worker", REPO / "workers/laya_worker.py")


def canonical(value):
    return local.sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def inputs(production):
    assert local.file_sha(production / "manifest.json") == MANIFEST, "WIRE_MANIFEST_CHANGED"
    path = production / "dev/fields.jsonl"
    assert local.file_sha(path) == FIELDS, "DEV_FIELDS_CHANGED"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(rows) == len({r["sample_id"] for r in rows}) == 755, "DEV_COHORT_CHANGED"
    for row in rows:
        q = row["wire"]["questions"]
        assert row["split"] == "dev" and set(q) == {"tool_name"}, "FIELD_SCOPE_CHANGED"
        assert q["tool_name"]["type"] == "choice" and set(q["tool_name"]["criteria"]) == {"yes", "no"}
    return rows


def model_identity(name, root):
    common = {"model_id": f"tool-selection/{name}", "checkpoint": CHECKPOINTS[name],
              "producer_sha256": local.file_sha(__file__), "worker_sha256": local.file_sha(worker.__file__)}
    if name == "clef8":
        frozen = json.loads((root / "artifacts8/manifest.json").read_text())
        assert frozen["checkpoint"] == CHECKPOINTS[name], "CLEF_CHECKPOINT_CHANGED"
        assets = {n: local.file_sha(root / "upstream" / n) for n in
                  ("joint_head.safetensors", "joint_head_config.json", "config.json")}
        assert all(h == frozen["identity"]["input_hashes"][n] for n, h in assets.items()), "CLEF_UPSTREAM_ASSET_CHANGED"
        package = Path(importlib.metadata.distribution("mlx-lm").locate_file("mlx_lm/models"))
        paths = [package / n for n in ("qwen3_5.py", "qwen3_next.py", "gated_delta.py", "base.py", "cache.py")]
        paths += [Path(local.__file__), root / "upstream/joint_schema_model.py"]
        common.update(runtime=local.runtime(), frozen_identity=frozen["identity"],
                      frozen_manifest_sha256=local.file_sha(root / "artifacts8/manifest.json"),
                      source_hashes={str(p): local.file_sha(p) for p in paths},
                      tokenizer_hashes={n: local.file_sha(root / "upstream" / n) for n in ("tokenizer.json", "tokenizer_config.json")},
                      upstream_asset_hashes=assets, precision="mlx8/BF16-head", max_tokens=2048)
    else:
        assert worker.fingerprint(root) == CHECKPOINTS[name], "LAYA_CHECKPOINT_CHANGED"
        package = Path(importlib.metadata.distribution("laya").locate_file("laya"))
        common.update(runtime={n: importlib.metadata.version(n) for n in ("laya", "torch", "transformers", "safetensors")},
                      source_hashes={n: local.file_sha(package / n) for n in ("agent.py", "common.py")},
                      config=json.loads((root / "rl_agent_config.json").read_text()),
                      weights_sha256=local.file_sha(root / "model.safetensors"),
                      tokenizer_hashes={p.name: local.file_sha(p) for p in sorted((root / "tokenizer").iterdir()) if p.is_file()},
                      precision="fp16", input_fit="lossless")
        assert common["runtime"]["laya"] == "0.3.4", "LAYA_VERSION_CHANGED"
    return common


def admission(name, root, rows):
    from transformers import AutoTokenizer
    records, details = [], []
    if name == "clef8":
        tokenizer = AutoTokenizer.from_pretrained(root / "upstream", local_files_only=True)
        official = local.official(root / "upstream")
    else:
        from laya.agent import Agent
        from laya.common import render_options, serialize_state
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
        agent = SimpleNamespace(tok=tokenizer, cfg=json.loads((root / "rl_agent_config.json").read_text()), _to_internal=Agent._to_internal)
    for row in rows:
        wire = row["wire"]
        info = {"sample_id": row["sample_id"], "input_identity": canonical({"state": wire["state"], "questions": wire["questions"]})}
        record = None
        try:
            if name == "clef8":
                record = official.encode_record(tokenizer, wire, max_length=1_000_000)
                info["tokens"] = {"sequence": len(record.input_ids)}
                if len(record.input_ids) > 2048:
                    raise ValueError("INPUT_TRUNCATED")
            else:
                q = agent._to_internal(wire["questions"]["tool_name"])
                head = tokenizer(f'{q["t"]} question: {q["ins"]}', add_special_tokens=False)["input_ids"]
                options = [tokenizer(" " + option, add_special_tokens=False)["input_ids"] for option in render_options(q)]
                state = tokenizer(serialize_state(wire["state"]), add_special_tokens=False)["input_ids"]
                info["tokens"] = {"head": len(head), "options": [len(x) for x in options], "state": len(state),
                                  "sequence": len(head) + sum(len(x) + 1 for x in options) + len(state) + 4}
                worker.assert_lossless(agent, wire["state"], wire["questions"])
        except ValueError as error:
            if str(error) not in ("INPUT_TRUNCATED", "INPUT_REWRITE_REFUSED"):
                raise
            info.update(status="unsupported_input", reason=str(error)); record = None
        else:
            info.update(status="ok", reason=None)
        details.append(info); records.append(record)
    return tokenizer, records, details


def prepare(name, root, production, output):
    assert not output.exists(), "OUTPUT_EXISTS"
    rows = inputs(production)
    identity = model_identity(name, root)
    _, _, details = admission(name, root, rows)
    manifest = {"revision": "tool-selection-producer-r1", "model": name, "root": str(root), "identity": identity,
                "wire_manifest_sha256": MANIFEST, "dev_fields_sha256": FIELDS, "sample_count": 755,
                "admission": dict(Counter(x["status"] for x in details)), "rows": details,
                "combined_metal_limit_bytes": LIMIT, "source_scope": "dev/fields.jsonl only; no reference or test reads",
                "output_contract": "sample_id/model_id/checkpoint/input_identity/status/choice/probabilities/elapsed_ms/UTC timestamps",
                "model_loads": 0, "inference_calls": 0, "prepared_at_utc": local.utc_now()}
    local.write_json(output / "manifest.json", manifest)
    print(json.dumps({"manifest": str(output / "manifest.json"), "sha256": local.file_sha(output / "manifest.json"),
                      "admission": manifest["admission"], "maximum_sequence_tokens": max(x["tokens"]["sequence"] for x in details)}))


def validate_answer(answer):
    probabilities = answer["probabilities"]
    assert answer["choice"] in ("yes", "no") and set(probabilities) == {"yes", "no"}, "OUTPUT_OPTIONS_CHANGED"
    assert all(type(x) in (int, float) and 0 <= x <= 1 for x in probabilities.values()), "INVALID_PROBABILITY"
    assert abs(sum(probabilities.values()) - 1) <= .02 and sum(probabilities.values()) > 0, "INVALID_MASS"
    return answer["choice"], probabilities


def run(name, root, production, output, expected):
    assert local.file_sha(output / "manifest.json") == expected, "PRODUCER_MANIFEST_CHANGED"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["model"] == name and manifest["identity"] == model_identity(name, root), "MODEL_OR_SOURCE_CHANGED"
    assert not (output / "predictions.jsonl").exists(), "PREDICTIONS_EXIST"
    rows = inputs(production)
    tokenizer, records, details = admission(name, root, rows)
    assert details == manifest["rows"], "ADMISSION_CHANGED"
    attempted, statuses, aborted = 0, Counter(), False
    loaded_at = local.utc_now(); load_started = time.monotonic()
    admitted = manifest["admission"].get("ok", 0)
    torch = mx = None
    effective_dtype = None
    memory = {"samples": 0, "maximum_sampled_driver_bytes": None, "maximum_sampled_torch_current_bytes": None,
              "mlx_allocator_peak_bytes": None, "torch_allocator_peak_bytes": "UNKNOWN: MPS exposes no peak counter here"}

    def observe_memory(boundary):
        driver, current = torch.mps.driver_allocated_memory(), torch.mps.current_allocated_memory()
        memory["samples"] += 1
        memory["maximum_sampled_driver_bytes"] = max(memory["maximum_sampled_driver_bytes"] or 0, driver)
        memory["maximum_sampled_torch_current_bytes"] = max(memory["maximum_sampled_torch_current_bytes"] or 0, current)
        if mx is not None:
            memory["mlx_allocator_peak_bytes"] = mx.get_peak_memory()
        assert driver <= LIMIT, boundary

    try:
        import torch
        if admitted and name == "clef8":
            import mlx.core as mx
            from mlx_lm.utils import load_model
            local.bounded_mlx(); mx.set_memory_limit(int(13.5 * 1024**3))
            for filename, expected_hash in manifest["identity"]["frozen_identity"]["converted_hashes"].items():
                assert local.file_sha(root / "mlx8" / filename) == expected_hash, "CLEF_WEIGHT_CHANGED"
            model, _ = load_model(root / "mlx8", lazy=False); model.eval()
            official = local.official(root / "upstream"); head = local.head(root / "upstream", official)
            mx.eval(model.language_model.lm_head.weight); mx.synchronize()
            lexical = torch.as_tensor(model.language_model.lm_head.weight)
            effective_dtype = {"embedding": str(model.language_model.model.embed_tokens.weight.dtype),
                               "head": str(next(head.parameters()).dtype)}
        elif admitted:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                agent, _ = worker.load_agent(str(root), "mps", "fp16")
            assert str(agent.device).split(":")[0] == "mps", "DEVICE_FALLBACK_REFUSED"
            effective_dtype = str(next(agent.model.parameters()).dtype)
        if admitted:
            torch.mps.synchronize(); observe_memory("LOAD_MEMORY_BUDGET")
    except Exception as error:
        try:
            if torch is not None:
                observe_memory("LOAD_FAILURE_MEMORY_BUDGET")
        except Exception:
            pass
        local.write_json(output / "completion.json", {"model_id": manifest["identity"]["model_id"],
                         "producer_manifest_sha256": expected, "status": "MODEL_LOAD_FAILED", "attempted_calls": 0,
                         "reason": str(error) if isinstance(error, AssertionError) else type(error).__name__,
                         "error_detail": str(error),
                         "model_load_started_at_utc": loaded_at, "model_load_seconds": time.monotonic() - load_started,
                         "completed_at_utc": local.utc_now(), "effective_parameter_dtype": effective_dtype, "memory": memory})
        raise
    load_seconds = time.monotonic() - load_started
    with (output / "predictions.jsonl").open("x") as stream:
        (output / "predictions.jsonl").chmod(0o600)
        for row, record, info in zip(rows, records, details):
            started_at = local.utc_now(); started = time.monotonic()
            result = {"sample_id": row["sample_id"], "model_id": manifest["identity"]["model_id"], "checkpoint": CHECKPOINTS[name],
                      "input_identity": info["input_identity"], "status": info["status"], "choice": None, "probabilities": None,
                      "started_at_utc": started_at, "producer_manifest_sha256": expected}
            if info["status"] != "ok":
                result["reason"] = info["reason"]
            elif aborted:
                result.update(status="error", reason="RUN_ABORTED_AFTER_INFERENCE_ERROR")
            else:
                attempted += 1
                try:
                    if name == "clef8":
                        hidden = model.model(mx.array([record.input_ids], dtype=mx.int32)); mx.eval(hidden); mx.synchronize()
                        observe_memory("BACKBONE_MEMORY_BUDGET")
                        batch = official.collate_records([record], tokenizer.pad_token_id, torch.device("mps"))
                        with torch.inference_mode():
                            logits = head(torch.as_tensor(hidden), batch["input_ids"], batch["attention_mask"], batch["records"], lexical)[0][0]
                            assert torch.isfinite(logits).all().item(), "NONFINITE_LOGITS"
                            probabilities = dict(zip(record.questions[0].option_ids, logits.float().softmax(-1).cpu().tolist()))
                            answer = official.systemone_answer(row["wire"]["questions"]["tool_name"], probabilities)
                        choice, _ = validate_answer(answer)
                        validate_answer({"choice": choice, "probabilities": probabilities})
                        del hidden, batch, logits
                    else:
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            answer = agent.system_one(row["wire"]["state"], row["wire"]["questions"])["answers"]["tool_name"]
                        assert str(agent.device).split(":")[0] == "mps", "DEVICE_FALLBACK_REFUSED"
                        choice, probabilities = validate_answer(answer)
                    torch.mps.synchronize()
                    observe_memory("INFERENCE_MEMORY_BUDGET")
                    result.update(choice=choice, probabilities=probabilities)
                except Exception as error:
                    result.update(status="error", reason=str(error) if isinstance(error, AssertionError) else type(error).__name__)
                    aborted = True
            result.update(completed_at_utc=local.utc_now(), elapsed_ms=(time.monotonic() - started) * 1000)
            stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"); stream.flush()
            statuses[result["status"]] += 1
            if admitted and name == "clef8":
                mx.clear_cache()
    local.write_json(output / "completion.json", {"model_id": manifest["identity"]["model_id"], "producer_manifest_sha256": expected,
                    "predictions_sha256": local.file_sha(output / "predictions.jsonl"), "attempted_calls": attempted, "status": dict(statuses),
                    "model_load_started_at_utc": loaded_at, "model_load_seconds": load_seconds, "completed_at_utc": local.utc_now(),
                    "effective_parameter_dtype": effective_dtype, "memory": memory,
                    "probability_semantics": "Clef unrounded softmax, official choice; Laya official probabilities already rounded internally, unmodified",
                    "quality_references_read": False, "test_inputs_read": False})
    print(json.dumps({"attempted_calls": attempted, "status": dict(statuses), "completion_sha256": local.file_sha(output / "completion.json")}))


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.model, args.root, args.production, args.output)
    else:
        if not args.manifest_sha:
            parser.error("run requires --manifest-sha")
        run(args.model, args.root, args.production, args.output, args.manifest_sha)
