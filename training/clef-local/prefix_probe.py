"""Bounded exact-prefix experiment. Preparation is CPU-only; --run is explicit."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import time

import evaluate as local

REVISION = "clef-prefix-probe-r1"
CHECKPOINT = "2d78acadca4a2d3865b6c9efd8402d1b2483c1dc6c43a802f81ecc72dcb45e63"
LIMIT = 14 * 1024**3
SEED = 42
THRESHOLDS = (.5, .8, .95)


def digest(value):
    return local.sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def identity(root):
    frozen = json.loads((root / "artifacts8/manifest.json").read_text())
    assert frozen["checkpoint"] == CHECKPOINT, "CHECKPOINT_CHANGED"
    package = Path(importlib.metadata.distribution("mlx-lm").locate_file("mlx_lm/models"))
    sources = {"probe": Path(__file__), "evaluator": Path(local.__file__),
               "official": root / "upstream/joint_schema_model.py"}
    sources.update({f"mlx_lm/{n}": package / n for n in
                    ("qwen3_5.py", "qwen3_next.py", "gated_delta.py", "base.py", "cache.py", "pipeline.py")})
    assets = {n: local.file_sha(root / "upstream" / n) for n in
              ("tokenizer.json", "tokenizer_config.json", "config.json", "joint_head_config.json", "joint_head.safetensors")}
    return {"checkpoint": CHECKPOINT, "frozen_manifest_sha256": local.file_sha(root / "artifacts8/manifest.json"),
            "model": frozen["identity"], "runtime": local.runtime(), "assets": assets,
            "sources": {n: local.file_sha(p) for n, p in sources.items()},
            "hidden_dtype": "bfloat16", "head_dtype": "bfloat16", "recurrent_dtype": "float32"}


def key(model_identity, prefix):
    return digest({"identity": model_identity, "exact_prefix_ids": prefix})


def invalidation_checks(model_identity, prefix, changed_prefix):
    original = key(model_identity, prefix)
    assert original == key(copy.deepcopy(model_identity), list(prefix))
    assert original != key(model_identity, changed_prefix), "CHANGED_STATE_CACHE_HIT"
    assert original != key(model_identity, [*prefix[:-1], prefix[-1] + 1]), "CHANGED_TOKEN_CACHE_HIT"
    for field, value in (("checkpoint", "changed"), ("hidden_dtype", "float32"), ("head_dtype", "float32"),
                         ("recurrent_dtype", "float16"), ("sources", {"changed": "source"})):
        assert original != key({**model_identity, field: value}, prefix), f"CHANGED_{field}_CACHE_HIT"


def prepare(root, output):
    assert not output.exists(), "OUTPUT_EXISTS"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root / "upstream", local_files_only=True)
    official = local.official(root / "upstream")
    model_identity = identity(root)
    questions = [
        {"type": "choice", "instructions": "Select the next routine software investigation using the complete state.",
         "criteria": {"inspect": "Inspect the stated failure evidence", "test": "Run a focused test", "unknown": "Evidence is insufficient"}},
        {"type": "choice", "instructions": "Select the validation boundary supported by the complete state.",
         "criteria": {"api": "Validate the API response", "parser": "Validate deterministic parser behavior", "unknown": "Evidence is insufficient"}},
    ]
    system = official._tokens(tokenizer, f"<|im_start|>system\n{official.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
    cases = []
    for target in (1000, 6000):
        state = {"task": "Synthetic public software diagnostic, independently authored; no private captures or reference answers.",
                 "observations": []}
        while len(system + official._tokens(tokenizer, official.render(state))) < target:
            i = len(state["observations"])
            state["observations"].append({"id": i, "component": f"parser-{i % 7}", "status": "test-passed" if i % 3 else "input-rejected",
                                          "evidence": "A local synthetic fixture reported a schema check; no deployment or model quality is established."})
        prefix = system + official._tokens(tokenizer, official.render(state))
        changed_state = {**state, "task": state["task"] + " Changed observation."}
        changed = system + official._tokens(tokenizer, official.render(changed_state))
        invalidation_checks(model_identity, prefix, changed)
        variants = []
        for n, question in enumerate(questions):
            request = {"model": "clef-flash", "state": state, "questions": {f"probe_{n}": question}}
            encoded = official.encode_record(tokenizer, request, max_length=1_000_000)
            ids = list(encoded.input_ids)
            assert ids[:len(prefix)] == prefix and len(ids) <= 16384, "PREFIX_OR_ADMISSION_CHANGED"
            variants.append({"request": request, "request_sha256": digest(request), "input_ids": ids,
                             "input_sha256": digest(ids), "option_ids": list(encoded.questions[0].option_ids)})
        cases.append({"target_prefix_tokens": target, "prefix_ids": prefix, "prefix_tokens": len(prefix),
                      "prefix_key": key(model_identity, prefix), "changed_state_key": key(model_identity, changed),
                      "variants": variants})
    spec = {"revision": REVISION, "identity": model_identity, "seed": SEED, "cases": cases,
            "provenance": "Independently authored synthetic state/questions; no quality labels or private source data",
            "max_backbone_forwards": 10, "combined_metal_limit_bytes": LIMIT,
            "acceptance": {"max_probability_delta": .002, "same_choices": True, "thresholds": THRESHOLDS,
                           "threshold_crossings": 0, "material_long_cached_to_full_max_ratio": .8},
            "timing": "One pair per schema/length, no p50/p95; prefix is separate; cloning/assembly/head included in warm request",
            "cache_types": {"ArraysCache": 24, "KVCache": 8}, "prepare_gpu_loads": 0}
    local.write_json(output / "spec.json", spec)
    print(json.dumps({"prepared": str(output / "spec.json"), "spec_sha256": local.file_sha(output / "spec.json"),
                      "prefix_tokens": [c["prefix_tokens"] for c in cases], "backbone_forwards": 10}))


def arrays(value):
    if isinstance(value, (tuple, list)):
        return [a for item in value for a in arrays(item)]
    if type(value).__name__ in ("ArraysCache", "KVCache"):
        return arrays(value.state)
    return [value] if type(value).__module__.startswith("mlx.core") and hasattr(value, "dtype") else []


def clone_cache(model, source):
    result = model.language_model.make_cache()
    assert len(result) == len(source) == 32
    for src, dst in zip(source, result):
        assert type(src) is type(dst) and src is not dst
        dst.state = copy.deepcopy(src.state)
        assert all(a is not b for a, b in zip(arrays(src.state), arrays(dst.state))), "ALIASED_CACHE_ARRAY"
    return result


def run(root, experiment, spec_sha):
    assert local.file_sha(experiment / "spec.json") == spec_sha, "SPEC_CHANGED"
    assert not (experiment / "result.json").exists(), "RESULT_EXISTS"
    spec = json.loads((experiment / "spec.json").read_text())
    assert spec["revision"] == REVISION and spec["identity"] == identity(root), "SOURCE_OR_MODEL_CHANGED"
    for n, expected in spec["identity"]["model"]["converted_hashes"].items():
        assert local.file_sha(root / "mlx8" / n) == expected, "WEIGHT_CHANGED"
    from transformers import AutoTokenizer
    import mlx.core as mx
    import torch
    from mlx_lm.utils import load_model
    mx.random.seed(SEED)
    torch.manual_seed(SEED)
    local.bounded_mlx()
    mx.set_memory_limit(int(13.5 * 1024**3))
    tokenizer = AutoTokenizer.from_pretrained(root / "upstream", local_files_only=True)
    official = local.official(root / "upstream")
    phases, pairs, calls = [], [], 0

    def phase(name, operation):
        mx.synchronize(); torch.mps.synchronize()
        started = time.monotonic()
        value = operation()
        mx.eval(*arrays(value)); mx.synchronize(); torch.mps.synchronize()
        elapsed = time.monotonic() - started
        memory = local.memory()
        phases.append({"phase": name, "seconds": elapsed, "memory": memory})
        assert memory["torch_mps_driver_bytes"] <= LIMIT, "COMBINED_METAL_BUDGET"
        local.write_json(experiment / "progress.json", {"phases": phases, "backbone_forwards": calls})
        return value, elapsed

    def forward(ids, cache=None):
        nonlocal calls
        calls += 1
        assert calls <= 10, "FORWARD_BUDGET"
        return model.model(mx.array([ids], dtype=mx.int32), cache=cache)

    loaded, _ = phase("model_load", lambda: load_model(root / "mlx8", lazy=False))
    model, _ = loaded
    model.eval()
    head, _ = phase("head_load", lambda: local.head(root / "upstream", official))
    lexical, _ = phase("lexical_bridge", lambda: torch.as_tensor(model.language_model.lm_head.weight))
    assert lexical.dtype == torch.bfloat16 and lexical.shape == (248320, 4096)
    head_warmed = False
    for case in spec["cases"]:
        prefix = case["prefix_ids"]
        assert key(spec["identity"], prefix) == case["prefix_key"]
        cache = model.language_model.make_cache()
        assert [type(c).__name__ for c in cache].count("ArraysCache") == 24
        assert [type(c).__name__ for c in cache].count("KVCache") == 8
        prefix_hidden, prefix_seconds = phase(f"{len(prefix)}/prefix_prefill", lambda: forward(prefix, cache))
        assert prefix_hidden.dtype == mx.bfloat16
        mx.eval(*arrays([c.state for c in cache])); mx.synchronize()
        cache_metadata = [{"type": type(c).__name__, "arrays": [{"shape": list(a.shape), "dtype": str(a.dtype)} for a in arrays(c.state)]} for c in cache]
        for index, variant in enumerate(case["variants"]):
            request = variant["request"]
            encoded = official.encode_record(tokenizer, request, max_length=1_000_000)
            assert digest(request) == variant["request_sha256"] and list(encoded.input_ids) == variant["input_ids"]
            assert list(encoded.questions[0].option_ids) == variant["option_ids"]
            question = next(iter(request["questions"].values()))
            label = f"{len(prefix)}/{index}"
            batch = official.collate_records([encoded], tokenizer.pad_token_id, torch.device("mps"))

            def score(hidden):
                with torch.inference_mode():
                    logits = head(torch.as_tensor(hidden), batch["input_ids"], batch["attention_mask"], batch["records"], lexical)[0][0]
                    assert torch.isfinite(logits).all().item(), "NONFINITE_LOGITS"
                    probabilities = dict(zip(encoded.questions[0].option_ids, logits.float().softmax(-1).cpu().tolist()))
                assert all(0 <= p <= 1 for p in probabilities.values()) and abs(sum(probabilities.values()) - 1) < 1e-5
                return {"probabilities": probabilities, "answer": official.systemone_answer(question, probabilities)}

            full, full_seconds = phase(label + "/full_backbone", lambda: forward(variant["input_ids"]))
            if not head_warmed:
                phase("head_first_invocation_warmup", lambda: score(full)); head_warmed = True
            full_score, full_head_seconds = phase(label + "/full_head_bridge_postprocess", lambda: score(full))
            del full
            cloned, clone_seconds = phase(label + "/cache_clone", lambda: clone_cache(model, cache))
            suffix, suffix_seconds = phase(label + "/cached_suffix", lambda: forward(variant["input_ids"][len(prefix):], cloned))
            whole, assembly_seconds = phase(label + "/hidden_assembly", lambda: mx.concatenate([prefix_hidden, suffix], axis=1))
            cached_score, cached_head_seconds = phase(label + "/cached_head_bridge_postprocess", lambda: score(whole))
            full_total = full_seconds + full_head_seconds
            cached_total = clone_seconds + suffix_seconds + assembly_seconds + cached_head_seconds
            a, b = full_score["probabilities"], cached_score["probabilities"]
            delta = max(abs(a[k] - b[k]) for k in a)
            crossings = sum((a[k] >= t) != (b[k] >= t) for k in a for t in THRESHOLDS)
            same = full_score["answer"]["choice"] == cached_score["answer"]["choice"]
            pair = {"prefix_tokens": len(prefix), "request_sha256": variant["request_sha256"], "input_tokens": len(encoded.input_ids),
                    "full": full_score, "cached": cached_score, "probability_delta": delta, "threshold_crossings": crossings,
                    "same_choice": same, "full_seconds": full_total, "cached_seconds": cached_total,
                    "ratio": cached_total / full_total, "prefix_seconds": prefix_seconds, "cache_metadata": cache_metadata}
            pairs.append(pair)
            local.write_json(experiment / "pairs.json", {"spec_sha256": spec_sha, "pairs": pairs,
                                                        "backbone_forwards": calls})
            assert delta <= .002 and crossings == 0 and same, "PARITY_FAILED"
            assert all(c.offset == len(prefix) for c in cache if type(c).__name__ == "KVCache"), "PREFIX_CACHE_MUTATED"
            del cloned, suffix, whole, batch
            mx.clear_cache()
        del cache, prefix_hidden
        mx.clear_cache()
    long = [p for p in pairs if p["prefix_tokens"] >= 6000]
    result = {"revision": REVISION, "spec_sha256": spec_sha, "identity": spec["identity"], "phases": phases, "pairs": pairs,
              "backbone_forwards": calls, "correctness": "PASS", "material_long_warm_optimization": all(p["ratio"] <= .8 for p in long),
              "timing_scope": "Single pairs, no p50/p95; cached totals exclude separately recorded prefix prefill",
              "A_or_model_quality_acceptance": False}
    local.write_json(experiment / "result.json", result)
    print(json.dumps({"result": str(experiment / "result.json"), "sha256": local.file_sha(experiment / "result.json"),
                      "backbone_forwards": calls, "material": result["material_long_warm_optimization"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec-sha")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.output)
    else:
        if not args.spec_sha:
            parser.error("run requires --spec-sha")
        run(args.root, args.output, args.spec_sha)
