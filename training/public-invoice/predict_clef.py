"""Blind direct-recorded-input Clef consumer; never opens gold or Jev answers."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time

from oracle import identity

INPUT_SHA = "f3e8826843fbca0f00ad93bcd63ccc272d1d19584be990d8e4622d2a9ddc2a94"
CHECKPOINT = "2d78acadca4a2d3865b6c9efd8402d1b2483c1dc6c43a802f81ecc72dcb45e63"
MODEL_MANIFEST_SHA = "bdafecf2240dd42beb6228b5355f3def4698b1a0681a1dbc892d5fcb7296791d"
OPTIONS = ("10", "15", "20", "30", "7", "none")
CONTRACT = "invoice-recorded-payload-v1"


def helper():
    spec = importlib.util.spec_from_file_location("invoice_clef_helpers", Path(__file__).parents[1] / "clef-local/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def direct_request(row):
    question, state = json.loads(row["question_json"]), json.loads(row["state_json"])
    if row["question_id"] != "discount_days" or row["node_id"] != "semantic" or row["kind"] != "choice":
        raise ValueError("INPUT_SCOPE_CHANGED")
    if question["type"] != "choice" or tuple(question["criteria"]) != OPTIONS:
        raise ValueError("OPTION_CONTRACT_CHANGED")
    if identity({"question": question, "state": state}) != row["canonical_input_sha256"]:
        raise ValueError("CANONICAL_INPUT_CHANGED")
    return {"model": "clef-flash", "state": state, "questions": {"discount_days": question}}


def self_test():
    question = {"type": "choice", "instructions": "fixture untouched", "criteria": {k: k for k in OPTIONS}}
    state = {"fixture": [1, False, None]}
    row = {"case_id": "fixture", "question_id": "discount_days", "node_id": "semantic", "kind": "choice",
           "question_json": json.dumps(question), "state_json": json.dumps(state), "canonical_input_sha256": identity({"question": question, "state": state})}
    request = direct_request(row)
    assert request == {"model": "clef-flash", "state": state, "questions": {"discount_days": question}}
    assert request["questions"]["discount_days"]["instructions"] == "fixture untouched"
    try:
        direct_request({**row, "state_json": "{}"})
    except ValueError as error:
        assert str(error) == "CANONICAL_INPUT_CHANGED"
    else:
        raise AssertionError("ALTERED_STATE_ACCEPTED")
    print(json.dumps({"direct_recorded_payload": "PASS", "altered_input_rejected": "PASS"}))


def run(root, invoice, output, comparison, comparison_sha, predict, max_tokens):
    if output.exists():
        raise ValueError("OUTPUT_EXISTS")
    local = helper()
    if local.file_sha(comparison) != comparison_sha:
        raise ValueError("COMPARISON_PREREG_CHANGED")
    inputs = invoice / "blindjev/discount-days-recorded-inputs.jsonl"
    if local.file_sha(inputs) != INPUT_SHA:
        raise ValueError("RECORDED_INPUT_CHANGED")
    # Hash-check preregistration only. Do not open gold.jsonl or parse label distributions.
    oracle_manifest = json.loads((invoice / "private-gold/manifest.json").read_text())
    prereg_path = invoice / "private-gold/preregistration.json"
    if local.file_sha(prereg_path) != oracle_manifest["preregistration.json"]:
        raise ValueError("ORACLE_PREREG_CHANGED")
    prereg = json.loads(prereg_path.read_text())
    if prereg["source"]["recorded_inputs_sha256"] != INPUT_SHA or prereg["coverage"]["eligible"] != 150 or prereg["recorded_input_match"]["matched"] != 150:
        raise ValueError("ORACLE_INPUT_COHORT_CHANGED")
    rows = [json.loads(line) for line in inputs.read_text().splitlines() if line]
    if len(rows) != 150 or len({r["case_id"] for r in rows}) != 150:
        raise ValueError("CASE_COHORT_CHANGED")
    if local.file_sha(root / "artifacts8/manifest.json") != MODEL_MANIFEST_SHA:
        raise ValueError("FROZEN_MODEL_MANIFEST_CHANGED")
    frozen = json.loads((root / "artifacts8/manifest.json").read_text())
    model_identity = frozen["identity"]
    if frozen["checkpoint"] != CHECKPOINT or local.runtime() != model_identity["runtime"]:
        raise ValueError("MODEL_RUNTIME_CHANGED")
    if local.file_sha(Path(__file__).parents[1] / "clef-local/evaluate.py") != model_identity["evaluator_source_sha256"]:
        raise ValueError("FORWARD_HELPERS_CHANGED")
    for name, expected in model_identity["converted_hashes"].items():
        if local.file_sha(root / "mlx8" / name) != expected:
            raise ValueError("CONVERTED_WEIGHT_CHANGED")
    upstream_manifest = root / "artifacts/full-upstream-hashes.json"
    if local.file_sha(upstream_manifest) != frozen["supplemental_full_upstream_hash_manifest_sha256"]:
        raise ValueError("UPSTREAM_MANIFEST_CHANGED")
    for entry in json.loads(upstream_manifest.read_text())["files"]:
        if not entry["file"].startswith("model-") and local.file_sha(root / "upstream" / entry["file"]) != entry["sha256"]:
            raise ValueError("UPSTREAM_ASSET_CHANGED")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root / "upstream", local_files_only=True)
    official = local.official(root / "upstream")
    encoded, metadata = [], []
    for row in rows:
        request = direct_request(row)
        item = {"case_id": row["case_id"], "question_id": "discount_days", "node_id": "semantic",
                "question_identity": identity(request["questions"]["discount_days"]), "state_identity": identity(request["state"]),
                "semantic_payload_identity": identity({"state": request["state"], "questions": request["questions"]}),
                "full_request_identity": identity(request), "canonical_input_sha256": row["canonical_input_sha256"]}
        try:
            record = official.encode_record(tokenizer, request, max_length=1_000_000)
            if len(record.input_ids) > max_tokens:
                raise ValueError("INPUT_TRUNCATED")
        except ValueError as error:
            if str(error) != "INPUT_TRUNCATED":
                raise
            record = None
            item["admission"] = "unsupported_input"
        else:
            if len(record.questions) != 1 or set(record.questions[0].option_ids) != set(OPTIONS):
                raise ValueError("ENCODED_OPTIONS_CHANGED")
            item.update(admission="ok", encoded_tokens=len(record.input_ids), head_option_order=list(record.questions[0].option_ids),
                        encoded_input_sha256=local.sha(json.dumps(record.input_ids, separators=(",", ":")).encode()))
        encoded.append(record)
        metadata.append(item)
    if max_tokens == 16384:
        # Deterministic input-length selection, before any prediction; this is one of 150 calls.
        first = max(range(len(encoded)), key=lambda i: len(encoded[i].input_ids) if encoded[i] else -1)
        order = [first] + [i for i in range(len(rows)) if i != first]
        rows, encoded, metadata = ([values[i] for i in order] for values in (rows, encoded, metadata))
    output.mkdir(parents=True, mode=0o700)
    contract = {"inference_contract": CONTRACT, "predictor_source_sha256": local.file_sha(__file__), "reference_checkpoint": CHECKPOINT,
                "reference_model_identity": model_identity, "runtime": local.runtime(), "comparison_prereg_sha256": comparison_sha,
                "reference_model_manifest_sha256": MODEL_MANIFEST_SHA,
                "input_path_note": "Direct recorded invoice payload; reuses computational helpers, not the original Pointsman wrapper/input path",
                "recorded_inputs_sha256": INPUT_SHA, "oracle_prereg_sha256": oracle_manifest["preregistration.json"],
                "full_upstream_manifest_sha256": local.file_sha(upstream_manifest), "case_count": 150,
                "max_new_calls": 150, "max_tokens": max_tokens, "admission_revision": "16k" if max_tokens == 16384 else "2048",
                "mlx_limit_bytes": int(13.5*1024**3), "combined_metal_limit_bytes": 14*1024**3, "cache_limit_bytes": 256*1024**2,
                "execution_case_order": [r["case_id"] for r in rows], "first_case_selection": "maximum_complete_input_length" if max_tokens == 16384 else "recorded_order",
                "max_encoded_tokens": max((len(r.input_ids) for r in encoded if r), default=None), "rows": metadata,
                "gold_or_jev_answers_opened": False, "pointsman_wrapper_added": False, "frozen_at_utc": local.utc_now()}
    local.write_json(output / "contract.json", contract)
    print(json.dumps({"preflight": "PASS", "contract_sha256": local.file_sha(output / "contract.json"),
                      "admitted": sum(r is not None for r in encoded), "unsupported": sum(r is None for r in encoded)}), flush=True)
    if not predict:
        return
    import mlx.core as mx
    import torch
    from mlx_lm.utils import load_model
    local.bounded_mlx()
    mx.set_memory_limit(int(13.5*1024**3))
    stage_start = local.utc_now()
    started = time.monotonic()
    model, _ = load_model(root / "mlx8", lazy=False)
    model.eval()
    head = local.head(root / "upstream", official)
    mx.eval(model.language_model.lm_head.weight); mx.synchronize()
    lexical = torch.as_tensor(model.language_model.lm_head.weight)
    assert lexical.dtype == torch.bfloat16 and lexical.shape == (248320, 4096)
    assert model.language_model.model.embed_tokens.weight.dtype == mx.bfloat16
    assert torch.mps.driver_allocated_memory() <= 14*1024**3, "MODEL_LOAD_MEMORY_BUDGET"
    load_seconds = time.monotonic()-started
    elapsed, statuses, attempted, aborted = [], {}, 0, False
    prediction_file = output / "predictions.jsonl"
    with prediction_file.open("x") as stream:
        prediction_file.chmod(0o600)
        for row, record, meta in zip(rows, encoded, metadata):
            started = time.monotonic()
            result = {**meta, "model_id": "clef_flash_mlx8_torch_bf16", "checkpoint": CHECKPOINT, "inference_contract": CONTRACT,
                      "admission_revision": contract["admission_revision"],
                      "predictor_source_sha256": contract["predictor_source_sha256"]}
            if record is None:
                result.update(status="unsupported_input", choice=None, probabilities=None, error="INPUT_TRUNCATED")
            elif aborted:
                result.update(status="error", choice=None, probabilities=None, error="RUN_ABORTED_AFTER_INFERENCE_ERROR")
            else:
                attempted += 1
                try:
                    hidden = model.model(mx.array([record.input_ids], dtype=mx.int32))
                    mx.eval(hidden); mx.synchronize()
                    assert torch.mps.driver_allocated_memory() <= 14*1024**3, "BACKBONE_MEMORY_BUDGET"
                    batch = official.collate_records([record], tokenizer.pad_token_id, torch.device("mps"))
                    with torch.inference_mode():
                        logits = head(torch.as_tensor(hidden), batch["input_ids"], batch["attention_mask"], batch["records"], lexical)[0][0]
                        assert torch.isfinite(logits).all().item(), "NONFINITE_LOGITS"
                        p = dict(zip(record.questions[0].option_ids, logits.float().softmax(-1).cpu().tolist()))
                    assert torch.mps.driver_allocated_memory() <= 14*1024**3, "HEAD_MEMORY_BUDGET"
                    answer = official.systemone_answer(json.loads(row["question_json"]), p)
                    probabilities = answer["probabilities"]
                    assert tuple(probabilities) == OPTIONS and all(0 <= v <= 1 for v in probabilities.values()) and abs(sum(probabilities.values())-1) <= .02
                    result.update(status="ok", choice=answer["choice"], probabilities=probabilities)
                    del hidden, logits, batch
                except Exception as error:
                    result.update(status="error", choice=None, probabilities=None,
                                  error=str(error) if isinstance(error, AssertionError) else type(error).__name__)
                    aborted = True
            torch.mps.synchronize()
            result["memory"] = local.memory()
            result["elapsedMs"] = (time.monotonic()-started)*1000
            elapsed.append(result["elapsedMs"]); statuses[result["status"]] = statuses.get(result["status"], 0)+1
            stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":"))+"\n"); stream.flush()
            mx.clear_cache()
            if len(elapsed) == 1 or len(elapsed)%10 == 0:
                print(json.dumps({"completed": len(elapsed), "total": 150, "attempted_calls": attempted, "status": statuses}), flush=True)
    warm = elapsed[1:]
    local.write_json(output / "manifest.json", {"inference_contract": CONTRACT, "predictor_source_sha256": local.file_sha(__file__),
                    "reference_checkpoint": CHECKPOINT, "reference_model_identity": model_identity, "runtime": local.runtime(),
                    "contract_sha256": local.file_sha(output / "contract.json"), "predictions_file_sha256": local.file_sha(prediction_file),
                    "sample_count": 150, "attempted_calls": attempted, "status": statuses, "stage_started_at_utc": stage_start,
                    "admission_revision": contract["admission_revision"], "max_tokens": max_tokens,
                    "execution_case_order": contract["execution_case_order"],
                    "stage_finished_at_utc": local.utc_now(), "model_load_seconds": load_seconds,
                    "elapsed_ms": {"first": elapsed[0], "warm_median": statistics.median(warm), "warm_p95": sorted(warm)[int(.95*len(warm))-1]},
                    "memory": local.memory(), "no_jev_speed_claim": True, "gold_or_jev_answers_opened": False})
    print(json.dumps({"output": str(output), "attempted_calls": attempted, "status": statuses, "manifest_sha256": local.file_sha(output / "manifest.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--invoice", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--comparison-sha")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--max-tokens", type=int, choices=(2048, 16384), default=2048)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not all((args.root, args.invoice, args.output, args.comparison, args.comparison_sha)):
            parser.error("root/invoice/output/comparison/comparison-sha required")
        run(args.root, args.invoice, args.output, args.comparison, args.comparison_sha, args.predict, args.max_tokens)
