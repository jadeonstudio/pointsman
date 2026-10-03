import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("tool_producer", Path(__file__).parents[1] / "training/tool-selection/predict.py")
producer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)


class ProducerContractTest(unittest.TestCase):
    def test_choice_and_raw_vector_preserved(self):
        probabilities = {"yes": .50001, "no": .49999}
        choice, result = producer.validate_answer({"choice": "no", "probabilities": probabilities})
        self.assertEqual(choice, "no")
        self.assertIs(result, probabilities)
        enum = {"candidate_000": .2, "none": .8}
        choice, result = producer.validate_answer({"choice": "candidate_000", "probabilities": enum}, enum)
        self.assertEqual(choice, "candidate_000")
        self.assertIs(result, enum)
        with self.assertRaisesRegex(AssertionError, "OUTPUT_OPTIONS_CHANGED"):
            producer.validate_answer({"choice": "candidate_000", "probabilities": enum})

    def test_invalid_vector_rejected(self):
        for values in ({"yes": float("nan"), "no": .5}, {"yes": .3}, {"yes": .1, "no": .1}, {"yes": True, "no": False}):
            with self.assertRaises(AssertionError):
                producer.validate_answer({"choice": "yes", "probabilities": values})

    def test_semantic_hash_excludes_model(self):
        semantic = {"state": {"text": "fixture"}, "questions": {"tool_name": {"type": "choice"}}}
        self.assertEqual(producer.canonical(semantic), producer.canonical({"questions": semantic["questions"], "state": semantic["state"]}))
        self.assertNotEqual(producer.canonical(semantic), producer.canonical({**semantic, "model": "routing"}))

    def test_runtime_load_failure_records_zero_attempts_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            identity = {"model_id": "fixture"}
            producer.local.write_json(output / "manifest.json", {"model": "base", "identity": identity,
                                       "admission": {"ok": 1}, "rows": []})
            real_import = __import__
            def fail_runtime(name, *args, **kwargs):
                if name == "torch":
                    raise ImportError("synthetic runtime unavailable")
                return real_import(name, *args, **kwargs)
            with patch.object(producer, "model_identity", return_value=identity), \
                 patch.object(producer, "inputs", return_value=[]), \
                 patch.object(producer, "admission", return_value=(None, [], [])), \
                 patch("builtins.__import__", side_effect=fail_runtime):
                with self.assertRaises(ImportError):
                    producer.run("base", output, output, output, producer.local.file_sha(output / "manifest.json"))
            result = json.loads((output / "completion.json").read_text())
            self.assertEqual(result["status"], "MODEL_LOAD_FAILED")
            self.assertEqual(result["attempted_calls"], 0)
            self.assertEqual(result["memory"]["samples"], 0)
            self.assertFalse((output / "predictions.jsonl").exists())

    def test_clef4_current_assets_and_weights_are_verified_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("upstream", "mlx4", "artifacts", "package"):
                (root / name).mkdir()
            upstream = ("joint_head.safetensors", "joint_head_config.json", "config.json", "joint_schema_model.py", "tokenizer.json", "tokenizer_config.json")
            sources = ("qwen3_5.py", "qwen3_next.py", "gated_delta.py", "base.py", "cache.py")
            for name in upstream:
                (root / "upstream" / name).write_text("fixture")
            for name in sources:
                (root / "package" / name).write_text("fixture")
            weight = root / "mlx4/model.safetensors"
            weight.write_text("fixture")
            frozen = {"checkpoint": producer.CHECKPOINTS["clef4"], "identity": {
                "input_hashes": {n: producer.local.file_sha(root / "upstream" / n) for n in upstream},
                "converted_hashes": {weight.name: producer.local.file_sha(weight)}}}
            producer.local.write_json(root / "artifacts/manifest.json", frozen)
            with patch.object(producer.importlib.metadata, "distribution") as distribution, \
                 patch.object(producer.local, "runtime", return_value={"fixture": "runtime"}):
                distribution.return_value.locate_file.return_value = root / "package"
                result = producer.model_identity("clef4", root)
                self.assertEqual(result["precision"], "mlx4/BF16-head")
                self.assertEqual(result["converted_storage_bytes"], {weight.name: 7})
                weight.write_text("changed")
                with self.assertRaisesRegex(AssertionError, "CLEF_WEIGHT_CHANGED"):
                    producer.model_identity("clef4", root)


if __name__ == "__main__":
    unittest.main()
