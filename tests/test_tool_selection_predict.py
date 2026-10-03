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


if __name__ == "__main__":
    unittest.main()
