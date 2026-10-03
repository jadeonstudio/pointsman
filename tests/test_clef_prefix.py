import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

DIRECTORY = Path(__file__).parents[1] / "training/clef-local"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("clef_prefix", DIRECTORY / "prefix_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
sys.path.remove(str(DIRECTORY))


class ClefPrefixTest(unittest.TestCase):
    def test_exact_identity_and_invalidation(self):
        identity = {"checkpoint": "frozen", "hidden_dtype": "bfloat16", "head_dtype": "bfloat16",
                    "recurrent_dtype": "float32", "sources": {"source": "hash"}}
        probe.invalidation_checks(identity, [1, 2, 3], [1, 2, 4])
        with self.assertRaisesRegex(AssertionError, "CHANGED_STATE_CACHE_HIT"):
            probe.invalidation_checks(identity, [1, 2, 3], [1, 2, 3])

    def test_native_cache_objects_and_nested_state_are_independent(self):
        class ArraysCache:
            def __init__(self):
                self.state = [[{"conv": [1, 2]}, {"recurrent": [3]}], None, None]
        class KVCache:
            def __init__(self):
                self.state = [[{"key": [1]}], [{"value": [2]}], 7]
        make = lambda: [ArraysCache() for _ in range(24)] + [KVCache() for _ in range(8)]
        original = make()
        model = SimpleNamespace(language_model=SimpleNamespace(make_cache=make))
        cloned = probe.clone_cache(model, original)
        cloned[0].state[0][0]["conv"][0] = 99
        cloned[24].state[0][0]["key"][0] = 99
        self.assertEqual(original[0].state[0][0]["conv"][0], 1)
        self.assertEqual(original[24].state[0][0]["key"][0], 1)
        self.assertEqual(cloned[24].state[-1], 7)


if __name__ == "__main__":
    unittest.main()
