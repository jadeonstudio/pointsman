"""Parity test: training/laya-kit/train_from_export.py copies fit_task_head()/assert_lossless()
verbatim from workers/laya_worker.py (the kit must stay standalone for Kaggle, so it cannot import
that module). This asserts both copies produce byte-identical output on the same synthetic cases,
offline, with no torch/laya/transformers dependency (both modules only import those lazily inside
functions this test never calls).
"""
import importlib.util
import json
import sys
import types
import unittest

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]

def load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

worker = load_module('laya_worker_for_kit_parity', 'workers/laya_worker.py')
kit = load_module('train_from_export_for_parity', 'training/laya-kit/train_from_export.py')

def make_synthetic_agent(max_len=120, head_max=60):
    class Tok:
        mask_token = '<mask>'
        def __call__(self, value, **kwargs): return {'input_ids': list(range(len(value)))}
    tok = Tok()
    render = lambda q: list(q['crit'])
    serialize = lambda state: state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    def build(t, state, q, max_len, head_max):
        count = len('%s question: %s' % (q['t'], q['ins'])) + sum(len(' ' + v) + 1 for v in render(q)) + len(serialize(state)) + 4
        return list(range(count)), list(range(len(render(q))))
    common = types.ModuleType('laya.common')
    common.render_options, common.serialize_state, common.build_sequence = render, serialize, build
    agent = types.SimpleNamespace(tok=tok, cfg={'max_len': max_len, 'head_max_len': head_max},
        _to_internal=lambda q: {'t': q['type'], 'ins': q['instructions'], 'crit': q['criteria']})
    return agent, common

class KitWorkerParity(unittest.TestCase):
    def setUp(self):
        self.agent, self.common = make_synthetic_agent()
        self.questions = {'worker': {'type': 'choice', 'instructions': 'Pick.', 'criteria': {'a': None, 'b': None}}}

    def run_both(self, state, questions):
        with self._patched():
            worker_out = worker.fit_task_head(self.agent, state, questions)
        with self._patched():
            kit_out = kit.fit_task_head(self.agent, state, questions)
        return worker_out, kit_out

    def _patched(self):
        from unittest.mock import patch
        return patch.dict(sys.modules, {'laya.common': self.common})

    def test_truncation_mark_constant_matches(self):
        self.assertEqual(worker.TRUNCATION_MARK, kit.TRUNCATION_MARK)

    def test_already_lossless_state_is_identical(self):
        state = {'task': 'short task', 'context': {'scope': 'local'}}
        (w_state, w_info), (k_state, k_info) = self.run_both(state, self.questions)
        self.assertEqual(w_state, k_state)
        self.assertEqual(w_info, k_info)

    def test_truncated_long_task_is_identical(self):
        for length in (50, 200, 499, 500, 501, 999):
            state = {'task': 'x' * length, 'context': {'a': 1, 'b': [1, 2, 3]}}
            (w_state, w_info), (k_state, k_info) = self.run_both(state, self.questions)
            self.assertEqual(w_state, k_state, f'length={length}')
            self.assertEqual(w_info, k_info, f'length={length}')

    def test_korean_like_multibyte_task_is_identical(self):
        state = {'task': ('\uC791\uC5C5 \uC9C0\uC2DC\uBB38\uC785\uB2C8\uB2E4 ' * 60), 'context': {'lang': 'ko'}}
        (w_state, w_info), (k_state, k_info) = self.run_both(state, self.questions)
        self.assertEqual(w_state, k_state)
        self.assertEqual(w_info, k_info)

    def test_no_prefix_fits_raises_identically_on_both(self):
        long_questions = {'worker': {'type': 'choice', 'instructions': 'Pick.', 'criteria': {'x' * 60: None, 'b': None}}}
        with self._patched():
            with self.assertRaises(ValueError) as w_cm:
                worker.fit_task_head(self.agent, {'task': 'x' * 10, 'context': {}}, long_questions)
        with self._patched():
            with self.assertRaises(ValueError) as k_cm:
                kit.fit_task_head(self.agent, {'task': 'x' * 10, 'context': {}}, long_questions)
        self.assertEqual(str(w_cm.exception), str(k_cm.exception))

    def test_mask_token_refusal_is_identical(self):
        with self._patched():
            with self.assertRaises(ValueError) as w_cm:
                worker.fit_task_head(self.agent, {'task': 'contains <mask> token', 'context': {}}, self.questions)
        with self._patched():
            with self.assertRaises(ValueError) as k_cm:
                kit.fit_task_head(self.agent, {'task': 'contains <mask> token', 'context': {}}, self.questions)
        self.assertEqual(str(w_cm.exception), 'INPUT_REWRITE_REFUSED')
        self.assertEqual(str(w_cm.exception), str(k_cm.exception))

if __name__ == '__main__':
    unittest.main()
