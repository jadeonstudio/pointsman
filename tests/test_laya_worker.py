"""Offline boundary tests. No model, torch, network, or training is used."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('laya_worker', ROOT / 'workers/laya_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

class WorkerBoundaries(unittest.TestCase):
    def assets(self, root):
        for name, content in {'model.safetensors': b'fixture-not-real-weights', 'rl_agent_config.json': b'{}',
                              'encoder/config.json': b'{}', 'tokenizer/tokenizer_config.json': b'{}'}.items():
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)

    def test_fingerprint_reproducible_and_tracks_all_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.assets(root)
            first = worker.fingerprint(str(root))
            self.assertEqual(first, worker.fingerprint(str(root)))
            (root / 'tokenizer/tokenizer_config.json').write_text('{"changed":true}')
            self.assertNotEqual(first, worker.fingerprint(str(root)))

    def test_missing_assets_remote_code_and_symlinks_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with self.assertRaises(ValueError): worker.fingerprint(str(root))
            self.assets(root)
            (root / 'encoder/config.json').write_text('{"auto_map":{"AutoModel":"remote.py"}}')
            with self.assertRaises(ValueError): worker.fingerprint(str(root))
            (root / 'encoder/config.json').write_text('{}')
            (root / 'alias').symlink_to(root / 'encoder', target_is_directory=True)
            with self.assertRaises(ValueError): worker.fingerprint(str(root))

    def test_root_symlink_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            target = root / 'assets'
            target.mkdir()
            self.assets(target)
            (root / 'alias').symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError): worker.fingerprint(str(root / 'alias'))

    def test_lossless_admission_with_synthetic_tokenizer(self):
        class Tok:
            mask_token = '<mask>'
            def __call__(self, value, **kwargs): return {'input_ids': list(range(len(value)))}
        tok = Tok()
        render = lambda q: list(q['crit'])
        serialize = lambda state: state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        def build(t, state, q, max_len, head_max):
            count = len('%s question: %s' % (q['t'], q['ins'])) + sum(len(' '+v)+1 for v in render(q)) + len(serialize(state)) + 4
            return list(range(count)), list(range(len(render(q))))
        common = types.ModuleType('laya.common')
        common.render_options, common.serialize_state, common.build_sequence = render, serialize, build
        agent = types.SimpleNamespace(tok=tok, cfg={'max_len':512,'head_max_len':192},
            _to_internal=lambda q: {'t':q['type'],'ins':q['instructions'],'crit':q['criteria']})
        q={'worker':{'type':'choice','instructions':'Pick.','criteria':{'a':None,'b':None}}}
        with patch.dict(sys.modules, {'laya.common':common}):
            worker.assert_lossless(agent,'small',q)
            with self.assertRaises(ValueError): worker.assert_lossless(agent,'x'*1000,q)
            with self.assertRaises(ValueError): worker.assert_lossless(agent,'contains <mask>',q)
            long={'worker':{'type':'choice','instructions':'Pick.','criteria':{'x'*60:None,'b':None}}}
            with self.assertRaises(ValueError): worker.assert_lossless(agent,'small',long)

def make_synthetic_agent():
    class Tok:
        mask_token = '<mask>'
        def __call__(self, value, **kwargs): return {'input_ids': list(range(len(value)))}
    tok = Tok()
    render = lambda q: list(q['crit'])
    serialize = lambda state: state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    def build(t, state, q, max_len, head_max):
        count = len('%s question: %s' % (q['t'], q['ins'])) + sum(len(' '+v)+1 for v in render(q)) + len(serialize(state)) + 4
        return list(range(count)), list(range(len(render(q))))
    common = types.ModuleType('laya.common')
    common.render_options, common.serialize_state, common.build_sequence = render, serialize, build
    agent = types.SimpleNamespace(tok=tok, cfg={'max_len': 120, 'head_max_len': 60},
        _to_internal=lambda q: {'t': q['type'], 'ins': q['instructions'], 'crit': q['criteria']})
    return agent, common

class FitTaskHead(unittest.TestCase):
    def setUp(self):
        self.agent, self.common = make_synthetic_agent()
        self.questions = {'worker': {'type': 'choice', 'instructions': 'Pick.', 'criteria': {'a': None, 'b': None}}}

    def test_returns_state_unchanged_when_already_lossless(self):
        with patch.dict(sys.modules, {'laya.common': self.common}):
            state = {'task': 'short task', 'context': {'scope': 'local'}}
            fitted, info = worker.fit_task_head(self.agent, state, self.questions)
        self.assertEqual(fitted, state)
        self.assertEqual(info, {'truncated': False})

    def test_truncates_only_the_task_field_and_keeps_context_intact(self):
        with patch.dict(sys.modules, {'laya.common': self.common}):
            state = {'task': 'x' * 500, 'context': {'scope': 'local', 'complete': True}}
            fitted, info = worker.fit_task_head(self.agent, state, self.questions)
            self.assertTrue(info['truncated'])
            self.assertEqual(info['original_chars'], 500)
            self.assertEqual(fitted['context'], state['context'])
            self.assertTrue(fitted['task'].endswith(worker.TRUNCATION_MARK))
            self.assertEqual(info['kept_chars'], len(fitted['task']) - len(worker.TRUNCATION_MARK))
            # the fitted state must itself pass the same admission rule
            worker.assert_lossless(self.agent, fitted, self.questions)

    def test_raises_when_no_prefix_fits_even_empty_task(self):
        with patch.dict(sys.modules, {'laya.common': self.common}):
            long_questions = {'worker': {'type': 'choice', 'instructions': 'Pick.', 'criteria': {'x' * 60: None, 'b': None}}}
            with self.assertRaises(ValueError):
                worker.fit_task_head(self.agent, {'task': 'x' * 10, 'context': {}}, long_questions)

    def test_raises_input_rewrite_refused_without_truncating(self):
        with patch.dict(sys.modules, {'laya.common': self.common}):
            with self.assertRaises(ValueError) as cm:
                worker.fit_task_head(self.agent, {'task': 'contains <mask> token', 'context': {}}, self.questions)
        self.assertEqual(str(cm.exception), 'INPUT_REWRITE_REFUSED')

    def test_non_dict_or_missing_task_state_raises_same_error_as_assert_lossless(self):
        with patch.dict(sys.modules, {'laya.common': self.common}):
            with self.assertRaises(ValueError):
                worker.fit_task_head(self.agent, 'x' * 1000, self.questions)
            with self.assertRaises(ValueError):
                worker.fit_task_head(self.agent, {'context': {}, 'blob': 'x' * 1000}, self.questions)

class PrecisionAndLoadMode(unittest.TestCase):
    def test_resolve_precision_defaults_to_fp32_and_rejects_unknown_values(self):
        self.assertEqual(worker.resolve_precision({}), 'fp32')
        self.assertEqual(worker.resolve_precision({'precision': 'fp32'}), 'fp32')
        self.assertEqual(worker.resolve_precision({'precision': 'fp16'}), 'fp16')
        with self.assertRaises(ValueError): worker.resolve_precision({'precision': 'bf16'})
        with self.assertRaises(ValueError): worker.resolve_precision({'precision': 'int8'})

    def _fake_laya(self):
        class FakeTensor:
            def __init__(self, dtype): self.dtype = dtype
        class FakeModel:
            def __init__(self):
                self.dtype = 'torch.float32'
                self.halved = False
            def parameters(self): return iter([FakeTensor(self.dtype)])
            def half(self):
                self.halved = True
                self.dtype = 'torch.float16'
        class FakeAgent:
            def __init__(self, model_path, device=None):
                self.model_path, self.device, self.model = model_path, device, FakeModel()
        autocast_calls = []
        def autocast(device_type, dtype=None, enabled=True, **kw):
            autocast_calls.append({'device_type': device_type, 'dtype': dtype, 'enabled': enabled})
            return contextlib.nullcontext()
        fake_torch = types.SimpleNamespace(float16='fake-torch-float16', autocast=autocast)
        agent_mod = types.ModuleType('laya.agent')
        agent_mod.Agent, agent_mod.torch = FakeAgent, fake_torch
        agent_mod._fix_tokenizer_config = lambda _: None
        laya_pkg = types.ModuleType('laya')
        laya_pkg.agent = agent_mod
        return laya_pkg, agent_mod, autocast_calls

    def test_load_agent_uses_no_init_weights_when_transformers_supports_it(self):
        laya_pkg, agent_mod, _ = self._fake_laya()
        calls = []
        class FakeCtx:
            def __enter__(self): calls.append('enter')
            def __exit__(self, *a): calls.append('exit')
        init_mod = types.ModuleType('transformers.initialization')
        init_mod.no_init_weights = lambda: FakeCtx()
        transformers_mod = types.ModuleType('transformers')
        transformers_mod.initialization = init_mod
        with patch.dict(sys.modules, {'laya': laya_pkg, 'laya.agent': agent_mod,
                                       'transformers': transformers_mod, 'transformers.initialization': init_mod}):
            agent, load_mode = worker.load_agent('/model', 'cpu', 'fp32')
        self.assertEqual(load_mode, 'no_init_weights')
        self.assertEqual(calls, ['enter', 'exit'])
        self.assertFalse(agent.model.halved)

    def test_load_agent_falls_back_to_default_when_no_init_weights_unavailable(self):
        laya_pkg, agent_mod, _ = self._fake_laya()
        # sys.modules[name] = None is the standard way to force a deterministic ImportError on
        # `import name` without depending on (or polluting via) whatever transformers/torch this
        # test process happens to have installed.
        with patch.dict(sys.modules, {'laya': laya_pkg, 'laya.agent': agent_mod, 'transformers': None, 'transformers.initialization': None}):
            agent, load_mode = worker.load_agent('/model', 'cpu', 'fp32')
        self.assertEqual(load_mode, 'default')

    def test_load_agent_fp16_halves_weights_and_forces_autocast_on_every_device(self):
        # B2: on MPS, half() alone crashes at inference because Agent.system_one only turns on
        # autocast for CUDA. The fp16 path must also force torch.autocast on for this process.
        laya_pkg, agent_mod, autocast_calls = self._fake_laya()
        with patch.dict(sys.modules, {'laya': laya_pkg, 'laya.agent': agent_mod, 'transformers': None, 'transformers.initialization': None}):
            agent, load_mode = worker.load_agent('/model', 'mps', 'fp16')
            self.assertTrue(agent.model.halved)
            self.assertEqual(str(next(agent.model.parameters()).dtype), 'torch.float16')
            with agent_mod.torch.autocast(device_type='mps', dtype='torch.float32', enabled=False):
                pass
        self.assertEqual(len(autocast_calls), 1)
        self.assertEqual(autocast_calls[0], {'device_type': 'mps', 'dtype': 'fake-torch-float16', 'enabled': True})

if __name__ == '__main__':
    unittest.main()
