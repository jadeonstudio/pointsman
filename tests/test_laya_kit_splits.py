"""Offline behavioral checks for dev/calibration selection and resume identity."""
import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('kit_splits', ROOT / 'training/laya-kit/train_from_export.py')
kit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kit)

class SplitConsumers(unittest.TestCase):
    def test_legacy_loads_but_cannot_select_epochs_on_calibration(self):
        legacy = {'exporter_version': kit.LEGACY_EXPORTER_VERSION}
        self.assertIsNone(kit.selection_split(legacy, False))
        with self.assertRaises(kit.KitError):
            kit.selection_split(legacy, True)
        self.assertEqual(kit.selection_split({'exporter_version': kit.EXPECTED_EXPORTER_VERSION}, True), 'dev')

    def test_epoch_callback_evaluates_only_its_development_input(self):
        node = next(n for n in ast.parse(kit.TRAIN_DDP_SCRIPT).body if isinstance(n, ast.FunctionDef) and n.name == 'make_epoch_end_fn')
        visited, dev, calibration = [], ['dev-item'], ['calibration-item']
        class Model:
            def train(self):
                visited.append('train')
        ns = {'collect_calib_logits': lambda model, items, *args: visited.append(items) or [],
              'compute_calib_agreement': lambda preds: {'choice': 1., 'score': None, 'noul': None, 'mean': 1.}}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<epoch>', 'exec'), ns)
        best = {'score': None, 'epoch': None, 'state_dict': None}
        fn = ns['make_epoch_end_fn'](Model(), dev, None, None, autocast_device='cpu', autocast_dtype=None, autocast_enabled=False,
             rank=0, world_size=1, dist_module=None, select_best_epoch=True, best_state=best, epoch_agreements=[], log_prefix='fixture', keep_best_in_memory=False)
        fn(0)
        self.assertIn(dev, visited)
        self.assertNotIn(calibration, visited)
        self.assertEqual(best['epoch'], 1)

    def test_resume_rejects_changed_dev_and_tokenizer_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); export = root/'export'; model = root/'model'; export.mkdir(); (model/'tokenizer').mkdir(parents=True)
            for name in ('manifest.json','train.jsonl','dev.jsonl','calibration.jsonl'):
                (export/name).write_text('{}\n')
            (model/'model.safetensors').write_bytes(b'weights')
            (model/'tokenizer'/'tokenizer.json').write_text('{}')
            args = dict(export_dir=export, manifest={'exporter_version':kit.EXPECTED_EXPORTER_VERSION}, resolved_model_dir=model,
                derived_model_name='fixture', epochs=1, micro_batch=1, grad_accum=1, device='cpu', mps_autocast='off', dropout=None,
                rdrop_alpha=0, select_best_epoch=True, cfg={}, method='supervised', input_fit='lossless')
            before = kit.build_resume_match_config(**args)
            (export/'dev.jsonl').write_text('{"changed":true}\n')
            (model/'tokenizer'/'tokenizer.json').write_text('{"changed":true}')
            after = kit.build_resume_match_config(**args)
            mismatch = kit.diff_resume_config(before,after)
            self.assertTrue(any('export_dev_sha256' in x for x in mismatch))
            self.assertTrue(any('tokenizer_sha256' in x for x in mismatch))

    def test_training_default_is_supervised_and_test_loading_is_opt_in(self):
        tree = ast.parse(Path(kit.__file__).read_text())
        main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        source = ast.get_source_segment(Path(kit.__file__).read_text(),main)
        self.assertIn('choices=["supervised", "rlcd-grpo"], default="supervised"',source)
        self.assertIn('if args.evaluate_sealed_test:',source)
        self.assertNotIn('"test": str(export_dir / "test.jsonl")',source)

if __name__ == '__main__':
    unittest.main()

try:
    import torch
except ImportError:
    torch = None

@unittest.skipUnless(torch is not None, 'torch unavailable')
class SupervisedObjective(unittest.TestCase):
    def test_choice_noul_score_ce_has_gradients_and_no_rl_noise(self):
        node = next(n for n in ast.parse(kit.TRAIN_DDP_SCRIPT).body if isinstance(n, ast.FunctionDef) and n.name == 'rl_ce_loss_terms')
        ns = {'torch': torch, 'TRAIN_METHOD': 'supervised', 'proper_reward': lambda *args, **kwargs: torch.zeros(3)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<objective>', 'exec'), ns)
        logits = torch.tensor([[1.,2.,99.],[2.,1.,99.],[1.,2.,3.]], requires_grad=True)
        targets = torch.tensor([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]])
        batch = {'marker_mask': torch.tensor([[1,1,0],[1,1,0],[1,1,1]],dtype=torch.bool), 'target': targets, 'qtype': torch.tensor([0,2,1])}
        from unittest import mock
        with mock.patch.object(torch,'randn',side_effect=AssertionError('supervised must not explore RL noise')):
            loss,_ = ns['rl_ce_loss_terms'](logits,batch,'cpu',4,.4)
        expected = (torch.nn.functional.cross_entropy(logits[:2,:2],torch.tensor([1,0]),reduction='sum') + torch.nn.functional.cross_entropy(logits[2:,:],torch.tensor([2]),reduction='sum'))/3
        self.assertAlmostEqual(loss.item(),expected.item(),places=6)
        loss.backward()
        self.assertGreater(logits.grad.abs().sum().item(),0)
        self.assertEqual(logits.grad[0,2].item(),0)
