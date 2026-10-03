"""Unit tests for train_from_export.py's --epochs / --resume / --keep-resume
(--local only, 2026-09-25).

- --epochs N drives both the loop count and the cosine schedule length (default 4).
- Every completed epoch writes <output-dir>/resume/epoch-000N/ atomically
  (temp dir + rename, then the LATEST pointer is replaced), keeping only the
  latest epoch plus the best-epoch snapshot.
- --resume refuses on any configuration mismatch, dies without a state, and
  continues so that the final weights/selection equal an uninterrupted run.

The argument/config tests are standard library only. The training tests exec
the embedded TRAIN_DDP_SCRIPT and run main_local() end to end on CPU with a tiny
stand-in model (same stubbing idea as tests/test_laya_kit_regularization.py);
they need torch + transformers + laya (the laya venv) and are skipped without them.
"""
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kit = load_module('train_from_export_for_resume_tests', 'training/laya-kit/train_from_export.py')

try:
    import torch
    import transformers  # noqa: F401
    import laya.common  # noqa: F401
    HAVE_TORCH_STACK = True
except ImportError:  # pragma: no cover - depends on the interpreter
    HAVE_TORCH_STACK = False


def fresh_script_ns():
    """A NEW exec of TRAIN_DDP_SCRIPT per call, so per-test patches of its
    globals never leak into other tests."""
    ns = {'__name__': 'train_ddp_script_resume_test'}
    exec(compile(kit.TRAIN_DDP_SCRIPT, '<TRAIN_DDP_SCRIPT>', 'exec'), ns)
    return ns


def write_fake_state(output_dir, config, epochs_completed=1, fmt=None):
    epoch_dir = Path(output_dir) / 'resume' / f'epoch-{epochs_completed:04d}'
    epoch_dir.mkdir(parents=True)
    (epoch_dir / 'training_state.pt').write_bytes(b'')
    (epoch_dir / 'state.json').write_text(json.dumps({
        'format': fmt or kit.RESUME_FORMAT, 'config': config, 'info': {},
        'epochs_completed': epochs_completed, 'global_step': 10,
    }), encoding='utf-8')
    (Path(output_dir) / 'resume' / 'LATEST').write_text(epoch_dir.name + '\n', encoding='utf-8')
    return epoch_dir


# ---------------------------------------------------------------------------
# 1. Argument validation (stdlib only).
# ---------------------------------------------------------------------------
class ValidateEpochsAndResumeArgs(unittest.TestCase):
    def test_default_is_four_epochs(self):
        self.assertEqual(kit.DEFAULT_EPOCHS, 4)
        self.assertEqual(kit.validate_epochs_and_resume_args(None, False, False, local=True), 4)
        self.assertEqual(kit.validate_epochs_and_resume_args(None, False, False, local=False), 4)

    def test_epochs_range(self):
        for ok in (1, 6, 20):
            self.assertEqual(kit.validate_epochs_and_resume_args(ok, False, False, local=True), ok)
        for bad in (0, -1, 21, 100):
            with self.assertRaises(kit.KitError, msg=repr(bad)):
                kit.validate_epochs_and_resume_args(bad, False, False, local=True)

    def test_flags_rejected_outside_local(self):
        for epochs, resume, keep in ((6, False, False), (4, False, False), (None, True, False), (None, False, True)):
            with self.assertRaises(kit.KitError) as ctx:
                kit.validate_epochs_and_resume_args(epochs, resume, keep, local=False)
            self.assertIn('--local only', str(ctx.exception))


class MainEpochsResumeCliWiring(unittest.TestCase):
    """main() validates the new flags right after argparse, before touching the export."""

    def run_main(self, *extra):
        argv = ['train_from_export.py', '--export-dir', '/nonexistent-export-for-test',
                '--output-dir', '/nonexistent-output-for-test', *extra]
        with mock.patch.object(sys, 'argv', argv), mock.patch('sys.stderr'):
            with self.assertRaises((kit.KitError, SystemExit)) as ctx:
                kit.main()
        return ctx.exception

    def test_epochs_out_of_range_dies_before_export_check(self):
        for value in ('0', '21'):
            exc = self.run_main('--local', '--epochs', value)
            self.assertIn('--epochs must satisfy', str(exc))

    def test_non_integer_epochs_rejected_by_argparse(self):
        self.assertIsInstance(self.run_main('--local', '--epochs', '2.5'), SystemExit)

    def test_new_flags_rejected_without_local(self):
        for extra in (('--epochs', '6'), ('--resume',), ('--keep-resume',)):
            self.assertIn('--local only', str(self.run_main(*extra)), extra)

    def test_valid_flags_reach_export_check(self):
        exc = self.run_main('--local', '--epochs', '6', '--resume', '--keep-resume')
        self.assertIsInstance(exc, kit.KitError)
        self.assertIn('manifest.json not found', str(exc))

    def test_local_cmd_passes_epochs_and_resume_argv_slots(self):
        source = (ROOT / 'training/laya-kit/train_from_export.py').read_text(encoding='utf-8')
        self.assertIn('str(resolved_epochs), json.dumps({"match": run_match, "info": run_info}),', source)
        self.assertIn('"epochs": resolved_epochs, "micro_batch": resolved_batch_size,', source)


# ---------------------------------------------------------------------------
# 2. Resume configuration checks (stdlib only).
# ---------------------------------------------------------------------------
class ResumeConfigChecks(unittest.TestCase):
    CONFIG = {'format': 'laya-kit-resume-v1', 'epochs': 6, 'dropout': 0.1, 'rdrop_alpha': 0.0,
              'export_manifest_sha256': 'aaa', 'micro_batch': 4, 'grad_accum': 16}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_format_constant_matches_embedded_script(self):
        self.assertIn(f'RESUME_FORMAT = "{kit.RESUME_FORMAT}"', kit.TRAIN_DDP_SCRIPT)
        self.assertIn(f'RESUME_DIRNAME = "{kit.RESUME_DIRNAME}"', kit.TRAIN_DDP_SCRIPT)

    def test_resume_without_state_dies_clearly(self):
        with self.assertRaises(kit.KitError) as ctx:
            kit.check_resume_request(self.out, dict(self.CONFIG), resume=True)
        self.assertIn('no resume state exists', str(ctx.exception))
        # A resume dir holding only a partial temp dir (interrupted first save) is still "no state".
        (Path(self.out) / 'resume' / '.epoch-0001.tmp-123').mkdir(parents=True)
        with self.assertRaises(kit.KitError):
            kit.check_resume_request(self.out, dict(self.CONFIG), resume=True)

    def test_fresh_run_refuses_to_overwrite_existing_state(self):
        write_fake_state(self.out, dict(self.CONFIG))
        with self.assertRaises(kit.KitError) as ctx:
            kit.check_resume_request(self.out, dict(self.CONFIG), resume=False)
        self.assertIn('Pass --resume', str(ctx.exception))

    def test_fresh_run_without_state_proceeds(self):
        self.assertIsNone(kit.check_resume_request(self.out, dict(self.CONFIG), resume=False))

    def test_config_mismatch_refused_listing_every_mismatch(self):
        write_fake_state(self.out, dict(self.CONFIG))
        current = dict(self.CONFIG, epochs=4, dropout=None, export_manifest_sha256='bbb')
        with mock.patch('builtins.print'), self.assertRaises(kit.KitError) as ctx:
            kit.check_resume_request(self.out, current, resume=True)
        msg = str(ctx.exception)
        self.assertIn('refusing to resume', msg)
        self.assertIn('epochs: saved=6 current=4', msg)
        self.assertIn('dropout: saved=0.1 current=None', msg)
        self.assertIn("export_manifest_sha256: saved='aaa' current='bbb'", msg)
        self.assertNotIn('grad_accum', msg)

    def test_extra_or_missing_key_is_a_mismatch(self):
        write_fake_state(self.out, dict(self.CONFIG))
        with self.assertRaises(kit.KitError) as ctx:
            kit.check_resume_request(self.out, dict(self.CONFIG, new_key=1), resume=True)
        self.assertIn('new_key: saved=None current=1', str(ctx.exception))

    def test_wrong_format_refused(self):
        write_fake_state(self.out, dict(self.CONFIG), fmt='other-v0')
        with self.assertRaises(kit.KitError) as ctx:
            kit.check_resume_request(self.out, dict(self.CONFIG), resume=True)
        self.assertIn("format: saved='other-v0'", str(ctx.exception))

    def test_matching_config_returns_state_with_epoch_dir(self):
        epoch_dir = write_fake_state(self.out, dict(self.CONFIG), epochs_completed=2)
        with mock.patch('builtins.print'):
            state = kit.check_resume_request(self.out, dict(self.CONFIG), resume=True)
        self.assertEqual(state['epochs_completed'], 2)
        self.assertEqual(state['epoch_dir'], str(epoch_dir))

    def test_latest_pointing_at_missing_dir_is_an_error_not_no_state(self):
        root = Path(self.out) / 'resume'
        root.mkdir()
        (root / 'LATEST').write_text('epoch-0003\n', encoding='utf-8')
        with self.assertRaises(kit.KitError) as ctx:
            kit.read_resume_state(self.out)
        self.assertIn('damaged', str(ctx.exception))

    def test_match_config_covers_run_settings_but_not_max_steps(self):
        export_dir = Path(self.out) / 'export'
        model_dir = Path(self.out) / 'model'
        export_dir.mkdir()
        model_dir.mkdir()
        for name in ('manifest.json', 'train.jsonl', 'calibration.jsonl'):
            (export_dir / name).write_text(name, encoding='utf-8')
        (model_dir / 'model.safetensors').write_bytes(b'weights')
        kwargs = dict(export_dir=export_dir, manifest={'exporter_version': 'v', 'dataset_version': 'd',
                                                       'source_data_sha256': 's'},
                      resolved_model_dir=model_dir, derived_model_name='m-pointsman-ft', epochs=6, micro_batch=4,
                      grad_accum=16, device='mps', mps_autocast='off', dropout=0.1, rdrop_alpha=0.0,
                      select_best_epoch=True, cfg={'max_len': 1024, 'head_max_len': 256})
        a = kit.build_resume_match_config(**kwargs)
        self.assertEqual(a, kit.build_resume_match_config(**kwargs))
        for key in ('epochs', 'micro_batch', 'grad_accum', 'dropout', 'rdrop_alpha', 'select_best_epoch',
                    'device', 'mps_autocast', 'max_len', 'head_max_len', 'export_manifest_sha256',
                    'export_train_sha256', 'model_weights_sha256', 'train_script_sha256'):
            self.assertIn(key, a)
        self.assertNotIn('max_steps', a)
        (export_dir / 'train.jsonl').write_text('changed', encoding='utf-8')
        b = kit.build_resume_match_config(**kwargs)
        self.assertEqual(kit.diff_resume_config(a, b)[0].split(':')[0], 'export_train_sha256')
        self.assertEqual(len(kit.diff_resume_config(a, b)), 1)

    def test_items_digest_is_deterministic_and_content_sensitive(self):
        items = [{'ids': [1, 2], 'markers': [1], 'qtype': 0, 'target': [0.25, 0.75], 'label': 1}]
        self.assertEqual(kit.items_digest(items), kit.items_digest(json.loads(json.dumps(items))))
        changed = [dict(items[0], target=[0.75, 0.25])]
        self.assertNotEqual(kit.items_digest(items), kit.items_digest(changed))
        self.assertNotEqual(kit.items_digest(items), kit.items_digest(items + items))


# ---------------------------------------------------------------------------
# 3. main_local() end to end with a tiny CPU stand-in model.
# ---------------------------------------------------------------------------
class Interrupted(BaseException):
    """Simulated Ctrl-C (BaseException like KeyboardInterrupt)."""


if HAVE_TORCH_STACK:
    class TinyEncoder(torch.nn.Module):
        def __init__(self, vocab=32, d=8):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, d)
            self.config = types.SimpleNamespace(
                save_pretrained=lambda path: (os.makedirs(path, exist_ok=True), Path(path, "tokenizer_config.json").write_text('{"tokenizer_class":"PreTrainedTokenizerFast"}')))

        def gradient_checkpointing_enable(self, **_kwargs):
            pass

    class TinyDecisionModel(torch.nn.Module):
        """DecisionModel's forward signature/outputs; its own dropout makes the
        torch RNG state matter, like the RLCD noise does."""
        interrupt_at_call = None

        def __init__(self, d=8):
            super().__init__()
            self.encoder = TinyEncoder(d=d)
            self.drop = torch.nn.Dropout(0.2)
            self.score = torch.nn.Linear(d, 1)
            self.act = torch.nn.Linear(d, 2)
            self.calls = 0

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            self.calls += 1
            if self.interrupt_at_call is not None and self.calls >= self.interrupt_at_call:
                raise Interrupted()
            h = self.drop(self.encoder.emb(input_ids))
            m = torch.gather(h, 1, marker_pos[:, :, None].expand(-1, -1, h.size(-1)))
            logits = self.score(m).squeeze(-1).float().masked_fill(~marker_mask, -1e4)
            return logits, self.act(h[:, 0])


def make_items(n, seed):
    import random as _random
    rng = _random.Random(seed)
    items = []
    for i in range(n):
        k = (2, 3, 4)[i % 3]
        target = [rng.random() for _ in range(k)]
        s = sum(target)
        target = [t / s for t in target]
        ids = [1] + [rng.randrange(3, 32) for _ in range(k + rng.randrange(0, 4))]
        items.append({'ids': ids, 'markers': list(range(1, k + 1)), 'target': target,
                      'qtype': (0, 1, 2)[i % 3], 'label': target.index(max(target))})
    return items


RUN_CONFIG = {'match': {'format': 'laya-kit-resume-v1', 'epochs': 3, 'test': True},
              'info': {'started_at': '2026-09-25T00:00:00Z'}}


@unittest.skipUnless(HAVE_TORCH_STACK, 'needs torch + transformers + laya (laya venv)')
class MainLocalResume(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.model_dir = self.tmp / 'model'
        self.model_dir.mkdir()
        (self.model_dir / 'rl_agent_config.json').write_text('{"head_layers": 2}', encoding='utf-8')
        torch.manual_seed(1234)
        self.base_weights = {k: v.clone() for k, v in TinyDecisionModel().state_dict().items()}
        self.train_path = self.tmp / 'train_items.pt'
        self.calib_path = self.tmp / 'calib_items.pt'
        torch.save(make_items(12, seed=1), self.train_path)
        torch.save(make_items(9, seed=2), self.calib_path)
        TinyDecisionModel.interrupt_at_call = None

    def tearDown(self):
        TinyDecisionModel.interrupt_at_call = None
        self._tmp.cleanup()

    def script(self):
        ns = fresh_script_ns()
        ns['build_model_with_encoder_dropout'] = lambda cfg, encoder_dir, dropout: (TinyDecisionModel(), None)
        ns['load_file'] = lambda path: {k: v.clone() for k, v in self.base_weights.items()}
        tok = types.SimpleNamespace(pad_token_id=0, save_pretrained=lambda path: (os.makedirs(path, exist_ok=True), Path(path, "tokenizer_config.json").write_text('{"tokenizer_class":"PreTrainedTokenizerFast"}')))
        ns['AutoTokenizer'] = types.SimpleNamespace(from_pretrained=lambda path: tok)
        # Large LRs so the tiny model's calibration agreement actually moves between epochs.
        ns['LR_ENCODER'] = 0.05
        ns['LR_HEAD'] = 0.05
        return ns

    def argv(self, out, epochs='3', resume='fresh', keep='clean', run_config=RUN_CONFIG, legacy=False):
        base = ['train_ddp.py', str(self.model_dir), str(out), str(self.train_path), str(self.calib_path),
                'm-pointsman-ft', 'multilingual', 'v', 'local', '1', 'cpu', '2', '2', 'off', '0', 'none', '0.0']
        if legacy:
            return base
        return base + [epochs, json.dumps(run_config), resume, keep]

    def run_local(self, argv, ns=None):
        # Epoch selection uses independent dev items; temperature fitting keeps calib_path.
        out = Path(argv[2]); out.mkdir(parents=True, exist_ok=True)
        torch.save(make_items(9, seed=3), out / "dev_items.pt")
        ns = ns or self.script()
        with mock.patch.object(sys, 'argv', argv), mock.patch('builtins.print'):
            ns['main_local']()
        return ns

    @staticmethod
    def outputs(out):
        out = Path(out)
        selection = json.loads((out / 'epoch_selection.json').read_text(encoding='utf-8'))
        cfg = json.loads((out / 'rl_agent_config.json').read_text(encoding='utf-8'))
        run = json.loads((out / 'local_training_run.json').read_text(encoding='utf-8'))
        return {
            'weights': (out / 'model.safetensors').read_bytes(),
            'selection': selection,
            'temperature': cfg['temperature'],
            'global_step': run['global_step'],
            'epochs_completed': run['epochs_completed'],
        }

    def test_default_argv_passes_four_epochs_to_loop_and_schedule_and_writes_no_resume(self):
        ns = self.script()
        seen = {}
        real_build, real_loop = ns['build_optimizer_and_scheduler'], ns['run_training_loop']

        def spy_build(named_params, n_items, micro_batch, grad_accum, epochs):
            seen['schedule_epochs'] = epochs
            opt, sched = real_build(named_params, n_items, micro_batch, grad_accum, epochs)
            seen['T_max'] = sched.T_max
            return opt, sched

        def spy_loop(*args, **kwargs):
            seen['loop_epochs'] = kwargs['epochs']
            seen['checkpoint_fn'] = kwargs['checkpoint_fn']
            return real_loop(*args, **kwargs)

        ns['build_optimizer_and_scheduler'] = spy_build
        ns['run_training_loop'] = spy_loop
        out = self.tmp / 'out-default'
        self.run_local(self.argv(out, legacy=True), ns)
        self.assertEqual(ns['EPOCHS'], 4)
        self.assertEqual(seen['schedule_epochs'], 4)
        self.assertEqual(seen['loop_epochs'], 4)
        self.assertEqual(seen['T_max'], (12 // (2 * 2)) * 4)
        self.assertIsNone(seen['checkpoint_fn'])
        self.assertFalse((out / 'resume').exists())
        run = json.loads((out / 'local_training_run.json').read_text(encoding='utf-8'))
        self.assertEqual((run['epochs'], run['epochs_completed']), (4, 4))
        self.assertIsNone(run['resumed_from_epoch'])

    def test_epochs_argv_drives_loop_and_schedule(self):
        ns = self.script()
        seen = {}
        real_build = ns['build_optimizer_and_scheduler']

        def spy_build(*args):
            opt, sched = real_build(*args)
            seen['T_max'] = sched.T_max
            return opt, sched

        ns['build_optimizer_and_scheduler'] = spy_build
        out = self.tmp / 'out-6'
        self.run_local(self.argv(out, epochs='6'), ns)
        self.assertEqual(seen['T_max'], (12 // (2 * 2)) * 6)
        run = json.loads((out / 'local_training_run.json').read_text(encoding='utf-8'))
        self.assertEqual((run['epochs'], run['epochs_completed'], run['global_step']), (6, 6, 36))
        self.assertEqual(len(json.loads((out / 'epoch_selection.json').read_text())['epoch_agreements']), 6)

    def test_checkpointed_run_matches_in_memory_best_state_run(self):
        # Same seed, same math: keeping the best epoch on disk instead of in RAM
        # must not change what is saved.
        a, b = self.tmp / 'out-mem', self.tmp / 'out-disk'
        self.run_local(self.argv(a, legacy=True) + ['3'])  # argv[17] only: no checkpoints
        self.run_local(self.argv(b))
        self.assertFalse((a / 'resume').exists())
        self.assertEqual(self.outputs(a), self.outputs(b))

    def _interrupted_then_resumed(self, name, interrupt):
        out = self.tmp / name
        ns = self.script()
        with self.assertRaises(Interrupted):
            interrupt(ns)
            self.run_local(self.argv(out), ns)
        TinyDecisionModel.interrupt_at_call = None
        state = kit.read_resume_state(out)
        self.assertEqual(state['epochs_completed'], 1)
        self.assertEqual(len(state['epoch_agreements']), 1)
        self.assertEqual(sorted(os.listdir(out / 'resume')), ['LATEST', 'epoch-0001'])
        self.run_local(self.argv(out, resume=state['epoch_dir']))
        return out

    def test_one_epoch_plus_resume_equals_uninterrupted_three_epochs(self):
        ref = self.tmp / 'out-ref'
        self.run_local(self.argv(ref))
        expected = self.outputs(ref)
        self.assertEqual(expected['epochs_completed'], 3)
        self.assertEqual(len(expected['selection']['epoch_agreements']), 3)
        self.assertIsNotNone(expected['selection']['selected_epoch'])

        def stop_after_first_checkpoint(ns):
            real = ns['write_resume_checkpoint']

            def wrapped(*args, **kwargs):
                result = real(*args, **kwargs)
                raise Interrupted()
            ns['write_resume_checkpoint'] = wrapped

        def stop_mid_epoch_two(ns):
            # 12 items / micro-batch 2 = 6 train forwards + 1 calibration forward per epoch.
            TinyDecisionModel.interrupt_at_call = 7 + 3

        for name, interrupt in (('out-cut-after-epoch', stop_after_first_checkpoint),
                                ('out-cut-mid-epoch', stop_mid_epoch_two)):
            out = self._interrupted_then_resumed(name, interrupt)
            got = self.outputs(out)
            self.assertEqual(got['weights'], expected['weights'], name)
            self.assertEqual(got['selection'], expected['selection'], name)
            self.assertEqual(got['temperature'], expected['temperature'], name)
            self.assertEqual(got['global_step'], expected['global_step'], name)
            run = json.loads((out / 'local_training_run.json').read_text(encoding='utf-8'))
            self.assertEqual(run['resumed_from_epoch'], 1)
            self.assertFalse((out / 'resume').exists(), 'resume dir must be removed after success')

    def test_selected_epoch_older_than_last_is_loaded_from_disk_snapshot(self):
        # Force epoch 1 to be the best so the final save must come from best_model.pt.
        ns = self.script()
        real = ns['compute_calib_agreement']
        calls = {'n': 0}

        def fake_agreement(preds):
            calls['n'] += 1
            agreement = real(preds)
            agreement['mean'] = 1.0 if calls['n'] == 1 else 0.1
            return agreement
        ns['compute_calib_agreement'] = fake_agreement
        out = self.tmp / 'out-best1'
        self.run_local(self.argv(out, keep='keep'), ns)
        selection = json.loads((out / 'epoch_selection.json').read_text(encoding='utf-8'))
        self.assertEqual(selection['selected_epoch'], 1)
        best = torch.load(out / 'resume' / 'epoch-0003' / 'best_model.pt')
        from safetensors.torch import load_file
        saved = load_file(str(out / 'model.safetensors'))
        for key, value in best.items():
            self.assertTrue(torch.equal(saved[key], value.half()), key)
        last = torch.load(out / 'resume' / 'epoch-0003' / 'training_state.pt', weights_only=False)['model']
        self.assertFalse(all(torch.equal(last[k], best[k]) for k in best), 'epoch 3 weights should differ from epoch 1')

    def test_keep_resume_keeps_only_latest_epoch_with_best_snapshot(self):
        out = self.tmp / 'out-keep'
        self.run_local(self.argv(out, keep='keep'))
        root = out / 'resume'
        self.assertEqual(sorted(os.listdir(root)), ['LATEST', 'epoch-0003'])
        self.assertEqual((root / 'LATEST').read_text().strip(), 'epoch-0003')
        self.assertEqual(sorted(os.listdir(root / 'epoch-0003')), ['best_model.pt', 'state.json', 'training_state.pt'])
        state = kit.read_resume_state(out)
        self.assertEqual(state['config'], RUN_CONFIG['match'])
        self.assertEqual(state['info'], RUN_CONFIG['info'])
        self.assertEqual((state['epochs_completed'], state['global_step']), (3, 18))
        ckpt = torch.load(root / 'epoch-0003' / 'training_state.pt', weights_only=False)
        self.assertEqual(set(ckpt), {'model', 'optimizer', 'scheduler', 'rng'})
        self.assertEqual(set(ckpt['rng']), {'python', 'numpy', 'torch_cpu'})

    def test_resume_with_all_epochs_done_goes_straight_to_final_save(self):
        ref = self.tmp / 'out-full'
        self.run_local(self.argv(ref, keep='keep'))
        expected = self.outputs(ref)
        state = kit.read_resume_state(ref)
        for name in ('model.safetensors', 'epoch_selection.json'):
            (ref / name).unlink()
        self.run_local(self.argv(ref, resume=state['epoch_dir']))
        self.assertEqual(self.outputs(ref), expected)

    def test_inner_resume_refuses_config_mismatch(self):
        out = self.tmp / 'out-mismatch'
        self.run_local(self.argv(out, keep='keep'))
        state = kit.read_resume_state(out)
        other = {'match': dict(RUN_CONFIG['match'], epochs=4), 'info': RUN_CONFIG['info']}
        with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
            self.run_local(self.argv(out, epochs='4', resume=state['epoch_dir'], run_config=other))


@unittest.skipUnless(HAVE_TORCH_STACK, 'needs torch + transformers + laya (laya venv)')
class AtomicResumeWrite(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self._tmp.name, 'resume')
        self.ns = fresh_script_ns()
        torch.manual_seed(0)
        self.model = TinyDecisionModel()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=4)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, epochs_completed, best_epoch, prev_epoch_dir):
        return self.ns['write_resume_checkpoint'](
            self.root, epochs_completed, model=self.model, optimizer=self.opt, scheduler=self.sched,
            device=torch.device('cpu'), global_step=epochs_completed * 5, epoch_agreements=[],
            best_state={'score': 0.5, 'epoch': best_epoch}, prev_epoch_dir=prev_epoch_dir,
            run_config=RUN_CONFIG, train_elapsed_s=1.0)

    def assert_only_epoch_one(self, first):
        self.assertEqual(sorted(os.listdir(self.root)), ['LATEST', 'epoch-0001'])
        self.assertEqual(Path(self.root, 'LATEST').read_text().strip(), 'epoch-0001')
        self.assertEqual(json.loads(Path(first, 'state.json').read_text())['epochs_completed'], 1)
        self.assertTrue(Path(first, 'best_model.pt').is_file())

    def test_failure_during_write_leaves_previous_checkpoint_and_no_partial_dir(self):
        first, _size = self.write(1, 1, None)
        real_save = torch.save
        for exc in (OSError('disk full'), Interrupted()):
            calls = {'n': 0}

            def failing_save(obj, path, *a, **kw):
                calls['n'] += 1
                if calls['n'] == 2:  # training_state.pt already written into the temp dir
                    raise exc
                return real_save(obj, path, *a, **kw)
            with mock.patch.object(torch, 'save', failing_save):
                with self.assertRaises(type(exc)):
                    self.write(2, 2, first)
            self.assert_only_epoch_one(first)

    def test_failure_at_rename_leaves_no_partial_dir(self):
        first, _ = self.write(1, 1, None)
        with mock.patch.object(os, 'rename', side_effect=OSError('rename failed')):
            with self.assertRaises(OSError):
                self.write(2, 1, first)
        self.assert_only_epoch_one(first)

    def test_best_snapshot_is_hard_linked_forward_and_old_epoch_pruned(self):
        first, _ = self.write(1, 1, None)
        ino = os.stat(os.path.join(first, 'best_model.pt')).st_ino
        second, size = self.write(2, 1, first)
        self.assertEqual(sorted(os.listdir(self.root)), ['LATEST', 'epoch-0002'])
        self.assertEqual(os.stat(os.path.join(second, 'best_model.pt')).st_ino, ino)
        self.assertEqual(size, sum(os.path.getsize(os.path.join(second, f)) for f in os.listdir(second))
                         + os.path.getsize(os.path.join(self.root, 'LATEST')))

    def test_missing_best_snapshot_fails_instead_of_dropping_it(self):
        with self.assertRaises(RuntimeError):
            self.write(2, 1, None)
        self.assertFalse(os.path.exists(os.path.join(self.root, 'LATEST')))
        self.assertEqual(os.listdir(self.root), [])


@unittest.skipUnless(HAVE_TORCH_STACK, 'needs torch + transformers + laya (laya venv)')
class CachedItemsReuse(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.train = Path(self._tmp.name, 'train_items.pt')
        self.calib = Path(self._tmp.name, 'calib_items.pt')
        self.items_t, self.items_c = make_items(5, 1), make_items(3, 2)
        torch.save(self.items_t, self.train)
        torch.save(self.items_c, self.calib)
        self.info = {'train_items_digest': kit.items_digest(self.items_t),
                     'calib_items_digest': kit.items_digest(self.items_c)}

    def tearDown(self):
        self._tmp.cleanup()

    def test_reused_when_digests_match(self):
        got = kit.load_cached_items_for_resume(self.train, self.calib, self.info, torch)
        self.assertEqual(got, (self.items_t, self.items_c))

    def test_reprocessed_when_digest_differs_or_file_missing(self):
        torch.save(make_items(5, 99), self.train)
        with mock.patch('builtins.print'):
            self.assertIsNone(kit.load_cached_items_for_resume(self.train, self.calib, self.info, torch))
        self.calib.unlink()
        self.assertIsNone(kit.load_cached_items_for_resume(self.train, self.calib, self.info, torch))


if __name__ == '__main__':
    unittest.main()
