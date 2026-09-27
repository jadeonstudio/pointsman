"""Unit tests for train_from_export.py's opt-in regularization flags (2026-09-24):

- --dropout P: train with encoder attention/embedding/mlp dropout = P, while the
  saved encoder/config.json keeps the base checkpoint's own values.
- --rdrop-alpha A: R-Drop (Liang et al., NeurIPS 2021) -- two train-mode passes,
  mean of their existing losses + A * symmetric KL over valid options.

Both default OFF; the default training loss must be exactly the pre-change
formula. The argument-validation tests are standard library only (like the
other tests/test_laya_kit_*.py files). The loss/model tests exec the embedded
TRAIN_DDP_SCRIPT and need torch + transformers + laya (the laya venv:
~/.local/share/laya/.venv/bin/python); they are skipped without them.
"""
import copy
import importlib.util
import json
import math
import os
import random
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


kit = load_module('train_from_export_for_regularization_tests', 'training/laya-kit/train_from_export.py')

try:
    import torch
    import transformers  # noqa: F401
    import laya.common  # noqa: F401
    HAVE_TORCH_STACK = True
except ImportError:  # pragma: no cover - depends on the interpreter
    HAVE_TORCH_STACK = False

_SCRIPT_NS = None


def script_ns():
    """Exec the embedded TRAIN_DDP_SCRIPT once as a module namespace (its
    `if __name__ == "__main__"` guard keeps main() from running)."""
    global _SCRIPT_NS
    if _SCRIPT_NS is None:
        ns = {'__name__': 'train_ddp_script_under_test'}
        exec(compile(kit.TRAIN_DDP_SCRIPT, '<TRAIN_DDP_SCRIPT>', 'exec'), ns)
        _SCRIPT_NS = ns
    return _SCRIPT_NS


# ---------------------------------------------------------------------------
# 1. Argument validation (stdlib only).
# ---------------------------------------------------------------------------
class ValidateRegularizationArgs(unittest.TestCase):
    def test_defaults_off_are_valid(self):
        kit.validate_regularization_args(None, 0.0, local=True)
        kit.validate_regularization_args(None, 0.0, local=False)

    def test_dropout_range(self):
        for ok in (0.01, 0.1, 0.5):
            kit.validate_regularization_args(ok, 0.0, local=True)
        for bad in (0.0, -0.1, 0.51, 1.0, float('nan'), float('inf')):
            with self.assertRaises(kit.KitError, msg=repr(bad)):
                kit.validate_regularization_args(bad, 0.0, local=True)

    def test_rdrop_alpha_range(self):
        kit.validate_regularization_args(0.1, 1.0, local=True)
        kit.validate_regularization_args(0.1, 0.0, local=True)
        for bad in (-0.1, float('nan'), float('inf')):
            with self.assertRaises(kit.KitError, msg=repr(bad)):
                kit.validate_regularization_args(0.1, bad, local=True)

    def test_rdrop_without_dropout_dies_with_clear_message(self):
        with self.assertRaises(kit.KitError) as ctx:
            kit.validate_regularization_args(None, 1.0, local=True)
        self.assertIn('--rdrop-alpha > 0 requires --dropout', str(ctx.exception))

    def test_flags_rejected_outside_local(self):
        for dropout, alpha in ((0.1, 0.0), (0.1, 1.0)):
            with self.assertRaises(kit.KitError) as ctx:
                kit.validate_regularization_args(dropout, alpha, local=False)
            self.assertIn('--local only', str(ctx.exception))


class MainRegularizationCliWiring(unittest.TestCase):
    """main() validates the flags right after argparse, before touching the export."""

    def run_main(self, *extra):
        argv = ['train_from_export.py', '--export-dir', '/nonexistent-export-for-test',
                '--output-dir', '/nonexistent-output-for-test', *extra]
        with mock.patch.object(sys, 'argv', argv):
            with self.assertRaises(kit.KitError) as ctx:
                kit.main()
        return str(ctx.exception)

    def test_rdrop_without_dropout_dies_before_export_check(self):
        msg = self.run_main('--local', '--rdrop-alpha', '1.0')
        self.assertIn('--rdrop-alpha > 0 requires --dropout', msg)

    def test_out_of_range_dropout_dies_before_export_check(self):
        msg = self.run_main('--local', '--dropout', '0.7')
        self.assertIn('--dropout must satisfy 0 < P <= 0.5', msg)

    def test_valid_flags_pass_validation_and_reach_export_check(self):
        msg = self.run_main('--local', '--dropout', '0.1', '--rdrop-alpha', '1.0')
        self.assertNotIn('--dropout', msg)
        self.assertNotIn('--rdrop-alpha', msg)

    def test_defaults_are_off(self):
        source = (ROOT / 'training/laya-kit/train_from_export.py').read_text(encoding='utf-8')
        self.assertIn('parser.add_argument("--dropout", type=float, default=None,', source)
        self.assertIn('parser.add_argument("--rdrop-alpha", type=float, default=0.0,', source)
        self.assertIn('"none" if args.dropout is None else str(args.dropout), str(args.rdrop_alpha),', source)
        self.assertIn('"dropout": args.dropout, "rdrop_alpha": args.rdrop_alpha if args.rdrop_alpha > 0 else None,', source)

    def test_main_local_parses_records_and_prints_both_settings(self):
        start = kit.TRAIN_DDP_SCRIPT.index('\ndef main_local(')
        rest = kit.TRAIN_DDP_SCRIPT[start + 1:]
        body = rest[:rest.find('\ndef ')]
        self.assertIn('sys.argv[15]', body)
        self.assertIn('sys.argv[16]', body)
        self.assertIn('"dropout": encoder_dropout, "rdrop_alpha": rdrop_alpha if rdrop_alpha > 0 else None,', body)
        self.assertIn("dropout={'off' if encoder_dropout is None else encoder_dropout}", body)
        self.assertIn('rdrop_alpha=rdrop_alpha', body)
        self.assertIn('encoder_config_restore=encoder_config_restore', body)


# ---------------------------------------------------------------------------
# Shared torch fixtures.
# ---------------------------------------------------------------------------
def make_items():
    # One question per item, like build_training_item(): choice (3 options),
    # score (4 levels), noul (2 options, false/true) -- different k per row.
    return [
        {'ids': [1, 5, 6, 7, 8, 9], 'markers': [1, 2, 3], 'target': [0.1, 0.7, 0.2], 'qtype': 0, 'label': 1},
        {'ids': [1, 10, 11, 12, 13, 14, 15], 'markers': [1, 2, 3, 4], 'target': [0.0, 0.2, 0.5, 0.3], 'qtype': 1, 'label': 2},
        {'ids': [1, 16, 17, 18], 'markers': [1, 2], 'target': [0.9, 0.1], 'qtype': 2, 'label': 0},
        {'ids': [1, 19, 20, 21, 22], 'markers': [1, 2, 3], 'target': [0.3, 0.3, 0.4], 'qtype': 0, 'label': 2},
    ]


if HAVE_TORCH_STACK:
    class TinyDecisionModel(torch.nn.Module):
        """Stand-in for DecisionModel's forward signature/outputs (masked marker
        logits + act logits), optionally with dropout so two train-mode passes differ."""

        def __init__(self, vocab=32, d=8, p=0.0):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, d)
            self.drop = torch.nn.Dropout(p)
            self.score = torch.nn.Linear(d, 1)
            self.act = torch.nn.Linear(d, 2)
            self.calls = 0

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            self.calls += 1
            h = self.drop(self.emb(input_ids))
            m = torch.gather(h, 1, marker_pos[:, :, None].expand(-1, -1, h.size(-1)))
            logits = self.score(m).squeeze(-1).float().masked_fill(~marker_mask, -1e4)
            return logits, self.act(h[:, 0])


def original_formula_loss(logits, act, batch, device, group_size, sigma, grad_accum):
    """Verbatim copy of the pre-change run_training_loop loss block (HEAD 5425a98)."""
    from laya.common import proper_reward
    logits = logits.float()
    mask = batch["marker_mask"].to(device)
    k = mask.sum(-1, keepdim=True).float()
    target = batch["target"].to(device)

    eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)

    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    loss = (loss_rl + 1.0 * loss_ce) / grad_accum + 0.0 * act.sum()
    return loss, r


def manual_symmetric_kl(logits1, logits2, mask):
    """Plain-python reference: per row, softmax over the VALID options only,
    0.5 * (KL(p1||p2) + KL(p2||p1)); mean over rows."""
    rows = []
    for l1, l2, m in zip(logits1.tolist(), logits2.tolist(), mask.tolist()):
        a = [x for x, keep in zip(l1, m) if keep]
        b = [x for x, keep in zip(l2, m) if keep]
        za, zb = sum(math.exp(x) for x in a), sum(math.exp(x) for x in b)
        p = [math.exp(x) / za for x in a]
        q = [math.exp(x) / zb for x in b]
        kl_pq = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
        kl_qp = sum(qi * math.log(qi / pi) for pi, qi in zip(p, q))
        rows.append(0.5 * (kl_pq + kl_qp))
    return sum(rows) / len(rows)


def loop_kwargs(**over):
    kw = dict(device=torch.device('cpu'), epochs=1, micro_batch=2, grad_accum=2, group_size=4,
              sigma_start=0.4, sigma_end=0.1, rank=0, world_size=1, use_scaler=False, scaler=None,
              autocast_device='cpu', autocast_dtype=None, autocast_enabled=False, max_steps=0,
              mem_log_fn=None, log_prefix='[test]')
    kw.update(over)
    return kw


# ---------------------------------------------------------------------------
# 2. Default path unchanged.
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_TORCH_STACK, 'needs torch + transformers + laya (laya venv)')
class DefaultLossUnchanged(unittest.TestCase):
    def setUp(self):
        self.ns = script_ns()
        self.device = torch.device('cpu')

    def test_micro_batch_loss_is_bitwise_the_original_formula(self):
        batch = self.ns['collate_train_batch'](make_items(), 0)
        torch.manual_seed(0)
        logits = torch.randn(4, 4).masked_fill(~batch['marker_mask'], -1e4)
        act = torch.randn(4, 2)
        for grad_accum, sigma in ((1, 0.4), (16, 0.1)):
            torch.manual_seed(123)
            new_loss, new_r = self.ns['micro_batch_loss'](logits, act, batch, self.device, 4, sigma, grad_accum)
            torch.manual_seed(123)
            old_loss, old_r = original_formula_loss(logits, act, batch, self.device, 4, sigma, grad_accum)
            self.assertTrue(torch.equal(new_loss, old_loss), (new_loss, old_loss))
            self.assertTrue(torch.equal(new_r, old_r))

    def _reference_step(self, model, items):
        """The pre-change loop for epochs=1, micro_batch=2, grad_accum=2, 4 items:
        shuffle, two micro-batches with the original loss, one clipped SGD step."""
        items = list(items)
        random.seed(42)
        random.shuffle(items)
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        opt.zero_grad(set_to_none=True)
        for b_idx in range(0, len(items), 2):
            batch = self.ns['collate_train_batch'](items[b_idx:b_idx + 2], 0)
            logits, act = model(batch['input_ids'], batch['attention_mask'], batch['marker_pos'],
                                batch['marker_mask'], batch['qtype'])
            loss, _ = original_formula_loss(logits, act, batch, self.device, 4, 0.4, 2)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
        opt.step()

    def _run_loop(self, model, items, **extra):
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _s: 1.0)
        tok = types.SimpleNamespace(pad_token_id=0)
        with mock.patch('builtins.print'):
            return self.ns['run_training_loop'](model, list(items), list(model.parameters()), opt, sched, tok,
                                                **loop_kwargs(), **extra)

    def test_run_training_loop_default_matches_pre_change_loop_bitwise(self):
        torch.manual_seed(0)
        base = TinyDecisionModel(p=0.0)
        for extra in ({}, {'rdrop_alpha': 0.0}):
            ref, new = copy.deepcopy(base), copy.deepcopy(base)
            torch.manual_seed(7)
            self._reference_step(ref, make_items())
            torch.manual_seed(7)
            result = self._run_loop(new, make_items(), **extra)
            self.assertEqual(result['global_step'], 2)
            self.assertEqual(new.calls, 2, 'default path must run one forward per micro-batch')
            for (name, a), (_, b) in zip(ref.named_parameters(), new.named_parameters()):
                self.assertTrue(torch.equal(a, b), f'{name} differs from the pre-change loop ({extra})')

    def test_rdrop_runs_two_forwards_per_micro_batch(self):
        torch.manual_seed(0)
        model = TinyDecisionModel(p=0.2)
        model.train()
        result = self._run_loop(model, make_items(), rdrop_alpha=1.0)
        self.assertEqual(result['global_step'], 2)
        self.assertEqual(model.calls, 4)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


# ---------------------------------------------------------------------------
# 3. R-Drop loss.
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_TORCH_STACK, 'needs torch + transformers + laya (laya venv)')
class RDropLoss(unittest.TestCase):
    def setUp(self):
        self.ns = script_ns()
        self.device = torch.device('cpu')
        self.batch = self.ns['collate_train_batch'](make_items(), 0)
        mask = self.batch['marker_mask']
        g = torch.Generator().manual_seed(3)
        self.logits1 = torch.randn(4, 4, generator=g).masked_fill(~mask, -1e4)
        self.logits2 = torch.randn(4, 4, generator=g).masked_fill(~mask, -1e4)
        self.act1 = torch.randn(4, 2, generator=g)
        self.act2 = torch.randn(4, 2, generator=g)

    def test_symmetric_kl_matches_manual_valid_option_reference(self):
        got = self.ns['masked_symmetric_kl'](self.logits1, self.logits2, self.batch['marker_mask']).item()
        want = manual_symmetric_kl(self.logits1, self.logits2, self.batch['marker_mask'])
        self.assertGreater(want, 0.0)
        self.assertAlmostEqual(got, want, places=5)

    def test_symmetric_kl_ignores_padded_option_slots(self):
        mask = self.batch['marker_mask']
        l2 = self.logits2.clone()
        l2[~mask] = 123.0  # garbage in padded slots must not matter
        a = self.ns['masked_symmetric_kl'](self.logits1, self.logits2, mask)
        b = self.ns['masked_symmetric_kl'](self.logits1, l2, mask)
        self.assertAlmostEqual(a.item(), b.item(), places=6)

    def test_rdrop_loss_is_mean_of_pass_losses_plus_alpha_times_symmetric_kl(self):
        alpha, grad_accum, sigma = 0.7, 4, 0.3
        torch.manual_seed(11)
        loss, r, kl = self.ns['rdrop_micro_batch_loss']((self.logits1, self.act1), (self.logits2, self.act2),
                                                         self.batch, self.device, 4, sigma, grad_accum, alpha)
        # Same RNG sequence: pass 1's RL noise, then pass 2's.
        torch.manual_seed(11)
        l1, r1 = original_formula_loss(self.logits1, self.act1, self.batch, self.device, 4, sigma, 1)
        l2, r2 = original_formula_loss(self.logits2, self.act2, self.batch, self.device, 4, sigma, 1)
        manual_kl = manual_symmetric_kl(self.logits1, self.logits2, self.batch['marker_mask'])
        want = (0.5 * (l1.item() + l2.item()) + alpha * manual_kl) / grad_accum
        self.assertAlmostEqual(loss.item(), want, places=5)
        self.assertAlmostEqual(kl.item(), manual_kl, places=5)
        self.assertTrue(torch.allclose(r, 0.5 * (r1 + r2)))

    def test_kl_term_zero_when_passes_identical(self):
        mask = self.batch['marker_mask']
        self.assertEqual(self.ns['masked_symmetric_kl'](self.logits1, self.logits1.clone(), mask).item(), 0.0)
        torch.manual_seed(5)
        loss, _, kl = self.ns['rdrop_micro_batch_loss']((self.logits1, self.act1), (self.logits1.clone(), self.act1),
                                                         self.batch, self.device, 4, 0.4, 2, 5.0)
        torch.manual_seed(5)
        l1, _ = self.ns['micro_batch_loss'](self.logits1, self.act1, self.batch, self.device, 4, 0.4, 1)
        l2, _ = self.ns['micro_batch_loss'](self.logits1, self.act1, self.batch, self.device, 4, 0.4, 1)
        self.assertEqual(kl.item(), 0.0)
        self.assertAlmostEqual(loss.item(), 0.5 * (l1.item() + l2.item()) / 2, places=6)

    def test_kl_gradient_flows_to_both_passes(self):
        l1 = self.logits1.clone().requires_grad_(True)
        l2 = self.logits2.clone().requires_grad_(True)
        self.ns['masked_symmetric_kl'](l1, l2, self.batch['marker_mask']).backward()
        self.assertGreater(l1.grad.abs().sum().item(), 0.0)
        self.assertGreater(l2.grad.abs().sum().item(), 0.0)


# ---------------------------------------------------------------------------
# 4. Encoder dropout on a real (tiny) ModernBERT built through laya's build_model.
# ---------------------------------------------------------------------------
def write_tiny_modernbert_encoder(encoder_dir):
    from transformers import ModernBertConfig
    config = ModernBertConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
        max_position_embeddings=128, local_attention=16, global_attn_every_n_layers=3,
        pad_token_id=0, bos_token_id=2, eos_token_id=1, cls_token_id=1, sep_token_id=1,
        attention_dropout=0.0, embedding_dropout=0.0, mlp_dropout=0.0, classifier_dropout=0.0)
    config.save_pretrained(encoder_dir)
    return json.loads((Path(encoder_dir) / 'config.json').read_text(encoding='utf-8'))


TINY_CFG = {'head_layers': 2, 'act_costs': {'escalate': 0.5}}


@unittest.skipUnless(HAVE_TORCH_STACK, 'needs torch + transformers + laya (laya venv)')
class EncoderDropoutApplied(unittest.TestCase):
    def setUp(self):
        self.ns = script_ns()
        self._tmp = tempfile.TemporaryDirectory()
        self.encoder_dir = os.path.join(self._tmp.name, 'encoder')
        self.original_config = write_tiny_modernbert_encoder(self.encoder_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, dropout):
        torch.manual_seed(0)
        with mock.patch('builtins.print'):
            return self.ns['build_model_with_encoder_dropout'](TINY_CFG, self.encoder_dir, dropout)

    def test_not_set_leaves_encoder_untouched(self):
        model, restore = self.build(None)
        self.assertIsNone(restore)
        report = self.ns['encoder_dropout_report'](model.encoder)
        self.assertEqual(report['active_dropout_modules'], 0)
        self.assertEqual(report['attention_dropout_values'], [0.0])
        for name in self.ns['ENCODER_DROPOUT_KEYS']:
            self.assertEqual(getattr(model.encoder.config, name), 0.0)
        attn = [m for m in model.encoder.modules() if hasattr(m, 'out_drop')]
        self.assertTrue(attn)
        self.assertTrue(all(isinstance(m.out_drop, torch.nn.Identity) for m in attn))
        with self.assertRaises(RuntimeError):
            self.ns['assert_encoder_dropout_applied'](model.encoder, 0.1)

    def test_set_applies_p_to_every_encoder_dropout_site(self):
        model, restore = self.build(0.1)
        self.assertEqual(restore, {'attention_dropout': 0.0, 'embedding_dropout': 0.0, 'mlp_dropout': 0.0})
        report = self.ns['assert_encoder_dropout_applied'](model.encoder, 0.1)
        # embeddings.drop + per layer (mlp.drop + attn.out_drop) = 1 + 2 * 3.
        self.assertEqual(report['active_dropout_modules'], 7)
        self.assertEqual(report['attention_modules'], 3)
        attn = [m for m in model.encoder.modules() if hasattr(m, 'out_drop')]
        self.assertTrue(all(isinstance(m.out_drop, torch.nn.Dropout) and m.out_drop.p == 0.1 for m in attn))
        # The source encoder dir on disk is never modified.
        self.assertEqual(json.loads((Path(self.encoder_dir) / 'config.json').read_text(encoding='utf-8')),
                         self.original_config)

    def test_dropout_is_live_in_train_mode_and_off_in_eval_mode(self):
        model, _ = self.build(0.3)
        ids = torch.randint(3, 64, (2, 12))
        att = torch.ones_like(ids)
        model.encoder.train()
        a = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        b = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        self.assertFalse(torch.allclose(a, b))
        model.encoder.eval()
        with torch.no_grad():
            c = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state
            d = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        self.assertTrue(torch.equal(c, d))

    def test_saved_encoder_config_keeps_original_dropout_values(self):
        calib_items = make_items()
        tok = types.SimpleNamespace(pad_token_id=0, save_pretrained=lambda path: os.makedirs(path, exist_ok=True))
        saved = {}
        for label, dropout in (('off', None), ('on', 0.2)):
            model, restore = self.build(dropout)
            model.train()
            out_dir = os.path.join(self._tmp.name, f'out-{label}')
            with mock.patch('builtins.print'):
                self.ns['finalize_and_save'](model, tok, calib_items, out_dir, 'm', 'multilingual', 'v', dict(TINY_CFG),
                                             device=torch.device('cpu'), autocast_device='cpu', autocast_dtype=None,
                                             autocast_enabled=False, encoder_config_restore=restore)
            # Temperature fitting read calibration logits in eval mode.
            self.assertFalse(model.training)
            self.assertFalse(any(m.training for m in model.modules()))
            if dropout is not None:
                # The live model keeps training-time dropout; only the saved copy is restored.
                self.assertEqual(model.encoder.config.attention_dropout, 0.2)
            saved[label] = json.loads(Path(out_dir, 'encoder', 'config.json').read_text(encoding='utf-8'))
        for key in ('attention_dropout', 'embedding_dropout', 'mlp_dropout', 'classifier_dropout'):
            self.assertEqual(saved['on'][key], self.original_config[key])
        self.assertEqual(saved['on'], saved['off'])

    def test_calibration_logits_are_collected_in_eval_mode(self):
        model, _ = self.build(0.3)
        model.train()
        tok = types.SimpleNamespace(pad_token_id=0)
        collect = self.ns['collect_calib_logits']
        a = collect(model, make_items(), tok, torch.device('cpu'), 'cpu', None, False)
        self.assertFalse(model.training)
        model.train()
        b = collect(model, make_items(), tok, torch.device('cpu'), 'cpu', None, False)
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
