"""Bounded native-component parity; CPU Torch reference, GPU MLX, no full model."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.gated_delta import compute_g, normalize_qk
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
LENGTHS = (17, 65, 129)
LIMIT = 2 * 1024**3


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    path.chmod(0o600)


def memory():
    result = {"mlx_peak_bytes": mx.get_peak_memory(), "mlx_active_bytes": mx.get_active_memory(),
              "torch_gpu_current_bytes": torch.mps.current_allocated_memory()}
    if result["mlx_peak_bytes"] + result["torch_gpu_current_bytes"] > LIMIT:
        raise RuntimeError("GPU_BUDGET_EXCEEDED")
    return result


class StopParity(Exception):
    pass


class Probe:
    def __init__(self, root, out):
        self.root, self.out, self.rows = root, out, []
        self.upstream = root / "upstream"
        self.index = json.loads((self.upstream / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(self, name, rows=None):
        with safe_open(self.upstream / self.index[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name) if rows is None else f.get_slice(name)[:rows, :]

    def compare(self, name, reference, value, *, diagnostic=False):
        mx.eval(value)
        mx.synchronize()
        actual = np.array(value.astype(mx.float32)).astype(np.float64)
        expected = reference.detach().float().numpy().astype(np.float64)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise StopParity(name + ":STRUCTURAL_OR_NONFINITE")
        a, b = actual.reshape(-1), expected.reshape(-1)
        denom = np.linalg.norm(b)
        relative = float(np.linalg.norm(a - b) / max(denom, 1e-30))
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * denom)) if denom and np.linalg.norm(a) else float(np.array_equal(a, b))
        passed = relative <= (1e-2 if diagnostic else 5e-4) and cosine >= (.9999 if diagnostic else .999999)
        row = {"operation": name, "shape": list(actual.shape), "relative_l2": relative, "cosine": cosine,
               "max_abs": float(np.abs(a-b).max()), "status": "PASS" if passed else "FAIL", "diagnostic": diagnostic,
               "memory": memory()}
        self.rows.append(row)
        save(self.out / "observations.json", self.rows)
        print(json.dumps({k: row[k] for k in ("operation", "status", "relative_l2", "max_abs")}), flush=True)
        if not passed and not diagnostic:
            raise StopParity(name + ":FP32_TOLERANCE")


class Capture(nn.Module):
    def __init__(self, inner, path, probe, expected, length):
        super().__init__()
        self.inner = inner
        self.path, self.probe, self.expected, self.length = path, probe, expected, length

    def __call__(self, *args, **kwargs):
        if self.path == "linear_attn.norm":
            ref = self.expected[self.path + ".input0"]
            self.probe.compare(f"T{self.length}/{self.path}.input0-recurrence", ref, args[0].reshape(ref.shape))
        result = self.inner(*args, **kwargs)
        ref = self.expected[self.path]
        self.probe.compare(f"T{self.length}/{self.path}", ref, result.reshape(ref.shape))
        return result


def wrap(model, path, capture):
    parts = path.split(".")
    for part in parts[:-1]:
        model = getattr(model, part)
    setattr(model, parts[-1], capture)


def standalone(probe, config, args, embeddings):
    final = probe.tensor("model.language_model.norm.weight")
    norm = hf.Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
    norm.weight = torch.nn.Parameter(final.float(), requires_grad=False)
    mnorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    mnorm.weight = mx.array((1 + final.float()).numpy())
    bnorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    bnorm.weight = mx.array(final + 1.0)
    bfnorm = hf.Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
    bfnorm.weight = torch.nn.Parameter(final, requires_grad=False)
    rope = hf.Qwen3_5TextRotaryEmbedding(config)
    mrope = initialize_rope(64, args.rope_theta, False, args.rope_scaling, args.max_position_embeddings)
    alog = probe.tensor("model.language_model.layers.0.linear_attn.A_log").float()
    dt = probe.tensor("model.language_model.layers.0.linear_attn.dt_bias").float()
    aw = probe.tensor("model.language_model.layers.0.linear_attn.in_proj_a.weight").float()
    bw = probe.tensor("model.language_model.layers.0.linear_attn.in_proj_b.weight").float()
    for length in LENGTHS:
        x = embeddings[:length].float().unsqueeze(0)
        mlx_x = mx.array(x.numpy())
        probe.compare(f"T{length}/final_norm_FP32", norm(x), mnorm(mlx_x))
        probe.compare(f"T{length}/final_norm_BF16_policy", bfnorm(x.bfloat16()), bnorm(mlx_x.astype(mx.bfloat16)), diagnostic=True)
        q = x.reshape(1, length, 16, 256).transpose(1, 2)
        positions = torch.arange(length).reshape(1, 1, length).expand(3, 1, length)
        cos, sin = rope(x, positions)
        rotated, _ = hf.apply_rotary_pos_emb(q, q, cos, sin)
        probe.compare(f"T{length}/partial_RoPE64_text", rotated, mrope(mx.array(q.numpy())))
        qk = x.reshape(1, length, 32, 128)[:, :, :16]
        mq, mk = normalize_qk(mx.array(qk.numpy()), mx.array(qk.numpy()), inv_scale=128**-.5, eps=1e-6)
        probe.compare(f"T{length}/GDN_q_L2_scale", hf.l2norm(qk) * 128**-.5, mq)
        probe.compare(f"T{length}/GDN_k_L2", hf.l2norm(qk), mk)
        a = F.linear(x, aw)
        b = F.linear(x, bw)
        decay = (-alog.exp() * F.softplus(a + dt)).exp()
        probe.compare(f"T{length}/GDN_decay_FP32", decay, compute_g(mx.array(alog.numpy()), mx.array(a.numpy()), mx.array(dt.numpy())))
        probe.compare(f"T{length}/GDN_beta_FP32", b.sigmoid(), mx.sigmoid(mx.array(b.numpy())))
    del mnorm, bnorm, mlx_x, q, a
    gc.collect()
    mx.clear_cache()


def block(probe, config, args, index, embeddings):
    prefix = f"model.language_model.layers.{index}."
    state = {name[len(prefix):]: probe.tensor(name).float() for name in probe.index if name.startswith(prefix)}
    with torch.device("meta"):
        reference = hf.Qwen3_5DecoderLayer(config, index)
    reference.load_state_dict(state, strict=True, assign=True)
    reference.eval()
    model = DecoderLayer(args, index)
    converted = {}
    for name, value in state.items():
        if name.endswith("conv1d.weight"):
            value = value.moveaxis(2, 1)
        if name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight")):
            value = value + 1.0
        converted[name] = mx.array(value.numpy())
    model.load_weights(list(converted.items()), strict=True)
    model.eval()
    del converted, state
    mx.eval(model.parameters())
    memory()
    paths = ["input_layernorm", "post_attention_layernorm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp"]
    paths += (["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a",
               "linear_attn.norm", "linear_attn.out_proj", "linear_attn"] if index == 0 else
              ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.q_norm", "self_attn.k_norm", "self_attn.o_proj", "self_attn"])
    expected, hooks = {}, []
    for path in paths:
        child = reference.get_submodule(path)
        hooks.append(child.register_forward_hook(lambda _, inputs, result, key=path: expected.__setitem__(key, (result[0] if isinstance(result, tuple) else result).detach())))
        if path == "linear_attn.norm":
            hooks.append(child.register_forward_pre_hook(lambda _, inputs: expected.__setitem__("linear_attn.norm.input0", inputs[0].detach())))
    # Native classes remain the computational implementation; wrappers only compare outputs.
    originals = {}
    for path in paths:
        inner = model
        for part in path.split("."):
            inner = getattr(inner, part)
        originals[path] = inner
    rope = hf.Qwen3_5TextRotaryEmbedding(config)
    try:
        for length in LENGTHS:
            expected.clear()
            x = embeddings[:length].float().unsqueeze(0)
            positions = torch.arange(length).reshape(1, 1, length).expand(3, 1, length)
            pos = rope(x, positions)
            mask = torch.full((length, length), float("-inf")).triu(1).reshape(1, 1, length, length) if index == 3 else None
            with torch.inference_mode():
                target = reference(x, position_embeddings=pos, attention_mask=mask)
            for path, inner in originals.items():
                wrap(model, path, Capture(inner, path, probe, expected, length))
            result = model(mx.array(x.numpy()), mask="causal" if index == 3 else None)
            probe.compare(f"T{length}/layer{index}.native_output", target, result)
            for path, inner in reversed(list(originals.items())):
                wrap(model, path, inner)
    finally:
        for hook in hooks:
            hook.remove()
    del model, reference, originals, expected
    gc.collect()
    mx.clear_cache()


def main(root):
    out = root / "parity"
    out.mkdir(mode=0o700)
    mx.set_memory_limit(1536 * 1024**2)
    mx.set_cache_limit(128 * 1024**2)
    config_dict = json.loads((root / "upstream/config.json").read_text())["text_config"]
    config = Qwen3_5TextConfig(**config_dict)
    config._attn_implementation = "eager"
    args = TextModelArgs.from_dict(json.loads(json.dumps(config_dict)))
    proof = root / "artifacts/full-upstream-hashes.json"
    contract = {"revision": REVISION, "full_upstream_manifest_sha256": digest(proof.read_bytes()),
                "probe_source_sha256": digest(Path(__file__).read_bytes()), "lengths": list(LENGTHS),
                "input": "pinned embedding rows 0..128; B1; no padding/cache; no dataset or gold access",
                "reference": "native HF CPU FP32; native MLX GPU FP32; no quantization",
                "norm_mapping": "FP32 before zero-centered +1; existing BF16 policy reported separately",
                "tolerances": {"FP32_relative_l2": 5e-4, "FP32_cosine_min": .999999, "BF16_diagnostic_relative_l2": .01, "BF16_diagnostic_cosine_min": .9999},
                "gpu_limit_bytes": LIMIT, "mlx_allocation_limit_bytes": 1536*1024**2, "mlx_cache_limit_bytes": 128*1024**2,
                "stop": "first structural/nonfinite/FP32 failure; no later block/length sweep",
                "started_at_utc": datetime.now(timezone.utc).isoformat()}
    save(out / "contract.json", contract)
    probe = Probe(root, out)
    started = time.monotonic()
    status, failure = "PASS_COMPONENTS_ONLY", None
    try:
        embeddings = probe.tensor("model.language_model.embed_tokens.weight", 129)
        lexical = probe.tensor("lm_head.weight", 129)
        for folder in ("mlx4", "mlx8"):
            ix = json.loads((root / folder / "model.safetensors.index.json").read_text())["weight_map"]
            for key, ref in (("language_model.model.embed_tokens.weight", embeddings), ("language_model.lm_head.weight", lexical)):
                with safe_open(root / folder / ix[key], framework="pt", device="cpu") as f:
                    actual = f.get_slice(key)[:129, :]
                if not torch.equal(actual, ref):
                    raise StopParity(folder + ":ROW_IDENTITY")
        standalone(probe, config, args, embeddings)
        block(probe, config, args, 0, embeddings)
        block(probe, config, args, 3, embeddings)
    except StopParity as error:
        status, failure = "FAIL_COMPONENT", str(error)
    except Exception as error:
        status, failure = "PREPARATION_ERROR", type(error).__name__ + ":" + str(error)
    result = {"status": status, "failure": failure, "contract_sha256": digest((out / "contract.json").read_bytes()),
              "elapsed_seconds": time.monotonic()-started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "memory": memory(), "observations": len(probe.rows), "full_backbone_equivalence": "UNKNOWN"}
    save(out / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
