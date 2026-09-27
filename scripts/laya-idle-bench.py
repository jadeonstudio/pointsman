"""First-call latency after idle gaps (sporadic hook traffic) for one device/precision.

Agent hooks call the resident server minutes apart, so the latency that matters is the first call after
an idle gap, not back-to-back warm latency. Offline; never modifies model files.

    <laya-venv>/bin/python scripts/laya-idle-bench.py --model-dir /abs/ckpt --device cpu --precision fp32 --gaps 0,5,30,60
"""
import argparse, contextlib, io, json, os, sys, time
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", USE_TF="0", USE_TORCH="1")
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
sys.path.insert(0, str(Path(__file__).resolve().parent))
bench = __import__("laya-bench")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--gaps", default="0,5,30,60")
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    import torch
    from transformers.initialization import no_init_weights
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import laya.agent
    laya.agent._fix_tokenizer_config = lambda _: None
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), no_init_weights():
        agent = laya.agent.Agent(a.model_dir, device=a.device)
    if a.precision == "fp16":
        agent.model.half(); orig = torch.autocast
        laya.agent.torch.autocast = lambda device_type, dtype=None, enabled=True, **kw: orig(device_type, dtype=torch.float16, enabled=True)
    def call(i):
        t = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            agent.system_one(bench.state_for(bench.TASKS[i % len(bench.TASKS)]), bench.ROUTE_QUESTIONS)
        if a.device == "mps": torch.mps.synchronize()
        return round((time.perf_counter() - t) * 1000, 1)
    for i in range(3): call(i)  # warm-up
    out = {"device": a.device, "precision": a.precision, "first_call_after_gap_ms": {}}
    for gap in [int(g) for g in a.gaps.split(",")]:
        samples = []
        for r in range(a.repeats):
            time.sleep(gap); samples.append(call(r))
        out["first_call_after_gap_ms"][f"{gap}s"] = samples
    print(json.dumps(out))

if __name__ == "__main__":
    main()
