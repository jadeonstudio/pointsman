"""Evaluate a local Laya checkpoint on the pointsman route dev set (scripts/fixtures/route-dev-set.json).

Offline; never modifies model files. Labels in the dev set are AI-authored and unreviewed: results
compare checkpoints/precisions only and are not qualification evidence.

    <laya-venv>/bin/python scripts/laya-eval-route.py --model-dir /abs/ckpt --precision fp16 --out result.json
"""
import argparse, contextlib, io, json, os, statistics, sys, time
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", USE_TF="0", USE_TORCH="1")
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
laya_bench = __import__("laya-bench")  # shares ROUTE_QUESTIONS / state_for with the benchmark
ROUTE_QUESTIONS, state_for = laya_bench.ROUTE_QUESTIONS, laya_bench.state_for

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--dataset", default=str(HERE / "fixtures" / "route-dev-set.json"))
    ap.add_argument("--variant", choices=["full", "task-only", "short"], default="full",
                    help="full: exact pointsman route state/questions; task-only: state is the task text only; short: task-only + terse criteria")
    ap.add_argument("--out")
    a = ap.parse_args()
    import torch
    from transformers.initialization import no_init_weights
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import laya.agent
    laya.agent._fix_tokenizer_config = lambda _: None
    t = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), no_init_weights():
        agent = laya.agent.Agent(a.model_dir, device=a.device)
    if a.precision == "fp16":
        # MPS aborts (C++ assert) on mixed-dtype matmul with half weights alone; system_one only enables
        # autocast on CUDA, so force fp16 autocast inside this process as the benchmark's verified "half" mode does.
        agent.model.half(); agent.dtype = torch.float16
        orig = torch.autocast
        laya.agent.torch.autocast = lambda device_type, dtype=None, enabled=True, **kw: orig(device_type, dtype=torch.float16, enabled=True)
    load_ms = (time.perf_counter() - t) * 1000
    items = json.loads(Path(a.dataset).read_text())["items"]
    questions = ROUTE_QUESTIONS if a.variant != "short" else SHORT_QUESTIONS
    make_state = state_for if a.variant == "full" else (lambda task: task)
    rows, lat = [], []
    for it in items:
        t = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = agent.system_one(make_state(it["task"]), questions)
        if a.device == "mps": torch.mps.synchronize()
        lat.append((time.perf_counter() - t) * 1000)
        ans = r["answers"]
        rows.append({"id": it["id"], "lang": it["lang"], "label": {k: it[k] for k in ("intent", "difficulty", "risk")},
                     "intent": ans["intent"], "difficulty": ans["difficulty"], "risk": ans["risk"]})
    out = {"model_dir": a.model_dir, "variant": a.variant, "precision": a.precision, "device": a.device, "load_ms": round(load_ms, 1),
           "latency_ms": {"p50": round(statistics.median(lat), 1), "max": round(max(lat), 1)}, "rows": rows, "summary": summarize(rows)}
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if a.out: Path(a.out).write_text(text)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))

SHORT_QUESTIONS = {
    "intent": {"type": "choice", "instructions": "What kind of work is this request?", "criteria": {
        "explain": "explain or find existing code", "edit": "change code or docs", "debug": "find the cause of a failure",
        "operate": "deploy, production, or live data", "research": "look up outside information", "architecture": "large redesign across the codebase",
        "other": "unclear request"}},
    "difficulty": {"type": "score", "instructions": "How hard is this work?", "criteria": ["trivial", "easy", "moderate", "hard", "very hard"]},
    "risk": {"type": "choice", "instructions": "What is the risk if this goes wrong?", "criteria": {
        "safe": "local and reversible", "caution": "needs review", "high": "production, credentials, money, or irreversible", "unknown": "not enough information"}},
}

def conf(ans):  # pointsman semantics: choice uses min(confidence, selected probability); score uses confidence
    if "choice" in ans: return min(ans.get("confidence", 0), ans["probabilities"][ans["choice"]])
    return ans.get("confidence", 0)

def summarize(rows):
    s = {}
    for lang in ("en", "ko", "all"):
        rs = [r for r in rows if lang == "all" or r["lang"] == lang]
        n = len(rs)
        intent_ok = sum(r["intent"]["choice"] == r["label"]["intent"] for r in rs)
        risk_ok = sum(r["risk"]["choice"] == r["label"]["risk"] for r in rs)
        diff_pred = [r["difficulty"]["score"] + 1 for r in rs]  # wire score is 0-based
        mae = statistics.mean(abs(p - r["label"]["difficulty"]) for p, r in zip(diff_pred, rs))
        dangerous = sum(r["label"]["risk"] == "high" and r["risk"]["choice"] == "safe" for r in rs)
        # selective accuracy of intent at confidence thresholds
        curve = []
        for th in (0.5, 0.7, 0.8, 0.9):
            cov = [r for r in rs if conf(r["intent"]) >= th]
            curve.append({"threshold": th, "coverage": f"{len(cov)}/{n}",
                          "accuracy": round(sum(r["intent"]["choice"] == r["label"]["intent"] for r in cov) / len(cov), 3) if cov else None})
        s[lang] = {"n": n, "intent_acc": round(intent_ok / n, 3), "risk_acc": round(risk_ok / n, 3), "difficulty_mae": round(mae, 2),
                   "high_risk_predicted_safe": dangerous, "intent_selective": curve}
    return s

if __name__ == "__main__":
    main()
