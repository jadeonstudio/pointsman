"""Local Laya benchmark: load-time breakdown, warm latency and 16-bit agreement on MPS/CPU.

Offline only (HF_HUB_OFFLINE=1). Never modifies model files: the official in-place tokenizer
helper is disabled. Run with the Laya runtime's own Python, one checkpoint per process:

    <laya-venv>/bin/python scripts/laya-bench.py --model-dir /abs/checkpoint --device mps --out result.json
"""
import argparse, contextlib, cProfile, io, json, os, pstats, statistics, sys, time

T0 = time.perf_counter()
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", USE_TF="0", USE_TORCH="1")
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)

# pointsman route questions (src/routing.mjs ROUTE_QUESTIONS), identical wording.
ROUTE_QUESTIONS = {
    "intent": {"type": "choice", "instructions": "Classify the work actually required. Use other when the request is unclear or outside these categories.", "criteria": {
        "explain": "Explain or locate existing code; do not change behavior", "edit": "Write or change a bounded piece of code or documentation",
        "debug": "Investigate the cause of a failure", "operate": "Operate infrastructure, deploy, or modify live data",
        "research": "Research information outside the repository", "architecture": "Cross-module design, whole-repository audit or major refactoring",
        "other": "Insufficient evidence or none of these intents"}},
    "difficulty": {"type": "score", "instructions": "Estimate the reasoning needed to finish, not prompt length. Account for unresolved dependencies and unknown scope. Do not infer that a short request is easy.", "criteria": [
        "1: Mechanical, exact edit with explicit location and no behavioral change", "2: Small local change with explicit requirements and known validation",
        "3: Moderate implementation requiring several related steps", "4: Difficult debugging or interacting components needing substantial investigation",
        "5: Deep reasoning, unknown repository-wide impact, architecture or long-horizon planning"]},
    "risk": {"type": "choice", "instructions": "Classify consequence and uncertainty, not permission. Safe requires evidence of local, reversible impact. Missing context is unknown. Risk is independent of difficulty.", "criteria": {
        "safe": "Known local, reversible work with no security, production, payment or financial impact",
        "caution": "Material uncertainty or changes needing additional review",
        "high": "Irreversible operation, production, credentials, permissions, financial or payment consequences",
        "unknown": "Not enough context to assess impact"}},
}
TASKS = [
    "Fix the typo 'teh' -> 'the' in README.md line 12. No code changes.",
    "Explain where the retry policy for HTTP requests is implemented.",
    "Add a --dry-run flag to the export command and a unit test for it.",
    "The nightly build fails with a segfault in the image decoder; find the cause.",
    "Rotate the production database credentials and redeploy all services.",
    "Redesign the plugin system so modules can be loaded lazily across the whole repo.",
    "README.md 12\uBC88\uC9F8 \uC904\uC758 \uC624\uD0C0 'teh'\uB97C 'the'\uB85C \uACE0\uCCD0\uC918. \uCF54\uB4DC\uB294 \uBC14\uAFB8\uC9C0 \uC54A\uC74C.",
    "HTTP \uC694\uCCAD \uC7AC\uC2DC\uB3C4 \uC815\uCC45\uC774 \uC5B4\uB514\uC5D0 \uAD6C\uD604\uB3FC \uC788\uB294\uC9C0 \uC124\uBA85\uD574\uC918.",
    "export \uBA85\uB839\uC5D0 --dry-run \uC635\uC158\uACFC \uB2E8\uC704 \uD14C\uC2A4\uD2B8\uB97C \uCD94\uAC00\uD574\uC918.",
    "\uC57C\uAC04 \uBE4C\uB4DC\uAC00 \uC774\uBBF8\uC9C0 \uB514\uCF54\uB354\uC5D0\uC11C segfault\uB85C \uC2E4\uD328\uD574. \uC6D0\uC778\uC744 \uCC3E\uC544\uC918.",
    "\uC6B4\uC601 DB \uBE44\uBC00\uBC88\uD638\uB97C \uAD50\uCCB4\uD558\uACE0 \uBAA8\uB4E0 \uC11C\uBE44\uC2A4\uB97C \uC7AC\uBC30\uD3EC\uD574\uC918.",
    "\uD50C\uB7EC\uADF8\uC778 \uC2DC\uC2A4\uD15C\uC744 \uC800\uC7A5\uC18C \uC804\uCCB4\uC5D0\uC11C \uC9C0\uC5F0 \uB85C\uB529\uB418\uB3C4\uB85D \uC7AC\uC124\uACC4\uD574\uC918.",
]

def state_for(task):
    return {"task": task, "context": {"complete": True, "scope": "local", "previousFailures": 0, "highImpact": False, "modelLocked": False, "exhaustive": False}}

def top_answer(ans):
    if "choice" in ans: return ans["choice"]
    if "score" in ans: return round(ans["score"])
    for k in ("probability", "probabilityTrue", "p_true"):
        if k in ans: return ans[k] >= 0.5
    return ans.get("value")

def summarize(ts):
    ts = sorted(ts)
    return {"n": len(ts), "p50": round(statistics.median(ts), 2), "p95": round(ts[max(0, int(0.95 * len(ts)) - 1)], 2), "min": round(ts[0], 2), "max": round(ts[-1], 2)}

def compare(ref, cur):
    agree = total = 0; max_dp = 0.0
    for a_, b_ in zip(ref, cur):
        for q in a_:
            total += 1; agree += top_answer(a_[q]) == top_answer(b_[q])
            rp, cp = a_[q].get("probabilities"), b_[q].get("probabilities")
            if isinstance(rp, dict) and isinstance(cp, dict): max_dp = max(max_dp, max(abs(rp[k] - cp.get(k, 0)) for k in rp))
            if "score" in a_[q]: max_dp = max(max_dp, abs(a_[q]["score"] - b_[q]["score"]))
    return {"agreement_vs_reference": f"{agree}/{total}", "max_prob_delta_vs_reference": round(max_dp, 5)}

def run_mode(a, agent, torch, call):
    import laya.agent as la
    base = a.mode.replace("-fastinit", "")
    dtype = {"amp-fp16": torch.float16, "amp-bf16": torch.bfloat16, "half": torch.float16}.get(base)
    if base == "half": agent.model.half()
    if dtype is not None:
        orig = torch.autocast
        # system_one only enables autocast on CUDA; force it for this device so MPS/CPU use 16-bit kernels.
        la.torch.autocast = lambda device_type, dtype_=None, enabled=True, **kw: orig(device_type, dtype=dtype, enabled=True)
    if a.device == "mps": torch.mps.empty_cache()
    one = {"intent": ROUTE_QUESTIONS["intent"]}
    for _ in range(3): call(ROUTE_QUESTIONS, TASKS[1])
    lat1, lat3 = [], []
    for i in range(a.n):
        t = time.perf_counter(); call(one, TASKS[i % len(TASKS)]); lat1.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter(); call(ROUTE_QUESTIONS, TASKS[i % len(TASKS)]); lat3.append((time.perf_counter() - t) * 1000)
    answers = [call(ROUTE_QUESTIONS, task)["answers"] for task in TASKS]
    mem = {}
    if a.device == "mps":
        mem = {"mps_allocated_mib": round(torch.mps.current_allocated_memory() / 2**20, 1), "mps_driver_mib": round(torch.mps.driver_allocated_memory() / 2**20, 1)}
    return {"latency_1q_ms": summarize(lat1), "latency_3q_ms": summarize(lat3), "memory": mem,
            "answers": [{q: top_answer(v) for q, v in ans.items()} for ans in answers], "answers_full": answers}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--modes", default="fp32,amp-fp16,amp-bf16,half")
    ap.add_argument("--mode", help="internal: run exactly one mode in this process")
    ap.add_argument("--profile-load", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = {"model_dir": a.model_dir, "device": a.device, "timings_ms": {}}

    t = time.perf_counter(); import torch; out["timings_ms"]["import_torch"] = round((time.perf_counter() - t) * 1000, 1)
    t = time.perf_counter(); import transformers  # noqa: F401
    out["timings_ms"]["import_transformers"] = round((time.perf_counter() - t) * 1000, 1)
    t = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import laya, laya.agent
    laya.agent._fix_tokenizer_config = lambda _: None  # never edit model files in place
    out["timings_ms"]["import_laya"] = round((time.perf_counter() - t) * 1000, 1)
    out["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                       "laya": getattr(laya, "__version__", None)}

    prof = cProfile.Profile() if a.profile_load else None
    t = time.perf_counter()
    if prof: prof.enable()
    fast = bool(a.mode and a.mode.endswith("-fastinit"))
    # transformers' official no_init_weights(): skip random init that load_state_dict(strict=True) overwrites anyway.
    from transformers.initialization import no_init_weights
    init_ctx = no_init_weights() if fast else contextlib.nullcontext()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), init_ctx:
        agent = laya.agent.Agent(a.model_dir, device=a.device)
    out["fast_init"] = fast
    if prof:
        prof.disable(); s = io.StringIO(); pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(18)
        out["load_profile_top"] = s.getvalue().splitlines()[:60]
    out["timings_ms"]["agent_init"] = round((time.perf_counter() - t) * 1000, 1)
    out["device_actual"] = str(agent.device)
    out["params"] = sum(p.numel() for p in agent.model.parameters())

    sync = (lambda: torch.mps.synchronize()) if a.device == "mps" else (lambda: None)
    def call(qs, task):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = agent.system_one(state_for(task), qs)
        sync(); return r

    t = time.perf_counter(); first = call(ROUTE_QUESTIONS, TASKS[0]); out["timings_ms"]["first_inference_3q"] = round((time.perf_counter() - t) * 1000, 1)
    out["timings_ms"]["cold_total_from_process_start"] = round((time.perf_counter() - T0) * 1000, 1)

    if a.mode:
        out["result"] = run_mode(a, agent, torch, call)
        print(json.dumps(out, ensure_ascii=False)); return
    results = {}
    for mode in a.modes.split(","):
        # Each mode runs in its own process: an unsupported MPS kernel aborts the process (C++ assert).
        import subprocess
        p = subprocess.run([sys.executable, __file__, "--model-dir", a.model_dir, "--device", a.device, "--n", str(a.n), "--mode", mode],
                           capture_output=True, text=True, timeout=900)
        try:
            child = json.loads(p.stdout.strip().splitlines()[-1])
            results[mode] = {**child["result"], "cold_ms": child["timings_ms"]}
        except Exception: results[mode] = {"error": f"exit {p.returncode}: " + (p.stderr.strip().splitlines() or ["?"])[-1][:240]}
    # The first mode that succeeded is the reference (normally fp32 or fp32-fastinit).
    ref_mode = next((m for m, r in results.items() if "answers_full" in r), None)
    ref = results[ref_mode]["answers_full"] if ref_mode else None
    for mode, r in results.items():
        if mode != ref_mode and ref and "answers_full" in r: r.update(compare(ref, r["answers_full"]), reference_mode=ref_mode)
    for r in results.values(): r.pop("answers_full", None)
    out["modes"] = results
    out["sample_answer_keys"] = sorted(first.get("answers", {}).get("intent", {}).keys())
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if a.out: open(a.out, "w").write(text)
    print(text)

if __name__ == "__main__":
    main()
