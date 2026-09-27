"""How much task text fits losslessly in a Laya checkpoint for the pointsman route questions.

Uses the same admission rule as workers/laya_worker.py (assert_lossless): no silent truncation of state.
Prints the largest prefix length (characters) of long English/Korean task texts that still passes.
Offline; tokenizer only (no model weights loaded).

    <laya-venv>/bin/python scripts/laya-budget.py --model-dir /abs/ckpt

With --fit, instead measures workers/laya_worker.py fit_task_head()'s added latency (the opt-in
`laya.inputFit: 'task-head'` path) on 2,500-char EN/KO tasks, over N runs (default 20):

    <laya-venv>/bin/python scripts/laya-budget.py --model-dir /abs/ckpt --fit --runs 20
"""
import argparse, contextlib, io, json, os, statistics, sys, time
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
bench = __import__("laya-bench")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workers"))
worker = __import__("laya_worker")

EN = ("Implement pagination for the /orders API endpoint. Keep the existing response shape, add cursor-based paging "
      "with a default page size of 50, update the TypeScript client in web/src/api/orders.ts, and add integration tests "
      "covering empty pages, the last page and invalid cursors. Do not change the database schema. ") * 8
KO = ("/orders API \uC5D4\uB4DC\uD3EC\uC778\uD2B8\uC5D0 \uD398\uC774\uC9C0\uB124\uC774\uC158\uC744 \uAD6C\uD604\uD574\uC918. \uAE30\uC874 \uC751\uB2F5 \uD615\uD0DC\uB294 \uC720\uC9C0\uD558\uACE0 \uAE30\uBCF8 \uD398\uC774\uC9C0 \uD06C\uAE30 50\uC758 \uCEE4\uC11C \uAE30\uBC18 \uD398\uC774\uC9D5\uC744 \uCD94\uAC00\uD55C \uB4A4, "
      "web/src/api/orders.ts\uC758 TypeScript \uD074\uB77C\uC774\uC5B8\uD2B8\uB97C \uACE0\uCE58\uACE0 \uBE48 \uD398\uC774\uC9C0·\uB9C8\uC9C0\uB9C9 \uD398\uC774\uC9C0·\uC798\uBABB\uB41C \uCEE4\uC11C\uB97C \uB2E4\uB8E8\uB294 \uD1B5\uD569 \uD14C\uC2A4\uD2B8\uB97C \uCD94\uAC00\uD574\uC918. "
      "DB \uC2A4\uD0A4\uB9C8\uB294 \uBC14\uAFB8\uC9C0 \uB9C8. ") * 8

def build_agent(model_dir):
    from transformers import AutoTokenizer
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import laya.agent  # noqa: F401 (common helpers used by assert_lossless/fit_task_head)
    cfg = json.loads((Path(model_dir) / "rl_agent_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
    class Shim:  # the parts of Agent that assert_lossless/fit_task_head read
        pass
    agent = Shim(); agent.tok = tok; agent.cfg = cfg; agent._to_internal = laya.agent.Agent._to_internal
    return agent, cfg

def run_budget(a):
    agent, cfg = build_agent(a.model_dir)
    def fits(text):
        try: worker.assert_lossless(agent, bench.state_for(text), bench.ROUTE_QUESTIONS); return True
        except ValueError: return False
    out = {"model_dir": a.model_dir, "max_len": cfg.get("max_len"), "head_max_len": cfg.get("head_max_len")}
    for name, text in (("en", EN), ("ko", KO)):
        lo, hi = 0, len(text)
        while lo < hi:  # largest prefix that passes
            mid = (lo + hi + 1) // 2
            if fits(text[:mid]): lo = mid
            else: hi = mid - 1
        out[f"{name}_max_chars"] = lo
        out[f"{name}_max_bytes"] = len(text[:lo].encode())
    print(json.dumps(out))

def run_fit(a):
    agent, cfg = build_agent(a.model_dir)
    # 2,500-char EN/KO tasks (matches the fit_task_head overhead target: "< 15ms added per call for
    # a 2,500-char task"), truncated/padded to exactly 2,500 chars from the repeating fixtures above.
    en_task = (EN * 3)[:2500]
    ko_task = (KO * 3)[:2500]
    out = {"model_dir": a.model_dir, "max_len": cfg.get("max_len"), "head_max_len": cfg.get("head_max_len"), "runs": a.runs}
    for name, task in (("en", en_task), ("ko", ko_task)):
        state = bench.state_for(task)
        kept_chars, times_ms = None, []
        for _ in range(a.runs):
            t0 = time.perf_counter()
            fitted_state, info = worker.fit_task_head(agent, state, bench.ROUTE_QUESTIONS)
            times_ms.append((time.perf_counter() - t0) * 1000)
            kept_chars = info.get("kept_chars")
        out[f"{name}_original_chars"] = len(task)
        out[f"{name}_kept_chars"] = kept_chars
        out[f"{name}_median_ms"] = round(statistics.median(times_ms), 3)
        out[f"{name}_max_ms"] = round(max(times_ms), 3)
    print(json.dumps(out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--fit", action="store_true", help="measure fit_task_head() overhead instead of the lossless budget")
    ap.add_argument("--runs", type=int, default=20)
    a = ap.parse_args()
    (run_fit if a.fit else run_budget)(a)

if __name__ == "__main__":
    main()
