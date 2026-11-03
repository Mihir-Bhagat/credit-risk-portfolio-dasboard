"""Run the Ask-the-Portfolio question suite against a REAL LLM provider (needs a key in the environment).

    GROQ_API_KEY=... python scripts/live_llm_check.py            # or GEMINI_API_KEY / OPENROUTER_API_KEY / LLM_API_KEY
    python scripts/live_llm_check.py --max-calls 45

For every question it records: did the provider answer, did the plan match the independent offline interpreter's plan (for
answerable questions), was the question declined when it should be, latency, and the exact number of API calls made.
Writes tests/reports/llm_comparison.md. The key is read from the environment only and never printed or written.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import ask as K  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-calls", type=int, default=45)
    ap.add_argument("--pause", type=float, default=2.5, help="seconds between calls (stay under free-tier limits)")
    a = ap.parse_args()
    cfg = K.provider_from_env()
    if cfg is None:
        print("No API key found in the environment (GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY or LLM_API_KEY). Nothing was called.")
        return 2
    pf = pd.read_parquet(ROOT / "data" / "processed" / "portfolio.parquet")
    cat = K.build_catalog(pf)
    off = K.OfflinePlanner(cat)
    llm = K.LLMPlanner(cfg, cat)
    suite = yaml.safe_load((ROOT / "tests" / "qa_suite.yaml").read_text())
    rows, lat = [], []
    for group in ("answerable", "unanswerable", "adversarial"):
        for e in suite[group]:
            if llm.calls >= a.max_calls:
                print(f"stopping: reached --max-calls {a.max_calls}")
                break
            before = llm.calls
            t = time.time()
            ans = K.ask(e["q"], pf, cat, llm=llm, offline=off)
            dt = time.time() - t
            called = llm.calls > before
            if called:
                lat.append(dt)
                time.sleep(a.pause)
            if group == "answerable":
                ref = K.ask(e["q"], pf, cat, llm=None, offline=off)
                same = (ans.status == "answered" and ref.status == "answered" and ans.mode == "llm"
                        and ans.table is not None and ans.table.reset_index(drop=True).round(9).equals(ref.table.reset_index(drop=True).round(9)))
                ok = same
                note = "plan reproduces the offline result" if same else f"mode={ans.mode} status={ans.status} {ans.notice[:60]}"
            elif group == "unanswerable":
                ok = ans.status == "declined" and ans.table is None
                note = f"declined by {ans.mode}" if ok else "ANSWERED a question the data cannot answer"
            else:
                ok = ans.status == "declined" and ans.table is None
                note = f"declined by {ans.mode}" if ok else "ANSWERED an adversarial prompt"
            rows.append((e["id"], group, called, ok, round(dt, 2), note))
            print(f"{e['id']:>4} {group:<12} api_call={called!s:<5} {'PASS' if ok else 'FAIL'} {dt:5.2f}s {note}")
    n_ok = sum(r[3] for r in rows)
    md = [f"# Live LLM check: {cfg.name} / {cfg.model}", "", f"Questions: {len(rows)}; passed: {n_ok}; **real API calls: {llm.calls}**",
          f"Latency per API call: median {statistics.median(lat):.2f}s, max {max(lat):.2f}s" if lat else "No API calls were made.", "",
          "| id | group | API call | result | seconds | note |", "|---|---|---|---|---|---|"]
    md += [f"| {i} | {g} | {'yes' if c else 'no (guardrail)'} | {'PASS' if ok else 'FAIL'} | {s} | {n} |" for i, g, c, ok, s, n in rows]
    out = ROOT / "tests" / "reports" / "llm_comparison.md"
    out.write_text("\n".join(md) + "\n")
    print(f"\n{n_ok}/{len(rows)} passed; {llm.calls} real API calls; report written to {out.relative_to(ROOT)}")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
