"""One command that runs the FULL verification and records it.

    python scripts/verify_all.py                # wipe generated files, rebuild from raw CSVs, run every test, log the run
    python scripts/verify_all.py --no-rebuild   # skip the wipe + rebuild (faster, for debugging only; not accepted as a final run)

Steps
  1. delete every generated artifact (processed data, models, reports, PDFs, previous results)
  2. python -m src.build        (the real pipeline, from data/raw)
  3. pytest tests/              (all groups, including the slow fresh-virtualenv and real-browser tests)
  4. map each test to the checklist items it verifies (@pytest.mark.item(...)):
       an item is PASS only if it has >= 1 test and every test passed; skipped / xfail / error / no test = FAIL;
       items the checklist marks MANUAL are listed with their steps and never counted as PASS
  5. append the run to tests/verify_log.md (+ tests/verify_history.json), rewrite the status column of the checklist
Exit code 0 = every automated item passed in this run. SUCCESS needs two consecutive such runs (the script says when).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKLIST = ROOT / "tests" / "acceptance_checklist.md"
LOG = ROOT / "tests" / "verify_log.md"
HISTORY = ROOT / "tests" / "verify_history.json"
RESULTS = ROOT / "tests" / ".verify_results.json"
GENERATED = ["data/processed", "reports", "models", "docs/architecture.pdf", "docs/solution_overview.pdf", "docs/data_quality_log.md",
             "tests/.verify_results.json", "tests/reports/screenshots", "tests/reports/pdf_previews"]
KEY_VARS = ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "LLM_API_KEY")


def parse_checklist() -> dict[int, dict]:
    items = {}
    for line in CHECKLIST.read_text().splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|(.*)\|\s*(PASS|FAIL|MANUAL)\s*\|\s*$", line)
        if m:
            cells = [c.strip() for c in m.group(2).split("|")]
            items[int(m.group(1))] = {"req": cells[0], "status0": m.group(3)}
    return items


def run(cmd, **kw):
    t = time.time()
    p = subprocess.run(cmd, cwd=ROOT, **kw)
    return p, time.time() - t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rebuild", action="store_true")
    ap.add_argument("-k", default=None, help="pytest -k expression (debugging only)")
    a = ap.parse_args()
    py = sys.executable
    history = json.loads(HISTORY.read_text()) if HISTORY.exists() else []
    iteration = len(history) + 1
    started = datetime.now(timezone.utc)
    print(f"=== verification iteration {iteration} at {started:%Y-%m-%d %H:%M:%SZ} ===")
    real_keys = [k for k in KEY_VARS if os.environ.get(k)]
    if real_keys:
        print(f"NOTE: {real_keys} are set in this shell; automated tests clear them and never call a real provider.")

    build_s = None
    if not a.no_rebuild:
        for g in GENERATED:
            p = ROOT / g
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        print("[1/3] rebuilding from raw CSVs ...")
        p, build_s = run([py, "-m", "src.build"])
        if p.returncode != 0:
            print("BUILD FAILED"); return 3
    print("[2/3] running the full test suite ...")
    RESULTS.unlink(missing_ok=True)
    cmd = [py, "-m", "pytest", "tests", "-q", "--tb=short", "-p", "no:cacheprovider"] + (["-k", a.k] if a.k else [])
    p, test_s = run(cmd)
    if not RESULTS.exists():
        print("no results file written: pytest crashed"); return 3
    res = json.loads(RESULTS.read_text())

    items = parse_checklist()
    per_item: dict[int, list] = {i: [] for i in items}
    unknown = set()
    for nodeid, r in res.items():
        for i in r["items"]:
            if i not in items:
                unknown.add(i)
            else:
                ok = r["outcome"] == "passed" and not r["wasxfail"]
                per_item[i].append((nodeid, ok, r["outcome"]))
    status, why = {}, {}
    for i, it in items.items():
        if it["status0"] == "MANUAL":
            status[i] = "MANUAL"; continue
        ts = per_item[i]
        if not ts:
            status[i], why[i] = "FAIL", "no test maps to this item"
        elif all(ok for _, ok, _ in ts):
            status[i] = "PASS"
        else:
            status[i] = "FAIL"; why[i] = "; ".join(f"{n} [{o}]" for n, ok, o in ts if not ok)
    if unknown:
        print(f"ERROR: tests reference unknown checklist items {sorted(unknown)}")
    unmarked = [n for n, r in res.items() if not r["items"]]
    failed_tests = [n for n, r in res.items() if r["outcome"] != "passed"]

    n_auto = sum(1 for s in status.values() if s != "MANUAL")
    n_pass = sum(1 for s in status.values() if s == "PASS")
    green = n_pass == n_auto and not failed_tests and not unknown and not unmarked
    entry = {"iteration": iteration, "time_utc": started.isoformat(timespec="seconds"), "build_seconds": round(build_s, 1) if build_s else None,
             "test_seconds": round(test_s, 1), "tests": len(res), "tests_failed": len(failed_tests), "status": status, "green": green,
             "rebuilt": not a.no_rebuild and not a.k, "real_llm_api_calls": 0}
    history.append(entry)
    HISTORY.write_text(json.dumps(history, indent=1))

    # ---- log
    lines = [f"\n## Iteration {iteration} - {started:%Y-%m-%d %H:%M:%SZ} - {'GREEN' if green else 'RED'}", "",
             f"Rebuilt from raw: {'yes' if entry['rebuilt'] else 'NO (debug run)'} · build {entry['build_seconds']}s · {len(res)} tests in {entry['test_seconds']}s · "
             f"failed tests: {len(failed_tests)} · real LLM API calls: 0", "",
             f"PASS {n_pass} / {n_auto} automated items · MANUAL {sum(1 for s in status.values() if s == 'MANUAL')}", "", "| item | status | detail |", "|---|---|---|"]
    for i in sorted(items):
        lines.append(f"| {i} | {status[i]} | {why.get(i, '')[:200]} |")
    if failed_tests:
        lines += ["", "Failed tests:"] + [f"- `{n}` ({res[n]['outcome']})" for n in failed_tests]
    first = not LOG.exists()
    with LOG.open("a") as f:
        if first:
            f.write("# Verification log\n\nEach section is one full run of `python scripts/verify_all.py`. SUCCESS = two consecutive GREEN runs that rebuilt from raw.\n")
        f.write("\n".join(lines) + "\n")

    # ---- checklist status column
    out = []
    for line in CHECKLIST.read_text().splitlines():
        m = re.match(r"^(\|\s*(\d+)\s*\|.*\|\s*)(PASS|FAIL|MANUAL)(\s*\|\s*)$", line)
        out.append(f"{m.group(1)}{status[int(m.group(2))]}{m.group(4)}" if m and int(m.group(2)) in status else line)
    CHECKLIST.write_text("\n".join(out) + "\n")

    # ---- console summary + stop conditions
    print("\n" + "=" * 70)
    print(f"iteration {iteration}: {n_pass}/{n_auto} automated items PASS, {len(failed_tests)} failed tests, {len(res)} tests total")
    for i in sorted(items):
        if status[i] == "FAIL":
            print(f"  FAIL item {i}: {items[i]['req'][:70]} -> {why.get(i, '')[:160]}")
    if unmarked:
        print(f"  tests without an item marker: {unmarked[:5]}")
    # same item failing three iterations in a row => STUCK
    for i in items:
        last3 = [h["status"].get(str(i)) for h in history[-3:]]
        if len(last3) == 3 and all(s == "FAIL" for s in last3):
            print(f"  STUCK: item {i} failed 3 iterations in a row")
    if iteration >= 12 and not green:
        print("  STUCK: 12 iterations reached")
    real = [h for h in history if h["rebuilt"]]
    if green and len(history) >= 2 and history[-2]["green"] and history[-2]["rebuilt"] and entry["rebuilt"]:
        print("SUCCESS: two consecutive GREEN full runs.")
    elif green:
        print("GREEN: run once more (fully rebuilt) to reach the two-consecutive-green stop condition.")
    manual = [i for i in sorted(items) if status[i] == "MANUAL"]
    print("MANUAL items (see tests/MANUAL_CHECKS.md):", ", ".join(f"#{i}" for i in manual))
    return 0 if green else 1


if __name__ == "__main__":
    sys.exit(main())
