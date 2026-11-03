"""Group K: LLM selection and adapter (items 88-91) plus the manual-check documents (items 13, 14, 51, 52, 71, 80-82)."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import requests

from fake_llm import FakeLLM
from src import ask as K

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.item(88)
def test_research_doc_has_shortlist_limits_and_sources():
    t = (ROOT / "tests" / "reports" / "llm_research.md").read_text()
    for need in ("Groq", "Gemini", "OpenRouter", "30 requests/min", "1,000 requests/day", "20 requests/min", "50 requests/day", "Decision",
                 "Data terms", "Sources", "MANUAL"):
        assert need in t, need
    urls = re.findall(r"\]\((https://[^)]+)\)", t)
    assert len(urls) >= 3 and all(u.startswith("https://") for u in urls)
    # every default model id in code is named in the research doc (so the doc cannot drift from the adapter)
    for name, p in K.PROVIDERS.items():
        if p["model"]:
            assert p["model"] in t, (name, p["model"])
    assert "Real LLM API calls made during development and verification: **0**" in t


@pytest.mark.item(91)
def test_live_check_script_refuses_without_key_and_makes_no_calls():
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY") and not k.startswith("LLM_")}
    r = subprocess.run([sys.executable, "scripts/live_llm_check.py"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "No API key found" in r.stdout and "Nothing was called" in r.stdout


@pytest.mark.item(91)
def test_live_check_script_runs_end_to_end_against_local_server(tmp_path):
    """The benchmark script itself is exercised (not mocked) against the fake OpenAI-compatible server."""
    fake = FakeLLM()
    url = fake.start()
    fake.default_plan = {"answerable": False, "reason": "no canned plan"}
    try:
        env = {**os.environ, "LLM_API_KEY": "sk-test-123456", "LLM_BASE_URL": url, "LLM_MODEL": "fake"}
        r = subprocess.run([sys.executable, "scripts/live_llm_check.py", "--pause", "0", "--max-calls", "60"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    finally:
        fake.stop()
    out = ROOT / "tests" / "reports" / "llm_comparison.md"
    try:
        assert "real API calls" in r.stdout, r.stdout + r.stderr
        n = int(re.search(r"(\d+) real API calls", r.stdout).group(1))
        assert n == len(fake.requests) and n > 0, (n, len(fake.requests))
        # a fake that declines everything must be caught as wrong on the answerable questions
        assert r.returncode == 1 and "FAIL" in r.stdout
        assert "sk-test-123456" not in r.stdout + out.read_text()
    finally:
        out.unlink(missing_ok=True)  # the fake-server run must not masquerade as a real comparison


@pytest.mark.item(91)
def test_call_counter_counts_http_requests_exactly():
    fake = FakeLLM()
    url = fake.start()
    try:
        fake.default_plan = {"answerable": True, "metric": "accounts", "group_by": [], "filters": []}
        import pandas as pd
        cat = K.build_catalog(pd.read_parquet(ROOT / "data" / "processed" / "portfolio.parquet"))
        llm = K.LLMPlanner(K.ProviderConfig("custom", url, "m", "k"), cat, retries=0)
        for q in ("How many accounts are there?", "How many accounts are there?", "count of accounts by region"):
            llm.plan(q)
        assert llm.calls == len(fake.requests) == 2  # identical question served from cache
    finally:
        fake.stop()


@pytest.mark.item(90)
def test_adapter_has_no_hardcoded_secret_and_reads_env_only():
    src = (ROOT / "src" / "ask.py").read_text()
    assert not re.search(r"(gsk_|AIza|sk-or-|sk-[A-Za-z0-9]{20,})", src)
    keys = set(re.findall(r'key_env="(\w+)"', src))
    env_example = (ROOT / ".env.example").read_text()
    for k in keys | {"LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL"}:
        assert re.search(rf"^{k}=$", env_example, re.M), k


@pytest.mark.item(13, 14, 51, 52, 71, 80, 81, 82)
def test_manual_checks_document_is_complete():
    t = (ROOT / "tests" / "MANUAL_CHECKS.md").read_text()
    for need in ("private/incognito window", "no login", "Record decision", "3 to 5 minutes", "exactly three links", "No attachments", "Streamlit Community",
                 "scripts/live_llm_check.py", "secret_scan.py", "14 days", "public, or private with the reviewer", "INTERVIEW_NOTES.md"):
        assert need.lower() in t.lower(), need
    assert (ROOT / "INTERVIEW_NOTES.md").is_file()
    notes = (ROOT / "INTERVIEW_NOTES.md").read_text()
    for need in ("Observation point", "Collections", "Split", "What is sent to the LLM"):
        assert need in notes
