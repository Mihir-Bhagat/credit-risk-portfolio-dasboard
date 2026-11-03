"""A. Repository layout, secrets, README (checklist items 1-19)."""
from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _readme() -> str:
    return (ROOT / "README.md").read_text()


@pytest.mark.item(1)
def test_readme_exists():
    assert (ROOT / "README.md").is_file() and len(_readme()) > 500


@pytest.mark.item(2)
def test_requirements_every_package_pinned_and_complete():
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert lines, "requirements.txt is empty"
    for l in lines:
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][A-Za-z0-9_.\-]*", l), f"not pinned with ==: {l}"
    pinned = {l.split("==")[0].lower().replace("_", "-") for l in lines}
    # every third-party import in app/ and src/ must be covered by a pin
    std = set(sys.stdlib_module_names)
    to_pkg = {"sklearn": "scikit-learn", "yaml": "pyyaml"}
    imported = set()
    for f in list((ROOT / "app").rglob("*.py")) + list((ROOT / "src").rglob("*.py")):
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else []
            imported |= {n for n in names if n not in std and n not in {"src", "app"}}
    missing = {to_pkg.get(i, i).lower().replace("_", "-") for i in imported} - pinned
    assert not missing, f"imported but not pinned: {missing}"


@pytest.mark.item(3)
def test_env_example_has_names_and_no_values():
    from src import ask
    lines = [l for l in (ROOT / ".env.example").read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert lines
    for l in lines:
        k, _, v = l.partition("=")
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", k) and v == "", f"must be KEY= with no value: {l!r}"
    keys = {l.split("=")[0] for l in lines}
    for p in ask.PROVIDERS.values():
        assert p["key_env"] in keys, f"{p['key_env']} (read by src/ask.py) missing from .env.example"


@pytest.mark.item(4)
def test_app_entry_point_exists():
    assert (ROOT / "app" / "app.py").is_file()


@pytest.mark.item(5)
def test_src_modules_exist_and_import():
    import importlib
    for m in ["ingestion", "features", "analytics", "model", "monitoring", "anomaly", "ask", "decisions", "build", "docs"]:
        assert (ROOT / "src" / f"{m}.py").is_file(), m
        importlib.import_module(f"src.{m}")


@pytest.mark.item(6)
def test_raw_csvs_unchanged():
    expected = dict(reversed(l.split()) for l in (ROOT / "tests" / "raw_data_sha256.txt").read_text().splitlines())
    csvs = ["customers.csv", "loan_accounts.csv", "bureau_credit.csv", "repayment_history.csv", "collections_events.csv"]
    for f in csvs:
        h = hashlib.sha256((ROOT / "data" / "raw" / f).read_bytes()).hexdigest()
        assert h == expected[f], f"{f} differs from the file provided"
    assert sorted(p.name for p in (ROOT / "data" / "raw").glob("*.csv")) == sorted(csvs)


@pytest.mark.item(7)
def test_architecture_pdf_exists():
    p = ROOT / "docs" / "architecture.pdf"
    assert p.is_file() and p.read_bytes()[:5] == b"%PDF-"


@pytest.mark.item(8)
def test_solution_overview_pdf_exists():
    p = ROOT / "docs" / "solution_overview.pdf"
    assert p.is_file() and p.read_bytes()[:5] == b"%PDF-"


@pytest.mark.item(9)
def test_tests_folder_present():
    assert len(list((ROOT / "tests").glob("test_*.py"))) >= 8


@pytest.mark.item(10)
def test_exactly_two_pdf_documents_and_no_slide_deck():
    pdfs = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.pdf") if ".git" not in p.parts and "venv" not in "".join(p.parts))
    assert pdfs == ["docs/architecture.pdf", "docs/solution_overview.pdf"], pdfs
    decks = [p for ext in ("pptx", "ppt", "key") for p in ROOT.rglob(f"*.{ext}")]
    assert not decks, decks


@pytest.mark.item(11)
def test_secret_scan_clean_tree_and_history():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "secret_scan.py"), str(ROOT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (ROOT / ".git").exists(), "git history must exist so the history scan is meaningful"


@pytest.mark.item(11)
def test_secret_scan_detects_planted_keys_including_in_history():
    """Positive control: the scanner must catch a key in the tree and one that was committed then deleted."""
    fake = "gsk_" + "A1b2C3d4E5" * 4
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        run = lambda *a: subprocess.run(a, cwd=d, capture_output=True, text=True, check=True)
        run("git", "init", "-q")
        run("git", "config", "user.email", "t@example.com")
        run("git", "config", "user.name", "t")
        (d / "a.py").write_text(f'KEY = "{fake}"\n')
        run("git", "add", "-A")
        run("git", "commit", "-qm", "oops")
        (d / "a.py").write_text("KEY = None\n")
        run("git", "commit", "-qam", "removed")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "secret_scan.py"), str(d)], capture_output=True, text=True)
        assert r.returncode == 1 and "git history" in r.stdout, r.stdout


@pytest.mark.item(12)
def test_gitignore_covers_secrets_and_os_files():
    for f in [".env", ".DS_Store", ".streamlit/secrets.toml", "sub/.DS_Store"]:
        r = subprocess.run(["git", "check-ignore", "-q", f], cwd=ROOT)
        assert r.returncode == 0, f"{f} is not git-ignored"
    r = subprocess.run(["git", "check-ignore", "-q", ".env.example"], cwd=ROOT)
    assert r.returncode == 1, ".env.example must stay tracked"


@pytest.mark.item(13)
def test_app_footer_shows_build_id_for_matching_repo_to_hosted_app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=120).run()
    assert any("Build" in c.value for c in at.sidebar.caption), [c.value for c in at.sidebar.caption]


@pytest.mark.item(15)
def test_readme_states_python_version_matching_runtime():
    m = re.search(r"Python version:\s*\*\*(\d+\.\d+)\*\*", _readme())
    assert m, "README must state 'Python version: **X.Y**'"
    assert (ROOT / ".python-version").read_text().strip() == m.group(1)
    assert f"{sys.version_info.major}.{sys.version_info.minor}" == m.group(1), "tests are not running on the documented Python version"


def _run_commands() -> list[str]:
    m = re.search(r"## Run it.*?```bash\n(.*?)```", _readme(), re.S)
    assert m, "README needs a 'Run it' section with a bash block"
    return [re.sub(r"\s+#.*$", "", l).strip() for l in m.group(1).splitlines() if l.strip()]


@pytest.mark.item(16)
def test_readme_run_instructions_at_most_three_commands():
    cmds = _run_commands()
    assert 1 <= len(cmds) <= 3, cmds
    assert cmds[0].startswith("pip install") and "requirements.txt" in cmds[0]
    assert "src.build" in cmds[1]
    assert cmds[2].startswith("streamlit run app/app.py")


@pytest.mark.item(17, 50, 26, 85)
@pytest.mark.slow
def test_fresh_virtualenv_runs_only_readme_commands_and_every_view_loads_in_browser():
    r = subprocess.run(["bash", str(ROOT / "scripts" / "fresh_venv_check.sh"), sys.executable], capture_output=True, text=True, cwd=ROOT,
                       env={**__import__("os").environ, "FRESH_PYTHON": sys.executable}, timeout=1500)
    print(r.stdout[-3000:], r.stderr[-1500:])
    assert r.returncode == 0 and "FRESH VENV CHECK: PASS" in r.stdout, r.stdout[-2500:] + r.stderr[-1500:]


@pytest.mark.item(18)
def test_readme_documents_key_setup_and_no_key_behaviour():
    t = _readme()
    assert re.search(r"GROQ_API_KEY|GEMINI_API_KEY|OPENROUTER_API_KEY", t)
    assert "secrets" in t.lower() and ".env.example" in t
    assert re.search(r"no key|without a key|offline mode", t, re.I) and re.search(r"visible notice", t, re.I)
    assert "preset" in t.lower() or "example questions" in t.lower()


@pytest.mark.item(19)
def test_readme_mapping_table_covers_brief_and_paths_exist():
    t = _readme()
    assert "Where each item in the brief lives" in t
    rows = [l for l in t.splitlines() if l.startswith("|") and not set(l) <= set("|- ")]
    body = "\n".join(rows)
    for needle in ["Data ingestion", "data-quality", "Portfolio analytics", "Default model", "monitor", "Dashboard", "Ask the Portfolio",
                   "Anomaly detection", "false positive", "architecture", "solution overview", "hosted dashboard", "GitHub", "demo video"]:
        assert needle.lower() in body.lower(), f"mapping table lacks '{needle}'"
    paths = set(re.findall(r"`((?:src|app|docs|reports|data|tests)/[A-Za-z0-9_./*]+|[A-Za-z_]+\.(?:md|txt))`", body))
    assert len(paths) >= 15
    for p in paths:
        if "*" in p or p.startswith("data/processed"):
            continue
        assert (ROOT / p).exists(), f"README maps to a path that does not exist: {p}"
