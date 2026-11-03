"""Secret scan: working tree (tracked + untracked, not ignored) AND the full git history.

Usage: python scripts/secret_scan.py [repo_dir]      exit code 0 = clean, 1 = findings.
Also searches for the literal values of any API-key environment variables currently set.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "groq key": r"gsk_[A-Za-z0-9]{20,}",
    "google api key": r"AIza[0-9A-Za-z_\-]{30,}",
    "openrouter key": r"sk-or-v1-[A-Za-z0-9]{16,}",
    "openai/anthropic style key": r"sk-(?:ant-)?[A-Za-z0-9_\-]{20,}",
    "github token": r"gh[pousr]_[A-Za-z0-9]{30,}",
    "aws access key": r"AKIA[0-9A-Z]{16}",
    "slack token": r"xox[baprs]-[A-Za-z0-9\-]{10,}",
    "private key block": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "assigned secret": r"""(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*["'][A-Za-z0-9_\-/+=.]{16,}["']""",
}
SKIP_SUFFIX = {".parquet", ".pdf", ".png", ".joblib", ".db", ".pyc", ".zip"}
SKIP_PARTS = {"__pycache__", ".git", "venv_dev", ".venv"}
# Files that legitimately contain secret-LOOKING strings: the scanner's own patterns and the tests that plant fake keys.
SELF = {"scripts/secret_scan.py", "tests/test_a_repo.py"}


def _git(args, cwd) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, errors="replace")


def tracked_and_untracked(root: Path) -> list[Path]:
    r = _git(["ls-files", "-co", "--exclude-standard"], root)
    if r.returncode != 0:
        return [p for p in root.rglob("*") if p.is_file() and not (set(p.parts) & SKIP_PARTS)]
    return [root / f for f in r.stdout.splitlines() if (root / f).is_file()]


def env_secret_values() -> list[str]:
    vals = []
    for k, v in os.environ.items():
        if re.search(r"(API_KEY|SECRET|TOKEN|PASSWORD)$", k) and len(v) >= 12 and k not in {"CLAUDE_CODE_OAUTH_TOKEN"}:
            vals.append(v)
    return vals


def scan_text(text: str, where: str, findings: list[str], extra: list[str]):
    for name, pat in PATTERNS.items():
        for m in re.finditer(pat, text):
            findings.append(f"{where}: {name}: {m.group(0)[:6]}...")
    for v in extra:
        if v in text:
            findings.append(f"{where}: literal value of an API-key environment variable")


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    extra = env_secret_values()
    for p in tracked_and_untracked(root):
        rel = p.relative_to(root).as_posix()
        if rel in SELF or p.suffix in SKIP_SUFFIX or (set(p.parts) & SKIP_PARTS):
            continue
        if rel in {".env", ".streamlit/secrets.toml"} or rel.endswith("/secrets.toml"):
            findings.append(f"{rel}: secret file present in the tree (must be git-ignored and untracked)")
        try:
            scan_text(p.read_text(errors="strict"), rel, findings, extra)
        except (UnicodeDecodeError, OSError):
            continue
    if (root / ".git").exists():
        files = _git(["ls-files"], root).stdout.splitlines()
        if any(f == ".env" or f.endswith("secrets.toml") for f in files):
            findings.append("a secrets file is TRACKED by git")
        hist = _git(["log", "--all", "-p", "--no-color", "--", ".", ":(exclude)scripts/secret_scan.py", ":(exclude)tests/test_a_repo.py"], root).stdout
        scan_text(hist, "git history", findings, extra)
        names = _git(["log", "--all", "--name-only", "--pretty=format:"], root).stdout.splitlines()
        for n in set(names):
            if n == ".env" or n.endswith("secrets.toml"):
                findings.append(f"git history: secrets file was committed at some point: {n}")
    return sorted(set(findings))


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    res = scan(target)
    if res:
        print("SECRET SCAN: FINDINGS")
        print("\n".join(res))
        sys.exit(1)
    print(f"SECRET SCAN: clean ({target})")
