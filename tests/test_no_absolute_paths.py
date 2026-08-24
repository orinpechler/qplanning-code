"""The release must contain no trace of the machine it was developed on.

The checks here are deliberately *categorical* rather than a list of known-bad strings. A
denylist of real usernames, hostnames and account names would have to contain those very
secrets in order to search for them, which defeats the purpose — the scanner becomes the
leak. Matching on shape instead also catches paths and identifiers nobody thought to
enumerate.

Site-specific strings can still be checked: put one substring per line in a gitignored
``tests/forbidden_strings.txt`` and it will be picked up if present.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

SCANNED_SUFFIXES = {".py", ".yaml", ".yml", ".sh", ".sbatch", ".toml", ".md", ".cfg",
                    ".example", ".env", ".cff"}

SKIPPED_DIRS = {".git", ".venv", "venv", "artifacts", "runs", "outputs", "slurm_out",
                "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}

# `.env` is the user's own file, gitignored, and is *supposed* to hold real paths.
SKIPPED_FILES = {".env", Path(__file__).name}

# Absolute POSIX paths of two or more segments. The lookbehind keeps URLs (`https://…`),
# relative paths (`configs/base.yaml`) and interpolation suffixes (`${cfg:x}/runs/`) from
# matching; the three-character floor on the first segment keeps `/../..` and toy fixtures
# like `/r/runs/demo` out while still catching any real machine path.
_ABSOLUTE_PATH = re.compile(
    r"(?<![:\w~$/.}\-])/[A-Za-z0-9_.-]{3,}/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]*"
)

# Roots that are the same on every machine, or are obviously illustrative.
_GENERIC_PREFIXES = (
    "/tmp/", "/usr/", "/etc/", "/var/", "/opt/", "/dev/", "/proc/", "/sys/", "/bin/",
    "/sbin/", "/lib/", "/srv/", "/mnt/", "/media/", "/root/",
    "/path/to/", "/abs/path", "/absolute/path", "/your/", "/some/",
)

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# A scheduler account/partition/QOS with a value baked in rather than left to a profile.
_SCHEDULER_VALUE = re.compile(
    r"(?:--account|--partition|--qos|QP_ACCOUNT|QP_PARTITION|QP_QOS|SBATCH\s+-[ApqA])"
    r"\s*[=\s]\s*[\"']?([A-Za-z0-9][\w.-]*)",
)


def _files_to_scan() -> list[Path]:
    return sorted(
        p for p in REPO_ROOT.rglob("*")
        if p.is_file()
        and not any(part in SKIPPED_DIRS for part in p.parts)
        and p.name not in SKIPPED_FILES
        and p.suffix in SCANNED_SUFFIXES
    )


def _lines(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:  # pragma: no cover
        return
    for lineno, line in enumerate(text.splitlines(), start=1):
        yield lineno, line


def _report(offenders: list[str], what: str, remedy: str) -> None:
    assert not offenders, f"found {what} in shipped files. {remedy}\n" + "\n".join(offenders)


def test_scan_covers_the_repository() -> None:
    """Guard against the scan silently matching nothing after a layout change."""
    files = _files_to_scan()
    assert len(files) > 10, f"expected to scan the whole repo, found only {len(files)} files"
    names = {f.name for f in files}
    assert {"pyproject.toml", ".env.example", "README.md"} <= names


def test_no_absolute_filesystem_paths() -> None:
    """No path that only exists on one machine."""
    offenders = []
    for path in _files_to_scan():
        for lineno, line in _lines(path):
            for match in _ABSOLUTE_PATH.findall(line):
                if match.startswith(_GENERIC_PREFIXES):
                    continue
                offenders.append(f"  {path.relative_to(REPO_ROOT)}:{lineno}: {match}")
    _report(offenders, "machine-specific absolute paths",
            "Every location must come from a config field or an environment variable; see .env.example.")


def test_no_home_directory_references() -> None:
    offenders = []
    # `~name/` is a home path; a bare `~name` is usually Sphinx cross-reference syntax
    # (:class:`~module.Class`), so the trailing slash is what distinguishes them.
    pattern = re.compile(r"(?:/home/|/Users/)[A-Za-z0-9_.-]+|(?<![`:\w])~[A-Za-z][A-Za-z0-9_.-]*/")
    for path in _files_to_scan():
        for lineno, line in _lines(path):
            for match in pattern.findall(line):
                offenders.append(f"  {path.relative_to(REPO_ROOT)}:{lineno}: {match}")
    _report(offenders, "home-directory references", "Use $HOME or a configured path instead.")


def test_no_email_addresses() -> None:
    offenders = []
    for path in _files_to_scan():
        for lineno, line in _lines(path):
            for match in _EMAIL.findall(line):
                offenders.append(f"  {path.relative_to(REPO_ROOT)}:{lineno}: {match}")
    _report(offenders, "email addresses",
            "Scheduler notifications and contact details belong in a local profile, not the repo.")


def test_no_hardcoded_scheduler_identifiers() -> None:
    """Account, partition and QOS names must come from a profile the user supplies."""
    offenders = []
    for path in _files_to_scan():
        for lineno, line in _lines(path):
            match = _SCHEDULER_VALUE.search(line)
            if match:
                offenders.append(f"  {path.relative_to(REPO_ROOT)}:{lineno}: {match.group(0).strip()}")
    _report(offenders, "hardcoded scheduler identifiers",
            "Leave these empty in scripts/slurm/profiles/generic.env and let users set their own.")


def test_optional_site_denylist() -> None:
    """Check any site-specific strings the maintainer keeps in a gitignored file."""
    denylist = REPO_ROOT / "tests" / "forbidden_strings.txt"
    if not denylist.exists():
        pytest.skip("no tests/forbidden_strings.txt; nothing site-specific to check")
    needles = [s.strip() for s in denylist.read_text().splitlines()
               if s.strip() and not s.startswith("#")]
    offenders = []
    for path in _files_to_scan():
        if path == denylist:
            continue
        for lineno, line in _lines(path):
            for needle in needles:
                if needle in line:
                    offenders.append(f"  {path.relative_to(REPO_ROOT)}:{lineno}")
    _report(offenders, "strings from the local denylist", "Remove them before publishing.")


def test_no_import_time_environment_mutation() -> None:
    """No module may latch an environment variable at *import* time.

    This is how an upstream default for a model directory leaks into any process that
    imports it: a module-scope ``os.environ.setdefault`` runs the moment the module loads,
    before any configuration has been read. Inside a function the same call is fine —
    ``load_dotenv`` uses it deliberately so real environment variables win over the file —
    so the check is scoped to module level rather than banning the call outright.
    """
    def module_level_nodes(tree: ast.Module):
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            yield from ast.walk(node)

    offenders = []
    for path in sorted((REPO_ROOT / "qplanning").rglob("*.py")):
        for node in module_level_nodes(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = getattr(node.func.value, "attr", None) or getattr(node.func.value, "id", None)
            if node.func.attr in {"setdefault", "__setitem__"} and owner == "environ":
                offenders.append(f"  {path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif node.func.attr == "putenv":
                offenders.append(f"  {path.relative_to(REPO_ROOT)}:{node.lineno}")
    _report(offenders, "environment mutation at import time",
            "Route it through qplanning._env_setup.prepare_environment() so the user's exports win.")
