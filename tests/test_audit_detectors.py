"""Tests for the deterministic audit detectors.

Two detectors, each targeting a failure class that recurred in real
incident data before being named:

  * dead env flags  → "built-never-wired" (a capability exists, nothing
    ever engages it). The detector finds env vars read with a falsy
    default that no config, doc, script, or assignment ever sets.
  * deviation register → "normalization of deviance" (an accepted,
    documented deroga silently becomes the status quo). The detector
    collects declared-deviation markers with their age, so old
    unrevisited ones become visible.

Everything here is deterministic: no LLM, no network, no subprocess
other than git (for ages).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from critic_orchestrator.audit_detectors import (
    audit_repo,
    find_dead_env_flags,
    find_deviations,
)


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# D1 — dead env flags
# ---------------------------------------------------------------------------

def test_unwired_default_off_flag_is_found(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/feature.py", (
        "import os\n"
        "def enabled():\n"
        "    return os.environ.get('MYAPP_SUPER_FEATURE', '0') == '1'\n"
    ))
    flags = find_dead_env_flags(tmp_path)
    assert len(flags) == 1
    f = flags[0]
    assert f["flag"] == "MYAPP_SUPER_FEATURE"
    assert f["tier"] == "unwired"
    assert f["read_sites"] == ["pkg/feature.py:3"]


def test_flag_set_in_config_is_wired_and_not_reported(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/feature.py",
           "import os\nX = os.getenv('MYAPP_FLAG', '')\n")
    _write(tmp_path, "deploy/run.ps1", "$env:MYAPP_FLAG = '1'\n")
    flags = find_dead_env_flags(tmp_path)
    assert flags == []


def test_flag_documented_in_readme_is_wired(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/feature.py",
           "import os\nX = os.environ.get('MYAPP_FLAG', 'off')\n")
    _write(tmp_path, "README.md", "Set MYAPP_FLAG=1 to enable.\n")
    assert find_dead_env_flags(tmp_path) == []


def test_flag_referenced_only_in_tests_is_test_only_tier(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/feature.py",
           "import os\nX = os.environ.get('MYAPP_FLAG', '0')\n")
    _write(tmp_path, "tests/test_feature.py",
           "def test_x(monkeypatch):\n"
           "    monkeypatch.setenv('MYAPP_FLAG', '1')\n")
    flags = find_dead_env_flags(tmp_path)
    assert len(flags) == 1
    assert flags[0]["tier"] == "test-only"


def test_truthy_default_flag_is_not_a_candidate(tmp_path: Path) -> None:
    """A flag defaulting ON is live by default — not this class."""
    _write(tmp_path, "pkg/feature.py",
           "import os\nX = os.environ.get('MYAPP_FLAG', '1')\n")
    assert find_dead_env_flags(tmp_path) == []


def test_or_fallback_truthy_default_is_not_a_candidate(tmp_path: Path) -> None:
    """`os.environ.get('X') or 90` defaults ON via the `or` fallback —
    found as a real false positive on the first live run (the flag
    ENGRAM_MODEL_LOCK_TIMEOUT_S was reported unwired while its
    effective default was 90)."""
    _write(tmp_path, "pkg/feature.py",
           "import os\n"
           "T = float(os.environ.get('MYAPP_TIMEOUT_S') or 90)\n")
    assert find_dead_env_flags(tmp_path) == []


def test_missing_default_counts_as_off(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/feature.py",
           "import os\nX = os.environ.get('MYAPP_LONELY')\n")
    flags = find_dead_env_flags(tmp_path)
    assert len(flags) == 1
    assert flags[0]["default"] is None


def test_multiple_read_sites_are_merged(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/a.py",
           "import os\nX = os.environ.get('MYAPP_FLAG', '0')\n")
    _write(tmp_path, "pkg/b.py",
           "import os\nY = os.getenv('MYAPP_FLAG', 'false')\n")
    flags = find_dead_env_flags(tmp_path)
    assert len(flags) == 1
    assert len(flags[0]["read_sites"]) == 2


def test_skips_vendored_and_hidden_dirs(tmp_path: Path) -> None:
    _write(tmp_path, ".venv/lib/mod.py",
           "import os\nX = os.environ.get('VENDORED_FLAG', '0')\n")
    _write(tmp_path, "node_modules/x/y.py",
           "import os\nX = os.environ.get('NODE_FLAG', '0')\n")
    assert find_dead_env_flags(tmp_path) == []


# ---------------------------------------------------------------------------
# D4 — deviation register
# ---------------------------------------------------------------------------

def test_deviation_markers_are_collected(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/mod.py", (
        "# KNOWN LIMIT: reachability is not health\n"
        "x = 1\n"
        "# TODO: revisit after the bench\n"
    ))
    devs, _ = find_deviations(tmp_path, with_age=False)
    markers = {d["marker"] for d in devs}
    assert markers == {"KNOWN LIMIT", "TODO"}
    assert devs[0]["file"] == "pkg/mod.py"
    assert devs[0]["line"] == 1


def test_deviation_age_from_git_blame(tmp_path: Path) -> None:
    _write(tmp_path, "mod.py", "# FIXME: temporary shim\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "x"],
        cwd=tmp_path, check=True,
    )
    devs, _ = find_deviations(tmp_path, with_age=True)
    assert len(devs) == 1
    assert devs[0]["age_days"] is not None
    assert devs[0]["age_days"] >= 0


def test_deviation_cap_is_declared_not_silent(tmp_path: Path) -> None:
    """No silent caps: when the register truncates, it says so."""
    lines = "\n".join(f"# TODO: item {i}" for i in range(30))
    _write(tmp_path, "mod.py", lines + "\n")
    devs, truncated = find_deviations(tmp_path, with_age=False, cap=10)
    assert len(devs) == 10 and truncated is True
    report = audit_repo(tmp_path, with_age=False, deviations_cap=10)
    assert report["deviations_truncated"] is True


# ---------------------------------------------------------------------------
# Whole-repo report
# ---------------------------------------------------------------------------

def test_audit_repo_shape_and_determinism(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/feature.py",
           "import os\nX = os.environ.get('MYAPP_FLAG', '0')\n")
    _write(tmp_path, "pkg/mod.py", "# KNOWN LIMIT: declared\n")
    r1 = audit_repo(tmp_path, with_age=False)
    r2 = audit_repo(tmp_path, with_age=False)
    assert r1 == r2  # deterministic
    assert r1["kind"] == "repo_audit"
    assert r1["summary"]["dead_flags"] == 1
    assert r1["summary"]["deviations"] == 1
    json.dumps(r1)  # serializable


def test_audit_repo_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        audit_repo(tmp_path / "nope")


def test_truncation_is_returned_not_stashed_on_the_function(tmp_path: Path) -> None:
    """Truncation was reported via `find_deviations._last_truncated`, a
    mutable attribute on the function object — with an 8-thread server
    executor. Two concurrent audits would overwrite each other's flag, so
    the very mechanism that prevents a silent cap could itself go silent.
    That is this tool's own state-leak-across-runs class."""
    import critic_orchestrator.audit_detectors as ad
    assert not hasattr(ad.find_deviations, "_last_truncated"), (
        "truncation must not live on the function object"
    )
    lines = "\n".join(f"# TODO: item {i}" for i in range(30))
    _write(tmp_path, "mod.py", lines + "\n")
    devs, truncated = find_deviations(tmp_path, with_age=False, cap=10)
    assert len(devs) == 10 and truncated is True
    devs2, truncated2 = find_deviations(tmp_path, with_age=False, cap=99)
    assert len(devs2) == 30 and truncated2 is False
    # The first result must be unaffected by the second call.
    assert truncated is True


def test_concurrent_audits_do_not_cross_contaminate(tmp_path: Path) -> None:
    """The race, exercised: interleaved audits with different caps must
    each report their OWN truncation."""
    from concurrent.futures import ThreadPoolExecutor
    lines = "\n".join(f"# TODO: item {i}" for i in range(40))
    _write(tmp_path, "mod.py", lines + "\n")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(audit_repo, tmp_path, with_age=False,
                        deviations_cap=cap)
            for cap in (5, 100, 5, 100, 5, 100, 5, 100)
        ]
        results = [f.result() for f in futures]
    for r in results:
        expected = len(r["deviations"]) < 40
        assert r["deviations_truncated"] is expected, (
            f"cap-mismatched truncation flag: {r['summary']}"
        )


def test_audit_is_reachable_as_an_mcp_tool(tmp_path: Path) -> None:
    """The mandate was an audit GATE an agent can invoke, not a script.
    On first ship `audit_detectors` was imported by its own tests only —
    zero production callers, no MCP tool: built-never-wired, the class
    this very module was written to detect."""
    import asyncio
    import json as _json
    from critic_orchestrator import mcp_server

    tools = asyncio.run(mcp_server._list_tools())
    assert "run_repo_audit" in [t.name for t in tools]

    _write(tmp_path, "pkg/feature.py",
           "import os\nX = os.environ.get('MYAPP_FLAG', '0')\n")
    _write(tmp_path, "pkg/mod.py", "# KNOWN LIMIT: declared\n")
    result = asyncio.run(mcp_server._call_tool_impl("run_repo_audit", {
        "project_dir": str(tmp_path), "with_age": False,
    }))
    payload = _json.loads(result[0].text)
    assert payload["kind"] == "repo_audit"
    assert payload["summary"]["dead_flags"] == 1
    assert payload["summary"]["deviations"] == 1


def test_audit_tool_reports_missing_dir_as_error() -> None:
    import asyncio
    import json as _json
    from critic_orchestrator import mcp_server
    result = asyncio.run(mcp_server._call_tool_impl("run_repo_audit", {
        "project_dir": "C:/definitely/not/here/xyzzy",
    }))
    payload = _json.loads(result[0].text)
    assert "error" in payload


def test_env_direct_subscript_access_counts_as_a_reference(tmp_path: Path) -> None:
    """`os.environ['X']` is a read the regex does not model. A flag read
    with .get() elsewhere but SET/read by subscript must not be called
    unwired on the strength of a blind spot."""
    _write(tmp_path, "pkg/a.py",
           "import os\nX = os.environ.get('MYAPP_FLAG', '0')\n")
    _write(tmp_path, "pkg/b.py",
           "import os\nos.environ['MYAPP_FLAG'] = '1'\n")
    assert find_dead_env_flags(tmp_path) == []
