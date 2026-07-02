"""Tests for the ghost-CLI backend (interactive hidden Claude sisters).

Everything here is mocked at the GhostSession boundary — no real
`claude.exe` is spawned, no `clp ai-eye` call leaves the test process.
The live path is exercised by `smoke_live.py --backend ghost_cli`
(subscription tokens, Windows only).
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import pytest

from critic_orchestrator import ghost_backend
from critic_orchestrator.backends import make_backend_from_env
from critic_orchestrator.ghost_backend import (
    GhostCLIBackend,
    GhostSession,
    close_all_sisters,
)
from critic_orchestrator.orchestrator import WorkerSpec

_SCHEMA = {
    "type": "object",
    "properties": {"claim_holds": {"type": "boolean"}},
    "required": ["claim_holds"],
}


def _spec(requires_execution: bool = True) -> WorkerSpec:
    return WorkerSpec(
        name="falsify", prompt="stash and run pytest", schema=_SCHEMA,
        extra_args=("--allowedTools", "Read Grep Glob Bash"),
        requires_execution=requires_execution,
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    ghost_backend._reset_registry_for_tests()
    yield
    ghost_backend._reset_registry_for_tests()


class FakeSession:
    """Stands in for GhostSession: no spawn, scripted behaviour."""

    def __init__(self, *, response: str | None = None,
                 inject_ok: bool = True,
                 start_error: str | None = None,
                 send_raises: bool = False) -> None:
        self.response = response
        self.inject_ok = inject_ok
        self.start_error = start_error
        self.send_raises = send_raises
        self.started = False
        self.closed = False
        self.sent_prompt_path: Path | None = None

    def start(self) -> None:
        if self.start_error:
            raise RuntimeError(self.start_error)
        self.started = True

    def send_prompt_file(self, prompt_path: Path, marker: str) -> bool:
        self.sent_prompt_path = prompt_path
        if self.send_raises:
            raise OSError("console gone")
        if not self.inject_ok:
            return False
        if self.response is not None:
            # The real sister reads the prompt file and writes the verdict
            # to the response path named inside it. Mimic that.
            text = prompt_path.read_text(encoding="utf-8")
            m = re.search(r"^\s*(\S+_response\.json)\s*$", text, re.M)
            assert m, "prompt file must name the response path"
            Path(m.group(1)).write_text(self.response, encoding="utf-8")
        return True

    def close(self) -> None:
        self.closed = True


def _backend(session: FakeSession, tmp_path: Path) -> GhostCLIBackend:
    return GhostCLIBackend(
        model="opus",
        session_factory=lambda **_kw: session,
        work_dir=tmp_path,
        poll_interval_s=0.01,
    )


# --------------------------------------------------------------------------
# run_worker happy path — execution workers are NOT skipped
# --------------------------------------------------------------------------

def test_ghost_runs_execution_worker(tmp_path: Path) -> None:
    verdict = {"claim_holds": True}
    session = FakeSession(response=json.dumps(verdict))
    be = _backend(session, tmp_path)

    result = be.run_worker(_spec(requires_execution=True), tmp_path, timeout=5)

    assert result.error is None
    assert result.verdict == verdict
    assert session.started and session.closed
    assert be.supports_execution is True


def test_ghost_prompt_file_contains_contract(tmp_path: Path) -> None:
    """The prompt file must carry the worker prompt, the JSON schema and
    the response path — that IS the filesystem-handshake contract."""
    session = FakeSession(response=json.dumps({"claim_holds": False}))
    be = _backend(session, tmp_path)
    be.run_worker(_spec(), tmp_path, timeout=5)

    text = (session.sent_prompt_path or Path()).read_text(encoding="utf-8")
    assert "stash and run pytest" in text          # original prompt verbatim
    assert '"claim_holds"' in text                 # schema inlined
    assert "_response.json" in text                # response path named
    assert "ONLY the JSON object" in text          # file-output override


def test_ghost_fenced_json_tolerated(tmp_path: Path) -> None:
    fenced = "```json\n{\"claim_holds\": true}\n```"
    session = FakeSession(response=fenced)
    be = _backend(session, tmp_path)
    result = be.run_worker(_spec(), tmp_path, timeout=5)
    assert result.error is None
    assert result.verdict == {"claim_holds": True}


# --------------------------------------------------------------------------
# failure paths — honest errors, never fabricated verdicts, always closed
# --------------------------------------------------------------------------

def test_ghost_malformed_response_is_error(tmp_path: Path) -> None:
    session = FakeSession(response="not json {{{")
    be = _backend(session, tmp_path)
    result = be.run_worker(_spec(), tmp_path, timeout=1)
    assert result.verdict is None
    assert "unparseable" in (result.error or "")
    assert session.closed


def test_ghost_non_object_response_is_error(tmp_path: Path) -> None:
    session = FakeSession(response="[1, 2]")
    be = _backend(session, tmp_path)
    result = be.run_worker(_spec(), tmp_path, timeout=1)
    assert result.verdict is None
    assert "JSON object" in (result.error or "")


def test_ghost_timeout_without_response(tmp_path: Path) -> None:
    session = FakeSession(response=None)  # sister never writes the file
    be = _backend(session, tmp_path)
    t0 = time.perf_counter()
    result = be.run_worker(_spec(), tmp_path, timeout=1)
    assert time.perf_counter() - t0 < 5
    assert result.verdict is None
    assert "timeout" in (result.error or "")
    assert session.closed


def test_ghost_inject_failure_is_error(tmp_path: Path) -> None:
    session = FakeSession(inject_ok=False)
    be = _backend(session, tmp_path)
    result = be.run_worker(_spec(), tmp_path, timeout=1)
    assert result.verdict is None
    assert "inject" in (result.error or "")
    assert session.closed


def test_ghost_start_failure_is_error(tmp_path: Path) -> None:
    session = FakeSession(start_error="cap reached (4)")
    be = _backend(session, tmp_path)
    result = be.run_worker(_spec(), tmp_path, timeout=1)
    assert result.verdict is None
    assert "cap reached" in (result.error or "")
    assert session.closed  # close() is idempotent even if start failed


def test_ghost_send_exception_captured_and_closed(tmp_path: Path) -> None:
    session = FakeSession(send_raises=True)
    be = _backend(session, tmp_path)
    result = be.run_worker(_spec(), tmp_path, timeout=1)
    assert result.verdict is None
    assert "console gone" in (result.error or "")
    assert session.closed


# --------------------------------------------------------------------------
# sister registry: cap + atexit sweep
# --------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, pid: int) -> None:
        self.pid = pid


def test_sister_cap_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRITIC_GHOST_MAX_SISTERS", "2")
    ghost_backend._acquire_slot()
    ghost_backend._acquire_slot()
    with pytest.raises(RuntimeError, match="cap"):
        ghost_backend._acquire_slot()


def test_slot_released_frees_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRITIC_GHOST_MAX_SISTERS", "1")
    ghost_backend._acquire_slot()
    ghost_backend._release_slot(None)  # spawn failed, pid never known
    ghost_backend._acquire_slot()      # capacity is back — must not raise


def test_close_all_sisters_kills_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed: list[int] = []
    monkeypatch.setattr(ghost_backend, "kill_process_tree",
                        lambda proc: killed.append(proc.pid))
    ghost_backend._acquire_slot()
    ghost_backend._commit_slot(_FakeProc(111))
    ghost_backend._acquire_slot()
    ghost_backend._commit_slot(_FakeProc(222))

    close_all_sisters()

    assert sorted(killed) == [111, 222]
    assert ghost_backend._live_sister_count() == 0
    close_all_sisters()  # idempotent on empty registry


def test_registry_is_thread_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRITIC_GHOST_MAX_SISTERS", "50")
    errors: list[BaseException] = []

    def _cycle(i: int) -> None:
        try:
            ghost_backend._acquire_slot()
            proc = _FakeProc(1000 + i)
            ghost_backend._commit_slot(proc)
            ghost_backend._release_slot(proc)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_cycle, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert ghost_backend._live_sister_count() == 0


# --------------------------------------------------------------------------
# GhostSession boot: trust dialog handling + honest boot failure
# --------------------------------------------------------------------------

class _ScriptedSession(GhostSession):
    """GhostSession with the console I/O scripted: no spawn, no clp."""

    def __init__(self, tails: list[str], **kw) -> None:
        super().__init__(project_dir=Path("."), model="opus",
                         extra_args=(), **kw)
        self._tails = list(tails)
        self.enters = 0

    def _spawn(self):  # type: ignore[override]
        return _FakeProc(4242)

    def _read_tail(self, n: int) -> str:  # type: ignore[override]
        return self._tails.pop(0) if self._tails else ""

    def _inject_enter(self) -> None:  # type: ignore[override]
        self.enters += 1

    def _sleep(self, _s: float) -> None:  # type: ignore[override]
        pass


def test_session_boot_confirms_trust_dialog() -> None:
    s = _ScriptedSession([
        "Do you trust the files in this folder?",
        "bypass permissions on  ❯",
    ], boot_timeout_s=5)
    s.start()
    assert s.enters == 1
    assert ghost_backend._live_sister_count() == 1
    s.close()
    assert ghost_backend._live_sister_count() == 0


def test_session_boot_timeout_raises_with_tail_preview() -> None:
    s = _ScriptedSession(["still booting..."], boot_timeout_s=0)
    with pytest.raises(RuntimeError, match="not become ready"):
        s.start()
    s.close()
    assert ghost_backend._live_sister_count() == 0


def test_session_close_is_idempotent() -> None:
    s = _ScriptedSession(["❯"], boot_timeout_s=5)
    s.start()
    s.close()
    s.close()
    assert ghost_backend._live_sister_count() == 0


# --------------------------------------------------------------------------
# env-driven selection
# --------------------------------------------------------------------------

def test_make_backend_ghost_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRITIC_BACKEND", "ghost_cli")
    monkeypatch.setenv("CRITIC_WORKER_MODEL", "opus")
    monkeypatch.setattr(ghost_backend, "_IS_WINDOWS", True)
    be = make_backend_from_env()
    assert isinstance(be, GhostCLIBackend)
    assert be.supports_execution is True
    assert be.model == "opus"


def test_make_backend_ghost_cli_rejects_non_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRITIC_BACKEND", "ghost_cli")
    monkeypatch.setattr(ghost_backend, "_IS_WINDOWS", False)
    with pytest.raises(ValueError, match="Windows-only"):
        make_backend_from_env()
