"""Ghost-CLI backend: each worker runs in a fresh, invisible INTERACTIVE
Claude session — no `claude --print` anywhere.

Why this exists: headless `claude --print` calls are moving to metered
billing, while interactive Claude CLI sessions stay on the flat
subscription. The API backends in `backends.py` are reasoning-only (they
honestly skip `requires_execution` workers). This backend is the only
alternative that supports EVERY worker, including `falsification`
(git stash + pytest) and `caller_verification` (grep call sites), because
the sister is a real Claude Code session with tool access.

Technique (validated live in Engram's interactive judge 2026-07-02,
decision agreement 10/10 vs the `claude -p` judge — see HippoAgent
`engram/interactive_judge.py`):

  * GHOST spawn — `claude.exe` is started directly with
    CREATE_NEW_CONSOLE + STARTUPINFO(SW_HIDE): the console EXISTS, so
    `clp ai-eye` can AttachConsole(pid) to read the buffer and inject
    keystrokes, but the window is never shown. No spawn-then-hide race.
  * Filesystem handshake — long prompts never travel through the console.
    The worker prompt (plus output contract) is written to a file; ONE
    short "read that file" line is injected (`ai-eye --inject --verify
    <marker>`); the sister writes its verdict JSON to a response file
    which we poll.
  * FRESH sister per worker — adversarial workers must be independent
    sessions (a counterexample worker that can see the falsification
    transcript is no longer an independent vote). The sister is killed
    (whole process tree) in a `finally`.

Lifecycle hardening:
  * module-level registry of live sisters + `atexit` sweep — an MCP
    server dying mid-review does not leak invisible `claude.exe`
    processes;
  * hard cap on concurrent sisters (`CRITIC_GHOST_MAX_SISTERS`,
    default 4) — a runaway caller fails fast with an honest error
    instead of silently filling the machine with hidden consoles.

Limits (honest): Windows-only (Win32 console API via the `clp` arsenal
CLI, which must be on PATH). Job cancellation cannot reach a sister
mid-flight — the worker ends at its timeout, and `close()`/atexit kill
the tree. Cost is reported as 0.0 because the whole point is flat
subscription usage; wall time is the real spend.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import threading
import time
from itertools import count
from pathlib import Path
from typing import Any, Callable

from .backends import BackendResult
from .orchestrator import WorkerSpec, kill_process_tree

_IS_WINDOWS = os.name == "nt"

# --------------------------------------------------------------------------
# Live-sister registry (cap + atexit sweep)
# --------------------------------------------------------------------------

_REG_LOCK = threading.Lock()
_LIVE: dict[int, Any] = {}   # pid -> Popen (or any object with .pid)
_RESERVED = 0                # slots acquired but pid not yet known
_STAMP = count(1)


def _max_sisters() -> int:
    raw = (os.environ.get("CRITIC_GHOST_MAX_SISTERS") or "4").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 4


def _acquire_slot() -> None:
    """Reserve capacity for one sister BEFORE spawning. Raises RuntimeError
    when the cap is hit — failing fast beats deadlocking or piling hidden
    consoles onto the machine."""
    global _RESERVED
    cap = _max_sisters()
    with _REG_LOCK:
        if len(_LIVE) + _RESERVED >= cap:
            raise RuntimeError(
                f"ghost sister cap reached ({cap} live); raise "
                "CRITIC_GHOST_MAX_SISTERS or investigate leaked claude.exe "
                "processes")
        _RESERVED += 1


def _commit_slot(proc: Any) -> None:
    """Bind a reserved slot to a spawned process."""
    global _RESERVED
    with _REG_LOCK:
        _RESERVED = max(0, _RESERVED - 1)
        _LIVE[proc.pid] = proc


def _release_slot(proc: Any | None) -> None:
    """Free a slot: pass the proc after kill, or None if spawn failed
    before a pid existed."""
    global _RESERVED
    with _REG_LOCK:
        if proc is None:
            _RESERVED = max(0, _RESERVED - 1)
        else:
            _LIVE.pop(proc.pid, None)


def _live_sister_count() -> int:
    with _REG_LOCK:
        return len(_LIVE)


def close_all_sisters() -> None:
    """Kill every registered sister's process tree. Registered atexit so a
    dying MCP server sweeps its hidden consoles; also callable directly."""
    with _REG_LOCK:
        procs = list(_LIVE.values())
        _LIVE.clear()
    for proc in procs:
        try:
            kill_process_tree(proc)
        except Exception:  # noqa: BLE001 — sweep must reach every sister
            pass


atexit.register(close_all_sisters)


def _reset_registry_for_tests() -> None:
    """Test hook: forget all registrations without killing anything."""
    global _RESERVED
    with _REG_LOCK:
        _LIVE.clear()
        _RESERVED = 0


# --------------------------------------------------------------------------
# One ghost sister
# --------------------------------------------------------------------------

class GhostSession:
    """One hidden interactive Claude CLI session, driven via `clp ai-eye`.

    The subprocess-facing methods (`_spawn`, `_read_tail`, `_inject`,
    `_inject_enter`, `_sleep`) are small and override-friendly so tests
    can script the console without spawning anything.
    """

    CREATE_NEW_CONSOLE = 0x00000010
    STARTF_USESHOWWINDOW = 0x00000001
    SW_HIDE = 0

    def __init__(self, *, project_dir: Path, model: str,
                 extra_args: tuple[str, ...] = (),
                 boot_timeout_s: float = 60.0) -> None:
        self.project_dir = Path(project_dir)
        self.model = model
        self.extra_args = tuple(extra_args)
        self.boot_timeout_s = boot_timeout_s
        self._proc: Any | None = None
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Acquire a cap slot, spawn the ghost, wait until the REPL is
        ready (confirming the folder-trust dialog if it appears)."""
        _acquire_slot()
        try:
            self._proc = self._spawn()
        except BaseException:
            _release_slot(None)
            raise
        _commit_slot(self._proc)
        self._wait_ready()

    def close(self) -> None:
        """Kill the sister's whole process tree. Idempotent; safe to call
        when start() never ran or failed halfway."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                kill_process_tree(proc)
            finally:
                _release_slot(proc)

    @staticmethod
    def _claude_exe() -> str:
        p = Path.home() / ".local" / "bin" / "claude.exe"
        return str(p) if p.exists() else "claude"

    def _spawn(self) -> Any:
        """Spawn claude.exe hidden FROM BIRTH (console exists, window never
        shown). Same flags as the validated Engram ghost sister, plus the
        worker model pin and the per-worker tool allowlist."""
        if not _IS_WINDOWS:  # pragma: no cover - guarded upstream too
            raise RuntimeError("ghost sessions are Windows-only")
        si = subprocess.STARTUPINFO()
        si.dwFlags |= self.STARTF_USESHOWWINDOW
        si.wShowWindow = self.SW_HIDE
        # --dangerously-skip-permissions: same rationale as the --print
        # path — nobody can answer a permission prompt inside a hidden
        # console; the tool set is still gated by --allowedTools in
        # extra_args.
        cmd = [self._claude_exe(), "--dangerously-skip-permissions",
               "--model", self.model, *self.extra_args]
        return subprocess.Popen(
            cmd, creationflags=self.CREATE_NEW_CONSOLE, startupinfo=si,
            cwd=str(self.project_dir))

    def _wait_ready(self) -> None:
        """Poll the console buffer until the REPL prompt shows up. The
        folder-trust dialog (first visit to a project_dir) is confirmed
        with one Enter — its default answer is Yes."""
        deadline = time.time() + self.boot_timeout_s
        trusted = False
        tail = ""
        while time.time() < deadline:
            tail = self._read_tail(12).lower()
            if not trusted and ("trust this folder" in tail
                                or "trust the files" in tail):
                self._inject_enter()
                trusted = True
                self._sleep(3)
                continue
            if "bypass permissions" in tail or "❯" in tail:
                return
            self._sleep(3)
        final = self._read_tail(20)
        if "claude" in final.lower():
            return
        raise RuntimeError(
            "ghost claude did not become ready within "
            f"{self.boot_timeout_s:.0f}s; last console tail: "
            f"{(final or tail)[-200:]!r}")

    # -- console I/O (subprocess boundary, overridden in tests) -------------

    def _read_tail(self, n: int) -> str:
        pid = self._proc.pid if self._proc else 0
        r = subprocess.run(["clp", "ai-eye", "--pid", str(pid), "--read",
                            "--tail", str(n)],
                           capture_output=True, text=True, timeout=30,
                           shell=True)
        return r.stdout or ""

    def _inject(self, text: str, verify: str) -> bool:
        pid = self._proc.pid if self._proc else 0
        r = subprocess.run(["clp", "ai-eye", "--pid", str(pid),
                            "--inject", text, "--verify", verify,
                            "--newline"],
                           capture_output=True, text=True, timeout=60,
                           shell=True)
        return '"ok": true' in (r.stdout or "").lower()

    def _inject_enter(self) -> None:
        pid = self._proc.pid if self._proc else 0
        subprocess.run(["clp", "ai-eye", "--pid", str(pid), "--inject", "",
                        "--newline"],
                       capture_output=True, text=True, timeout=30,
                       shell=True)

    def _sleep(self, s: float) -> None:
        time.sleep(s)

    # -- work ---------------------------------------------------------------

    def send_prompt_file(self, prompt_path: Path, marker: str) -> bool:
        """Inject the ONE short line that points the sister at the prompt
        file. Returns True iff ai-eye verified the marker in the buffer."""
        posix = str(prompt_path).replace("\\", "/")
        return self._inject(
            f"Read the file {posix} and follow its instructions exactly. "
            f"[{marker}]",
            marker)


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$")


def _work_dir() -> Path:
    return Path(os.environ.get("TEMP", "/tmp")) / "critic_orchestrator" / "ghost"


class GhostCLIBackend:
    """Runs each WorkerSpec inside a fresh ghost sister (see module doc).

    `session_factory` is the test seam: it receives project_dir / model /
    extra_args / boot_timeout_s keywords and must return a GhostSession-
    shaped object (start / send_prompt_file / close).
    """

    supports_execution = True

    def __init__(self, *, model: str = "opus",
                 boot_timeout_s: float = 60.0,
                 session_factory: Callable[..., Any] | None = None,
                 work_dir: Path | None = None,
                 poll_interval_s: float = 2.0) -> None:
        self.model = model
        self.boot_timeout_s = boot_timeout_s
        self._session_factory = session_factory or (
            lambda **kw: GhostSession(**kw))
        self._work_dir = Path(work_dir) if work_dir else _work_dir()
        self.poll_interval_s = poll_interval_s
        self.name = f"ghost_cli:{model}"

    # -- prompt-file rendering ----------------------------------------------

    def _render_prompt(self, spec: WorkerSpec, response_path: Path) -> str:
        posix = str(response_path).replace("\\", "/")
        return (
            f"# Adversarial worker: {spec.name}\n\n"
            f"{spec.prompt}\n\n"
            "---\n"
            "# OUTPUT CONTRACT — overrides any earlier instruction about\n"
            "# where the final JSON goes (including 'last message must be\n"
            "# the JSON object')\n\n"
            "Do the work described above. When you are done, WRITE the\n"
            "final JSON object to this file (create it; overwrite if it\n"
            "exists):\n\n"
            f"    {posix}\n\n"
            "The file must contain ONLY the JSON object - no markdown\n"
            "fences, no prose, no explanation. It must conform to this\n"
            "JSON schema:\n\n"
            f"{json.dumps(spec.schema, indent=2)}\n\n"
            "After writing the file, reply in chat with exactly: DONE\n")

    # -- response polling -----------------------------------------------------

    def _await_response(self, path: Path, timeout: int) -> tuple[dict | None, str | None, str]:
        """Poll for the response file. Returns (verdict, error, raw_preview)."""
        deadline = time.time() + timeout
        raw = ""
        while time.time() < deadline:
            if path.exists():
                try:
                    raw = path.read_text(encoding="utf-8").strip()
                except OSError:
                    time.sleep(self.poll_interval_s)
                    continue
                if raw:
                    parsed = self._parse(raw)
                    if parsed is not None:
                        return parsed, None, raw[:500]
                    # partially written or fenced garbage: give the write
                    # one settle window, then judge it on the next loop
                    time.sleep(self.poll_interval_s)
                    settled = path.read_text(encoding="utf-8").strip()
                    parsed = self._parse(settled)
                    if parsed is not None:
                        return parsed, None, settled[:500]
                    if not isinstance(self._loads(settled), dict) \
                            and self._loads(settled) is not None:
                        return None, "response was not a JSON object", settled[:500]
                    return None, f"unparseable response file: {settled[:120]!r}", settled[:500]
            time.sleep(self.poll_interval_s)
        return None, f"ghost worker timeout after {timeout}s (no response file)", raw[:500]

    @staticmethod
    def _loads(text: str) -> Any | None:
        try:
            return json.loads(text)
        except ValueError:
            return None

    def _parse(self, text: str) -> dict | None:
        """JSON object from the response text; tolerates markdown fences
        (a sister occasionally wraps output despite the contract)."""
        for candidate in (text, _FENCE_RE.sub("", text).strip()):
            obj = self._loads(candidate)
            if isinstance(obj, dict):
                return obj
        return None

    # -- main entry -----------------------------------------------------------

    def run_worker(self, spec: WorkerSpec, project_dir: Path,
                   timeout: int) -> BackendResult:
        self._work_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{int(time.time() * 1000):x}_{next(_STAMP)}"
        prompt_path = self._work_dir / f"{spec.name}_{stamp}_prompt.md"
        response_path = self._work_dir / f"{spec.name}_{stamp}_response.json"
        prompt_path.write_text(self._render_prompt(spec, response_path),
                               encoding="utf-8")
        marker = f"CRIT-{stamp}"
        session = self._session_factory(
            project_dir=project_dir, model=self.model,
            extra_args=spec.extra_args, boot_timeout_s=self.boot_timeout_s)
        try:
            try:
                session.start()
            except Exception as exc:
                return BackendResult(None, f"ghost session failed: {exc}")
            try:
                sent = session.send_prompt_file(prompt_path, marker)
            except Exception as exc:
                return BackendResult(None, f"ghost inject error: {exc}")
            if not sent:
                return BackendResult(
                    None, "ghost inject failed (ai-eye did not verify the "
                          "marker in the console buffer)")
            verdict, error, preview = self._await_response(response_path,
                                                           timeout)
            return BackendResult(verdict, error, 0.0, preview)
        finally:
            session.close()


def make_ghost_backend_from_env() -> GhostCLIBackend:
    """Build a GhostCLIBackend from the environment. Raises ValueError on
    non-Windows hosts so the MCP server surfaces a clear error."""
    if not _IS_WINDOWS:
        raise ValueError(
            "CRITIC_BACKEND=ghost_cli is Windows-only (Win32 console "
            "transport via the clp ai-eye arsenal)")
    model = (os.environ.get("CRITIC_WORKER_MODEL") or "opus").strip() or "opus"
    raw_boot = (os.environ.get("CRITIC_GHOST_BOOT_TIMEOUT") or "").strip()
    try:
        boot = float(raw_boot) if raw_boot else 60.0
    except ValueError:
        boot = 60.0
    return GhostCLIBackend(model=model, boot_timeout_s=boot)


__all__ = [
    "GhostCLIBackend",
    "GhostSession",
    "close_all_sisters",
    "make_ghost_backend_from_env",
]
