"""Execution policy: the layer that assumes the CALLER is hostile.

THE MISTAKE THIS MODULE CORRECTS
================================
`pinned_exec` removes the model's ability to compose a command: the argv
is built by code from a caller-supplied selector. The reasoning was "the
command comes from the caller, not the model, so an injection has no
leverage" — and that reasoning is incomplete. It moves trust from the
model to the CALLER without ever establishing that the caller deserves it.

This package is an MCP server. Its tools are reachable by any MCP-capable
agent: another instance, Codex, Cursor, a scheduled job, or an agent that
is itself carrying a prompt injection picked up from a web page. Both
`project_dir` and `test_path` arrive from there. With execution wired up,
a caller-chosen `project_dir` means running pytest on a directory of the
caller's choosing — and pytest executes that directory's own conftest and
plugins. "The caller decides" is not a security property.

So execution is gated by four independent conditions:

  1. OFF BY DEFAULT — mounting the server grants no execution. An
     OPERATOR enables it (``CRITIC_ALLOW_EXEC=1``). Anyone wiring this
     into a third-party agent gets the read-only product unless they
     deliberately choose otherwise.
  2. ROOT ALLOWLIST — even enabled, only paths under roots the operator
     named (``CRITIC_EXEC_ROOTS``) are eligible. Enabling WITHOUT roots
     is refused rather than treated as "anywhere": a fail-open default
     here would be the entire hole.
  3. A GIT REPOSITORY — execution runs in an ephemeral worktree, so a
     non-repo cannot be isolated.
  4. CONTAINMENT BY PATH COMPONENTS after resolution — never a string
     prefix (``/work-evil`` must not pass for ``/work``), and symlinks
     cannot walk out.

The selector itself is validated separately in `pinned_exec`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_ENV_ENABLE = "CRITIC_ALLOW_EXEC"
_ENV_ROOTS = "CRITIC_EXEC_ROOTS"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


class ExecPolicyError(Exception):
    """Execution refused. The message names what an operator must set."""


@dataclass
class ExecPolicy:
    """Where, if anywhere, this server may execute commands."""

    enabled: bool = False
    roots: list[Path] = field(default_factory=list)

    def check(self, project_dir: Path | str) -> Path:
        """Return the resolved directory, or raise with a reason.

        Order matters: the cheapest, most categorical refusals first, so a
        deployment that never enabled execution gets one clear message
        instead of a path-shaped one.
        """
        if not self.enabled:
            raise ExecPolicyError(
                "execution is not enabled for this server. An operator must "
                f"set {_ENV_ENABLE}=1 and list allowed roots in "
                f"{_ENV_ROOTS} (path-separated). Reviewers that need "
                "execution are skipped until then, never answered without "
                "running anything."
            )
        if not self.roots:
            raise ExecPolicyError(
                f"{_ENV_ENABLE} is set but no execution root is configured. "
                f"Set {_ENV_ROOTS} to the repositories where execution is "
                "allowed — an empty list means nowhere, deliberately, "
                "because the alternative is everywhere."
            )
        target = Path(project_dir).resolve()
        inside = False
        for root in self.roots:
            try:
                root_real = Path(root).resolve()
            except OSError:  # pragma: no cover - unreadable root
                continue
            try:
                # relative_to compares path COMPONENTS: /work-evil is not
                # inside /work, which a string prefix test would accept.
                target.relative_to(root_real)
            except ValueError:
                continue
            inside = True
            break
        if not inside:
            raise ExecPolicyError(
                f"{target} is outside every configured execution root "
                f"({', '.join(str(r) for r in self.roots)}). The caller does "
                "not choose where this server executes; the operator does."
            )
        if not (target / ".git").exists():
            raise ExecPolicyError(
                f"{target} is not a git repository. Execution runs in an "
                "ephemeral git worktree so the real tree is never touched, "
                "which cannot be arranged for a plain directory."
            )
        return target


def policy_from_env() -> ExecPolicy:
    """Build the policy from the operator's environment."""
    enabled = (os.environ.get(_ENV_ENABLE) or "").strip().lower() in _TRUTHY
    raw = (os.environ.get(_ENV_ROOTS) or "").strip()
    roots: list[Path] = []
    if raw:
        for part in raw.split(os.pathsep):
            part = part.strip().strip('"').strip("'")
            if not part:
                continue
            p = Path(part)
            # A stale entry must not disable the valid ones — and must not
            # widen the policy either, so it is simply dropped.
            if p.is_dir():
                roots.append(p)
    return ExecPolicy(enabled=enabled, roots=roots)


__all__ = ["ExecPolicy", "ExecPolicyError", "policy_from_env"]
