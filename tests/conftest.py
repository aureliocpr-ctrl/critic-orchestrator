"""Pytest configuration for critic_orchestrator tests."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `import critic_orchestrator` when running tests without
# installing the package (test from a worktree, CI, etc.).
_PKG_PARENT = Path(__file__).resolve().parent.parent.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))
