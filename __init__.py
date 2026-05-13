"""Critic-Orchestrator — adversarial review layer.

Layer 3 of Aurelio's anti-confabulation system:
  1. HippoAgent memory rules (passive: rules read by future instances)
  2. Stop/PostToolUse hook (auto-trigger of Layer 3)
  3. This package: spawns N fresh Claude CLI workers in parallel, each
     with adversarial role + JSON-schema-constrained output, aggregates
     their verdicts, and surfaces a consensus.

Pure subprocess against the `claude` binary: uses the user's Claude Code
subscription, no Anthropic API key required.
"""

__version__ = "0.1.0"
