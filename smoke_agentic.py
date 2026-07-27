"""Live smoke test for the agentic API backend.

Proves, against a REAL endpoint, the one thing the mocked tests cannot:
that a third-party reasoning model actually drives the fs_* tools and
returns a schema-shaped verdict — i.e. that a design lens really runs on
Kimi / GLM / DeepSeek.

Reads keys from ~/.clp/keys.env (never printed). Reviews a small file so
the run stays cheap and fast.

  python -m critic_orchestrator.smoke_agentic            # all configured
  python -m critic_orchestrator.smoke_agentic deepseek   # one provider
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from .agentic_api import AgenticApiBackend
from .design_workers import build_design_workers

#: Optional key file, one `NAME=value` per line. Override with
#: CRITIC_KEYS_ENV. Plain environment variables work too and take
#: precedence in spirit — the file is a convenience, not a requirement.
_KEYS_ENV = Path(
    os.environ.get("CRITIC_KEYS_ENV")
    or (Path.home() / ".clp" / "keys.env")
)

#: (label, key-env-name, base_url-env-name-or-literal, model)
#: Model ids are NOT guessed: they were read from each provider's
#: /v1/models endpoint (Moonshot listed kimi-k3, kimi-k2.7-code, … while
#: a plausible-looking `kimi-k2-0905-preview` 404'd).
PROVIDERS: list[tuple[str, str, str, str]] = [
    ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1",
     "deepseek-chat"),
    ("kimi", "MOONSHOT_API_KEY", "MOONSHOT_BASE_URL", "kimi-k3"),
    ("glm", "ZAI_API_KEY", "https://api.z.ai/api/paas/v4", "glm-4.6"),
]


def _load_keys() -> dict[str, str]:
    """Parse KEY=value lines. Values are used, never echoed."""
    out: dict[str, str] = {}
    if not _KEYS_ENV.is_file():
        return out
    for line in _KEYS_ENV.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _repo_root() -> Path:
    """The directory that CONTAINS the `critic_orchestrator` package.

    Derived from this file, never hardcoded: the smoke has to run for
    anyone who clones the repo, not just on the machine it was written on.
    """
    return Path(__file__).resolve().parent.parent


def _target_file(root: Path) -> str:
    """A small, self-contained module: enough to review, cheap to read."""
    return "critic_orchestrator/job_registry.py"


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    only = {a.lower() for a in argv} or None
    keys = _load_keys()
    root = _repo_root()
    target = _target_file(root)
    assert (root / target).is_file(), f"{target} not found under {root}"

    # One lens (premortem) is enough for a smoke: it exercises tool use,
    # the schema, and the sandbox in one pass.
    lens = build_design_workers(module_paths=[target])[0]

    any_run = False
    rc = 0
    for label, key_env, base, model in PROVIDERS:
        if only and label not in only:
            continue
        key = keys.get(key_env) or os.environ.get(key_env) or ""
        if not key:
            print(f"[{label}] SKIP — no {key_env} configured", flush=True)
            continue
        # `base` may name an env var holding the URL (providers move hosts).
        if not base.startswith("http"):
            base = (keys.get(base) or os.environ.get(base) or "").strip()
            if not base:
                print(f"[{label}] SKIP — no base URL configured", flush=True)
                continue
        any_run = True
        backend = AgenticApiBackend(
            base_url=base, api_key=key, model=model,
            max_steps=14, read_budget_bytes=200_000,
        )
        print(f"[{label}] model={model} — running the premortem lens on "
              f"{target} …", flush=True)
        t0 = time.time()
        res = backend.run_worker(lens, root, timeout=300)
        dt = time.time() - t0
        if res.error:
            print(f"[{label}] ERROR after {dt:.1f}s: {res.error}", flush=True)
            if res.raw_preview:
                print(f"[{label}] raw: {res.raw_preview[:300]}", flush=True)
            rc = 1
            continue
        v = res.verdict or {}
        findings = v.get("findings") or []
        print(f"[{label}] OK in {dt:.1f}s — purpose: "
              f"{str(v.get('inferred_purpose'))[:110]}…", flush=True)
        print(f"[{label}] findings: {len(findings)}, "
              f"confidence: {v.get('confidence')}", flush=True)
        for f in findings[:4]:
            print(f"[{label}]   - [{f.get('severity')}] {f.get('title')}",
                  flush=True)
        out = Path(os.environ.get("TEMP", "/tmp")) / f"smoke_agentic_{label}.json"
        out.write_text(json.dumps(v, indent=2), encoding="utf-8")
        print(f"[{label}] verdict written to {out}", flush=True)

    if not any_run:
        print("no provider configured — nothing ran", flush=True)
        return 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
