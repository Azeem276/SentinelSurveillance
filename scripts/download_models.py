#!/usr/bin/env python
"""Provision every local AI model Sentinel needs.

Idempotent: already-present files are left alone, so this is safe to run on
every deploy or as a container entrypoint step.

    python scripts/download_models.py            # download what is missing
    python scripts/download_models.py --list     # show inventory only
    python scripts/download_models.py --force    # re-download everything
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.core.exceptions import ModelNotAvailableError  # noqa: E402
from app.intelligence.model_registry import (  # noqa: E402
    MODEL_SPECS, ensure_model, model_path, registry_status,
)


def print_inventory() -> None:
    print(f"\nModel directory: {get_settings().model_path}\n")
    header = f"{'KEY':<18}{'FILE':<42}{'SIZE':>10}  {'STATUS':<14}LICENCE"
    print(header)
    print("-" * len(header))
    for entry in registry_status():
        size = f"{entry['size_bytes'] / 1e6:.1f} MB" if entry["installed"] else f"~{entry['approx_mb']:.1f} MB"
        status = "installed" if entry["installed"] else "MISSING"
        print(f"{entry['key']:<18}{entry['filename']:<42}{size:>10}  {status:<14}{entry['licence']}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Sentinel AI models")
    parser.add_argument("--list", action="store_true", help="show inventory and exit")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    get_settings().ensure_directories()

    if args.list:
        print_inventory()
        return 0

    total = sum(s.approx_mb for k, s in MODEL_SPECS.items()
                if args.force or not model_path(k).exists())
    if total > 0:
        print(f"Provisioning local AI models (~{total:.1f} MB). All inference stays on this machine.")
    else:
        print("All models already installed.")

    failures = []
    for key, spec in MODEL_SPECS.items():
        path = model_path(key)
        if args.force and path.exists():
            path.unlink()
        if path.exists():
            print(f"  [ok]      {key:<16} {path.name}")
            continue
        print(f"  [fetch]   {key:<16} {spec.filename}  (~{spec.approx_mb:.1f} MB, {spec.licence})")
        try:
            ensure_model(key, allow_download=True)
            print(f"  [done]    {key:<16} {model_path(key)}")
        except ModelNotAvailableError as exc:
            failures.append((key, str(exc)))
            print(f"  [FAILED]  {key:<16} {exc}")

    print_inventory()
    if failures:
        print("Some models could not be downloaded. Fix connectivity or install manually.")
        return 1
    print("All models ready. Inference is fully local - nothing is sent to any cloud service.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
