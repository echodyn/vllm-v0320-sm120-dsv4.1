#!/usr/bin/env python3
"""Verify pinned source blobs, apply the bundle in a temporary tree, run CPU tests.

No image, container, model weights or GPU is used. --source-root is the pristine
vllm package directory (not the repository root) for fully offline verification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import patch_sm120 as bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    if args.require_torch and importlib.util.find_spec("torch") is None:
        parser.error("torch is required for CPU tensor regressions")
    assets = REPO / "patches"
    manifest = bundle.read_manifest(assets)
    with tempfile.TemporaryDirectory(prefix="vllm-v0320-verify-") as tmp:
        root = Path(tmp) / "vllm"
        for name, expected in {
            **manifest["files"],
            **manifest.get("test_sources", {}),
        }.items():
            if args.source_root:
                data = (args.source_root / name).read_bytes()
            else:
                base = manifest["baseline"]
                url = f"https://raw.githubusercontent.com/{base['repository']}/{base['commit']}/vllm/{name}"
                with urllib.request.urlopen(url, timeout=45) as response:
                    data = response.read()
            digest = hashlib.sha1(
                b"blob " + str(len(data)).encode() + b"\0" + data
            ).hexdigest()
            if (
                digest != expected["base_git_blob"]
                or hashlib.sha256(data).hexdigest() != expected["base_sha256"]
            ):
                raise bundle.PatchError(f"{name}: pristine baseline blob mismatch")
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        originals = {name: (root / name).read_bytes() for name in manifest["files"]}
        changed = bundle.apply_bundle(root, assets, check=True)
        assert all(
            (root / name).read_bytes() == data for name, data in originals.items()
        )
        assert bundle.apply_bundle(root, assets) == changed
        assert bundle.apply_bundle(root, assets) == []
        bundle.verify_bundle(root, assets)
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=REPO,
            env=dict(os.environ, VLLM_BACKPORT_TEST_ROOT=str(root)),
        )
        if result.returncode:
            raise SystemExit(result.returncode)
        print(
            json.dumps(
                {
                    "modified_files": len(changed),
                    "patches": len(manifest["backports"]),
                    "idempotent": True,
                    "source_blobs_verified": True,
                    "gpu_tested": False,
                }
            )
        )


if __name__ == "__main__":
    main()
