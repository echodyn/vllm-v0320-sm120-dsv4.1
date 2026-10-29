#!/usr/bin/env python3
"""Download reviewed Linux wheels during the online Docker build.

FlashInfer wheels come from the official GitHub release. Other reviewed wheels
come from their locked PyPI URLs (or an explicitly selected PyPI mirror).
Downloads resume through .part files and become installable only after SHA-256
verification. This script downloads files without installing any packages.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit

from packaging.tags import compatible_tags, cpython_tags

import install_dependencies as deps


def download_target(architecture=None, python_version=None, *, runtime=False):
    if not runtime:
        return architecture or "x86_64", python_version or "3.12"
    if architecture is not None or python_version is not None:
        raise deps.DependencyError(
            "--runtime-target cannot be combined with --architecture or --python-version"
        )
    if platform.system() != "Linux":
        raise deps.DependencyError("--runtime-target requires a Linux build container")
    machine = platform.machine().lower()
    aliases = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }
    if machine not in aliases:
        raise deps.DependencyError(f"Unsupported build architecture: {machine}")
    return aliases[machine], f"{sys.version_info.major}.{sys.version_info.minor}"


def linux_tags(python_version, architecture):
    try:
        version = tuple(int(part) for part in python_version.split("."))
    except ValueError as exc:
        raise deps.DependencyError("Python version must be MAJOR.MINOR") from exc
    if len(version) != 2 or version[0] != 3 or not 10 <= version[1] <= 14:
        raise deps.DependencyError("This lock supports CPython 3.10 through 3.14")
    interpreter = f"cp{version[0]}{version[1]}"
    platforms = [f"manylinux_2_{minor}_{architecture}" for minor in range(39, 16, -1)]
    platforms += [f"manylinux2014_{architecture}", f"linux_{architecture}"]
    return [
        *cpython_tags(version, abis=[interpreter], platforms=platforms),
        *compatible_tags(version, interpreter=interpreter, platforms=platforms),
    ]


def select_wheels(lock, python_version, architecture):
    tags = linux_tags(python_version, architecture)
    return {
        name: deps.choose_wheel(entry, tags)
        for name, entry in sorted(
            {**lock["packages"], **lock["kernel_packages"]}.items()
        )
    }


def download_url(name, wheel, lock, sources):
    if name.startswith("flashinfer-"):
        wheel = dict(wheel)
        wheel["url"] = (
            "https://github.com/flashinfer-ai/flashinfer/releases/download/"
            f"v{lock['flashinfer']}/{wheel['filename']}"
        )
        url = deps.wheel_download_url(name, wheel, sources, kernel=True)
    else:
        url = deps.wheel_download_url(name, wheel, sources)
    return urlunsplit(urlsplit(url)._replace(fragment=""))


def prepare_one(name, wheel, url, directory):
    target = directory / wheel["filename"]
    reused = target.is_file()
    if reused:
        if deps.sha256_file(target) != wheel["sha256"]:
            raise deps.DependencyError(f"Existing wheel has wrong SHA-256: {target}")
        print(f"Verified existing: {target.name}", flush=True)
    else:
        partial = target.with_name(target.name + ".part")
        print(f"Downloading: {target.name}", flush=True)
        result = subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "4",
                "--retry-delay",
                "3",
                "--connect-timeout",
                "15",
                "--speed-limit",
                "1024",
                "--speed-time",
                "60",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise deps.DependencyError(
                f"Download failed for {target.name}: {result.stderr.strip()} "
                "(partial file retained for resuming)"
            )
        if deps.sha256_file(partial) != wheel["sha256"]:
            raise deps.DependencyError(
                f"Downloaded SHA-256 differs from lock: {target.name}; "
                "partial file retained, no installable wheel published"
            )
        partial.replace(target)
        print(
            f"Downloaded and verified: {target.name} ({target.stat().st_size} bytes)",
            flush=True,
        )
    return {
        "package": name,
        "filename": target.name,
        "bytes": target.stat().st_size,
        "sha256": wheel["sha256"],
        "source_url": url,
        "reused": reused,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["x86_64", "aarch64"], default=None)
    parser.add_argument("--python-version", default=None)
    parser.add_argument(
        "--runtime-target",
        action="store_true",
        help="Use the Linux build container's architecture and Python version",
    )
    parser.add_argument("--output-dir", type=Path, default=deps.WHEELS)
    parser.add_argument("--jobs", type=int, choices=range(1, 9), default=3)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check local files without network access",
    )
    parser.add_argument("--pypi-mirror", choices=deps.PYPI_MIRRORS, default="official")
    parser.add_argument("--pypi-index-url", default="")
    parser.add_argument("--github-proxy-url", default="")
    parser.add_argument("--kernel-wheel-base-url", default="")
    args = parser.parse_args()
    try:
        lock = deps.read_lock()
        architecture, python_version = download_target(
            args.architecture, args.python_version, runtime=args.runtime_target
        )
        wheels = select_wheels(lock, python_version, architecture)
        directory = args.output_dir.resolve()
        if args.verify_only:
            deps.local_wheel_urls(wheels, directory)
            print(
                json.dumps(
                    {
                        "verified": True,
                        "wheels": len(wheels),
                        "directory": str(directory),
                    }
                )
            )
            return
        if not shutil.which("curl"):
            raise deps.DependencyError("curl is required for resumable wheel downloads")
        sources = deps.download_sources(
            args.pypi_mirror,
            args.pypi_index_url,
            args.github_proxy_url,
            args.kernel_wheel_base_url,
        )
        # Resolve all URLs before starting any downloads.
        urls = {
            name: download_url(name, wheel, lock, sources)
            for name, wheel in wheels.items()
        }
        directory.mkdir(parents=True, exist_ok=True)
        print(
            f"Preparing {len(wheels)} wheels for Linux {architecture}, CPython {python_version}",
            flush=True,
        )
        records = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(prepare_one, name, wheel, urls[name], directory)
                for name, wheel in wheels.items()
            ]
            for future in as_completed(futures):
                records.append(future.result())
        records.sort(key=lambda item: item["filename"])
        manifest = {
            "schema": 1,
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "platform": "linux",
            "architecture": architecture,
            "python": python_version,
            "cuda": lock["cuda"],
            "lock_sha256": hashlib.sha256(deps.LOCK.read_bytes()).hexdigest(),
            "total_bytes": sum(item["bytes"] for item in records),
            "wheels": records,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (directory / "SHA256SUMS").write_text(
            "".join(f"{item['sha256']}  {item['filename']}\n" for item in records)
        )
        print(
            json.dumps(
                {
                    "prepared": True,
                    "wheels": len(records),
                    "bytes": manifest["total_bytes"],
                    "directory": str(directory),
                }
            ),
            flush=True,
        )
    except (deps.DependencyError, OSError) as exc:
        raise SystemExit(f"Wheel preparation failed: {exc}") from exc


if __name__ == "__main__":
    main()
