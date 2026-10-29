#!/usr/bin/env python3
"""Install the reviewed FlashInfer overlay inside the base image, without a resolver.

--install is explicit and is used only by Dockerfile. --verify is read-only.
This module does not install anything when imported by CPU tests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import parse_qs, quote, unquote, urljoin, urlsplit, urlunsplit
import urllib.request

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "dependencies" / "lock.json"
REPORT = REPO / "dependency-report.json"
WHEELS = REPO / "dependencies" / "wheels"
PYPI_MIRRORS = {
    "official": None,
    "tuna": "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
    "aliyun": "https://mirrors.aliyun.com/pypi/simple",
}


class DependencyError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_wheel_urls(wheels, directory):
    """Validate every required local wheel before allowing package mutations."""
    directory = Path(directory).resolve()
    urls = []
    for wheel in wheels.values():
        path = directory / wheel["filename"]
        if not path.is_file():
            raise DependencyError(
                f"Missing local wheel: {path}. Re-run the image build to populate "
                "its download cache. No network fallback is allowed in the installer."
            )
        if sha256_file(path) != wheel["sha256"]:
            raise DependencyError(f"Local wheel SHA-256 differs from lock: {path.name}")
        urls.append(path.as_uri() + "#sha256=" + wheel["sha256"])
    return urls


def source_base_url(value, label):
    """Validate a public index or proxy prefix; never disable TLS verification."""
    value = value.strip().rstrip("/")
    try:
        parts = urlsplit(value)
        valid = (
            parts.scheme in ("http", "https")
            and parts.hostname
            and not parts.username
            and not parts.password
            and not parts.query
            and not parts.fragment
        )
        parts.port  # Validate an explicitly provided port as well.
    except ValueError:
        valid = False
    if not valid:
        raise DependencyError(
            f"{label} must be an HTTP(S) base URL without credentials, query or fragment"
        )
    return value


def download_sources(
    pypi_mirror="official",
    pypi_index_url="",
    github_proxy_url="",
    kernel_wheel_base_url="",
):
    if pypi_mirror not in PYPI_MIRRORS:
        raise DependencyError(f"Unknown PyPI mirror: {pypi_mirror}")
    index = pypi_index_url.strip() or PYPI_MIRRORS[pypi_mirror]
    return {
        "pypi_mirror": "custom" if pypi_index_url.strip() else pypi_mirror,
        "pypi_index_url": source_base_url(index, "PyPI index") if index else None,
        "github_proxy_url": source_base_url(github_proxy_url, "GitHub proxy")
        if github_proxy_url.strip()
        else None,
        "kernel_wheel_base_url": source_base_url(
            kernel_wheel_base_url, "Kernel wheel directory"
        )
        if kernel_wheel_base_url.strip()
        else None,
    }


class WheelLinks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def mirror_wheel_url(name, wheel, index_url):
    """Find the same locked filename on a PEP 503 index, without resolution."""
    project_url = f"{index_url}/{canonicalize_name(name)}/"
    request = urllib.request.Request(
        project_url,
        headers={"Accept": "text/html", "User-Agent": "vllm-sm120-backports/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            project_url = response.geturl()
            html = response.read().decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise DependencyError(f"Cannot read mirror index for {name}: {exc}") from exc
    parser = WheelLinks()
    parser.feed(html)
    for href in parser.links:
        candidate = urljoin(project_url, href)
        parts = urlsplit(candidate)
        if unquote(parts.path.rsplit("/", 1)[-1]) != wheel["filename"]:
            continue
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise DependencyError(f"Mirror returned a non-HTTP(S) wheel for {name}")
        hashes = parse_qs(parts.fragment, keep_blank_values=True).get("sha256")
        if hashes is not None and [value.lower() for value in hashes] != [
            wheel["sha256"].lower()
        ]:
            raise DependencyError(
                f"Mirror SHA-256 differs from lock: {wheel['filename']}"
            )
        return candidate
    raise DependencyError(
        f"Mirror does not contain locked wheel {wheel['filename']}; "
        "choose another mirror or official. No fallback was performed."
    )


def wheel_download_url(name, wheel, sources, *, kernel=False):
    url = wheel["url"]
    if kernel and sources.get("kernel_wheel_base_url"):
        url = f"{sources['kernel_wheel_base_url']}/{quote(wheel['filename'], safe='')}"
    elif kernel and sources["github_proxy_url"]:
        if urlsplit(url).hostname != "github.com":
            raise DependencyError(f"Expected a locked GitHub release URL for {name}")
        url = f"{sources['github_proxy_url']}/{url}"
    elif not kernel and sources["pypi_index_url"]:
        url = mirror_wheel_url(name, wheel, sources["pypi_index_url"])
    # Keep the reviewed digest even when a mirror does not publish a fragment.
    return urlunsplit(urlsplit(url)._replace(fragment=f"sha256={wheel['sha256']}"))


def versions():
    return {
        canonicalize_name(d.metadata["Name"]): d.version
        for d in metadata.distributions()
        if d.metadata["Name"]
    }


def active(requirement, extras=()):
    return requirement.marker is None or any(
        requirement.marker.evaluate({"extra": extra}) for extra in ("", *extras)
    )


def plan_installs(lock, installed):
    """Keep satisfying base dependencies; install only exact reviewed replacements."""
    catalog = {**lock["packages"], **lock["kernel_packages"]}
    selected = {
        name: catalog[name] for name in ["flashinfer-python", *lock["kernel_packages"]]
    }
    visited = set()

    def requirements_of(name, extras=()):
        key = (name, tuple(sorted(extras)))
        if key in visited:
            return
        visited.add(key)
        for text in catalog[name].get("requires", ()):
            req = Requirement(text)
            if not active(req, extras):
                continue
            dep = canonicalize_name(req.name)
            actual = selected[dep]["version"] if dep in selected else installed.get(dep)
            if actual is not None and (
                not req.specifier or Version(actual) in req.specifier
            ):
                if dep in selected:
                    requirements_of(dep, req.extras)
                continue
            if dep not in catalog:
                raise DependencyError(
                    f"Base dependency {text!r} is missing/incompatible; refusing an unreviewed upgrade"
                )
            candidate = catalog[dep]
            if req.specifier and Version(candidate["version"]) not in req.specifier:
                raise DependencyError(
                    f"Locked {dep}=={candidate['version']} cannot satisfy {text}"
                )
            selected[dep] = candidate
            requirements_of(dep, req.extras)

    for name in list(selected):
        requirements_of(name)
    return selected


def choose_wheel(package, supported=None):
    supported = list(sys_tags()) if supported is None else supported
    rank = {tag: i for i, tag in enumerate(supported)}
    candidates = []
    for wheel in package["wheels"]:
        _, version, _, tags = parse_wheel_filename(wheel["filename"])
        if str(version) != package["version"]:
            raise DependencyError(f"Unexpected wheel version: {wheel['filename']}")
        matches = [rank[tag] for tag in tags if tag in rank]
        if matches:
            candidates.append((min(matches), wheel["filename"], wheel))
    if not candidates:
        raise DependencyError(
            f"No locked binary wheel for Python/platform: {package['version']}"
        )
    return min(candidates, key=lambda item: item[:2])[2]


def conflicts():
    installed = versions()
    result = set()
    for dist in metadata.distributions():
        owner = canonicalize_name(dist.metadata["Name"] or "")
        for text in dist.requires or ():
            req = Requirement(text)
            if not active(req):
                continue
            name = canonicalize_name(req.name)
            value = installed.get(name)
            if value is None or (req.specifier and Version(value) not in req.specifier):
                result.add((owner, str(req), value))
    return result


def declared_override(conflict):
    owner, text, actual = conflict
    req = Requirement(text)
    return (
        owner == "vllm"
        and canonicalize_name(req.name) == "flashinfer-python"
        and str(req.specifier) == "==0.7.0.post1"
        and actual == "0.7.2"
    )


def package_root():
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise DependencyError("vllm is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def native_fingerprint(root):
    paths = set(root.rglob("*.so"))
    dg = root / "third_party" / "deep_gemm"
    if not dg.is_dir():
        raise DependencyError("The base image's vendored DeepGEMM is missing")
    paths.update(
        p
        for p in dg.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    if not any(p.suffix == ".so" for p in paths):
        raise DependencyError("The base image's native vLLM extensions are missing")
    result = {}
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[path.relative_to(root).as_posix()] = digest.hexdigest()
    return result


def read_lock():
    raw = LOCK.read_bytes()
    manifest = json.loads((REPO / "patches" / "manifest.json").read_text())
    if hashlib.sha256(raw).hexdigest() != manifest["dependency_lock_sha256"]:
        raise DependencyError("Dependency lock hash mismatch")
    return json.loads(raw)


def verify(lock):
    report = json.loads(REPORT.read_text())
    current = versions()
    if current != report["after_versions"]:
        raise DependencyError(
            "Installed distributions changed after overlay installation"
        )
    if native_fingerprint(package_root()) != report["native_sha256"]:
        raise DependencyError("Native vLLM/DeepGEMM files changed")
    if hashlib.sha256(LOCK.read_bytes()).hexdigest() != report["lock_sha256"]:
        raise DependencyError("Dependency report was made for another lock")
    for name, entry in lock["kernel_packages"].items():
        if current.get(name) != entry["version"]:
            raise DependencyError(f"Wrong kernel package: {name}")
    if current.get("flashinfer-python") != lock["flashinfer"]:
        raise DependencyError("Wrong FlashInfer Python package")
    unexpected = conflicts() - {tuple(item) for item in report["before_conflicts"]}
    unexpected = {item for item in unexpected if not declared_override(item)}
    if unexpected:
        raise DependencyError(f"New dependency conflicts: {sorted(unexpected)}")
    import torch

    if (
        str(torch.__version__) != report["torch_version"]
        or torch.version.cuda != report["torch_cuda"]
    ):
        raise DependencyError("PyTorch/CUDA changed")
    print(
        json.dumps(
            {
                "verified": True,
                "flashinfer": current["flashinfer-python"],
                "torch": report["torch_version"],
                "cuda": report["torch_cuda"],
                "native_files": len(report["native_sha256"]),
                "download_sources": report.get("download_sources"),
                "declared_override": "vllm requires flashinfer-python==0.7.0.post1; overlay installs 0.7.2",
            }
        )
    )


def install(lock, wheel_dir=WHEELS):
    wheel_dir = Path(wheel_dir).resolve()
    sources = {"mode": "local-wheelhouse", "wheel_dir": str(wheel_dir)}
    before = versions()
    if Version(before.get("vllm", "0")).base_version != "0.32.0":
        raise DependencyError("Use an exact vLLM v0.32.0 base image")
    import torch

    if torch.version.cuda != lock["cuda"]:
        raise DependencyError(
            "This lock targets the default CUDA 13.0 image, not cu129/Rubin variants"
        )
    native = native_fingerprint(package_root())
    initial_conflicts = conflicts()
    selected = plan_installs(lock, before)
    wheels = {name: choose_wheel(entry) for name, entry in sorted(selected.items())}
    urls = local_wheel_urls(wheels, wheel_dir)
    print("Dependency installation source: " + json.dumps(sources), flush=True)
    # Old providers not required by the new shim must not remain discoverable.
    orphaned = sorted(
        name
        for name in before
        if name.startswith("flashinfer-jit-cache-")
        and name not in lock["kernel_packages"]
    )
    if orphaned:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", *orphaned], check=True
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            *urls,
        ],
        check=True,
    )
    after = versions()
    for name, entry in selected.items():
        if after.get(name) != entry["version"]:
            raise DependencyError(f"Installed version does not match lock: {name}")
    changed = {
        name
        for name in before.keys() | after.keys()
        if before.get(name) != after.get(name)
    }
    if changed - selected.keys() - set(orphaned):
        raise DependencyError(f"Unreviewed distributions changed: {sorted(changed)}")
    if native_fingerprint(package_root()) != native:
        raise DependencyError("Dependency installation changed native vLLM/DeepGEMM")
    unexpected = {
        item for item in conflicts() - initial_conflicts if not declared_override(item)
    }
    if unexpected:
        raise DependencyError(f"New dependency conflicts: {sorted(unexpected)}")
    report = {
        "schema": 1,
        "before_versions": before,
        "after_versions": after,
        "before_conflicts": sorted(initial_conflicts, key=str),
        "changed": sorted(changed),
        "removed_old_providers": orphaned,
        "native_sha256": native,
        "torch_version": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
        "download_sources": sources,
        "declared_overrides": [
            list(item) for item in conflicts() if declared_override(item)
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    verify(lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--install", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        default=WHEELS,
        help="Local directory of locked wheels; missing files fail without downloading",
    )
    args = parser.parse_args()
    try:
        lock = read_lock()
        if args.install:
            install(lock, args.wheel_dir)
        else:
            verify(lock)
    except (DependencyError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Dependency validation failed: {exc}") from exc


if __name__ == "__main__":
    main()
