"""Exercise build-time target selection and curl transfers using tiny local fixtures."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location(
    "online_wheel_downloads", REPO / "scripts/download_wheels.py"
)
downloads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloads)


class RuntimeTargetTests(unittest.TestCase):
    def test_linux_runtime_target_uses_container_architecture_and_python(self):
        for machine, expected in (
            ("x86_64", "x86_64"),
            ("AMD64", "x86_64"),
            ("aarch64", "aarch64"),
            ("arm64", "aarch64"),
        ):
            with (
                self.subTest(machine=machine),
                patch.object(downloads.platform, "system", return_value="Linux"),
                patch.object(downloads.platform, "machine", return_value=machine),
                patch.object(
                    downloads.sys, "version_info", SimpleNamespace(major=3, minor=13)
                ),
            ):
                self.assertEqual(
                    downloads.download_target(runtime=True), (expected, "3.13")
                )

    def test_runtime_target_rejects_non_linux_and_unlocked_architectures(self):
        with patch.object(downloads.platform, "system", return_value="Darwin"):
            with self.assertRaisesRegex(downloads.deps.DependencyError, "Linux"):
                downloads.download_target(runtime=True)
        with (
            patch.object(downloads.platform, "system", return_value="Linux"),
            patch.object(downloads.platform, "machine", return_value="ppc64le"),
        ):
            with self.assertRaisesRegex(
                downloads.deps.DependencyError, "Unsupported build architecture"
            ):
                downloads.download_target(runtime=True)

    def test_runtime_target_cannot_be_overridden_by_host_target_flags(self):
        for kwargs in ({"architecture": "aarch64"}, {"python_version": "3.11"}):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(
                    downloads.deps.DependencyError, "cannot be combined"
                ),
            ):
                downloads.download_target(runtime=True, **kwargs)


@unittest.skipUnless(shutil.which("curl"), "curl is required for local transfer tests")
class OnlineTransferTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.body = b"reviewed wheel fixture bytes\n" * 512
        self.digest = hashlib.sha256(self.body).hexdigest()
        self.filename = "fixture-1.0-py3-none-any.whl"
        self.requests = []
        self.assets = {self.filename: self.body}
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                path = urlsplit(self.path).path
                requested_range = self.headers.get("Range")
                test.requests.append((path, requested_range))
                if path == "/simple/fixture/":
                    body = (
                        f'<a href="../../{test.filename}#sha256={test.digest}">wheel</a>'
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = test.assets.get(path.rsplit("/", 1)[-1])
                if body is None:
                    self.send_error(404)
                    return
                if requested_range:
                    match = re.fullmatch(r"bytes=(\d+)-", requested_range)
                    if match is None or int(match.group(1)) >= len(body):
                        self.send_error(416)
                        return
                    start = int(match.group(1))
                    self.send_response(206)
                    self.send_header(
                        "Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}"
                    )
                    transfer = body[start:]
                else:
                    self.send_response(200)
                    transfer = body
                self.send_header("Content-Length", str(len(transfer)))
                self.end_headers()
                self.wfile.write(transfer)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True
        )
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.wheel = {
            "filename": self.filename,
            "sha256": self.digest,
            "url": self.base_url + "/" + self.filename,
        }
        self.lock = {
            "cuda": "13.0",
            "flashinfer": "0.7.2",
            "kernel_packages": {},
            "packages": {"fixture": {"version": "1.0", "wheels": [self.wheel]}},
        }
        self.lock_path = self.root / "lock.json"

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def run_cli(self, directory, *extra):
        self.lock_path.write_text(json.dumps(self.lock))
        with (
            patch.object(downloads.deps, "read_lock", return_value=self.lock),
            patch.object(downloads.deps, "LOCK", self.lock_path),
            patch.object(
                downloads, "download_target", return_value=("x86_64", "3.12")
            ) as target,
            patch.object(
                downloads.sys,
                "argv",
                [
                    "download_wheels.py",
                    "--runtime-target",
                    "--output-dir",
                    str(directory),
                    *extra,
                ],
            ),
        ):
            downloads.main()
        target.assert_called_once_with(None, None, runtime=True)

    def test_empty_build_cache_downloads_and_valid_cache_needs_no_network(self):
        cache = self.root / "fresh-cache"
        self.assertFalse(cache.exists())
        self.run_cli(cache)
        self.assertEqual((cache / self.filename).read_bytes(), self.body)
        first = json.loads((cache / "manifest.json").read_text())
        self.assertFalse(first["wheels"][0]["reused"])
        self.assertEqual(first["architecture"], "x86_64")
        self.assertEqual(first["python"], "3.12")
        self.assertEqual(
            first["lock_sha256"],
            hashlib.sha256(self.lock_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(len(self.requests), 1)
        with patch.object(
            downloads.subprocess, "run", side_effect=AssertionError("network used")
        ):
            self.run_cli(cache)
        self.assertTrue(
            json.loads((cache / "manifest.json").read_text())["wheels"][0]["reused"]
        )

    def test_interrupted_download_resumes_actual_http_range_and_verifies(self):
        cache = self.root / "cache"
        cache.mkdir()
        partial = cache / (self.filename + ".part")
        partial.write_bytes(self.body[:1187])
        downloads.prepare_one("fixture", self.wheel, self.wheel["url"], cache)
        self.assertEqual(self.requests, [("/" + self.filename, "bytes=1187-")])
        self.assertFalse(partial.exists())
        self.assertEqual((cache / self.filename).read_bytes(), self.body)

    def test_failed_download_retains_partial_without_publishing_wheel(self):
        cache = self.root / "cache"
        cache.mkdir()
        partial = cache / (self.filename + ".part")
        partial.write_bytes(self.body[:17])
        with self.assertRaisesRegex(downloads.deps.DependencyError, "Download failed"):
            downloads.prepare_one(
                "fixture", self.wheel, self.base_url + "/missing.whl", cache
            )
        self.assertEqual(partial.read_bytes(), self.body[:17])
        self.assertFalse((cache / self.filename).exists())

    def test_modified_cached_wheel_fails_without_downloading_replacement(self):
        cache = self.root / "cache"
        cache.mkdir()
        (cache / self.filename).write_bytes(b"modified cached wheel")
        with self.assertRaisesRegex(downloads.deps.DependencyError, "wrong SHA-256"):
            downloads.prepare_one("fixture", self.wheel, self.wheel["url"], cache)
        self.assertEqual(self.requests, [])

    def test_build_mirror_arguments_select_python_index_and_kernel_directory(self):
        filename = "flashinfer_python-0.7.2-py3-none-any.whl"
        self.assets[filename] = self.body
        self.lock["packages"]["flashinfer-python"] = {
            "version": "0.7.2",
            "wheels": [
                {
                    "filename": filename,
                    "sha256": self.digest,
                    "url": "https://unreachable.invalid/" + filename,
                }
            ],
        }
        self.lock["packages"]["fixture"]["wheels"][0]["url"] = (
            "https://unreachable.invalid/ordinary.whl"
        )
        cache = self.root / "cache"
        self.run_cli(
            cache,
            "--pypi-mirror",
            "aliyun",
            "--pypi-index-url",
            self.base_url + "/simple",
            "--github-proxy-url",
            self.base_url + "/unused-proxy",
            "--kernel-wheel-base-url",
            self.base_url + "/flashinfer",
        )
        paths = [path for path, _ in self.requests]
        self.assertIn("/simple/fixture/", paths)
        self.assertIn("/" + self.filename, paths)
        self.assertIn("/flashinfer/" + filename, paths)
        self.assertFalse(any("unused-proxy" in path for path in paths))
        self.assertEqual(
            len(json.loads((cache / "manifest.json").read_text())["wheels"]), 2
        )


if __name__ == "__main__":
    unittest.main()
