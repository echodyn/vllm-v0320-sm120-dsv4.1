import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location(
    "host_wheel_downloads", REPO / "scripts/download_wheels.py"
)
downloads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloads)


class WheelhouseTests(unittest.TestCase):
    def setUp(self):
        self.lock = json.loads((REPO / "dependencies/lock.json").read_text())

    def test_host_selects_complete_linux_target_without_using_macos_tags(self):
        wheels = downloads.select_wheels(self.lock, "3.12", "x86_64")
        self.assertEqual(
            set(wheels), set(self.lock["packages"]) | set(self.lock["kernel_packages"])
        )
        self.assertEqual(len(wheels), 17)
        self.assertIn("-cp312-cp312-", wheels["cuda-bindings"]["filename"])
        for name, wheel in wheels.items():
            with self.subTest(package=name):
                self.assertNotIn("aarch64", wheel["filename"])
                self.assertNotIn("macosx", wheel["filename"])
                self.assertTrue(
                    "x86_64" in wheel["filename"] or "none-any" in wheel["filename"]
                )

    def test_flashinfer_downloads_use_github_without_mutating_lock(self):
        original = json.loads(json.dumps(self.lock))
        wheels = downloads.select_wheels(self.lock, "3.12", "x86_64")
        for name, wheel in wheels.items():
            url = downloads.download_url(
                name, wheel, self.lock, downloads.deps.download_sources()
            )
            with self.subTest(package=name):
                self.assertEqual(
                    urlsplit(url).hostname == "github.com",
                    name.startswith("flashinfer-"),
                )
        self.assertEqual(self.lock, original)

    def test_download_publishes_only_after_hash_verification_and_reuses_valid_file(
        self,
    ):
        content = b"downloaded wheel fixture"
        wheel = {
            "filename": "example-1.0-py3-none-any.whl",
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)

            def curl(args, **kwargs):
                self.assertIn("--continue-at", args)
                Path(args[args.index("--output") + 1]).write_bytes(content)
                self.assertFalse((directory / wheel["filename"]).exists())
                return subprocess.CompletedProcess(args, 0, "", "")

            with patch.object(downloads.subprocess, "run", side_effect=curl) as run:
                record = downloads.prepare_one(
                    "example", wheel, "https://files.example/wheel", directory
                )
                downloads.prepare_one(
                    "example", wheel, "https://files.example/wheel", directory
                )
            self.assertEqual(run.call_count, 1)
            self.assertEqual((directory / wheel["filename"]).read_bytes(), content)
            self.assertEqual(record["bytes"], len(content))
            self.assertFalse((directory / (wheel["filename"] + ".part")).exists())

    def test_corrupt_download_is_not_published_as_installable_wheel(self):
        wheel = {"filename": "example-1.0-py3-none-any.whl", "sha256": "0" * 64}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)

            def curl(args, **kwargs):
                Path(args[args.index("--output") + 1]).write_bytes(b"wrong bytes")
                return subprocess.CompletedProcess(args, 0, "", "")

            with patch.object(downloads.subprocess, "run", side_effect=curl):
                with self.assertRaisesRegex(
                    downloads.deps.DependencyError, "SHA-256 differs"
                ):
                    downloads.prepare_one(
                        "example", wheel, "https://files.example/wheel", directory
                    )
            self.assertFalse((directory / wheel["filename"]).exists())


if __name__ == "__main__":
    unittest.main()
