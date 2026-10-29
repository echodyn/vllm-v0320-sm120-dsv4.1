import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch
from urllib.parse import quote, urlsplit

from packaging.tags import Tag

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "overlay_dependencies", REPO / "scripts/install_dependencies.py"
)
deps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deps)


class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.lock = json.loads((REPO / "dependencies/lock.json").read_text())
        self.base = {
            "apache-tvm-ffi": "0.1.11",
            "click": "8.2.1",
            "cuda-python": "13.0.3",
            "cuda-bindings": "13.0.3",
            "cuda-tile": "1.4.0",
            "einops": "0.8.1",
            "nccl-extensions": "0.1.0",
            "nccl4py": "0.4.1",
            "ninja": "1.13.0",
            "numpy": "2.2.6",
            "nvidia-cudnn-frontend": "1.31.0",
            "nvidia-cutlass-dsl": "4.8.0",
            "nvidia-ml-py": "13.580.82",
            "packaging": "25.0",
            "requests": "2.32.5",
            "tabulate": "0.9.0",
            "torch": "2.13.0",
            "tqdm": "4.67.1",
            "typing-extensions": "4.15.0",
        }

    def test_satisfying_base_dependencies_are_retained(self):
        plan = deps.plan_installs(self.lock, self.base)
        self.assertEqual(
            set(plan), {"flashinfer-python", *self.lock["kernel_packages"]}
        )
        self.assertNotIn("torch", plan)
        self.assertNotIn("nvidia-cutlass-dsl", plan)
        self.assertEqual(plan["flashinfer-jit-cache-sm120f"]["version"], "0.7.2+cu130")

    def test_missing_new_dependencies_have_fixed_transitive_versions(self):
        base = {
            name: value
            for name, value in self.base.items()
            if name
            not in ["nccl4py", "nccl-extensions", "cuda-tile", "nvidia-cudnn-frontend"]
        }
        plan = deps.plan_installs(self.lock, base)
        self.assertEqual(plan["cuda-core"]["version"], "1.0.0")
        self.assertEqual(plan["cuda-pathfinder"]["version"], "1.5.4")
        self.assertEqual(plan["nccl4py"]["version"], "0.4.1")
        self.assertEqual(plan["nvidia-cudnn-frontend"]["version"], "1.31.0")
        self.assertNotIn("cuda-python", plan)
        self.assertNotIn("torch", plan)

    def test_missing_protected_base_dependency_fails_without_resolver(self):
        del self.base["torch"]
        with self.assertRaisesRegex(deps.DependencyError, "unreviewed upgrade"):
            deps.plan_installs(self.lock, self.base)

    def test_only_original_vllm_flashinfer_declaration_is_exempted(self):
        self.assertTrue(
            deps.declared_override(("vllm", "flashinfer-python==0.7.0.post1", "0.7.2"))
        )
        for item in [
            ("another", "flashinfer-python==0.7.0.post1", "0.7.2"),
            ("vllm", "torch==2.13.0", "2.14.0"),
            ("vllm", "flashinfer-python==0.7.0.post1", "0.7.3"),
        ]:
            self.assertFalse(deps.declared_override(item))

    def test_matching_kernel_wheels_and_lock_integrity(self):
        for architecture in ["x86_64", "aarch64"]:
            tags = [
                Tag("cp39", "abi3", f"manylinux_2_28_{architecture}"),
                Tag("py3", "none", "any"),
            ]
            for entry in self.lock["kernel_packages"].values():
                wheel = deps.choose_wheel(entry, tags)
                self.assertEqual(len(wheel["sha256"]), 64)
                self.assertIn("/v0.7.2/", wheel["url"])
        manifest = json.loads((REPO / "patches/manifest.json").read_text())
        self.assertEqual(
            hashlib.sha256((REPO / "dependencies/lock.json").read_bytes()).hexdigest(),
            manifest["dependency_lock_sha256"],
        )

    def test_unsupported_wheel_target_fails(self):
        entry = self.lock["kernel_packages"]["flashinfer-jit-cache-sm120f"]
        with self.assertRaises(deps.DependencyError):
            deps.choose_wheel(entry, [Tag("cp312", "cp312", "macosx_14_0_arm64")])

    def mirror_response(
        self, html, final_url="https://mirror.example/simple/flashinfer-python/"
    ):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = html.encode()
        response.geturl.return_value = final_url
        return response

    def test_default_downloads_remain_locked_official_urls(self):
        sources = deps.download_sources()
        with patch.object(deps.urllib.request, "urlopen") as opener:
            for category in ("packages", "kernel_packages"):
                name = next(iter(self.lock[category]))
                wheel = self.lock[category][name]["wheels"][0]
                self.assertEqual(
                    deps.wheel_download_url(
                        name, wheel, sources, kernel=category == "kernel_packages"
                    ),
                    wheel["url"] + "#sha256=" + wheel["sha256"],
                )
        opener.assert_not_called()

    def test_mirror_presets_and_explicit_custom_index(self):
        self.assertEqual(
            deps.download_sources("tuna")["pypi_index_url"],
            "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
        )
        self.assertEqual(
            deps.download_sources("aliyun")["pypi_index_url"],
            "https://mirrors.aliyun.com/pypi/simple",
        )
        sources = deps.download_sources(
            "tuna", "https://internal.example/simple/", "https://proxy.example/"
        )
        self.assertEqual(sources["pypi_mirror"], "custom")
        self.assertEqual(sources["pypi_index_url"], "https://internal.example/simple")
        self.assertEqual(sources["github_proxy_url"], "https://proxy.example")

    def test_simple_index_uses_exact_filename_redirect_and_locked_hash(self):
        wheel = self.lock["packages"]["flashinfer-python"]["wheels"][0]
        wrong = wheel["filename"].replace("0.7.2", "0.7.3")
        html = (
            f'<a href="../../packages/{wrong}">newer version</a>'
            f'<a href="../../packages/{quote(wheel["filename"], safe="")}#sha256={wheel["sha256"]}">locked</a>'
        )
        response = self.mirror_response(
            html, "https://cdn.example/pypi/simple/flashinfer-python/"
        )
        with patch.object(
            deps.urllib.request, "urlopen", return_value=response
        ) as opener:
            url = deps.wheel_download_url(
                "flashinfer-python", wheel, deps.download_sources("tuna")
            )
        self.assertEqual(urlsplit(url).hostname, "cdn.example")
        self.assertEqual(urlsplit(url).path, "/pypi/packages/" + wheel["filename"])
        self.assertEqual(urlsplit(url).fragment, "sha256=" + wheel["sha256"])
        request = opener.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/flashinfer-python/",
        )

    def test_mirror_without_hash_still_passes_locked_digest_to_pip(self):
        wheel = self.lock["packages"]["flashinfer-python"]["wheels"][0]
        response = self.mirror_response(
            f'<a href="https://files.example/{wheel["filename"]}">wheel</a>'
        )
        with patch.object(deps.urllib.request, "urlopen", return_value=response):
            url = deps.wheel_download_url(
                "flashinfer-python", wheel, deps.download_sources("aliyun")
            )
        self.assertEqual(urlsplit(url).fragment, "sha256=" + wheel["sha256"])

    def test_mirror_hash_mismatch_fails_without_official_fallback(self):
        wheel = self.lock["packages"]["flashinfer-python"]["wheels"][0]
        response = self.mirror_response(
            f'<a href="{wheel["filename"]}#sha256={"0" * 64}">bad</a>'
        )
        with patch.object(
            deps.urllib.request, "urlopen", return_value=response
        ) as opener:
            with self.assertRaisesRegex(deps.DependencyError, "SHA-256 differs"):
                deps.wheel_download_url(
                    "flashinfer-python", wheel, deps.download_sources("tuna")
                )
        self.assertEqual(opener.call_count, 1)

    def test_missing_locked_wheel_fails_instead_of_selecting_another_version(self):
        wheel = self.lock["packages"]["flashinfer-python"]["wheels"][0]
        response = self.mirror_response(
            '<a href="flashinfer_python-0.7.3-py3-none-any.whl">new</a>'
        )
        with patch.object(deps.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(deps.DependencyError, "No fallback"):
                deps.wheel_download_url(
                    "flashinfer-python", wheel, deps.download_sources("tuna")
                )

    def test_github_proxy_only_rewrites_kernel_downloads(self):
        sources = deps.download_sources(github_proxy_url="https://proxy.example/")
        wheel = self.lock["kernel_packages"]["flashinfer-cubin"]["wheels"][0]
        url = deps.wheel_download_url("flashinfer-cubin", wheel, sources, kernel=True)
        self.assertEqual(
            url, "https://proxy.example/" + wheel["url"] + "#sha256=" + wheel["sha256"]
        )
        python_wheel = self.lock["packages"]["flashinfer-python"]["wheels"][0]
        with patch.object(deps.urllib.request, "urlopen") as opener:
            self.assertEqual(
                deps.wheel_download_url("flashinfer-python", python_wheel, sources),
                python_wheel["url"] + "#sha256=" + python_wheel["sha256"],
            )
        opener.assert_not_called()

    def test_kernel_directory_overrides_proxy_and_preserves_all_locked_wheels(self):
        base = "https://packages.internal.example/flashinfer/v0.7.2"
        sources = deps.download_sources(
            github_proxy_url="https://unreachable-proxy.example",
            kernel_wheel_base_url=base + "/",
        )
        self.assertEqual(sources["kernel_wheel_base_url"], base)
        with patch.object(deps.urllib.request, "urlopen") as opener:
            for name, entry in self.lock["kernel_packages"].items():
                for wheel in entry["wheels"]:
                    with self.subTest(filename=wheel["filename"]):
                        url = deps.wheel_download_url(name, wheel, sources, kernel=True)
                        self.assertEqual(
                            url,
                            base
                            + "/"
                            + quote(wheel["filename"], safe="")
                            + "#sha256="
                            + wheel["sha256"],
                        )
            wheel = self.lock["packages"]["flashinfer-python"]["wheels"][0]
            self.assertEqual(
                deps.wheel_download_url("flashinfer-python", wheel, sources),
                wheel["url"] + "#sha256=" + wheel["sha256"],
            )
        opener.assert_not_called()

    def test_kernel_directory_keeps_python_packages_on_selected_pypi_mirror(self):
        sources = deps.download_sources(
            "aliyun", kernel_wheel_base_url="https://kernel-mirror.example/v0.7.2"
        )
        wheel = self.lock["packages"]["flashinfer-python"]["wheels"][0]
        response = self.mirror_response(
            f'<a href="https://pypi-files.example/{wheel["filename"]}">wheel</a>'
        )
        with patch.object(
            deps.urllib.request, "urlopen", return_value=response
        ) as opener:
            url = deps.wheel_download_url("flashinfer-python", wheel, sources)
        self.assertEqual(urlsplit(url).hostname, "pypi-files.example")
        self.assertEqual(urlsplit(url).fragment, "sha256=" + wheel["sha256"])
        self.assertEqual(
            opener.call_args.args[0].full_url,
            "https://mirrors.aliyun.com/pypi/simple/flashinfer-python/",
        )

    def test_invalid_source_configuration_and_non_http_wheel_are_rejected(self):
        for value in (
            "file:///tmp/simple",
            "https://host/simple?token=secret",
            "https://host/#frag",
            "https://user:password@host/simple",
            "https://host:invalid/simple",
        ):
            with self.subTest(value=value), self.assertRaises(deps.DependencyError):
                deps.download_sources(pypi_index_url=value)
            with (
                self.subTest(kernel_base=value),
                self.assertRaises(deps.DependencyError),
            ):
                deps.download_sources(kernel_wheel_base_url=value)
        with self.assertRaises(deps.DependencyError):
            deps.download_sources("unknown")
        wheel = self.lock["packages"]["flashinfer-python"]["wheels"][0]
        response = self.mirror_response(
            f'<a href="file:///tmp/{wheel["filename"]}">local</a>'
        )
        with patch.object(deps.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(deps.DependencyError, "non-HTTP"):
                deps.wheel_download_url(
                    "flashinfer-python", wheel, deps.download_sources("tuna")
                )

    def test_missing_local_wheel_fails_before_any_package_mutation(self):
        wheel = self.lock["packages"]["flashinfer-python"]["wheels"][0]
        before = {
            **self.base,
            "vllm": "0.32.0",
            "flashinfer-jit-cache-unused": "0.7.0+cu130",
        }
        with (
            patch.object(deps, "versions", return_value=before),
            patch.object(deps, "package_root"),
            patch.object(deps, "native_fingerprint"),
            patch.object(deps, "conflicts", return_value=set()),
            patch.object(
                deps,
                "plan_installs",
                return_value={
                    "flashinfer-python": self.lock["packages"]["flashinfer-python"]
                },
            ),
            patch.object(deps, "choose_wheel", return_value=wheel),
            patch.object(
                deps,
                "local_wheel_urls",
                side_effect=deps.DependencyError("missing wheel"),
            ),
            patch.object(deps, "wheel_download_url") as network_url,
            patch.object(deps.subprocess, "run") as run,
            patch.dict(deps.sys.modules, {"torch": Mock(version=Mock(cuda="13.0"))}),
        ):
            with self.assertRaisesRegex(deps.DependencyError, "missing wheel"):
                deps.install(self.lock)
        run.assert_not_called()
        network_url.assert_not_called()

    def test_local_wheels_require_locked_hash_and_never_fall_back_to_network(self):
        wheel = dict(self.lock["kernel_packages"]["flashinfer-jit-cache"]["wheels"][1])
        content = b"verified local wheel fixture"
        wheel["sha256"] = hashlib.sha256(content).hexdigest()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                deps.urllib.request,
                "urlopen",
                side_effect=AssertionError("network used"),
            ),
        ):
            path = Path(tmp) / wheel["filename"]
            with self.assertRaisesRegex(deps.DependencyError, "Missing local wheel"):
                deps.local_wheel_urls({"shim": wheel}, tmp)
            path.write_bytes(content)
            self.assertEqual(
                deps.local_wheel_urls({"shim": wheel}, tmp),
                [path.resolve().as_uri() + "#sha256=" + wheel["sha256"]],
            )
            path.write_bytes(content + b"corrupt")
            with self.assertRaisesRegex(deps.DependencyError, "SHA-256 differs"):
                deps.local_wheel_urls({"shim": wheel}, tmp)

    def test_install_uses_only_local_files_and_no_index(self):
        before = {**self.base, "vllm": "0.32.0", "flashinfer-python": "0.7.0.post1"}
        after = {**before, "flashinfer-python": "0.7.2"}
        content = b"local flashinfer fixture"
        wheel = dict(self.lock["packages"]["flashinfer-python"]["wheels"][0])
        wheel["sha256"] = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / wheel["filename"]).write_bytes(content)
            report = directory / "report.json"
            with (
                patch.object(deps, "versions", side_effect=[before, after]),
                patch.object(deps, "package_root"),
                patch.object(
                    deps, "native_fingerprint", return_value={"native.so": "digest"}
                ),
                patch.object(deps, "conflicts", return_value=set()),
                patch.object(
                    deps,
                    "plan_installs",
                    return_value={
                        "flashinfer-python": self.lock["packages"]["flashinfer-python"]
                    },
                ),
                patch.object(deps, "choose_wheel", return_value=wheel),
                patch.object(
                    deps,
                    "wheel_download_url",
                    side_effect=AssertionError("network URL used"),
                ),
                patch.object(
                    deps.urllib.request,
                    "urlopen",
                    side_effect=AssertionError("network used"),
                ),
                patch.object(deps.subprocess, "run") as run,
                patch.object(deps, "REPORT", report),
                patch.object(deps, "verify"),
                patch.dict(
                    deps.sys.modules,
                    {"torch": Mock(__version__="2.13.0", version=Mock(cuda="13.0"))},
                ),
            ):
                deps.install(self.lock, directory)
            args = run.call_args.args[0]
            self.assertIn("--no-index", args)
            self.assertIn("--no-deps", args)
            self.assertEqual(
                args[-1],
                (directory / wheel["filename"]).resolve().as_uri()
                + "#sha256="
                + wheel["sha256"],
            )
            self.assertEqual(
                json.loads(report.read_text())["download_sources"]["mode"],
                "local-wheelhouse",
            )


if __name__ == "__main__":
    unittest.main()
