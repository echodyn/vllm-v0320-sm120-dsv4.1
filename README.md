# vLLM v0.32.0 — SM120 / DeepSeek V4.1 Flash backports

[中文说明](README_CN.md) · [Backport details](BACKPORTS.md) · [Apache-2.0](LICENSE)

An online-build overlay for `vllm/vllm-openai:v0.32.0` (CUDA 13.0), bundling
14 reviewed upstream backports across 25 Python files. The vLLM baseline is
`5df63a7203cdb9328d824a27b682e77c7b9928be`; patch and dependency hashes are pinned.

**Run `docker compose build` to download and verify dependencies inside the build.**
No wheel files or Python setup are required on the host. The first default
Linux x86_64 / Python 3.12 build downloads about 2.97 GB of wheels. BuildKit
keeps a resumable download cache and excludes wheel archives from image layers.

The matching offline baseline has been reported to start successfully by its
operator. This online variant has passed the source checks recorded in
[VALIDATION.json](VALIDATION.json); a complete online image build and GPU run
have not been performed. These checks do not establish 512K capacity, 48-request
admission, CUDA numerical accuracy or throughput.

## Requirements

- A Linux NVIDIA server with four visible SM120 GPUs, NVIDIA Container Toolkit,
  and sufficient GPU and host memory for your workload.
- Docker Compose v2 with BuildKit enabled.
- A local DeepSeek-V4.1-Flash weights directory.
- Build-time access to the base-image registry, GitHub Releases and PyPI,
  or explicitly configured download mirrors.

## Quick start

```bash
git clone https://github.com/echodyn/vllm-v0320-sm120-dsv4.1.git
cd vllm-v0320-sm120-dsv4.1
cp .env.example .env
```

Edit `MODEL_PATH` in `.env` to your existing local weights directory. `BASE_IMAGE`
may point to a mirror or digest of the exact default CUDA 13.0 v0.32.0 image.
Other vLLM versions, cu129 and nightly images are outside this dependency lock.

```bash
docker compose config
docker compose build
```

Run the read-only preflight before loading the model:

```bash
docker compose run --rm --no-deps --entrypoint python3 vllm \
  /opt/vllm-sm120-backports/scripts/check_runtime.py
```

Start the service:

```bash
docker compose up -d
docker compose logs -f vllm
```

The online variant uses image `vllm:v0.32.0-sm120-dsv41-online`, Compose project
`vllm-dsv41-sm120-online` and the matching container name. It listens on port
8000 and serves `dsv41-flash-1017`. Review available ports and GPU resources
before starting another deployment on the same server.

## Downloads, mirrors and cache

The build checks the base sources, selects wheels for the **build container's**
Linux architecture and Python version, downloads them, verifies every SHA-256,
installs from the verified cache, and applies and verifies the bundled patches.
It never fetches floating PR branches during a build.

The nine FlashInfer wheels come from the official GitHub `v0.7.2` release.
The other eight reviewed dependency wheels use locked PyPI URLs. All 17 are
prepared for the default target; satisfying dependencies already in the base
image are retained during installation.

Set these optional values in `.env` or pass them as build arguments:

| Option | Default | Meaning |
| --- | --- | --- |
| `PYPI_MIRROR` | `official` | `official`, `tuna` or `aliyun` for ordinary dependency wheels |
| `PYPI_INDEX_URL` | empty | Custom PEP 503 index; overrides `PYPI_MIRROR` |
| `GITHUB_PROXY_URL` | empty | Proxy prefix supporting `PREFIX/https://github.com/...` |
| `KERNEL_WHEEL_BASE_URL` | empty | Directory serving identical FlashInfer wheel filenames; overrides the GitHub proxy |

For example:

```bash
docker compose build --build-arg PYPI_MIRROR=aliyun
docker compose build --build-arg PYPI_MIRROR=tuna
```

PyPI mirrors cover the eight ordinary wheels; GitHub-hosted FlashInfer assets
need their own route. A source-code repository URL is not a wheel directory.
Mirrors must serve the exact locked filenames and bytes. TLS verification and
SHA-256 checks remain enabled; the build does not switch versions or sources
silently. Generic `PIP_INDEX_URL` does not replace these locked wheel URLs.

BuildKit mounts the download cache with `sharing=locked`, so simultaneous builds
cannot write it concurrently. A fresh cache downloads automatically. Interrupted
transfers leave `.part` files for the next build to resume; valid cached wheels
are rehashed and reused. Bad hashes fail the build before dependency installation.
Downloaded archives stay in BuildKit's cache rather than the final image.

Installation uses `pip --no-index --no-deps`. PyTorch, CUDA and vendored DeepGEMM
are retained; native vLLM and DeepGEMM file hashes are compared before and after
installation. FlashInfer Python/cubin are fixed at `0.7.2`; the JIT shim and all
six providers (`sm80`, `sm89`, `sm90a`, `sm100a`, `sm103a`, `sm120f`) use
`0.7.2+cu130`.

The base vLLM metadata may still require `flashinfer-python==0.7.0.post1`.
The deliberate override to `0.7.2` is recorded without rewriting that metadata.
Existing base conflicts are retained; other new conflicts fail the build.
Inspect `/opt/vllm-sm120-backports/dependency-report.json` and
`dependency-downloads.json` in the image for installation checks, requested
source URLs, target details and cache reuse.

## Runtime defaults

| Setting | Default |
| --- | --- |
| Tensor / expert parallelism | TP4 / EP4; GPU IDs 0–3 |
| MoE / all-reduce | `deep_gemm` / NCCL; custom all-reduce disabled |
| KV / indexer cache / block size | FP8 / FP8 / 64 |
| Engram | CPU offload enabled |
| Model runner / CUDA graphs | V2 / `FULL_AND_PIECEWISE` |
| Speculation | DSpark, 5 tokens, adaptive verification disabled |
| Context / prefill budget / max sequences | 524288 / 16384 / 48 |
| GPU memory utilization | **0.94** |
| Other | FlashInfer autotune, prefix caching, V4.1 parsers |

512K is a per-request limit; 48 is a scheduler limit. The combination does not
promise 48 resident 512K requests. Startup determines actual KV capacity.
Engram offload also requires substantial host RAM. Adjust runtime limits to the
hardware and workload when capacity validation fails.

## Verification and serving checks

For development, source verification requires an existing Python environment
with `packaging` and CPU PyTorch. It downloads only pinned upstream Python source
files, applies the bundle in a temporary tree and runs the CPU regressions:

```bash
python3 scripts/verify_backports.py --require-torch
```

After building the image, verify installed dependencies and patched sources:

```bash
docker compose exec vllm python3 /opt/vllm-sm120-backports/patch_sm120.py --verify
docker compose exec vllm python3 /opt/vllm-sm120-backports/scripts/install_dependencies.py --verify
```

Against an already-ready server:

```bash
python3 scripts/acceptance.py --smoke-only --output smoke-report.json
python3 scripts/acceptance.py --output acceptance-report.json
```

Set `VLLM_API_KEY` when needed. Smoke checks cover health, Chinese, arithmetic,
parsed reasoning, streaming, tools and prefix metrics. Full acceptance also
sends a roughly 512K request and 48 concurrent requests; reserve a suitable test
window. The script does not start containers or download weights. These are
serving checks rather than accuracy or CUDA numerical benchmarks.

## Provenance and license

[BACKPORTS.md](BACKPORTS.md) describes every included PR, adaptation and exclusion.
`patches/manifest.json` contains the pinned commits and source/result hashes;
`dependencies/lock.json` contains reviewed versions, wheel names and digests.
The project is distributed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE) for
upstream attribution. Dependency distributions retain their own licenses.
