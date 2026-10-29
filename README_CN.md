# vLLM v0.32.0 — SM120 / DeepSeek V4.1 Flash 补丁包

[English](README.md) · [回填与适配说明](BACKPORTS.md) · [Apache-2.0](LICENSE)

基于 `vllm/vllm-openai:v0.32.0`（CUDA 13.0）的在线构建版本，整合 14 项经过核对的
上游回填，涉及 25 个 Python 文件。vLLM 基线提交固定为
`5df63a7203cdb9328d824a27b682e77c7b9928be`，补丁和依赖均校验固定哈希。

**直接执行 `docker compose build`，由构建步骤下载并校验依赖。**
主机无需预下载 wheel，也无需准备 Python 环境。默认 Linux x86_64 / Python 3.12
目标首次构建约下载 2.97 GB wheel。BuildKit 提供断点续传缓存，wheel 压缩包不进入最终镜像层。

对应的离线基线已由使用者反馈启动成功。本在线版的源码检查结果记录在
[VALIDATION.json](VALIDATION.json)；尚未执行完整的在线镜像构建或 GPU 运行验证。
这些检查不能证明 512K 容量、48 请求准入、CUDA 数值正确性或吞吐性能。

## 环境要求

- Linux NVIDIA 服务器，4 张可见 SM120 GPU，已配置 NVIDIA Container Toolkit，
  并具备满足目标负载的显存和主机内存。
- Docker Compose v2，启用 BuildKit。
- 已准备好的本地 DeepSeek-V4.1-Flash 权重目录。
- 构建时能够访问基础镜像仓库、GitHub Releases 和 PyPI，或已明确配置下载镜像。

## 快速开始

```bash
git clone https://github.com/echodyn/vllm-v0320-sm120-dsv4.1.git
cd vllm-v0320-sm120-dsv4.1
cp .env.example .env
```

编辑 `.env` 中的 `MODEL_PATH`，填写已有本地权重目录。`BASE_IMAGE` 可替换为同一默认
CUDA 13.0 v0.32.0 镜像的镜像地址或 digest。其他 vLLM 版本、cu129 和 nightly 不在本依赖锁范围内。

```bash
docker compose config
docker compose build
```

加载模型前执行只读预检：

```bash
docker compose run --rm --no-deps --entrypoint python3 vllm \
  /opt/vllm-sm120-backports/scripts/check_runtime.py
```

启动服务：

```bash
docker compose up -d
docker compose logs -f vllm
```

在线版镜像为 `vllm:v0.32.0-sm120-dsv41-online`，Compose 项目为
`vllm-dsv41-sm120-online`，容器使用对应名称。API 监听 8000 端口，服务名为
`dsv41-flash-1017`。在同一服务器启动另一个部署前，请检查端口及 GPU 资源是否可用。

## 下载、镜像与缓存

构建先检查基础源码，再按**构建容器自身的** Linux 架构和 Python 版本选择 wheel，
下载后逐个验证 SHA-256，从已验证缓存安装，最后应用并复核本地补丁。
构建期间不会拉取浮动的 PR 分支。

其中 9 个 FlashInfer wheel 来自 GitHub 官方 `v0.7.2` Release，其他 8 个经过核对的
普通依赖来自锁定 PyPI 地址。默认目标共准备 17 个 wheel；安装时，基础镜像中已满足要求的
依赖仍然保留。

可在 `.env` 中配置以下选项，或通过 build args 传入：

| 选项 | 默认值 | 作用 |
| --- | --- | --- |
| `PYPI_MIRROR` | `official` | 普通依赖可选 `official`、`tuna`、`aliyun` |
| `PYPI_INDEX_URL` | 空 | 自定义 PEP 503 索引，优先于 `PYPI_MIRROR` |
| `GITHUB_PROXY_URL` | 空 | 支持 `前缀/https://github.com/...` 格式的下载代理 |
| `KERNEL_WHEEL_BASE_URL` | 空 | 提供同名 FlashInfer wheel 的目录，优先于 GitHub 代理 |

例如：

```bash
docker compose build --build-arg PYPI_MIRROR=aliyun
docker compose build --build-arg PYPI_MIRROR=tuna
```

PyPI 镜像用于 8 个普通 wheel；GitHub 上的 FlashInfer 附件需要单独保证下载链路可用。
源码仓库地址不能充当 wheel 目录。镜像必须提供锁定的同名文件和相同字节内容。
TLS 和 SHA-256 验证保持启用，构建不会静默切换版本或来源。
普通 `PIP_INDEX_URL` 不能替换这些固定的 wheel 下载地址。

BuildKit 使用 `sharing=locked` 挂载下载缓存，避免多个构建并发写入。
空缓存会自动下载；传输中断会保留 `.part` 文件供下一次构建续传；已缓存的完整 wheel
会重新校验后复用。哈希不符会在安装依赖前终止。压缩包保留在 BuildKit 缓存中，不额外复制进镜像层。

安装器采用 `pip --no-index --no-deps`。保留 PyTorch、CUDA 和 vendored DeepGEMM，
并比较安装前后的 vLLM 原生扩展和 DeepGEMM 文件哈希。
FlashInfer Python/cubin 固定为 `0.7.2`；JIT shim 与全部六个 provider
（`sm80`、`sm89`、`sm90a`、`sm100a`、`sm103a`、`sm120f`）固定为 `0.7.2+cu130`。

原 vLLM 元数据可能仍声明 `flashinfer-python==0.7.0.post1`。
本包明确记录覆盖为 `0.7.2`，并保留原始元数据。原镜像已有冲突会记录，其他新增冲突导致构建失败。
镜像内 `/opt/vllm-sm120-backports/dependency-report.json` 和 `dependency-downloads.json`
可供核对安装检查、请求的下载地址、目标环境及缓存复用情况。

## 运行参数

| 配置 | 默认值 |
| --- | --- |
| Tensor / Expert Parallel | TP4 / EP4；GPU 0–3 |
| MoE / all-reduce | `deep_gemm` / NCCL；关闭 custom all-reduce |
| KV / indexer 缓存 / block size | FP8 / FP8 / 64 |
| Engram | 启用 CPU offload |
| Model Runner / CUDA graphs | V2 / `FULL_AND_PIECEWISE` |
| 推测解码 | DSpark，5 tokens，关闭自适应验证 |
| 上下文 / prefill 预算 / 最大序列数 | 524288 / 16384 / 48 |
| 显存利用率 | **0.94** |
| 其他 | FlashInfer autotune、前缀缓存、V4.1 解析器 |

512K 是单请求长度上限，48 是调度器上限，不承诺 48 条完整 512K 请求同时驻留。
实际 KV 容量由启动时测量确定。Engram offload 还需要较大主机内存。
容量检查不通过时，应根据硬件和负载调整运行限制。

## 验证与服务检查

开发验证需要已有的 Python、`packaging` 和 CPU PyTorch 环境。源码验证只下载固定提交的
必要 Python 文件，在临时目录应用补丁并执行 CPU 回归：

```bash
python3 scripts/verify_backports.py --require-torch
```

构建镜像后，复核已安装依赖和补丁：

```bash
docker compose exec vllm python3 /opt/vllm-sm120-backports/patch_sm120.py --verify
docker compose exec vllm python3 /opt/vllm-sm120-backports/scripts/install_dependencies.py --verify
```

针对已经 Ready 的服务：

```bash
python3 scripts/acceptance.py --smoke-only --output smoke-report.json
python3 scripts/acceptance.py --output acceptance-report.json
```

需要 API key 时设置 `VLLM_API_KEY`。Smoke 检查涵盖健康、中文、算术、推理解析、流式、
工具及前缀缓存指标。完整验收还会发送接近 512K 的请求和 48 个并发请求，应预留适当测试窗口。
脚本不会启动容器或下载权重。这些是服务功能检查，不是准确率或 CUDA 数值基准。

## 上游出处与许可

[BACKPORTS.md](BACKPORTS.md) 说明每项 PR、适配和排除内容。
`patches/manifest.json` 保存固定提交和输入/结果哈希；`dependencies/lock.json`
保存已核对的版本、wheel 文件名和摘要。
本项目采用 [Apache-2.0](LICENSE)，上游归属见 [NOTICE](NOTICE)。依赖发行包保留各自许可证。
