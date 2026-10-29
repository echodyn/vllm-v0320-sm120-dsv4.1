# DeepSeek V4.1 Flash / SM120 回填清单

核对日期：**2026-10-28**。基础镜像为官方默认 CUDA 13.0 的
`vllm/vllm-openai:v0.32.0`，源码提交为
`5df63a7203cdb9328d824a27b682e77c7b9928be`。

适用性以固定 tag 的实际源码为准。发布时间不能代替源码/提交包含关系判断；
这里既包含基线缺少的已合并修复，也包含仍在审查的相关 PR。
状态是本次核对的快照，不代表之后的仓库状态。

`patches/manifest.json` 保存每项 PR 的完整 head/base/merge 提交、状态、
适配说明、执行顺序和补丁 SHA-256，以及基础文件 Git blob/SHA-256 和最终 SHA-256。
未合并 PR 的 GitHub 测试合并提交单独放在 `github_test_merge_commit`，不视为已合并。

## 纳入的 14 项

| PR | 状态 | 本包行为 |
|---|---|---|
| [#60762](https://github.com/vllm-project/vllm/pull/60762) | Open | SM12x indexer 与 FlashInfer sparse 后端声明 `[64]`。不修改 SM120 不会选择的 FlashMLA base，也不改变独立 SWA/compressor 布局。 |
| [#57292](https://github.com/vllm-project/vllm/pull/57292) | Open | 构造 SM120 attention 时检查公共 runtime-page-size 能力；旧 FlashInfer 给出明确错误。 |
| [#55264](https://github.com/vllm-project/vllm/pull/55264) | Open | 仅在稀疏 MLA 调用内进入调优，各 rank 独立校准；按可执行的 capture 桶 refine；传递配置的 skip-ops。 |
| [#63473](https://github.com/vllm-project/vllm/pull/63473) | Open | 拒绝 SM120 上的 FlashInfer CUTLASS MXFP4×MXFP8 MoE 组合；SM121、其他 activation 和 NVFP4 变体不被此 guard 禁用。 |
| [#63533](https://github.com/vllm-project/vllm/pull/63533) | Open | 小批量每 expert 预计 M≤64 时，在 permutation、两个 expert apply 路径和 warmup 一致地使用 64 对齐。 |
| [#59986](https://github.com/vllm-project/vllm/pull/59986) | Merged | 五个 V4.1 metadata builder 改为 `UNIFORM_BATCH`，保留以 `max_decode_query_len` 为边界的可变长度 decode 捕获。 |
| [#62769](https://github.com/vllm-project/vllm/pull/62769) | Open | 自适应 DSpark 的主机预算和设备端裁剪边界均以 TP rank 0 为准；优先复用 PyNCCL communicator。默认配置仍关闭自适应验证。 |
| [#63155](https://github.com/vllm-project/vllm/pull/63155) | Merged | MRV2 将模型捕获流传给 decoder replay；有仍存活的早期 aux hidden states 时继续单独捕获。 |
| [#60613](https://github.com/vllm-project/vllm/pull/60613) | Merged | caller-owned combined sparse-index workspace 在内核写入前填充 `-1`，避免读取残留尾部索引。 |
| [#60874](https://github.com/vllm-project/vllm/pull/60874) | Open | 按一个 prefill chunk 实际可达的范围限制 indexer K-gather 和 FlashMLA gather workspace，并保留一个完整请求的下限。 |
| [#63047](https://github.com/vllm-project/vllm/pull/63047) | Open | 缩小 expanded decode table；释放上一 chunk 的 logits 与 Engram previous_post；profile 阶段计入 Engram 激活。 |
| [#62860](https://github.com/vllm-project/vllm/pull/62860) | Open | logits 预算取 prefill 上限与对齐后的 decode 峰值之最大值，使用基础 native API 的 FP32 字节宽度。 |
| [#60996](https://github.com/vllm-project/vllm/pull/60996) | Open | 仅回填 shared KV pool 按每请求实际字节计费的分组选择，包含 stride 对齐；不引入批量 SWA metadata 重构。 |
| [#59849](https://github.com/vllm-project/vllm/pull/59849) | Open | 普通 CUDA Engram 主机表按精确字节注册；普通页保留 sorted lookup，原 THP 行为和分配失败回退保留。 |

合计 **14 个补丁、25 个修改文件**。本包以固定基线定义补丁状态；
完整补丁可重复应用，混合 pristine/final 文件可继续完成，未知文件内容会被拒绝。

## 版本适配

- **保留 DeepGEMM native。** 基线固定的是 `1e1842a833699298f7afc02eefb2ed168fff6938`。
  不引入 [#61149](https://github.com/vllm-project/vllm/pull/61149) 的新版 native/API 迁移。
  因此 #62860 按 FP32 logits 计费；不复制新版 SM100 BF16 输出、额外 FP32 转换或新的
  `clean_logits` 接口。decode 行数还受 `max_num_batched_tokens` 限制。
- **保留旧预热缓存机制。** #55264 的行为适配到 v0.32.0 的 legacy 文件缓存，
  不引入 [#55671](https://github.com/vllm-project/vllm/pull/55671) 的 managed-cache 重构。
  同类配置 skip-ops 传递已覆盖，不单独重复叠加
  [#49072](https://github.com/vllm-project/vllm/pull/49072)。V2 microbatching 不执行不受支持的串行预热。
- **保留旧模型状态结构。** #63155 的 stream 参数是可选的；旧调用仍创建新流。
  基线中只有 V4.1 和通用 interface 定义 `capture_inner_cudagraphs`。
  后来增加的 specialized-decode 文件不存在于此基线；AMD 专项修改不纳入。
- **KV 只回填计费策略。** #60996 的新批量 SWA module 和 model-state 改造未导入。
  分组增多时的主机开销仍需实测，不能引用上游其他硬件上的吞吐结果。
- **Engram 只改 CUDA 分配。** 不为此先引入
  [#63009](https://github.com/vllm-project/vllm/pull/63009) 的 ROCm 分配变化。
  注册失败返回原有 pinned-memory 回退；这不保证所有主机内存/锁页限制均已解决。

## FlashInfer 与未纳入项目

固定使用 [FlashInfer 0.7.2](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.2)，
源码提交 `d84335d8c54e0c8f9f7e22faa15c7e34a58abf94`。官方 wheel 和新依赖锁在
`dependencies/lock.json`；构建时验证下载 SHA-256，不使用浮动版本。
默认使用官方直链；也可在构建参数中选择清华/阿里云 PyPI 源、自定义索引及 GitHub 代理前缀。
下载地址可替换，锁定文件名、版本和 SHA-256 保留。详见 [下载、镜像与缓存](README_CN.md#下载镜像与缓存)。

- [#59846](https://github.com/vllm-project/vllm/pull/59846) 当前 head
  `3ebdb142b4a60b8dd7f9b525c1eae76af0bba2ec` 使用 `fp8_dsv41_inline` 和
  `fp8_dsv41_inline_fp4_ca`。已发布 0.7.2 的
  [接口](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.2/flashinfer/mla/_core.py)
  没有这两个格式。不能仅凭 PR 描述中的“≥0.7.1”判断当前代码兼容；本包不纳入，
  保留基础版 V4 格式的 FP8 KV，也不新增 SM120 `nvfp4_ds_mla` 支持。
- [#56509](https://github.com/vllm-project/vllm/pull/56509) 的旧页几何方案不与 runtime page size
  支持叠加。[#57028](https://github.com/vllm-project/vllm/pull/57028) 是几何、multi-stream、
  empty-batch 的组合修改，本包没有整包导入，也不把其中其他行为声称为已修复。
- [#47751](https://github.com/vllm-project/vllm/pull/47751) 所述 custom all-reduce 问题，
  在部署配置中通过 `--disable-custom-all-reduce` 使用 NCCL。它是配置选择，不计入上述 14 项源码回填。
- 不引入 ROCm、GB300 specialized decode、PCP/PP/DP 新功能、视觉专项优化或完整框架升级。

官方 JIT shim 0.7.2+cu130 的依赖为 sm80、sm89、sm90a、sm100a、sm103a、sm120f 六个
同版本 provider。本包锁定全部六个，避免最小安装留下未满足的 shim 依赖。
安装器仅允许记录中的 vLLM 原 FlashInfer 固定声明冲突；其他新增依赖冲突会失败。
满足要求的基础依赖不升级，缺失或过旧依赖只能使用清单中的精确候选。

## 验证范围

固定提交的必要源码先验证 Git blob 和 SHA-256，再在临时目录执行补丁。
CPU 回归执行真实修改函数或方法；为避免导入整个 CUDA/vLLM 运行时，通过 AST 选取定义并
显式提供依赖。CUDA kernel launch、CUDA host registration、通信和 graph capture 使用 stub。

覆盖补丁完整应用、重复执行、未知源码拒绝、资源哈希、语法、路径/符号链接拒绝和写入回滚；
覆盖 block 声明、旧 FlashInfer 拒绝、MoE 对齐边界、graph 分类、TP 决策同步、索引初始化、
workspace 预算、实际 KV 分组/对齐计费、Engram 精确分配及预热作用域。
另检查依赖选择/锁、SSE 完整性、前缀指标与验收脚本的 token 预算。

对应离线基线已由使用者反馈启动成功；在线版仅执行源码、CPU 和构建配置检查，
尚未完整构建在线镜像或执行其 GPU 验收。没有据此声明 512K admission、48 并发、
CUDA 数值正确性、准确率或性能已经验证。GPU 验收脚本已提供，结果应另行保存。
