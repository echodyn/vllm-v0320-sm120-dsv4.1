import ast
from collections import Counter, defaultdict
import contextlib
from contextvars import ContextVar
import dataclasses
from enum import Enum, IntEnum
from fractions import Fraction
import functools
import math
import sys
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import Mock, patch

from source_loader import load, tree

try:
    import torch
    import numpy as np
except ImportError:
    torch = np = None


def platform(family):
    return NS(
        is_device_capability_family=lambda value: value == family,
        get_device_capability=lambda: (
            (12, family - 120) if family >= 120 else (family // 10, 0)
        ),
    )


class CapabilityTests(unittest.TestCase):
    def test_sm120_block_declarations_and_other_architectures(self):
        for family, expected in [(120, 64), (90, 64), (100, 128)]:
            fn = load(
                "v1/attention/backends/mla/indexer.py",
                ["get_supported_kernel_block_sizes"],
                {"current_platform": platform(family)},
                "DeepseekV41IndexerBackend",
            )["get_supported_kernel_block_sizes"]
            self.assertEqual(fn(), [expected])
        for family, expected in [(120, 64), (90, 128), (100, 128)]:
            fn = load(
                "models/deepseek_v41/nvidia/flashinfer_sparse.py",
                ["get_supported_kernel_block_sizes"],
                {"current_platform": platform(family)},
                "DeepseekV4FlashInferMLASparseBackend",
            )["get_supported_kernel_block_sizes"]
            self.assertEqual(fn(), [expected])

    def test_runtime_page_capability_requires_the_public_flag(self):
        for config, expected in [
            (None, False),
            (NS(page_block_size=64), False),
            (NS(page_block_size_is_runtime=False), False),
            (NS(page_block_size_is_runtime=True), True),
        ]:
            module = NS(supported_sparse_mla_sm120_configs=lambda: {"dsv4": config})
            fn = load(
                "utils/flashinfer.py",
                ["has_flashinfer_sparse_mla_sm120_runtime_page_size"],
                {
                    "has_flashinfer_sparse_mla_sm120": lambda: True,
                    "_get_submodule": lambda _: module,
                },
            )["has_flashinfer_sparse_mla_sm120_runtime_page_size"]
            self.assertEqual(fn(), expected)

    def test_missing_flashinfer_and_missing_config_api_fail_closed(self):
        for present in [False, True]:
            fn = load(
                "utils/flashinfer.py",
                ["has_flashinfer_sparse_mla_sm120_runtime_page_size"],
                {
                    "has_flashinfer_sparse_mla_sm120": lambda: present,
                    "_get_submodule": lambda _: NS(),
                },
            )["has_flashinfer_sparse_mla_sm120_runtime_page_size"]
            self.assertFalse(fn())

    def test_old_flashinfer_rejected_when_attention_is_constructed(self):
        parent = type("Parent", (), {"__init__": lambda self, *a, **kw: None})
        # Run the actual constructor prefix through the capability gate.
        node = next(
            n
            for n in tree("models/deepseek_v41/nvidia/flashinfer_sparse.py").body
            if isinstance(n, ast.ClassDef)
            and n.name == "DeepseekV4FlashInferSM120Attention"
        )
        init = next(
            n
            for n in node.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        gate_end = next(i for i, n in enumerate(init.body) if isinstance(n, ast.If))
        cls = ast.ClassDef(
            name="Attention",
            bases=[ast.Name(id="Parent", ctx=ast.Load())],
            keywords=[],
            body=[init],
            decorator_list=[],
        )
        init.body = init.body[: gate_end + 1]
        api = ModuleType("vllm.utils.flashinfer")
        api.has_flashinfer_sparse_mla_sm120_config = lambda *a: True
        api.has_flashinfer_sparse_mla_sm120_runtime_page_size = lambda: False
        env = {"Parent": parent, "VllmConfig": object}
        with patch.dict(sys.modules, {"vllm.utils.flashinfer": api}):
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                    "<gate>",
                    "exec",
                ),
                env,
            )
            with self.assertRaisesRegex(RuntimeError, "FlashInfer >= 0.7.1"):
                env["Attention"](None)

    def test_mxfp8_moe_guard_is_scoped_to_sm120(self):
        key = object()
        base = NS(is_supported_config=lambda *args: (True, None))
        for family, activation, expected in [
            (120, key, False),
            (121, key, True),
            (100, key, True),
            (120, None, True),
        ]:
            fn = load(
                "model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py",
                ["is_supported_config"],
                {
                    "current_platform": platform(family),
                    "kMxfp8Dynamic": key,
                    "_SM120_UNSUPPORTED_REASON": "bad backend",
                    "mk": NS(FusedMoEExpertsModular=base),
                },
                "FlashInferExperts",
            )["is_supported_config"]
            self.assertEqual(fn(object, None, None, activation, None)[0], expected)

    def test_five_builders_restrict_full_graphs_and_keep_spec_width(self):
        cases = [
            ("models/deepseek_v41/sparse_mla.py", "DeepseekV4FlashMLAMetadataBuilder"),
            ("models/deepseek_v41/compressor.py", "CompressorMetadataBuilder"),
            (
                "models/deepseek_v41/nvidia/flashmla.py",
                "DeepseekSparseSWAFlashMLAMetadataBuilder",
            ),
            (
                "models/deepseek_v41/nvidia/flashinfer_sparse.py",
                "DeepseekV4FlashInferSparseMLAMetadataBuilder",
            ),
            (
                "models/deepseek_v41/nvidia/flashinfer_sparse.py",
                "DeepseekSparseSWAFlashInferMetadataBuilder",
            ),
        ]
        maximum = load("v1/attention/backend.py", ["max_decode_query_len"])[
            "max_decode_query_len"
        ]
        for path, owner in cases:
            cls = next(
                n
                for n in tree(path).body
                if isinstance(n, ast.ClassDef) and n.name == owner
            )
            assignment = next(
                n
                for n in cls.body
                if isinstance(n, ast.AnnAssign) and n.target.id == "_cudagraph_support"
            )
            self.assertEqual(assignment.value.attr, "UNIFORM_BATCH")
            fn = load(
                path,
                ["get_varlen_cudagraph_max_query_len"],
                {"max_decode_query_len": maximum},
                owner,
            )["get_varlen_cudagraph_max_query_len"]
            for tokens in [0, 5, 7]:
                cfg = NS(
                    speculative_config=NS(
                        num_speculative_tokens=tokens, parallel_drafting=False
                    ),
                    use_v2_model_runner=True,
                )
                self.assertEqual(fn(None, cfg, None), tokens + 1)


@unittest.skipIf(torch is None, "torch required for CPU tensors")
class TensorTests(unittest.TestCase):
    def test_host_adaptive_budget_uses_tp_leader_result(self):
        group = NS(broadcast_object=Mock(return_value=2))
        fn = load(
            "v1/worker/gpu/spec_decode/adaptive_verification.py",
            ["get_num_tokens"],
            {"np": np, "get_tp_group": lambda: group},
            "AdaptiveVerificationManager",
        )["get_num_tokens"]
        obj = NS(
            cost_tables=(np.ones(3), np.arange(1, 20)),
            req_states=NS(
                req_id_to_index={"a": 0, "b": 1},
                num_computed_tokens_np=np.array([10, 10]),
                prefill_len=NS(np=np.array([1, 1])),
            ),
            _stale_confidences=[NS(np=np.zeros((2, 2)))],
            _stale_idx=0,
            num_speculative_steps=2,
            num_bonus_tokens=1,
            _max_total_logits=128,
        )
        self.assertEqual(fn(obj, {"a": 3, "b": 3}, {"a": [1, 2], "b": [3, 4]}), 4)
        self.assertEqual(obj._batch_budget[2], 2)
        group.broadcast_object.assert_called_once_with(0, src=0)

    def test_reused_sparse_index_workspace_is_seeded_before_kernel(self):
        def kernel(indices, lens, *args, **kwargs):
            self.assertTrue(torch.all(indices == -1))
            indices[0, 0] = 3
            lens.fill_(1)

        fn = load(
            "models/deepseek_v41/common/ops/cache_utils.py",
            ["combine_topk_swa_indices"],
            {
                "torch": torch,
                "_SPARSE_PREFILL_TOPK_ALIGNMENT": 128,
                "_COMBINE_TOPK_SWA_INDICES_KERNEL": kernel,
            },
        )["combine_topk_swa_indices"]
        for supplied in [False, True]:
            out = (
                (
                    torch.full((2, 128), 999, dtype=torch.int32),
                    torch.empty(2, dtype=torch.int32),
                )
                if supplied
                else None
            )
            indices, _ = fn(
                torch.zeros(2, 8, dtype=torch.int32),
                None,
                None,
                None,
                16,
                1,
                8,
                2,
                2,
                out=out,
            )
            self.assertEqual(indices[0, 0], 3)
            self.assertTrue(torch.all(indices[:, 1:] == -1))

    def test_moe_permute_alignment_boundary(self):
        class Stop(Exception):
            pass

        for family, m, expected in [(120, 64, 64), (120, 65, 128), (100, 1, 128)]:

            def capture(**kwargs):
                self.assertEqual(kwargs["alignment"], expected)
                raise Stop

            fn = load(
                "model_executor/layers/fused_moe/deep_gemm_utils.py",
                ["deepgemm_moe_permute"],
                {
                    "current_platform": platform(family),
                    "get_mk_alignment_for_contiguous_layout": lambda: (128, 128),
                    "compute_aligned_M_and_alignment": capture,
                },
            )["deepgemm_moe_permute"]
            with self.assertRaises(Stop):
                fn(
                    torch.empty(m, 128),
                    torch.empty(m, 1),
                    torch.zeros(m, 1, dtype=torch.int32),
                    1,
                    None,
                    None,
                )

    def test_moe_warmup_matches_runtime_alignment(self):
        observed = []

        def aligned(**kwargs):
            observed.append((kwargs["M"], kwargs["alignment"]))
            a = kwargs["alignment"]
            return math.ceil(kwargs["M"] / a) * a, a

        fn = load(
            "model_executor/warmup/deep_gemm_warmup.py",
            ["_get_grouped_gemm_params"],
            {
                "torch": torch,
                "current_platform": platform(120),
                "get_mk_alignment_for_contiguous_layout": lambda: (128, 128),
                "get_dp_group": lambda: NS(world_size=1),
                "_generate_optimal_warmup_m_values": lambda *a: [64, 65],
                "compute_aligned_M_and_alignment": aligned,
            },
        )["_get_grouped_gemm_params"]
        _, _, cases = fn(torch.empty(1, 128, 128), torch.empty(1, 128, 128), 1, 65)
        self.assertEqual(observed, [(64, 64), (65, 128)])
        self.assertEqual([a for _, a, _ in cases], [64, 128])

    def test_pynccl_broadcast_and_disabled_fallback(self):
        for enabled in [True, False, None]:
            pynccl = (
                NS(disabled=not enabled, broadcast=Mock())
                if enabled is not None
                else None
            )
            group = NS(device_communicator=NS(pynccl_comm=pynccl), broadcast=Mock())
            fn = load(
                "v1/worker/gpu/spec_decode/adaptive_verification.py",
                ["_broadcast_from_tp_leader"],
                {"get_tp_group": lambda: group},
            )["_broadcast_from_tp_leader"]
            tensor = torch.tensor([1, 2])
            fn(tensor)
            (pynccl.broadcast if enabled else group.broadcast).assert_called_once_with(
                tensor, src=0
            )
            if enabled:
                group.broadcast.assert_not_called()

    def test_adaptive_reallocation_uses_leader_boundaries(self):
        path = "v1/worker/gpu/spec_decode/adaptive_verification.py"

        def hostcopy(array, out):
            out.copy_(torch.from_numpy(array))

        broadcasts = []

        def broadcast(capacity):
            broadcasts.append(capacity.clone())
            capacity.copy_(torch.tensor([0, 2], dtype=capacity.dtype))

        def assign(conf, mapping, capacity, *args):
            capacity.copy_(torch.tensor([2, 0], dtype=capacity.dtype))

        fn = load(
            path,
            ["reallocate_drafts"],
            {
                "torch": torch,
                "np": np,
                "async_tensor_h2d": hostcopy,
                "_assign_draft_token_budget_compiled": assign,
                "_broadcast_from_tp_leader": broadcast,
            },
            "AdaptiveVerificationManager",
        )["reallocate_drafts"]
        for budget in [0, 2, 4]:
            broadcasts.clear()
            obj = NS(
                _batch_budget=({"a": 2, "b": 2}, {"a": 1, "b": 1}, budget),
                _batch_draft_capacity=torch.empty(2, dtype=torch.int32),
                _confidence_probs=None,
                num_speculative_steps=2,
                num_bonus_tokens=1,
                _num_non_draft_tokens=torch.empty(2, dtype=torch.int32),
                _cu_num_logits=torch.empty(3, dtype=torch.int32),
                query_start_loc=torch.empty(4, dtype=torch.int32),
            )
            cu, q, _ = fn(obj, ["a", "b"], torch.arange(2))
            self.assertEqual(len(broadcasts), int(budget == 2))
            if budget == 2:
                self.assertEqual(q.tolist(), [0, 1, 4, 4])
                self.assertEqual(cu.tolist(), [0, 1, 4])


class BudgetTests(unittest.TestCase):
    def test_decode_logits_reservation_stride_and_token_cap(self):
        fn = load(
            "model_executor/layers/sparse_attn_indexer.py",
            ["_profile_indexer_logits_bytes"],
            {"math": math, "round_up": lambda x, a: math.ceil(x / a) * a},
        )["_profile_indexer_logits_bytes"]
        cfg = NS(
            scheduler_config=NS(max_num_seqs=48, max_num_batched_tokens=16384),
            num_speculative_tokens=5,
        )
        self.assertEqual(fn(524289, cfg, 1, False), 48 * 6 * 524544 * 4)
        cfg.scheduler_config.max_num_batched_tokens = 100
        self.assertEqual(fn(1024, cfg, 1, False), 100 * 1024 * 4)
        self.assertEqual(fn(1024, None, 4096, False), 4096)
        self.assertEqual(fn(1024, cfg, 1 << 30, False), 1 << 30)

    def test_indexer_gather_budget_tracks_compression_and_logit_cap(self):
        fn = load(
            "models/deepseek_v41/attention.py",
            ["_indexer_prefill_buffer_size"],
            {
                "math": math,
                "envs": NS(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1),
                "get_max_prefill_buffer_size": lambda cfg: 1 << 24,
            },
        )["_indexer_prefill_buffer_size"]
        cfg = NS(model_config=NS(max_model_len=1048576))
        self.assertEqual(fn(cfg, 1), 1048576)
        self.assertEqual(
            fn(cfg, 2), 524288
        )  # Always retain one full compressed request.
        cfg.model_config.max_model_len = 262144
        self.assertEqual(fn(cfg, 2), math.isqrt((1024 * 1024 // 4) * 131072))

    def test_capture_stream_is_optional_and_preserves_old_callers(self):
        calls = []
        new_stream = object()
        torch_stub = NS(cuda=NS(Stream=lambda **kw: calls.append(kw) or new_stream))

        @contextlib.contextmanager
        def capture(ctx):
            yield

        group = NS(graph_capture=capture)
        fn = load(
            "distributed/parallel_state.py",
            ["graph_capture"],
            {
                "torch": torch_stub,
                "GraphCaptureContext": lambda stream: NS(stream=stream),
                "get_tp_group": lambda: group,
                "get_pp_group": lambda: group,
                "get_dp_group": lambda: group,
            },
        )["graph_capture"]
        fn = contextlib.contextmanager(fn)
        chosen = object()
        with fn("cuda:0", chosen) as context:
            self.assertIs(context.stream, chosen)
        self.assertEqual(calls, [])
        with fn("cuda:0") as context:
            self.assertIs(context.stream, new_stream)
        self.assertEqual(len(calls), 1)

    def test_replay_with_live_auxiliary_states_keeps_separate_capture(self):
        path = "models/deepseek_v41/nvidia/decoder_replay_cudagraph.py"
        fn = load(
            path, ["capture_replay_graphs"], owner="DecoderReplayCudaGraphManager"
        )["capture_replay_graphs"]
        chosen = object()
        for has_aux in [False, True]:
            capture = Mock()
            obj = NS(
                layers=NS(has_early_aux=lambda: has_aux, replay_batch=object()),
                capture=capture,
            )
            fn(obj, lambda *a: None, chosen)
            self.assertIs(capture.call_args.args[2], None if has_aux else chosen)
            self.assertIsNone(obj.layers.replay_batch)


@unittest.skipIf(torch is None, "torch required for cache dtype sizes")
class PackingTests(unittest.TestCase):
    def definitions(self):
        import copy

        env = {
            "__name__": "__main__",
            "dataclass": dataclasses.dataclass,
            "field": dataclasses.field,
            "fields": dataclasses.fields,
            "replace": dataclasses.replace,
            "Enum": Enum,
            "IntEnum": IntEnum,
            "Counter": Counter,
            "Fraction": Fraction,
            "torch": torch,
            "math": math,
            "copy": copy,
            "cached_property": functools.cached_property,
            "get_dtype_size": lambda dt: dt.itemsize,
            "cdiv": lambda a, b: -(-a // b),
            "round_up": lambda a, b: -(-a // b) * b,
        }
        env = load(
            "v1/kv_cache_interface.py",
            [
                "KVQuantMode",
                "KVCacheSpec",
                "AttentionSpec",
                "CircularBufferSpec",
                "UniformTypeKVCacheSpecs",
                "KVCacheGroupRole",
                "KVCacheGroupSpec",
            ],
            env,
        )
        base = env["KVCacheSpec"]

        @dataclasses.dataclass(frozen=True)
        class Spec(base):
            size: int = 64
            pages: int = 100

            @property
            def page_size_bytes(self):
                return self.size

            def max_memory_usage_bytes(self, config):
                return self.size * self.pages

            def is_uniform_with_collection(self, specs):
                return all(type(value) is type(self) for value in specs.values())

        env.update(
            defaultdict=defaultdict,
            HiSparseHotSpec=type("Hot", (), {}),
            MambaSpec=type("Mamba", (), {}),
            SlidingWindowSpec=type("SWA", (), {}),
            _annotate_eagle_groups=lambda *a, **kw: None,
            _uses_trailing_mtp_layers=lambda *a: False,
            _warn_if_unannotated_eagle_mamba=lambda *a: None,
        )
        funcs = load(
            "v1/core/kv_cache_utils.py",
            [
                "_get_per_layer_spec",
                "_get_kv_cache_bytes_per_block",
                "_pool_bytes_per_block",
                "_max_memory_usage_bytes_from_groups",
                "_get_packed_kv_cache_groups",
            ],
            env,
        )
        return Spec, funcs

    def test_pooled_cost_and_layer_partition(self):
        Spec, env = self.definitions()
        cfg = NS(
            cache_config=NS(
                get_resolved_kv_cache_layout=lambda: NS(is_block_outermost=True)
            )
        )
        for alignment, expected_cost, attention_groups in [
            (1, 24080, 3),
            (128, 25856, 1),
        ]:
            specs = {
                f"a{i}": Spec(64, size=64, block_stride_alignment=alignment)
                for i in range(3)
            }
            specs.update(
                {
                    f"b{i}": Spec(64, size=16, block_stride_alignment=alignment)
                    for i in range(3)
                }
            )
            specs["ring"] = env["CircularBufferSpec"](
                block_size=1,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.uint8,
                page_size_padded=64,
            )
            module = ModuleType("vllm.v1.attention.backend")
            module.MultipleOf = type("MultipleOf", (), {})
            with patch.dict(sys.modules, {"vllm.v1.attention.backend": module}):
                groups = env["_get_packed_kv_cache_groups"](cfg, specs)
                self.assertEqual(
                    env["_max_memory_usage_bytes_from_groups"](cfg, groups),
                    expected_cost,
                )
            names = [name for group in groups for name in group.layer_names]
            self.assertEqual(sorted(names), sorted(specs))
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(len(groups), attention_groups + 1)

    def test_no_packing_for_uniform_or_layer_layout(self):
        Spec, env = self.definitions()
        cfg = NS(
            cache_config=NS(
                get_resolved_kv_cache_layout=lambda: NS(is_block_outermost=True)
            )
        )
        self.assertIsNone(env["_get_packed_kv_cache_groups"](cfg, {"one": Spec(64)}))
        cfg.cache_config.get_resolved_kv_cache_layout = lambda: NS(
            is_block_outermost=False
        )
        self.assertIsNone(
            env["_get_packed_kv_cache_groups"](
                cfg, {"one": Spec(64), "two": Spec(64, size=16)}
            )
        )


@unittest.skipIf(np is None, "numpy required")
class EngramTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch required")
    def test_private_tables_use_exact_storage_without_disabling_sorted_lookup(self):
        for success in [True, False]:
            allocated = []

            def register(size, huge_pages):
                allocated.append((size, huge_pages))
                return torch.empty(size, dtype=torch.uint8) if success else None

            pinned = []

            def empty(*args, **kwargs):
                pinned.append(kwargs.pop("pin_memory", False))
                return torch.empty(*args, **kwargs)

            env = {
                "current_platform": NS(is_cuda=lambda: True),
                "_allocate_registered_storage": register,
                "torch": NS(
                    empty=empty, uint8=torch.uint8, float8_e4m3fn=torch.float8_e4m3fn
                ),
            }
            fn = load(
                "models/deepseek_v41/common/engram.py",
                ["_allocate_weights"],
                env,
                "ParallelEngramEmbedding",
            )["_allocate_weights"]
            obj = NS(
                dp_shared_memory=False,
                cpu_offload=True,
                use_thp=False,
                part_num_embeddings=8,
                dim=64,
                block_size=32,
                _packed=None,
            )
            weight, scales = fn(obj)
            self.assertEqual(allocated, [(528, False)])
            self.assertEqual(tuple(weight.shape), (8, 64))
            self.assertEqual(tuple(scales.shape), (8, 2))
            self.assertIsNone(obj._packed)
            self.assertEqual(pinned, [] if success else [True, True])

    def test_exact_size_registration_and_optional_huge_pages(self):
        import weakref

        for huge in [False, True]:

            class Mapping(bytearray):
                def madvise(self, advice):
                    self.advice = advice

            mapped = Mapping(b"\xab" * 128)

            class Tensor:
                def __init__(self, owner):
                    self.owner = owner

                def data_ptr(self):
                    return 1234

                def is_pinned(self):
                    return True

            register = Mock(return_value=NS(value=0))
            torch_stub = NS(
                from_numpy=Tensor, cuda=NS(cudart=lambda: NS(cudaHostRegister=register))
            )
            env = {
                "sys": NS(platform="linux"),
                "np": np,
                "torch": torch_stub,
                "mmap": NS(
                    mmap=lambda *a, **kw: mapped,
                    MAP_PRIVATE=1,
                    MAP_ANONYMOUS=2,
                    MADV_HUGEPAGE=3,
                    PAGESIZE=16,
                ),
                "weakref": weakref,
                "DPSharedEngramStorage": NS(_unregister=Mock()),
                "logger": Mock(),
            }
            fn = load(
                "models/deepseek_v41/common/engram.py",
                ["_allocate_registered_storage"],
                env,
            )["_allocate_registered_storage"]
            tensor = fn(128, huge)
            self.assertIsNotNone(tensor)
            register.assert_called_once_with(1234, 128, 0)
            self.assertEqual(hasattr(mapped, "advice"), huge)
            self.assertEqual(mapped[0], 0 if huge else 0xAB)

    def test_registration_failure_returns_existing_fallback_signal(self):
        fn = load(
            "models/deepseek_v41/common/engram.py",
            ["_allocate_registered_storage"],
            {"sys": NS(platform="darwin")},
        )["_allocate_registered_storage"]
        self.assertIsNone(fn(1024, False))


class WarmupTests(unittest.TestCase):
    def make_worker(self, rank, *, v2=True, micro=False):
        runner = NS(
            max_num_reqs=48,
            decode_query_len=6,
            uniform_decode_query_len=6,
            vllm_config=NS(use_v2_model_runner=v2),
            ubatch_runner=object() if micro else None,
            _dummy_run=Mock(),
        )
        cfg = NS(
            kernel_config=NS(enable_flashinfer_autotune=True),
            compilation_config=NS(cudagraph_capture_sizes=[1, 6, 12, 12, 18, 64, 66]),
        )
        return NS(
            model_runner=runner,
            vllm_config=cfg,
            scheduler_config=NS(max_num_batched_tokens=16384),
            execute_model=Mock(),
            sample_tokens=Mock(),
        )

    def test_refine_buckets_are_executable_and_bounded(self):
        env = load(
            "model_executor/warmup/flashinfer_sparse_mla_warmup.py",
            ["_uses_v2_model_runner", "_sparse_mla_refine_tokens"],
            {"cast": lambda typ, value: value, "_SPARSE_MLA_REFINE_TOKEN_CAP": 64},
        )
        worker = self.make_worker(0)
        self.assertEqual(env["_sparse_mla_refine_tokens"](worker), (6, 12, 18))
        worker.scheduler_config.max_num_batched_tokens = 12
        self.assertEqual(env["_sparse_mla_refine_tokens"](worker), (6, 12))
        worker.model_runner.vllm_config.use_v2_model_runner = False
        self.assertEqual(env["_sparse_mla_refine_tokens"](worker), (1, 6, 12))

    def test_v2_microbatching_skips_unsupported_serial_warmup(self):
        env = {
            "_flashinfer_sparse_mla_decode_label": lambda *args: "DSv4.1",
            "has_flashinfer": lambda: True,
            "current_platform": platform(120),
            "logger": Mock(),
        }
        funcs = load(
            "model_executor/warmup/flashinfer_sparse_mla_warmup.py",
            ["_uses_v2_model_runner", "_run_flashinfer_sparse_mla_decode_autotune"],
            env,
        )
        worker = self.make_worker(0, micro=True)
        self.assertFalse(
            funcs["_run_flashinfer_sparse_mla_decode_autotune"](worker, 16, frozenset())
        )
        worker.model_runner._dummy_run.assert_not_called()

    def test_sparse_scope_tunes_only_wrapped_attention_and_restores_context(self):
        state = ContextVar("test_sparse_tuning", default=None)
        tuner = NS(is_tuning_mode=False)
        module = ModuleType("flashinfer.autotuner")
        module.AutoTuner = NS(get=lambda: tuner)
        events = []

        @contextlib.contextmanager
        def tuning():
            events.append("enter")
            yield
            events.append("exit")

        env = load(
            "utils/flashinfer.py",
            ["autotune_sparse_mla_only", "_autotune_sparse_mla_call"],
            {"_sparse_mla_autotune_context": state, "functools": functools},
        )
        scope = contextlib.contextmanager(env["autotune_sparse_mla_only"])
        wrapped = env["_autotune_sparse_mla_call"](lambda: events.append("attention"))
        with patch.dict(sys.modules, {"flashinfer.autotuner": module}):
            with scope(tuning):
                events.append("gemm")
                wrapped()
            self.assertEqual(events, ["gemm", "enter", "attention", "exit"])
            self.assertIsNone(state.get())
            tuner.is_tuning_mode = True
            with self.assertRaises(RuntimeError):
                with scope(tuning):
                    pass

    def test_every_rank_tunes_with_skip_ops_on_legacy_cache(self):
        import tempfile
        from pathlib import Path

        for rank in [0, 1, 3]:
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as tmp:
                worker = self.make_worker(rank)
                cache = Path(tmp) / "tune.json"
                tuner = NS(
                    _dirty=True,
                    load_configs=Mock(),
                    save_configs=lambda path: Path(path).write_bytes(b"{}"),
                )
                module = ModuleType("flashinfer.autotuner")
                module.AutoTuner = NS(get=lambda: tuner)
                group = NS(
                    rank_in_group=rank,
                    broadcast_object=lambda value, **kw: b"{}",
                    barrier=Mock(),
                )
                group_module = ModuleType("vllm.distributed.parallel_state")
                group_module.get_world_group = lambda: group
                contexts = []

                @contextlib.contextmanager
                def tune(*args, **kwargs):
                    contexts.append(kwargs)
                    yield

                @contextlib.contextmanager
                def scope(factory):
                    with factory():
                        yield

                def mixed(*args, mixed_step_context, **kwargs):
                    with mixed_step_context:
                        return True

                env = {
                    "torch": NS(inference_mode=contextlib.nullcontext),
                    "has_flashinfer": lambda: True,
                    "current_platform": platform(120),
                    "logger": Mock(),
                    "cast": lambda typ, x: x,
                    "_flashinfer_sparse_mla_decode_label": lambda *a: "DSv4.1",
                    "resolve_flashinfer_autotune_file": lambda *a: cache,
                    "write_flashinfer_autotune_cache": lambda p, data: p.write_bytes(
                        data
                    ),
                    "_sparse_mla_refine_tokens": lambda *a: (6, 12),
                    "_run_sparse_mla_refine_bucket": Mock(),
                    "run_mixed_prefill_decode_warmup": mixed,
                    "flashinfer_autotune": tune,
                    "autotune_sparse_mla_only": scope,
                    "envs": NS(VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=["moe"]),
                    "contextlib": contextlib,
                }
                funcs = load(
                    "model_executor/warmup/flashinfer_sparse_mla_warmup.py",
                    [
                        "_uses_v2_model_runner",
                        "_sparse_mla_autotune_kwargs",
                        "_run_flashinfer_sparse_mla_decode_autotune",
                    ],
                    env,
                )
                with patch.dict(
                    sys.modules,
                    {
                        "flashinfer.autotuner": module,
                        "vllm.distributed.parallel_state": group_module,
                    },
                ):
                    self.assertTrue(
                        funcs["_run_flashinfer_sparse_mla_decode_autotune"](
                            worker, 16, frozenset()
                        )
                    )
                self.assertEqual(
                    contexts, [{"skip_ops": {"moe"}}, {"skip_ops": {"moe"}}]
                )
                self.assertEqual(env["_run_sparse_mla_refine_bucket"].call_count, 2)


if __name__ == "__main__":
    unittest.main()
