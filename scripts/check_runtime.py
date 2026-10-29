#!/usr/bin/env python3
"""Read-only preflight inside a built container; does not load the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import patch_sm120


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/models/dsv41-flash"))
    parser.add_argument("--expect-gpus", type=int, default=4)
    args = parser.parse_args()
    patch_sm120.verify_bundle(patch_sm120.installed_root(), REPO / "patches")
    subprocess.run(
        [sys.executable, str(REPO / "scripts/install_dependencies.py"), "--verify"],
        check=True,
    )
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < args.expect_gpus:
        raise SystemExit(f"Need at least {args.expect_gpus} visible NVIDIA GPUs")
    devices = []
    for i in range(args.expect_gpus):
        capability = torch.cuda.get_device_capability(i)
        if capability != (12, 0):
            raise SystemExit(f"GPU {i} is {capability}; this deployment targets SM120")
        properties = torch.cuda.get_device_properties(i)
        devices.append(
            {
                "index": i,
                "name": properties.name,
                "memory_GiB": round(properties.total_memory / 2**30, 2),
                "capability": list(capability),
            }
        )
    config = json.loads((args.model_path / "config.json").read_text())
    if config.get("model_type") != "deepseek_v41" and not any(
        name.startswith("DeepseekV41") for name in config.get("architectures", ())
    ):
        raise SystemExit("Model config is not DeepSeek V4.1")
    index_path = args.model_path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        missing = [
            name
            for name in set(weight_map.values())
            if not (args.model_path / name).is_file()
        ]
        if missing:
            raise SystemExit(f"Missing weight shards: {missing[:5]}")
    elif not (args.model_path / "model.safetensors").is_file():
        raise SystemExit("No local safetensors weights/index found")
    from vllm.utils.flashinfer import (
        has_flashinfer_sparse_mla_sm120_config,
        has_flashinfer_sparse_mla_sm120_runtime_page_size,
    )
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    if not has_flashinfer_sparse_mla_sm120_runtime_page_size():
        raise SystemExit("FlashInfer runtime page-size support is missing")
    for topk in (128, 192):
        if not has_flashinfer_sparse_mla_sm120_config(16, topk):
            raise SystemExit(f"FlashInfer lacks the TP4 sparse MLA shape (16, {topk})")
    if not is_deep_gemm_supported():
        raise SystemExit("The base image's DeepGEMM is unavailable on this GPU/toolkit")
    print(
        json.dumps(
            {
                "preflight": "passed",
                "devices": devices,
                "model": str(args.model_path),
                "model_loaded": False,
                "note": "Actual KV capacity and 512K admission are checked when vLLM loads the model.",
            }
        )
    )


if __name__ == "__main__":
    main()
