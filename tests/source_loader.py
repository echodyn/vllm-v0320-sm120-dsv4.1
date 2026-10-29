"""Execute selected real source definitions without importing CUDA/vLLM.

Tests provide the omitted runtime dependencies explicitly. GPU launches are
stubbed; these tests make no assertion about CUDA numerics or performance.
"""

import ast
import copy
import os
from pathlib import Path

ROOT = Path(os.environ.get("VLLM_BACKPORT_TEST_ROOT", ".cache/upstream/patched/vllm"))


def tree(path):
    return ast.parse((ROOT / path.removeprefix("vllm/")).read_text())


def load(path, names, env=None, owner=None):
    module = tree(path)
    nodes = module.body
    if owner:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    selected = []
    for name in names:
        node = copy.deepcopy(next(n for n in nodes if getattr(n, "name", None) == name))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
        selected.append(node)
    scope = dict(env or {})
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), path, "exec"), scope)
    return scope
