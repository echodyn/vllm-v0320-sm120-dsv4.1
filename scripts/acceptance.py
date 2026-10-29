#!/usr/bin/env python3
"""Explicit server acceptance: chat/tools/stream/APC, near-512K, 48-way mixed load.

Run this yourself after starting the patched server. It never downloads weights
or starts containers. Default full mode sends a near-512K request and 48 concurrent
requests; --smoke-only skips those two workloads. No cache-reset endpoint is used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time
import urllib.request


def parse_sse(lines):
    chunks = []
    done = False
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            break
        item = json.loads(payload)
        if "error" in item:
            raise RuntimeError(item["error"])
        chunks.append(item)
    if not done or not chunks:
        raise RuntimeError("Incomplete/empty SSE stream")
    return chunks


def cache_hits(metrics):
    result = 0.0
    found = False
    for line in metrics.splitlines():
        if line.startswith("#"):
            continue
        if re.match(r"^vllm:prefix_cache_hits(?:_total)?(?:\{|\s)", line):
            result += float(line.split()[-1])
            found = True
    if not found:
        raise RuntimeError("No prefix_cache_hits metric is exposed")
    return result


class Client:
    def __init__(self, url, model, timeout=7200):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def open(self, path, body=None):
        headers = {"Content-Type": "application/json"}
        if key := os.environ.get("VLLM_API_KEY"):
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(
            self.url + path,
            data=None
            if body is None
            else json.dumps(body, ensure_ascii=False).encode(),
            headers=headers,
        )
        return urllib.request.urlopen(request, timeout=self.timeout)

    def json(self, path, body=None):
        with self.open(path, body) as response:
            item = json.load(response)
        if isinstance(item, dict) and "error" in item:
            raise RuntimeError(item["error"])
        return item

    def metrics(self):
        with self.open("/metrics") as response:
            return response.read().decode()

    def request(self, prompt, **extra):
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 1024,
            "chat_template_kwargs": {"thinking": False},
            **extra,
        }

    def chat(self, prompt, **extra):
        item = self.json("/v1/chat/completions", self.request(prompt, **extra))
        choice = item["choices"][0]
        message = choice["message"]
        if not (message.get("content") or message.get("tool_calls")):
            raise RuntimeError(
                f"No final assistant content/tool call: finish_reason={choice.get('finish_reason')}"
            )
        return item

    def token_count(self, prompt):
        item = self.json(
            "/tokenize",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "add_generation_prompt": True,
                "chat_template_kwargs": {"thinking": False},
            },
        )
        return item["count"] if "count" in item else len(item["tokens"])


def long_prompt(client, target):
    prefix = "记住文档开头的校验码：739251。\n"
    unit = "背景资料仅供占位，请继续阅读。\n"
    suffix = "\n文档已经结束。请只回答开头的六位校验码。"
    n = max(1, target // max(1, client.token_count(unit)))
    for _ in range(8):
        text = prefix + unit * n + suffix
        count = client.token_count(text)
        if target - 256 <= count <= target:
            return text, count
        n = max(1, int(n * (target - 128) / max(count, 1)))
    raise RuntimeError(
        f"Cannot fit long-context prompt to target {target}; last count {count}"
    )


def run(client, args):
    report = {
        "base_url": client.url,
        "model": client.model,
        "checks": {},
        "gpu_numerics_benchmark": False,
    }

    def check(name, fn):
        started = time.monotonic()
        result = fn()
        report["checks"][name] = {
            "passed": True,
            "seconds": round(time.monotonic() - started, 3),
            **(result or {}),
        }
        print(f"PASS {name}", flush=True)

    def health():
        with client.open("/health") as response:
            assert response.status == 200
        models = client.json("/v1/models")["data"]
        assert any(item["id"] == client.model for item in models), (
            "served model name mismatch"
        )

    check("health_and_models", health)

    def chinese():
        item = client.chat("请用中文只回答：中国的首都是哪座城市？")
        assert "北京" in item["choices"][0]["message"]["content"]

    check("chinese", chinese)

    def arithmetic():
        item = client.chat("17乘以19等于多少？请只回答计算结果。")
        text = item["choices"][0]["message"]["content"]
        assert re.search(r"(?<!\d)323(?!\d)", text), text

    check("arithmetic", arithmetic)

    def reasoning():
        item = client.chat(
            "请简短思考后回答：2加2等于多少？",
            chat_template_kwargs={"thinking": True},
            max_tokens=2048,
        )
        message = item["choices"][0]["message"]
        assert message.get("reasoning") or message.get("reasoning_content"), (
            "No parsed reasoning field"
        )
        assert "4" in message["content"] or "四" in message["content"]

    check("reasoning_parser", reasoning)

    def streaming():
        with client.open(
            "/v1/chat/completions",
            client.request("请用中文简短解释什么是张量并行。", stream=True),
        ) as response:
            chunks = parse_sse(response)
        assert any(
            choice.get("finish_reason")
            for item in chunks
            for choice in item.get("choices", [])
        )
        assert any(
            choice.get("delta", {}).get("content")
            for item in chunks
            for choice in item.get("choices", [])
        )
        return {"chunks": len(chunks)}

    check("streaming", streaming)

    def tools():
        item = client.chat(
            "请调用 get_weather 查询上海的天气，不要编造天气。",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "查询城市天气",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        call = item["choices"][0]["message"]["tool_calls"][0]["function"]
        assert call["name"] == "get_weather"
        assert json.loads(call["arguments"])["city"] in ("上海", "Shanghai", "上海市")

    check("tool_call", tools)

    def prefix():
        prompt = "以下重复内容用于前缀缓存测试。" * 256 + "\n请只回答：缓存测试完成。"
        client.chat(prompt, max_tokens=256)
        before = cache_hits(client.metrics())
        client.chat(prompt, max_tokens=256)
        after = cache_hits(client.metrics())
        assert after > before, "No prefix cache hit increase on the repeated request"
        return {"hit_delta": after - before}

    check("prefix_cache", prefix)

    if not args.smoke_only:

        def long_context():
            prompt, count = long_prompt(client, args.long_context_tokens)
            item = client.chat(prompt, max_tokens=1024)
            assert "739251" in item["choices"][0]["message"]["content"]
            return {"input_tokens": count, "target_tokens": args.long_context_tokens}

        check("long_context", long_context)

        def concurrency():
            def request(i):
                # Long inputs arrive while other requests are generating tokens.
                prompt = (
                    (("并发预填充的背景材料。" * 1024) + "\n请简要概括这段背景材料。")
                    if i % 4 == 0
                    else (f"这是第{i}个测试。请连续列出从1到100的整数，以逗号分隔。")
                )
                return client.chat(prompt, max_tokens=256)

            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(request, i) for i in range(args.concurrency)]
                for future in as_completed(futures):
                    future.result()
            return {"requests": args.concurrency, "mixed_prefill_decode": True}

        check("concurrent_mixed_requests", concurrency)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="dsv41-flash-1017")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--long-context-tokens", type=int, default=522240)
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("acceptance-report.json"))
    args = parser.parse_args()
    if (
        not 1024 <= args.long_context_tokens <= 522240
        or not 1 <= args.concurrency <= 48
    ):
        parser.error("Use 1024..522240 input tokens and 1..48 concurrent requests")
    try:
        report = run(Client(args.base_url, args.model, args.timeout), args)
    except Exception as exc:
        args.output.write_text(
            json.dumps(
                {"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2
            )
            + "\n"
        )
        raise SystemExit(f"FAIL: {exc}") from exc
    report["passed"] = True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
