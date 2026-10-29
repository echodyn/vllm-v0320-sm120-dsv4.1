import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "acceptance", REPO / "scripts/acceptance.py"
)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class AcceptanceTests(unittest.TestCase):
    def test_sse_requires_done_and_propagates_error(self):
        chunks = api.parse_sse(
            [b": keepalive\n", b'data: {"choices":[]}\n', b"\n", b"data: [DONE]\n"]
        )
        self.assertEqual(chunks, [{"choices": []}])
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            api.parse_sse([b'data: {"choices":[]}\n'])
        with self.assertRaisesRegex(RuntimeError, "failed"):
            api.parse_sse([b'data: {"error":"failed"}\n'])

    def test_prefix_metric_counts_labels_and_ignores_creation_time(self):
        text = '# HELP\nvllm:prefix_cache_hits_total{engine="0"} 8\nvllm:prefix_cache_hits_total{engine="1"} 3\nvllm:prefix_cache_hits_created 100000\n'
        self.assertEqual(api.cache_hits(text), 11)
        with self.assertRaises(RuntimeError):
            api.cache_hits("vllm:unrelated_total 5\n")

    def test_reasoning_only_or_empty_response_does_not_pass_chat(self):
        client = api.Client("http://127.0.0.1:8000", "test")
        client.json = Mock(
            return_value={
                "choices": [
                    {
                        "message": {"reasoning": "thinking", "content": None},
                        "finish_reason": "length",
                    }
                ]
            }
        )
        with self.assertRaises(RuntimeError):
            client.chat("test")
        client.json.return_value = {
            "choices": [{"message": {"content": None, "tool_calls": [{"id": "1"}]}}]
        }
        self.assertTrue(client.chat("test"))

    def test_long_prompt_fits_token_target_and_retains_both_markers(self):
        client = Mock(token_count=lambda text: 32 + len(text) // 3)
        prompt, count = api.long_prompt(client, 4096)
        self.assertLessEqual(count, 4096)
        self.assertGreaterEqual(count, 3840)
        self.assertIn("739251", prompt[:100])
        self.assertIn("六位校验码", prompt[-100:])

    def test_bulk_probes_disable_thinking_and_reasoning_probe_can_override(self):
        client = api.Client("http://localhost:8000", "test")
        self.assertEqual(
            client.request("hi")["chat_template_kwargs"], {"thinking": False}
        )
        self.assertEqual(
            client.request("hi", chat_template_kwargs={"thinking": True})[
                "chat_template_kwargs"
            ],
            {"thinking": True},
        )


if __name__ == "__main__":
    unittest.main()
