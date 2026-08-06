import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "parse_run.py"
SPEC = importlib.util.spec_from_file_location("parse_run", MODULE_PATH)
parse_run = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(parse_run)


class ParseRunTests(unittest.TestCase):
    def test_extracts_delegation_tokens_and_context(self):
        result = {
            "usage": {
                "input_tokens": 2,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 20,
                "output_tokens": 3,
            },
            "modelUsage": {
                "codex": {
                    "inputTokens": 5,
                    "cacheCreationInputTokens": 30,
                    "cacheReadInputTokens": 60,
                    "outputTokens": 10,
                }
            },
        }
        transcript = [
            {
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Agent",
                            "input": {
                                "description": "Verify architecture",
                                "subagent_type": "Explore",
                                "prompt": "private prompt",
                            },
                        }
                    ]
                }
            },
            {
                "message": {
                    "content": [{"type": "tool_result", "tool_use_id": "tool-1"}]
                },
                "toolUseResult": {
                    "agentId": "agent-1",
                    "resolvedModel": "codex-sonnet",
                },
            },
            {
                "attachment": {
                    "type": "hook_additional_context",
                    "content": [
                        "Memory\n- Source: one\nRevision status: exact-match\n- Source: two"
                    ],
                }
            },
            {"attachment": {"type": "hook_success", "durationMs": 12}},
        ]

        metrics = self._parse(result, transcript)

        self.assertEqual(metrics["parent_tokens"]["total"], 35)
        self.assertEqual(metrics["all_model_tokens"]["total"], 105)
        self.assertEqual(metrics["subagent_tokens"]["total"], 70)
        self.assertEqual(metrics["subagent_count"], 1)
        self.assertEqual(metrics["delegations"][0]["purpose"], "Verify architecture")
        self.assertEqual(metrics["delegations"][0]["agent_id"], "agent-1")
        self.assertEqual(
            metrics["delegations"][0]["resolved_model"], "codex-sonnet"
        )
        self.assertNotIn("private prompt", json.dumps(metrics))
        self.assertEqual(metrics["retrieved_documents"], 2)
        self.assertEqual(metrics["exact_revision_memories"], 1)
        self.assertEqual(metrics["hook_latency_ms"], 12)

    def test_missing_token_telemetry_is_null(self):
        metrics = self._parse({}, [])

        self.assertIsNone(metrics["parent_tokens"]["total"])
        self.assertIsNone(metrics["all_model_tokens"]["total"])
        self.assertIsNone(metrics["subagent_tokens"])
        self.assertIsNone(metrics["unattributed_non_parent_tokens"]["total"])

    def test_parses_codex_exec_jsonl_into_shared_metrics(self):
        events = [
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "command": "rg --files"},
            },
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "command": "rg --files"},
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 60,
                    "cache_write_input_tokens": 5,
                    "output_tokens": 10,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "codex.jsonl"
            log_path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
            metrics = parse_run.parse_codex_exec_log(log_path)

        self.assertEqual(metrics["parent_tokens"]["input_tokens"], 40)
        self.assertEqual(metrics["parent_tokens"]["cache_read_input_tokens"], 60)
        self.assertEqual(metrics["parent_tokens"]["cache_creation_input_tokens"], 5)
        self.assertEqual(metrics["parent_tokens"]["output_tokens"], 10)
        self.assertEqual(metrics["parent_tokens"]["total"], 115)
        self.assertEqual(metrics["tool_calls"], 2)
        self.assertEqual(metrics["repeated_tool_calls"], 1)

    def _parse(self, result, transcript):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            transcript_path = root / "transcript.jsonl"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            transcript_path.write_text(
                "\n".join(json.dumps(event) for event in transcript), encoding="utf-8"
            )
            return parse_run.parse_run(result_path, transcript_path)


if __name__ == "__main__":
    unittest.main()
