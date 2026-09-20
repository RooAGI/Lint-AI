import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "parse_run.py"
SPEC = importlib.util.spec_from_file_location("muse_parse_run", MODULE_PATH)
parse_run = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(parse_run)


def _record(payload_type, payload):
    return json.dumps({"payload_type": payload_type, "payload": payload})


ECHO_LOG = "\n".join(
    [
        _record(
            "turn.input.user",
            {"kind": "turn_input_user", "prompt": "Say hello"},
        ),
        _record(
            "run.output.delta",
            {"kind": "run_output_delta", "text": "echo: Say hel"},
        ),
        _record(
            "run.output.delta",
            {"kind": "run_output_delta", "text": "echo: Say hello"},
        ),
        _record(
            "run.terminal.completed",
            {"kind": "run_terminal", "terminal": "completed", "text": "echo: Say hello"},
        ),
        "not json at all",
    ]
)


class ParseMuseOutputTests(unittest.TestCase):
    def test_extracts_terminal_text_not_deltas(self):
        self.assertEqual(
            parse_run.extract_muse_last_message(ECHO_LOG), "echo: Say hello"
        )

    def test_extract_returns_none_without_terminal_record(self):
        self.assertIsNone(parse_run.extract_muse_last_message("not json\n"))

    def test_parse_counts_records_and_no_tool_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            path.write_text(ECHO_LOG, encoding="utf-8")
            metrics = parse_run.parse_muse_output(path)
        self.assertEqual(metrics["record_count"], 4)
        self.assertEqual(metrics["tool_calls"], 0)
        self.assertEqual(metrics["parent_tokens"], {})
        self.assertEqual(metrics["injected_context_bytes"], 0)

    def test_tool_like_records_deduped_by_task_id(self):
        lines = [
            _record(
                "task.lifecycle.proposed",
                {"task_kind": "tool.shell", "task_id": "task-1"},
            ),
            _record(
                "task.lifecycle.completed",
                {"task_kind": "tool.shell", "task_id": "task-1"},
            ),
            _record(
                "task.lifecycle.proposed",
                {"task_kind": "tool.read", "task_id": "task-2"},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            path.write_text("\n".join(lines), encoding="utf-8")
            metrics = parse_run.parse_muse_output(path)
        self.assertEqual(metrics["tool_calls"], 2)


if __name__ == "__main__":
    unittest.main()
