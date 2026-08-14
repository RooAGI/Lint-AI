import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def _load(module_file: str, module_name: str):
    module_path = Path(__file__).parents[1] / "src" / module_file
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scorer = _load("scorer.py", "codex_scorer_test")
report = _load("report.py", "codex_report_test")


class ScorerReportTests(unittest.TestCase):
    def test_scores_and_summarizes_results(self):
        scenario = {
            "expected_facts": [
                {"id": "codex-tool-hooks", "match_any": ["PreToolUse", "PostToolUse"]},
            ],
            "forbidden_facts": [
                {"id": "tool-hooks-ignored", "match_any": ["unsupported or ignored"]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuation_last = root / "continuation.last"
            continuation_last.write_text(
                "PreToolUse and PostToolUse keep retrieval bounded.\n",
                encoding="utf-8",
            )
            continuation_metrics = root / "continuation.metrics.json"
            continuation_metrics.write_text(
                """{
  "parent_tokens": {
    "input_tokens": 10,
    "cache_creation_input_tokens": 2,
    "cache_read_input_tokens": 20,
    "output_tokens": 3,
    "total": 35
  },
  "all_model_tokens": {
    "input_tokens": 10,
    "cache_creation_input_tokens": 2,
    "cache_read_input_tokens": 20,
    "output_tokens": 3,
    "total": 35
  },
  "tool_calls": 2,
  "repeated_tool_calls": 1
}
""",
                encoding="utf-8",
            )
            result = {
                "scenario_id": "codex-tool-use-retrieval",
                "arm": "lint-ai",
                "repetition": 2,
                "success": True,
                "validators": [{"name": "validator-a"}],
                "continuation": {
                    "last_path": str(continuation_last),
                    "metrics_path": str(continuation_metrics),
                },
            }
            score = scorer.score_result(result, scenario)
            self.assertEqual(score.scenario_id, "codex-tool-use-retrieval")
            self.assertEqual(score.arm, "lint-ai")
            self.assertEqual(score.validator_count, 1)
            self.assertEqual(score.expected_fact_ids_found, ["codex-tool-hooks"])
            self.assertEqual(score.forbidden_fact_ids_found, [])
            summary = report.summarize_results(
                [result], [score.__dict__], {"codex-tool-use-retrieval": 1}
            )
        self.assertEqual(summary["result_count"], 1)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["scenarios"]["codex-tool-use-retrieval"]["lint-ai"], 1)
        self.assertEqual(summary["scores"][0]["scenario_id"], "codex-tool-use-retrieval")
        self.assertEqual(summary["final_results"]["status"], "passed")
        self.assertEqual(summary["final_results"]["successful_runs"], 1)
        self.assertEqual(summary["final_results"]["scenarios"][0]["input_tokens"], 30)
        self.assertIsNone(
            summary["final_results"]["scenarios"][0]["interaction_round_latency_ms"]
        )
        self.assertEqual(summary["final_results"]["scenarios"][0]["repeated_tool_calls"], 1)
        self.assertTrue(
            summary["final_results"]["comparison_note"].startswith(
                "This report contains only the lint-ai arm"
            )
        )
        self.assertEqual(
            summary["final_results"]["scenarios"][0]["expected_fact_ids_found"],
            ["codex-tool-hooks"],
        )


if __name__ == "__main__":
    unittest.main()
