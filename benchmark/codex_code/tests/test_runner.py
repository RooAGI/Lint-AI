import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).parents[1] / "src" / "runner.py"
SPEC = importlib.util.spec_from_file_location("codex_runner", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_discovers_and_validates_scenarios(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenarios = root / "scenarios"
            scenarios.mkdir()
            scenario_path = scenarios / "smoke.json"
            scenario_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "id": "smoke",
                        "category": "decision-recall",
                        "description": "Smoke test",
                        "repository": {"path": ".", "revision": "HEAD"},
                        "setup_messages": [{"prompt": "remember", "establishes_fact_ids": []}],
                        "continuation_prompt": "what happened?",
                        "expected_facts": [],
                        "forbidden_facts": [],
                        "validators": [],
                        "limits": {
                            "timeout_seconds": 1,
                            "max_turns": 1,
                            "max_budget_usd": 0.1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            discovered = runner.discover_scenarios(root)
            self.assertEqual(discovered, [scenario_path])
            self.assertEqual(runner.validate_scenario(runner.load_scenario(scenario_path)), [])
            plan = runner.build_run_plan(discovered, 3)
            self.assertEqual(plan.repetitions, 3)
            self.assertEqual(plan.scenarios[0].id, "smoke")


if __name__ == "__main__":
    unittest.main()
