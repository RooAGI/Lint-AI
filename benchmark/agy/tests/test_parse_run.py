import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "parse_run.py"
SPEC = importlib.util.spec_from_file_location("agy_parse_run", MODULE_PATH)
parse_run = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(parse_run)


class ParseRunTests(unittest.TestCase):
    def test_extracts_gemini_usage_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agy.jsonl"
            path.write_text(
                json.dumps({
                    "usageMetadata": {
                        "promptTokenCount": 12,
                        "cachedContentTokenCount": 4,
                        "candidatesTokenCount": 7,
                        "totalTokenCount": 23,
                    }
                }) + "\n",
                encoding="utf-8",
            )
            metrics = parse_run.parse_agy_output(path)
            self.assertEqual(metrics["parent_tokens"]["input_tokens"], 12)
            self.assertEqual(metrics["parent_tokens"]["cache_read_input_tokens"], 4)
            self.assertEqual(metrics["parent_tokens"]["output_tokens"], 7)
            self.assertEqual(metrics["parent_tokens"]["total"], 23)

    def test_missing_usage_is_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agy.log"
            path.write_text("terminal output\n", encoding="utf-8")
            metrics = parse_run.parse_agy_output(path)
            self.assertIsNone(metrics["parent_tokens"]["total"])


if __name__ == "__main__":
    unittest.main()
