import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from metrics import canonical_metrics, normalize_tokens  # noqa: E402


def test_normalize_google_usage_metadata():
    tokens = normalize_tokens(
        {
            "promptTokenCount": 12,
            "cachedContentTokenCount": 4,
            "candidatesTokenCount": 7,
            "totalTokenCount": 23,
        }
    )
    # Unknown provider-specific names must remain unavailable rather than be
    # confused with the Claude/Codex accounting fields.
    assert tokens["total"] == 23
    assert tokens["input_tokens"] is None


def test_canonical_metrics_preserves_missing_usage():
    metrics = canonical_metrics({"parent_tokens": {"inputTokens": 10}}, provider="agy")
    assert metrics["provider"] == "agy"
    assert metrics["parent_tokens"]["input_tokens"] == 10
    assert metrics["parent_tokens"]["output_tokens"] is None
    assert metrics["parent_tokens"]["total"] is None
