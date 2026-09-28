"""Regression tests for scripts/behood_query.py question-phrase extraction.

The NeMo eval question "What is the user's weekend exercise routine?" used to
yield only the interrogative chunk "What" from spaCy noun_chunks, so bekind
never saw the sought phrase. These tests pin the dependency-parse fallback
that recovers the predicate nominal's full subtree. The bekind binary is not
required: tests run against the descriptor-building layer only.
"""

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from behood_query import _load_spacy, question_np_descriptors  # noqa: E402


class QuestionPhraseExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = _load_spacy()
        if cls.nlp is None:
            raise unittest.SkipTest("spaCy en_core_web_sm not available")

    def descriptors(self, question):
        return question_np_descriptors(self.nlp(question))

    def test_weekend_exercise_routine_reaches_bekind(self):
        ds = self.descriptors("What is the user's weekend exercise routine?")
        texts = [d["text"] for d in ds]
        self.assertTrue(
            any("weekend exercise routine" in t for t in texts),
            f"sought phrase missing from descriptors: {texts}",
        )
        self.assertFalse(
            all(t.strip().lower() == "what" for t in texts),
            f"lone 'What' is the only descriptor: {texts}",
        )

    def test_fallback_descriptor_shape_matches_chunk_descriptors(self):
        ds = self.descriptors("What is the user's weekend exercise routine?")
        fb = [d for d in ds if d["id"].startswith("q:fb")]
        self.assertEqual(len(fb), 1, f"expected one fallback descriptor: {ds}")
        for key in ("id", "text", "head_lemma", "head_pos", "ner_label", "modifiers"):
            self.assertIn(key, fb[0])

    def test_contentful_questions_do_not_get_fallback(self):
        # noun_chunks already yields content -> behavior unchanged.
        for q in (
            "Which city have both Jean and John visited?",
            "Who wrote Hamlet?",
            "What is the capital of France?",
        ):
            with self.subTest(question=q):
                ds = self.descriptors(q)
                self.assertFalse(
                    any(d["id"].startswith("q:fb") for d in ds),
                    f"unexpected fallback descriptor for {q!r}: {ds}",
                )

    def test_temporal_question_word_handling_intact(self):
        # "what time" is contentful for noun_chunks; no fallback injected.
        ds = self.descriptors("What time is the meeting?")
        self.assertFalse(any(d["id"].startswith("q:fb") for d in ds))


if __name__ == "__main__":
    unittest.main()
