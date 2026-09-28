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

from behood_query import _load_spacy, analyze_scope, question_np_descriptors  # noqa: E402


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


class UnchunkedNominalRecoveryTests(unittest.TestCase):
    """spaCy mis-tags "cilantro" ADV in the mem-08 fact, so noun_chunks
    drops it. Recovery is by nominal dependency slot, not per-question:
    any uncovered nominal-slot token becomes a descriptor."""

    @classmethod
    def setUpClass(cls):
        cls.nlp = _load_spacy()
        if cls.nlp is None:
            raise unittest.SkipTest("spaCy en_core_web_sm not available")

    def test_cilantro_recovered_from_mem08_fact(self):
        ds = question_np_descriptors(
            self.nlp("The user dislikes cilantro and always asks for it to be left out.")
        )
        texts = [d["text"] for d in ds]
        self.assertIn("cilantro", texts, f"cilantro not recovered: {texts}")

    def test_recovery_does_not_emit_verbs(self):
        # "asks" is a conj VERB in the mem-08 parse: must not leak in.
        ds = question_np_descriptors(
            self.nlp("The user dislikes cilantro and always asks for it to be left out.")
        )
        for d in ds:
            self.assertNotIn(
                d["head_pos"], ("VERB", "AUX"), f"verb leaked into descriptors: {d}"
            )

    def test_recovered_descriptor_shape(self):
        ds = question_np_descriptors(
            self.nlp("The user dislikes cilantro and always asks for it to be left out.")
        )
        rec = [d for d in ds if d["id"].startswith("q:rec")]
        self.assertTrue(rec, f"no recovered descriptors: {ds}")
        for key in ("id", "text", "head_lemma", "head_pos", "ner_label", "modifiers"):
            self.assertIn(key, rec[0])

    def test_no_duplicates_when_chunk_covers_token(self):
        # "it" is chunked; recovery must not duplicate it.
        ds = question_np_descriptors(
            self.nlp("The user dislikes cilantro and always asks for it to be left out.")
        )
        texts = [d["text"] for d in ds]
        self.assertEqual(texts.count("it"), 1, f"duplicate descriptors: {texts}")

    def test_clean_sentence_gains_no_junk(self):
        # Chunks cover everything here: recovery adds nothing.
        ds = question_np_descriptors(self.nlp("The cat sat on the mat."))
        texts = sorted(d["text"] for d in ds)
        self.assertEqual(texts, ["The cat", "the mat"], f"unexpected descriptors: {texts}")


class ScopeVerdictWiringTests(unittest.TestCase):
    """analyze_scope is additive and fail-open; no bekind binary needed."""

    def test_fail_open_without_binary(self):
        # Deterministic: a missing binary path must fail open to [].
        self.assertEqual(analyze_scope(["weekend routine"], binary="/nonexistent"), [])

    def test_empty_input_returns_empty(self):
        # Empty input short-circuits before touching the binary.
        self.assertEqual(analyze_scope([], binary="/nonexistent"), [])


if __name__ == "__main__":
    unittest.main()
