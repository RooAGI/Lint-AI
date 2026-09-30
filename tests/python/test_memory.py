import unittest

import lint_ai


class MemoryBindingTests(unittest.TestCase):
    def test_public_surface_uses_memory_service(self):
        self.assertTrue(hasattr(lint_ai, "Memory"))
        self.assertTrue(hasattr(lint_ai, "version"))
        # 0.3.0 breaking change: RemoteMemory removed (Memory covers base_url).
        self.assertFalse(hasattr(lint_ai, "RemoteMemory"))
        self.assertFalse(hasattr(lint_ai, "IndexStore"))

    def test_constructor_rejects_bad_config(self):
        with self.assertRaises(ValueError):
            lint_ai.Memory(language="xx")
        with self.assertRaises(ValueError):
            lint_ai.Memory(ner_provider="bert")
        with self.assertRaises(ValueError):
            lint_ai.Memory(path="/tmp/x", base_url="http://localhost:1")

    def test_constructor_accepts_config_knobs(self):
        memory = lint_ai.Memory(language="en", ner_provider="heuristic")
        memory.refresh()
        memory = lint_ai.Memory(
            language="zh", ner_provider="spacy", spacy_model="zh_core_web_sm"
        )
        memory.refresh()

    def test_memory_lifecycle(self):
        memory = lint_ai.Memory()

        added = memory.add(
            "request-1",
            "user-a",
            "session-a",
            [{"role": "user", "content": "I prefer dark mode in every editor"}],
        )
        self.assertTrue(added["success"])

        listed = memory.list("user-a")
        self.assertEqual(len(listed["data"]), 1)
        record = listed["data"][0]
        self.assertEqual(record["user_id"], "user-a")
        self.assertEqual(record["role"], "user")

        found = memory.search("dark mode editor", "user-a", 5)
        self.assertGreaterEqual(len(found), 1)
        self.assertEqual(found[0]["user_id"], "user-a")

        fetched = memory.get(record["id"], "user-a")
        self.assertEqual(fetched["id"], record["id"])

        updated = memory.update(
            record["id"], "user-a", "I prefer light mode in every editor"
        )
        self.assertIn("light mode", updated["content"])

        self.assertTrue(memory.delete("user-a", record["id"]))
        self.assertIsNone(memory.get(record["id"], "user-a"))
        memory.refresh()

    def test_search_with_session_and_filters(self):
        memory = lint_ai.Memory()
        memory.add(
            "request-s1",
            "user-b",
            "session-b",
            [{"role": "user", "content": "The deploy target is production us-west"}],
        )
        memory.refresh()

        # session_id + filters + scope are forwarded to SearchRequest.
        found = memory.search(
            "deploy target",
            "user-b",
            top_k=5,
            session_id="session-b",
            filters={},
            scope=None,
        )
        self.assertGreaterEqual(len(found), 1)

        # Unknown filter keys follow engine semantics (ignored when the
        # ownership filter matches); the call must still succeed and
        # return the owned doc.
        found_unfiltered = memory.search(
            "deploy target", "user-b", top_k=5, filters={"provider": "other"}
        )
        self.assertGreaterEqual(len(found_unfiltered), 1)

    def test_add_batch_lifecycle(self):
        memory = lint_ai.Memory()
        responses = memory.add_batch(
            [
                {
                    "request_id": "batch-1",
                    "user_id": "user-c",
                    "session_id": "session-c",
                    "messages": [
                        {"role": "user", "content": "Batch memory alpha about penguins"}
                    ],
                },
                {
                    "request_id": "batch-2",
                    "user_id": "user-c",
                    "session_id": "session-c",
                    "messages": [
                        {"role": "user", "content": "Batch memory beta about walruses"}
                    ],
                },
            ]
        )
        self.assertEqual(len(responses), 2)
        self.assertTrue(all(r["success"] for r in responses))
        memory.refresh()

        found = memory.search("penguins walruses", "user-c", top_k=5)
        self.assertGreaterEqual(len(found), 2)

        with self.assertRaises(ValueError):
            memory.add_batch([])


if __name__ == "__main__":
    unittest.main()
