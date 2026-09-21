import unittest

import lint_ai


class MemoryBindingTests(unittest.TestCase):
    def test_public_surface_uses_memory_service(self):
        self.assertTrue(hasattr(lint_ai, "Memory"))
        self.assertTrue(hasattr(lint_ai, "RemoteMemory"))
        self.assertFalse(hasattr(lint_ai, "IndexStore"))

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


if __name__ == "__main__":
    unittest.main()
