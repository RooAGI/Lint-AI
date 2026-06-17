import unittest

import lint_ai


class IndexStoreBindingTests(unittest.TestCase):
    def test_upsert_query_and_remove_lifecycle(self):
        store = lint_ai.IndexStore()

        self.assertTrue(store.is_empty())
        self.assertEqual(store.len(), 0)
        self.assertFalse(store.is_dirty())

        store.upsert("doc-1", "Docker install guide for Ubuntu hosts")

        self.assertFalse(store.is_empty())
        self.assertEqual(store.len(), 1)
        self.assertTrue(store.is_dirty())

        results = store.query("docker ubuntu", 5)

        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "doc-1")
        self.assertFalse(store.is_dirty())

        self.assertTrue(store.remove("doc-1"))
        self.assertTrue(store.is_empty())
        self.assertEqual(store.len(), 0)
        self.assertTrue(store.is_dirty())
        self.assertFalse(store.remove("doc-1"))


if __name__ == "__main__":
    unittest.main()
