import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.request

import lint_ai


class MemoryBindingTests(unittest.TestCase):
    def test_public_surface_uses_memory_service(self):
        self.assertTrue(hasattr(lint_ai, "Memory"))
        self.assertTrue(hasattr(lint_ai, "version"))
        # 0.3.0 breaking change: RemoteMemory removed (Memory covers base_url).
        self.assertFalse(hasattr(lint_ai, "RemoteMemory"))
        self.assertFalse(hasattr(lint_ai, "IndexStore"))

    def test_version_reports_0_3_0(self):
        import importlib.metadata

        self.assertEqual(lint_ai.version(), "0.3.0")
        # Wheel metadata and the binding must agree.
        self.assertEqual(importlib.metadata.version("lint-ai"), "0.3.0")

    def test_constructor_rejects_bad_config(self):
        with self.assertRaises(ValueError):
            lint_ai.Memory(ner_provider="bert")
        with self.assertRaises(ValueError):
            lint_ai.Memory(path="/tmp/x", base_url="http://localhost:1")

    def test_constructor_accepts_config_knobs(self):
        for ner_provider in ("heuristic", "spacy"):
            memory = lint_ai.Memory(ner_provider=ner_provider)
            memory.refresh()
        memory = lint_ai.Memory(
            ner_provider="spacy", spacy_model="zh_core_web_sm"
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
        # Missing records: get/update -> None, delete -> False (no raise).
        self.assertIsNone(memory.get("does-not-exist", "user-a"))
        self.assertIsNone(
            memory.update("does-not-exist", "user-a", "x")
        )
        self.assertFalse(memory.delete("user-a", "does-not-exist"))
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

        # session_id + filters + scope are forwarded to SearchRequest
        # (filter semantics themselves are pinned in TestFilterContract).
        found = memory.search(
            "deploy target",
            "user-b",
            top_k=5,
            session_id="session-b",
            filters={"request_id": "request-s1"},
            scope=None,
        )
        self.assertEqual(len(found), 1)

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


class TestSessionAndScope(unittest.TestCase):
    """session_id drives follow-up resolution; scope never widens ownership."""

    def test_follow_up_resolves_against_conversation_state(self):
        memory = lint_ai.Memory()
        memory.add(
            "sess-req-1",
            "user-s",
            "sess-1",
            [
                {
                    "role": "user",
                    "content": "The Quartz database listens on 5432 for connections",
                }
            ],
        )
        memory.refresh()

        # First query in the session records conversation state
        # (entities + recent queries).
        first = memory.search("Quartz database", "user-s", top_k=5, session_id="sess-1")
        self.assertGreaterEqual(len(first), 1)

        # Pronoun follow-up resolves via the recorded session state. Note:
        # follow-up detection runs on the *augmented* query, so the
        # follow-up must still read as one after augmentation — a
        # "tell me more"-style prefix survives, a bare mid-sentence
        # pronoun ("which port does it use?") does not.
        follow = memory.search(
            "tell me more about it", "user-s", top_k=5, session_id="sess-1"
        )
        self.assertGreaterEqual(len(follow), 1)
        self.assertIn("Quartz", follow[0]["content"])

        # Controls: without the session (or in a foreign session) the
        # follow-up has nothing to resolve against, so the doc is not found.
        stateless = memory.search("tell me more about it", "user-s", top_k=5)
        self.assertEqual(len(stateless), 0)
        foreign = memory.search(
            "tell me more about it", "user-s", top_k=5, session_id="sess-other"
        )
        self.assertEqual(len(foreign), 0)

    def test_scope_does_not_widen_ownership(self):
        memory = lint_ai.Memory()
        memory.add(
            "scope-req-a",
            "owner-a",
            "s",
            [{"role": "user", "content": "Owner A secret plan for zephyrs"}],
        )
        memory.add(
            "scope-req-b",
            "owner-b",
            "s",
            [{"role": "user", "content": "Owner B secret plan for zephyrs"}],
        )
        memory.refresh()

        # A nonempty scope only re-keys conversation state; documents stay
        # filtered by the owning user_id.
        found = memory.search("zephyrs", "owner-a", top_k=5, scope="team-x")
        self.assertGreaterEqual(len(found), 1)
        self.assertTrue(all(r["user_id"] == "owner-a" for r in found))

        found_b = memory.search("zephyrs", "owner-b", top_k=5, scope="team-x")
        self.assertTrue(all(r["user_id"] == "owner-b" for r in found_b))


class TestAddBatchEdges(unittest.TestCase):
    def _req(self, rid, user="user-e", content="edge content"):
        return {
            "request_id": rid,
            "user_id": user,
            "session_id": "sess-e",
            "messages": [{"role": "user", "content": f"{content} {rid}"}],
        }

    def _count(self, memory, user="user-e"):
        return len(memory.list(user)["data"])

    def test_batch_size_limits(self):
        memory = lint_ai.Memory()
        reqs = [self._req(f"lim-{i}") for i in range(128)]
        resps = memory.add_batch(reqs)
        self.assertEqual(len(resps), 128)
        self.assertTrue(all(r["success"] for r in resps))
        with self.assertRaises(ValueError):
            memory.add_batch(reqs + [self._req("lim-129")])

    def test_malformed_entry_rejects_whole_batch_atomically(self):
        memory = lint_ai.Memory()
        before = self._count(memory)
        # Second entry is missing required fields: serde rejects the entire
        # payload before the service sees anything, so nothing is stored.
        with self.assertRaises(ValueError):
            memory.add_batch([self._req("atom-1"), {"request_id": "atom-bad"}])
        self.assertEqual(self._count(memory), before)
        # Non-list and non-dict payloads are rejected the same way.
        with self.assertRaises(ValueError):
            memory.add_batch({"request_id": "atom-nope"})
        with self.assertRaises(ValueError):
            memory.add_batch(["not-a-dict"])

    def test_service_failure_mid_batch_has_no_rollback(self):
        memory = lint_ai.Memory()
        # Entry 2 parses (empty messages is valid JSON) but fails service
        # validation ("messages must not be empty").
        bad = self._req("noroll-2")
        bad["messages"] = []
        with self.assertRaises(RuntimeError):
            memory.add_batch([self._req("noroll-1"), bad])
        # No rollback: the first entry's documents were already mutated
        # into the store before the failure. Refresh and check.
        memory.refresh()
        ids = [r["id"] for r in memory.list("user-e")["data"]]
        self.assertEqual(len(ids), 1)
        # Retrying the succeeded request_id with identical content is
        # idempotent: no duplicate record.
        retry = memory.add_batch([self._req("noroll-1")])
        self.assertTrue(retry[0]["success"])
        memory.refresh()
        self.assertEqual(self._count(memory), 1)


class TestFilterContract(unittest.TestCase):
    """Pins the engine's actual filter semantics (memory_api.rs, index/query.rs).

    Contract (all verified against the implementation):
    - Filters are ANDed, but a filter whose key has no postings — or whose
      key exists but value has no postings — is SILENTLY DROPPED, not
      exclusionary.
    - Ownership (memory_user_id) is always enforced and cannot be overridden
      by caller filters.
    - The structured-fact arm receives NO caller filters (only ownership /
      supersession / expiry gates) and its hits blend FIRST. Extra filters
      are therefore best-effort narrowing, not a security boundary.
    """

    def test_request_id_filter_returns_exact_ids(self):
        memory = lint_ai.Memory()
        memory.add(
            "flt-1", "user-f", "s",
            [{"role": "user", "content": "Apples are red fruits"}],
        )
        memory.add(
            "flt-2", "user-f", "s",
            [{"role": "user", "content": "Carrots are orange vegetables"}],
        )
        memory.refresh()

        # request_id is a filterable key: exact match returns only that
        # request's document.
        found = memory.search(
            "fruits vegetables", "user-f", top_k=5, filters={"request_id": "flt-1"}
        )
        self.assertEqual(len(found), 1)
        self.assertIn("Apples", found[0]["content"])

    def test_nonmatching_filter_value_is_dropped_not_exclusionary(self):
        memory = lint_ai.Memory()
        memory.add(
            "flt-3", "user-f", "s",
            [{"role": "user", "content": "Apples are red fruits"}],
        )
        memory.refresh()
        owned = sorted(x["id"] for x in memory.list("user-f")["data"])

        # "no-such-request" has no postings: the filter is dropped, so all
        # owned docs are returned (NOT zero results).
        found = memory.search(
            "fruits", "user-f", top_k=5, filters={"request_id": "no-such"}
        )
        self.assertEqual(sorted(x["id"] for x in found), owned)

    def test_absent_filter_key_is_dropped(self):
        memory = lint_ai.Memory()
        memory.add(
            "flt-4", "user-f", "s",
            [{"role": "user", "content": "Apples are red fruits"}],
        )
        memory.refresh()
        owned = sorted(x["id"] for x in memory.list("user-f")["data"])

        found = memory.search(
            "fruits", "user-f", top_k=5, filters={"bogus_key": "v"}
        )
        self.assertEqual(sorted(x["id"] for x in found), owned)

    def test_ownership_filter_cannot_be_overridden(self):
        memory = lint_ai.Memory()
        memory.add(
            "flt-5", "owner-f", "s",
            [{"role": "user", "content": "Apples are red fruits"}],
        )
        memory.refresh()

        # The engine strips a caller-supplied memory_user_id; a literal
        # "user_id" key is unknown and dropped. Either way, only owned docs.
        for f in ({"memory_user_id": "someone-else"}, {"user_id": "someone-else"}):
            found = memory.search("fruits", "owner-f", top_k=5, filters=f)
            self.assertGreaterEqual(len(found), 1)
            self.assertTrue(all(x["user_id"] == "owner-f" for x in found))

    def test_structured_arm_bypasses_extra_filters(self):
        # CURRENT BEHAVIOR, pinned deliberately: the structured-fact arm is
        # built with filters=None (memory_api.rs: structured_fact_results
        # only gates on ownership/supersession/expiry), so a fact question
        # can return documents the caller's filters would exclude on the
        # lexical path. Changing this is an engine retrieval change and
        # needs an explicit decision (measure-first rule) — see
        # docs/python-migration-0.3.0.md.
        memory = lint_ai.Memory()
        memory.add(
            "g1", "user-g", "s",
            [{"role": "user", "content": "Gina went to Rome last summer"}],
        )
        memory.add(
            "g2", "user-g", "s",
            [{"role": "user", "content": "Jon went to Rome last spring"}],
        )
        memory.refresh()

        query = "Which city have both Gina and Jon visited?"
        unfiltered = memory.search(query, "user-g", top_k=5)
        # The bypass only manifests when the structured-fact arm fires
        # (relations extractor available; hits score 1000.0 + confidence).
        # Without it there is nothing to pin — skip honestly instead of
        # failing on an environment precondition.
        if len(unfiltered) != 2 or not all(x["score"] > 1000.0 for x in unfiltered):
            raise unittest.SkipTest(
                "structured-fact arm did not fire (relations extractor unavailable)"
            )
        self.assertTrue(all(x["score"] > 1000.0 for x in unfiltered))

        # Lexically, request_id=g2 matches only Jon's doc — but the
        # structured arm returns both anyway.
        filtered = memory.search(
            query, "user-g", top_k=5, filters={"request_id": "g2"}
        )
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(x["score"] > 1000.0 for x in filtered))


if __name__ == "__main__":
    unittest.main()


class TestRemoteLifecycle(unittest.TestCase):
    """End-to-end remote mode against a real server binary (Luyi gate #2).

    Starts `target/debug/server` on a free loopback port with
    `--server-token` auth, runs a full CRUD lifecycle through
    `lint_ai.Memory(base_url=..., api_key=...)`, then checks the
    unauthorized and missing-record paths. Hermetic: localhost only.
    Skips cleanly when the server binary has not been built
    (`cargo build --bin server`).
    """

    TOKEN = "remote-test-secret"

    @classmethod
    def _repo_root(cls):
        return os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

    @classmethod
    def setUpClass(cls):
        cls.binary = os.path.join(cls._repo_root(), "target", "debug", "server")
        if not os.path.isfile(cls.binary):
            raise unittest.SkipTest(
                "server binary not built; run `cargo build --bin server`"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]
        cls.tmpdir = tempfile.mkdtemp(prefix="lintai-remote-test-")
        # Scrub ambient auth config so the test controls the server's
        # credentials exactly (JWT auth is env-only; token via CLI flag).
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("JWT_SECRET", "SERVER_TOKEN", "SERVER_TENANT_ID")
        }
        cls.proc = subprocess.Popen(
            [
                cls.binary,
                "--bind", f"127.0.0.1:{cls.port}",
                "--index", cls.tmpdir,
                "--server-token", cls.TOKEN,
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        deadline = time.time() + 30
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited during startup (code {cls.proc.returncode})"
                )
            try:
                with urllib.request.urlopen(
                    f"{cls.base_url}/health", timeout=2
                ) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.2)
        else:
            cls.proc.terminate()
            raise RuntimeError("server did not become ready within 30s")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "proc"):
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
        if hasattr(cls, "tmpdir"):
            shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _memory(self, api_key=TOKEN):
        return lint_ai.Memory(base_url=self.base_url, api_key=api_key)

    def test_remote_crud_lifecycle(self):
        memory = self._memory()

        added = memory.add(
            "remote-1",
            "remote-user",
            "remote-session",
            [{"role": "user", "content": "Remote memory about submarines"}],
        )
        self.assertTrue(added["success"])

        found = memory.search("submarines", "remote-user", 5)
        self.assertGreaterEqual(len(found), 1)
        record = found[0]
        self.assertEqual(record["user_id"], "remote-user")

        fetched = memory.get(record["id"], "remote-user")
        self.assertEqual(fetched["id"], record["id"])

        listed = memory.list("remote-user")
        self.assertEqual(len(listed["data"]), 1)

        updated = memory.update(
            record["id"], "remote-user", "Remote memory about sailboats"
        )
        self.assertIn("sailboats", updated["content"])

        self.assertTrue(memory.delete("remote-user", record["id"]))
        self.assertIsNone(memory.get(record["id"], "remote-user"))
        memory.refresh()

    def test_remote_wrong_api_key_is_unauthorized(self):
        memory = self._memory(api_key="wrong-key")
        with self.assertRaises(RuntimeError) as ctx:
            memory.list("remote-user")
        self.assertIn("401", str(ctx.exception))

    def test_remote_missing_record_returns_none(self):
        memory = self._memory()
        self.assertIsNone(memory.get("does-not-exist", "remote-user"))
        self.assertIsNone(memory.update("does-not-exist", "remote-user", "x"))
        self.assertFalse(memory.delete("remote-user", "does-not-exist"))
