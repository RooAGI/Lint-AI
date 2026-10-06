"""Unit tests for the hermes-plugin-lintai directory plugin.

Stdlib unittest only — no dependencies. Run from the plugin directory:
    python3 -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import __init__ as plugin
from __init__ import (
    LintaiPlugin,
    LintaiClient,
    WriteQueue,
    build_session_record,
    build_subagent_start_record,
    build_subagent_stop_record,
    build_tool_record,
    build_turn_record,
    format_recall_context,
    load_config,
    truncate,
)


class FakeClient:
    def __init__(self, hits=None):
        self.hits = hits if hits is not None else []
        self.searches = []
        self.batches = []

    def search(self, query, top_k=5):
        self.searches.append((query, top_k))
        return self.hits

    def add_batch(self, requests):
        self.batches.append(list(requests))
        return {"ok": True}


class FakeCtx:
    def __init__(self):
        self.hooks = {}

    def register_hook(self, name, fn):
        self.hooks.setdefault(name, []).append(fn)


class LintaiClientRouteTest(unittest.TestCase):
    def test_uses_provider_memory_http_routes(self):
        client = LintaiClient("http://127.0.0.1:8080", "hermes")
        client._post = Mock(side_effect=[{"data": [{"content": "hit"}]}, {"ok": True}])

        self.assertEqual(client.search("needle", top_k=3), [{"content": "hit"}])
        self.assertEqual(client._post.call_args_list[0].args[0], "/provider-memory/search")
        self.assertEqual(client._post.call_args_list[0].args[1]["user_id"], "hermes")

        batch = [{"request_id": "r1", "messages": [], "session_id": "s1"}]
        client.add_batch(batch)
        self.assertEqual(client._post.call_args_list[1].args[0], "/provider-memory/add/batch")
        self.assertEqual(batch[0]["user_id"], "hermes")


class TruncateTest(unittest.TestCase):
    def test_short_passthrough(self):
        self.assertEqual(truncate("abc", 10), "abc")

    def test_long_truncated(self):
        out = truncate("x" * 100, 10)
        self.assertTrue(out.startswith("x" * 10))
        self.assertIn("[truncated]", out)

    def test_none(self):
        self.assertEqual(truncate(None), "")


class TurnRecordTest(unittest.TestCase):
    def _record(self, **kw):
        args = dict(session_id="s1", turn_id="t1",
                    user_message="hello", assistant_response="hi",
                    conversation_history=None, model="m", platform="p",
                    task_id="task1")
        args.update(kw)
        return build_turn_record(**args)

    def test_request_id_is_dedupe_key(self):
        r1 = self._record()
        r2 = self._record()
        self.assertEqual(r1["request_id"], r2["request_id"])
        self.assertEqual(r1["request_id"], "hermes:turn:s1:t1")

    def test_session_namespaced(self):
        self.assertEqual(self._record()["session_id"], "hermes:s1")

    def test_messages_present(self):
        msgs = self._record()["messages"]
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles[:2], ["user", "assistant"])
        self.assertIn("hello", msgs[0]["content"])
        self.assertIn("hi", msgs[1]["content"])

    def test_history_tail_bounded(self):
        hist = [{"role": "user", "content": "m%d" % i} for i in range(50)]
        msgs = self._record(conversation_history=hist)["messages"]
        ctx = [m for m in msgs if m["content"].startswith("[context]")]
        self.assertEqual(len(ctx), 1)
        # only the last 20 messages make it in
        self.assertIn("m49", ctx[0]["content"])
        self.assertNotIn("m0\n", ctx[0]["content"].replace("m0]", ""))

    def test_turn_budget(self):
        big = "z" * 50000
        msgs = self._record(user_message=big, assistant_response=big,
                            conversation_history=[{"role": "u", "content": big}])["messages"]
        total = sum(len(m["content"]) for m in msgs)
        self.assertLessEqual(total, 24 * 1024 + 4096)

    def test_user_id_filled_by_plugin(self):
        self.assertIsNone(self._record()["user_id"])


class ToolRecordTest(unittest.TestCase):
    def test_fields_and_dedupe(self):
        r = build_tool_record(session_id="s1", turn_id="t1",
                              tool_call_id="tc-9", function_name="read_file",
                              function_args='{"p": "/x"}', result="ok",
                              duration_ms=12, status="success")
        self.assertEqual(r["request_id"], "hermes:tool:tc-9")
        content = r["messages"][0]["content"]
        self.assertIn("read_file", content)
        self.assertIn("success", content)
        self.assertIn("12", content)

    def test_failure_carries_error(self):
        r = build_tool_record(session_id="s1", turn_id="t1",
                              tool_call_id="tc-1", function_name="bash",
                              status="error", error_type="Timeout",
                              error_message="killed")
        self.assertIn("Timeout", r["messages"][0]["content"])

    def test_long_args_truncated(self):
        r = build_tool_record(session_id="s", turn_id="t", tool_call_id="c",
                              function_name="f", function_args="a" * 9000)
        self.assertLess(len(r["messages"][0]["content"]), 9000)


class SessionRecordTest(unittest.TestCase):
    def test_start_and_close_have_distinct_request_ids(self):
        # The server rejects the same request_id with different content, so
        # start and close must NOT share an idempotency key.
        s = build_session_record(session_id="s1", model="m", platform="p")
        c = build_session_record(session_id="s1", closed=True,
                                 reason="session_boundary")
        self.assertNotEqual(s["request_id"], c["request_id"])
        self.assertEqual(s["request_id"], "hermes:session:s1:start")
        self.assertEqual(c["request_id"], "hermes:session:s1:close")
        self.assertIn("session_start", s["messages"][0]["content"])
        self.assertIn("session_close", c["messages"][0]["content"])


class SubagentRecordTest(unittest.TestCase):
    def test_start_is_scoped_to_child_and_keeps_parent_link(self):
        record = build_subagent_start_record(
            parent_session_id="parent-session",
            parent_turn_id="parent-turn",
            parent_subagent_id="parent-child",
            child_session_id="child-session",
            child_subagent_id="child-1",
            child_role="leaf",
            child_goal="Inspect the parser",
        )
        self.assertEqual(record["request_id"], "hermes:subagent:start:child-1")
        self.assertEqual(record["session_id"], "hermes:child-session")
        content = record["messages"][0]["content"]
        self.assertIn("parent_session_id=parent-session", content)
        self.assertIn("parent_turn_id=parent-turn", content)
        self.assertIn("parent_subagent_id=parent-child", content)
        self.assertIn("Inspect the parser", content)

    def test_stop_keeps_summary_and_only_safe_tool_metadata(self):
        record = build_subagent_stop_record(
            parent_session_id="parent-session",
            parent_turn_id="parent-turn",
            child_session_id="child-session",
            child_subagent_id="child-1",
            child_role="leaf",
            child_summary="Parser bug is in the empty-input branch",
            child_status="completed",
            tool_call_history=[{
                "tool_name": "read_file",
                "tool_input": {"path": "/private/project.rs"},
                "input_bytes": 42,
                "output_bytes": 128,
                "status": "success",
            }],
            duration_ms=1250,
        )
        self.assertEqual(record["request_id"], "hermes:subagent:stop:child-session")
        self.assertEqual(record["session_id"], "hermes:child-session")
        content = record["messages"][0]["content"]
        self.assertIn("status=completed", content)
        self.assertIn("duration_ms=1250", content)
        self.assertIn("Parser bug is in the empty-input branch", content)
        self.assertIn("input_bytes=42", content)
        self.assertNotIn("/private/project.rs", content)

    def test_missing_child_identity_is_ignored(self):
        self.assertIsNone(build_subagent_start_record(
            "parent", "turn", None, None, None, "leaf", "goal"))
        self.assertIsNone(build_subagent_stop_record(
            "parent", "turn", None, None, "leaf", "summary", "completed",
            [], 10))


class RecallFormatTest(unittest.TestCase):
    def test_empty_hits(self):
        self.assertEqual(format_recall_context([]), "")
        self.assertEqual(format_recall_context(None), "")

    def test_hits_formatted(self):
        out = format_recall_context(
            [{"content": "user likes tea", "score": 0.9},
             {"content": "x" * 9000, "score": 0.1}], max_hits=5)
        self.assertIn("user likes tea", out)
        self.assertIn("0.90", out)
        self.assertLess(len(out), 3000)


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        cfg = load_config(env={}, hermes_home="/nonexistent-dir")
        self.assertEqual(cfg["server_url"], "http://127.0.0.1:8080")
        self.assertTrue(cfg["capture"] and cfg["recall"])
        self.assertEqual(cfg["recall_top_k"], 5)

    def test_env_wins_over_file(self):
        with tempfile.TemporaryDirectory() as home:
            with open(os.path.join(home, "lintai.json"), "w") as f:
                json.dump({"user_id": "file-user", "recall_top_k": 3}, f)
            env = {"HERMES_HOME": home, "LINTAI_USER_ID": "env-user"}
            cfg = load_config(env=env, hermes_home=home)
            self.assertEqual(cfg["user_id"], "env-user")
            self.assertEqual(cfg["recall_top_k"], 3)

    def test_bool_parsing(self):
        cfg = load_config(env={"LINTAI_CAPTURE": "off", "LINTAI_RECALL": "0"},
                          hermes_home="/nonexistent-dir")
        self.assertFalse(cfg["capture"])
        self.assertFalse(cfg["recall"])

    def test_server_url_trailing_slash_stripped(self):
        cfg = load_config(env={"LINTAI_SERVER_URL": "http://x:1/"},
                          hermes_home="/nonexistent-dir")
        self.assertEqual(cfg["server_url"], "http://x:1")


class QueueTest(unittest.TestCase):
    def test_drop_oldest_when_full(self):
        client = FakeClient()
        q = WriteQueue.__new__(WriteQueue)  # no background thread
        import queue as _queue
        q.client = client
        q.queue = _queue.Queue(maxsize=2)
        q.dropped = 0
        q.enqueue({"request_id": "a"})
        q.enqueue({"request_id": "b"})
        q.enqueue({"request_id": "c"})  # drops "a"
        self.assertEqual(q.dropped, 1)
        remaining = []
        while not q.queue.empty():
            remaining.append(q.queue.get_nowait()["request_id"])
        self.assertEqual(remaining, ["b", "c"])

    def test_flush_waits_for_drain_thread(self):
        client = FakeClient()
        q = WriteQueue(client, maxsize=10)
        try:
            q.enqueue({"request_id": "a"})
            q.enqueue({"request_id": "b"})
            self.assertTrue(q.flush(timeout_s=5.0))
            self.assertEqual(len(client.batches), 1)
            self.assertEqual(len(client.batches[0]), 2)
        finally:
            q.stop()

    def test_flush_timeout_when_drain_stalled(self):
        class SlowClient(FakeClient):
            def add_batch(self, requests):
                import time as _time
                _time.sleep(30)

        q = WriteQueue(SlowClient(), maxsize=10)
        try:
            q.enqueue({"request_id": "a"})
            self.assertFalse(q.flush(timeout_s=0.5))
        finally:
            q.stop()

    def test_overflow_drop_decrements_unfinished_tasks(self):
        # P2: dropping the oldest item on overflow must pair get_nowait()
        # with task_done(); otherwise the queue's unfinished count leaks and
        # flush() waits the full timeout on an already-drained queue.
        client = FakeClient()
        q = WriteQueue.__new__(WriteQueue)  # no background thread
        import queue as _queue
        q.client = client
        q.queue = _queue.Queue(maxsize=2)
        q.dropped = 0
        q.enqueue({"request_id": "a"})
        q.enqueue({"request_id": "b"})
        q.enqueue({"request_id": "c"})  # drops "a"
        self.assertEqual(q.dropped, 1)
        self.assertEqual(q.queue.unfinished_tasks, 2)

    def test_flush_returns_promptly_after_overflow_and_drain(self):
        # End-to-end symptom: after the dropped item is accounted for and the
        # rest drained, flush() must return True immediately, not after the
        # full timeout.
        import time as _time
        client = FakeClient()
        q = WriteQueue.__new__(WriteQueue)  # no background thread
        import queue as _queue
        q.client = client
        q.queue = _queue.Queue(maxsize=2)
        q.dropped = 0
        q.enqueue({"request_id": "a"})
        q.enqueue({"request_id": "b"})
        q.enqueue({"request_id": "c"})  # drops "a"
        while not q.queue.empty():  # drain like the background thread does
            q.queue.get_nowait()
            q.queue.task_done()
        start = _time.time()
        self.assertTrue(q.flush(timeout_s=5.0))
        self.assertLess(_time.time() - start, 4.0)


class PluginHookTest(unittest.TestCase):
    def _plugin(self, **cfg_overrides):
        cfg = load_config(env={}, hermes_home="/nonexistent-dir")
        cfg.update(cfg_overrides)
        p = LintaiPlugin(config=cfg)
        p.client = FakeClient(hits=[{"content": "user likes tea",
                                     "score": 0.95}])
        # swap the queue for a threadless one
        import queue as _queue
        p.writes.stop()
        p.writes = WriteQueue.__new__(WriteQueue)
        p.writes.client = p.client
        p.writes.queue = _queue.Queue()
        p.writes.dropped = 0
        p.writes.errors = 0
        return p

    def _drain_queue(self, p):
        items = []
        while not p.writes.queue.empty():
            items.append(p.writes.queue.get_nowait())
        return items

    def test_register_subscribes_hooks(self):
        ctx = FakeCtx()
        plugin.register(ctx)
        for name in ("pre_llm_call", "post_tool_call", "post_llm_call",
                     "on_session_start", "on_session_finalize",
                     "on_session_reset", "subagent_start", "subagent_stop"):
            self.assertIn(name, ctx.hooks, name)
        # the naming-trap hook is deliberately NOT subscribed
        self.assertNotIn("on_session_end", ctx.hooks)
        self.assertNotIn("pre_tool_call", ctx.hooks)

    def test_subagent_hooks_enqueue_parent_linked_records(self):
        p = self._plugin()
        p.on_subagent_start(
            parent_session_id="parent", parent_turn_id="turn",
            child_session_id="child", child_subagent_id="agent-1",
            child_role="leaf", child_goal="Check cache behavior")
        p.on_subagent_stop(
            parent_session_id="parent", parent_turn_id="turn",
            child_session_id="child", child_subagent_id="agent-1",
            child_role="leaf", child_summary="Cache is refreshed once",
            child_status="completed", tool_call_history=[], duration_ms=25)

        records = self._drain_queue(p)
        self.assertEqual(len(records), 2)
        self.assertTrue(records[0]["request_id"].endswith(":start:agent-1"))
        self.assertTrue(records[1]["request_id"].endswith(":stop:child"))

    def test_pre_llm_call_injects_context(self):
        p = self._plugin()
        out = p.on_pre_llm_call(session_id="s1", turn_id="t1",
                                user_message="what tea?", task_id="k")
        self.assertIn("context", out)
        self.assertIn("user likes tea", out["context"])
        self.assertEqual(p.client.searches[0][0], "what tea?")

    def test_pre_llm_call_empty_message_no_query(self):
        p = self._plugin()
        self.assertIsNone(p.on_pre_llm_call(session_id="s", turn_id="t",
                                            user_message="   "))
        self.assertEqual(p.client.searches, [])

    def test_pre_llm_call_recall_disabled(self):
        p = self._plugin(recall=False)
        self.assertIsNone(p.on_pre_llm_call(session_id="s", turn_id="t",
                                            user_message="hi"))
        self.assertEqual(p.client.searches, [])

    def test_pre_llm_call_caches_per_turn(self):
        p = self._plugin()
        kw = dict(session_id="s", turn_id="t", user_message="hi")
        p.on_pre_llm_call(**kw)
        p.on_pre_llm_call(**kw)
        self.assertEqual(len(p.client.searches), 1)

    def test_pre_llm_call_fail_open(self):
        p = self._plugin()
        p.client.search = lambda *a, **k: 1 / 0
        self.assertIsNone(p.on_pre_llm_call(session_id="s", turn_id="t",
                                            user_message="hi"))

    def test_post_llm_call_enqueues_turn(self):
        p = self._plugin()
        self.assertIsNone(p.on_post_llm_call(
            session_id="s1", turn_id="t1", user_message="u",
            assistant_response="a", conversation_history=[]))
        items = self._drain_queue(p)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["request_id"], "hermes:turn:s1:t1")

    def test_post_tool_call_enqueues_tool_event(self):
        p = self._plugin()
        self.assertIsNone(p.on_post_tool_call(
            session_id="s1", turn_id="t1", tool_call_id="tc-1",
            function_name="bash", function_args="ls",
            result="ok", duration_ms=5, status="success"))
        items = self._drain_queue(p)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["request_id"], "hermes:tool:tc-1")

    def test_capture_disabled_enqueues_nothing(self):
        p = self._plugin(capture=False)
        p.on_post_llm_call(session_id="s", turn_id="t", user_message="u",
                           assistant_response="a")
        p.on_post_tool_call(session_id="s", turn_id="t", tool_call_id="c",
                            function_name="f")
        self.assertTrue(p.writes.queue.empty())

    def test_handlers_never_raise(self):
        p = self._plugin()
        p.writes.enqueue = lambda *a: 1 / 0  # broken queue
        for fn, kw in [
                (p.on_post_llm_call, dict(session_id="s", turn_id="t")),
                (p.on_post_tool_call, dict(session_id="s", tool_call_id="c")),
                (p.on_session_start, dict(session_id="s")),
                (p.on_session_finalize, dict(session_id="s"))]:
            self.assertIsNone(fn(**kw))


if __name__ == "__main__":
    unittest.main()
