"""lint-ai memory plugin for Hermes Agent.

Automatic capture/recall via hooks — the "next level" after the `--hermes-serve`
MCP adapter. The MCP adapter makes memory available when the agent *chooses* to call
a tool; this plugin makes capture/recall *automatic*.

How it works
------------
* ``pre_llm_call``  -> recall: query the lint-ai server and return
  ``{"context": ...}``; Hermes stamps it into the user message (verified live).
* ``post_tool_call`` -> structured tool-event record (name/args/result/duration/status).
* ``post_llm_call``  -> per-turn transcript record.
* ``on_session_start`` -> session registry entry.
* ``on_session_finalize`` / ``on_session_reset`` -> boundary markers
  (no transcript — per-turn accumulation is the authoritative record).

Transport: the plugin is Python running inside Hermes' process, so it cannot call
the Rust core in-process the way our Rust MCP adapters do. It talks to a running
lint-ai server over HTTP keep-alive (``POST /search``, ``POST /add/batch``) —
the same pattern mem0's Hermes plugin uses. No Hermes changes, no Rust changes.

Dedupe is stateless (no state file): turns key on ``(session_id, turn_id)``,
tool events on ``tool_call_id`` — the lint-ai ``request_id`` is the idempotency key.

Fail-open: every handler swallows exceptions; a dead server degrades to "no
automatic memory", never to a broken agent.

Config (env wins over $HERMES_HOME/lintai.json wins over defaults):
  LINTAI_SERVER_URL   default http://127.0.0.1:8080
  LINTAI_USER_ID      default "hermes"
  LINTAI_QUEUE_MAX    default 1000 (bounded write queue; drops oldest when full)
  LINTAI_CAPTURE      on/off, default on
  LINTAI_RECALL       on/off, default on
  LINTAI_RECALL_TOP_K default 5
"""

import http.client
import json
import os
import queue
import threading
import time
import urllib.parse
from datetime import datetime, timezone

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

_DEFAULTS = {
    "server_url": "http://127.0.0.1:8080",
    "user_id": "hermes",
    "queue_max": 1000,
    "capture": True,
    "recall": True,
    "recall_top_k": 5,
}

_ENV_MAP = {
    "LINTAI_SERVER_URL": "server_url",
    "LINTAI_USER_ID": "user_id",
    "LINTAI_QUEUE_MAX": "queue_max",
    "LINTAI_CAPTURE": "capture",
    "LINTAI_RECALL": "recall",
    "LINTAI_RECALL_TOP_K": "recall_top_k",
}

_INT_KEYS = {"queue_max", "recall_top_k"}
_BOOL_KEYS = {"capture", "recall"}


def _to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def load_config(env=None, hermes_home=None):
    """Resolve plugin config: defaults < $HERMES_HOME/lintai.json < env."""
    env = os.environ if env is None else env
    cfg = dict(_DEFAULTS)
    home = hermes_home or env.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    try:
        with open(os.path.join(home, "lintai.json"), "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        if isinstance(file_cfg, dict):
            for key, value in file_cfg.items():
                if key in cfg:
                    cfg[key] = value
    except (OSError, ValueError):
        pass
    for env_name, key in _ENV_MAP.items():
        if env_name in env:
            raw = env[env_name]
            if key in _INT_KEYS:
                try:
                    cfg[key] = int(raw)
                except (TypeError, ValueError):
                    pass
            elif key in _BOOL_KEYS:
                cfg[key] = _to_bool(raw)
            else:
                cfg[key] = raw
    cfg["server_url"] = cfg["server_url"].rstrip("/")
    return cfg


# --------------------------------------------------------------------------
# Truncation helpers
# --------------------------------------------------------------------------

MAX_FIELD_CHARS = 2000
MAX_TURN_BYTES = 24 * 1024
HISTORY_TAIL = 20


def truncate(text, limit=MAX_FIELD_CHARS):
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "…[truncated]"


# --------------------------------------------------------------------------
# Record builders (pure functions — unit-tested)
# --------------------------------------------------------------------------

def _session_ns(session_id):
    return "hermes:%s" % (session_id,)


def _meta_preamble(**fields):
    parts = ["[meta]"]
    for key in sorted(fields):
        value = fields[key]
        if value is not None:
            parts.append("%s=%s" % (key, truncate(value, 200)))
    return " ".join(parts)


def build_turn_record(session_id, turn_id, user_message, assistant_response,
                       conversation_history=None, model=None, platform=None,
                       task_id=None, is_first_turn=None):
    """One record per turn. Idempotency key: (session_id, turn_id)."""
    request_id = "hermes:turn:%s:%s" % (session_id, turn_id)
    messages = [
        {"role": "user", "timestamp": None,
         "content": _meta_preamble(model=model, platform=platform,
                                   task_id=task_id,
                                   is_first_turn=is_first_turn)
                    + "\n" + truncate(user_message)},
        {"role": "assistant", "timestamp": None,
         "content": truncate(assistant_response)},
    ]
    tail = _history_tail(conversation_history)
    if tail:
        messages.append({"role": "user", "timestamp": None,
                         "content": "[context] recent conversation:\n" + tail})
    _enforce_turn_budget(messages)
    return {
        "request_id": request_id,
        "user_id": None,  # filled in by the plugin from config
        "session_id": _session_ns(session_id),
        "messages": messages,
    }


def _history_tail(conversation_history):
    if not conversation_history:
        return ""
    items = list(conversation_history)[-HISTORY_TAIL:]
    lines = []
    for item in items:
        if isinstance(item, dict):
            role = item.get("role", "?")
            content = item.get("content", "")
        else:
            role, content = "?", item
        if isinstance(content, (list, tuple)):
            content = " ".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p)
                for p in content)
        lines.append("[%s] %s" % (role, truncate(content, 500)))
    return "\n".join(lines)


def _enforce_turn_budget(messages):
    total = sum(len(m.get("content", "")) for m in messages)
    if total <= MAX_TURN_BYTES:
        return
    # Shrink the context tail first; never drop the user/assistant pair.
    for m in messages:
        if m.get("content", "").startswith("[context]"):
            m["content"] = truncate(m["content"], MAX_TURN_BYTES // 4)
    total = sum(len(m.get("content", "")) for m in messages)
    if total > MAX_TURN_BYTES:
        for m in messages:
            if not m.get("content", "").startswith("[meta]"):
                m["content"] = truncate(m["content"], MAX_TURN_BYTES // 4)


def build_tool_record(session_id, turn_id, tool_call_id, function_name,
                      function_args=None, result=None, duration_ms=None,
                      status=None, error_type=None, error_message=None):
    """One structured record per tool call. Idempotency key: tool_call_id."""
    request_id = "hermes:tool:%s" % (tool_call_id,)
    header = ("[meta] turn_id=%s\n" % (turn_id,) if turn_id else "")
    header += ("tool_call function_name=%s status=%s duration_ms=%s"
               % (function_name, status, duration_ms))
    parts = [header,
             "args:\n" + truncate(function_args),
             "result:\n" + truncate(result)]
    if error_type or error_message:
        parts.append("error: %s %s" % (truncate(error_type, 200),
                                       truncate(error_message)))
    return {
        "request_id": request_id,
        "user_id": None,
        "session_id": _session_ns(session_id),
        # NOTE: the server only accepts role "user" | "assistant" on /add/batch
        # (memory_api.rs validates this); the [tool_call] prefix marks the kind.
        "messages": [{"role": "user", "timestamp": None,
                      "content": "\n".join(parts)}],
    }


def build_session_record(session_id, model=None, platform=None,
                         parent_session_id=None, closed=False, reason=None):
    """Session registry entry.

    Start and close use DISTINCT request_ids: the server treats request_id as an
    idempotency key and rejects the same id with different content, so a single
    id cannot be rewritten from start to close.
    """
    phase = "close" if closed else "start"
    request_id = "hermes:session:%s:%s" % (session_id, phase)
    if closed:
        content = ("session_close reason=%s" % (reason,))
    else:
        content = (_meta_preamble(model=model, platform=platform,
                                  parent_session_id=parent_session_id)
                   + "\nsession_start")
    return {
        "request_id": request_id,
        "user_id": None,
        "session_id": _session_ns(session_id),
        # NOTE: the server only accepts role "user" | "assistant" on /add/batch;
        # the session_start/session_close prefix marks the kind.
        "messages": [{"role": "user", "timestamp": None,
                      "content": content}],
    }


def format_recall_context(hits, max_hits=5):
    """Format /search hits into the context block injected via pre_llm_call."""
    lines = ["[lint-ai memory — recalled for this turn]"]
    for hit in (hits or [])[:max_hits]:
        content = hit.get("content", "") if isinstance(hit, dict) else str(hit)
        score = hit.get("score") if isinstance(hit, dict) else None
        suffix = " (score %.2f)" % (score,) if isinstance(score, (int, float)) else ""
        lines.append("- " + truncate(content, 500).replace("\n", " ") + suffix)
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


# --------------------------------------------------------------------------
# HTTP client (stdlib only, keep-alive per thread)
# --------------------------------------------------------------------------

class LintaiClient:
    def __init__(self, server_url, user_id, timeout_s=5.0):
        self.server_url = server_url.rstrip("/")
        self.user_id = user_id
        self.timeout_s = timeout_s
        self._local = threading.local()

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            parts = urllib.parse.urlparse(self.server_url)
            if parts.scheme == "https":
                conn = http.client.HTTPSConnection(parts.hostname,
                                                   parts.port or 443,
                                                   timeout=self.timeout_s)
            else:
                conn = http.client.HTTPConnection(parts.hostname,
                                                  parts.port or 80,
                                                  timeout=self.timeout_s)
            self._local.conn = conn
        return conn

    def _post(self, path, payload):
        body = json.dumps(payload).encode("utf-8")
        try:
            conn = self._conn()
            conn.request("POST", path, body,
                         {"Content-Type": "application/json",
                          "Content-Length": str(len(body))})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status != 200:
                raise IOError("HTTP %s on %s" % (resp.status, path))
            return json.loads(data.decode("utf-8")) if data else {}
        except Exception:
            # Drop the connection so the next call reconnects.
            self._local.conn = None
            raise

    def search(self, query, top_k=5):
        resp = self._post("/search", {"query": query,
                                     "user_id": self.user_id,
                                     "top_k": top_k})
        return resp.get("data", []) if isinstance(resp, dict) else []

    def add_batch(self, requests):
        for req in requests:
            req["user_id"] = self.user_id
        return self._post("/add/batch", requests)


# --------------------------------------------------------------------------
# Async bounded write queue
# --------------------------------------------------------------------------

class WriteQueue:
    """Bounded, best-effort, drop-oldest queue drained by a daemon thread."""

    def __init__(self, client, maxsize=1000):
        self.client = client
        self.queue = queue.Queue(maxsize=max(1, maxsize))
        self.dropped = 0
        self.errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain, daemon=True,
                                        name="lintai-write-queue")
        self._thread.start()

    def enqueue(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            try:
                self.queue.get_nowait()
                # The dropped item will never be processed: pair the removal
                # with task_done() or the unfinished-task count leaks and
                # flush() waits the full timeout on a drained queue.
                self.queue.task_done()
            except queue.Empty:
                pass
            self.dropped += 1
            try:
                self.queue.put_nowait(record)
            except queue.Full:
                self.dropped += 1

    def _drain(self):
        while not self._stop.is_set():
            batch = []
            try:
                batch.append(self.queue.get(timeout=0.5))
            except queue.Empty:
                continue
            while len(batch) < 128:
                try:
                    batch.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            # One retry on transient failures (e.g. server 429 "writer busy").
            # Retries are safe: identical request_ids replay idempotently.
            for attempt in range(2):
                try:
                    self.client.add_batch(batch)
                    break
                except Exception as exc:
                    if attempt == 1:
                        self.errors += 1
                        print("lintai: dropped batch of %d (write error: %s)"
                              % (len(batch), exc), flush=True)
                    else:
                        time.sleep(1.0)
            for _ in batch:
                self.queue.task_done()

    def flush(self, timeout_s=5.0):
        """Cooperative drain: wait for the background thread to finish pending
        writes. Never writes directly — that would race the drain thread for
        the server's writer lock (HTTP 429)."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.queue.unfinished_tasks == 0:
                return True
            time.sleep(0.1)
        return self.queue.unfinished_tasks == 0

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------
# Plugin
# --------------------------------------------------------------------------

class LintaiPlugin:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.client = LintaiClient(self.config["server_url"],
                                   self.config["user_id"])
        self.writes = WriteQueue(self.client,
                                 maxsize=self.config["queue_max"])
        self._recall_cache = {}
        self._lock = threading.Lock()

    # -- helpers ---------------------------------------------------------
    def _fail_open(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            return None

    # -- hooks -----------------------------------------------------------
    def on_pre_llm_call(self, **kwargs):
        """Recall + inject. Runs synchronously (the turn blocks on it)."""
        if not self.config["recall"]:
            return None

        def _recall():
            session_id = kwargs.get("session_id")
            turn_id = kwargs.get("turn_id")
            cache_key = (session_id, turn_id)
            with self._lock:
                if cache_key in self._recall_cache:
                    return self._recall_cache[cache_key]
            user_message = kwargs.get("user_message") or ""
            if not str(user_message).strip():
                return None
            hits = self.client.search(
                str(user_message),
                top_k=self.config["recall_top_k"])
            context = format_recall_context(hits)
            result = {"context": context} if context else None
            with self._lock:
                if len(self._recall_cache) > 64:
                    self._recall_cache.clear()
                self._recall_cache[cache_key] = result
            return result

        return self._fail_open(_recall)

    def on_post_tool_call(self, **kwargs):
        def _capture():
            if not self.config["capture"]:
                return None
            record = build_tool_record(
                session_id=kwargs.get("session_id"),
                turn_id=kwargs.get("turn_id"),
                tool_call_id=kwargs.get("tool_call_id"),
                function_name=kwargs.get("function_name"),
                function_args=kwargs.get("function_args"),
                result=kwargs.get("result"),
                duration_ms=kwargs.get("duration_ms"),
                status=kwargs.get("status"),
                error_type=kwargs.get("error_type"),
                error_message=kwargs.get("error_message"),
            )
            self.writes.enqueue(record)
            return None

        return self._fail_open(_capture)

    def on_post_llm_call(self, **kwargs):
        def _capture():
            if not self.config["capture"]:
                return None
            record = build_turn_record(
                session_id=kwargs.get("session_id"),
                turn_id=kwargs.get("turn_id"),
                user_message=kwargs.get("user_message"),
                assistant_response=kwargs.get("assistant_response"),
                conversation_history=kwargs.get("conversation_history"),
                model=kwargs.get("model"),
                platform=kwargs.get("platform"),
                task_id=kwargs.get("task_id"),
            )
            self.writes.enqueue(record)
            return None

        return self._fail_open(_capture)

    def on_session_start(self, **kwargs):
        def _capture():
            if not self.config["capture"]:
                return None
            self.writes.enqueue(build_session_record(
                session_id=kwargs.get("session_id"),
                model=kwargs.get("model"),
                platform=kwargs.get("platform")))
            return None

        return self._fail_open(_capture)

    def _on_boundary(self, **kwargs):
        reason = kwargs.pop("reason", None)

        def _capture():
            if not self.config["capture"]:
                return None
            self.writes.enqueue(build_session_record(
                session_id=kwargs.get("session_id"),
                platform=kwargs.get("platform"),
                closed=True, reason=reason or "session_boundary"))
            # Short synchronous flush is acceptable here (CLI thread, no
            # timeout bound) — but still best-effort.
            self.writes.flush(timeout_s=5.0)
            return None

        return self._fail_open(_capture)

    def on_session_finalize(self, **kwargs):
        return self._on_boundary(**kwargs)

    def on_session_reset(self, **kwargs):
        kwargs.setdefault("reason", "new_session")
        return self._on_boundary(**kwargs)


def register(ctx):
    """Hermes plugin entry point."""
    plugin = LintaiPlugin()
    ctx.register_hook("pre_llm_call", plugin.on_pre_llm_call)
    ctx.register_hook("post_tool_call", plugin.on_post_tool_call)
    ctx.register_hook("post_llm_call", plugin.on_post_llm_call)
    ctx.register_hook("on_session_start", plugin.on_session_start)
    ctx.register_hook("on_session_finalize", plugin.on_session_finalize)
    ctx.register_hook("on_session_reset", plugin.on_session_reset)
    return plugin
