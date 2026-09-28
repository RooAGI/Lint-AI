#!/usr/bin/env python3
"""Query-time behood analysis.

Takes a question string, runs spaCy to extract noun phrases, sends them to
the behood binary via the JSON protocol, and outputs (text, kind) pairs.

This is the query-time half of Luyi's design: "the behood provide people as
the source, then we have place and thing." Behood judges the question's
entities; lint-ai uses the text for matching and the kind for filtering.

Usage:
    echo "Which city have both Jean and John visited?" | python3 behood_query.py
    python3 behood_query.py "Which city have both Jean and John visited?"
    python3 behood_query.py --serve   # one {"question": ...} per stdin line,
                                     # one {"entities": [...]} per stdout line
    python3 behood_query.py --serve-socket /tmp/bq.sock
                                     # shared daemon on a unix socket: one
                                     # daemon per user, warm for every process

Daemon protocols (both modes serve one JSON object per line):
    request:  {"question": "..."}
    response: {"entities": [{"text": ..., "kind": ...}, ...]}

The unix-socket mode additionally sends one handshake line per connection:
    {"ready": true, "protocol": 1}

Output (JSON to stdout):
    {"entities": [{"text": "Jean", "kind": "person"}, ...]}

Fail-open: on any error, outputs {"entities": []} and exits 0.
"""

import errno
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

# Daemon wire-protocol version. The Rust client checks this on the socket
# handshake; bump it (and the socket path) if the protocol ever changes.
PROTOCOL_VERSION = 1
# Socket daemon exits after this many idle seconds with no connections.
SOCKET_IDLE_TIMEOUT_SECS = 300


def _behood_bin():
    """Path to the compiled `bekind` classifier, if available."""
    env = os.environ.get("BEHOOD_BIN")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    # Project renamed behood -> bekind; try the new binary name first,
    # fall back to the old one during transition.
    found = shutil.which("bekind") or shutil.which("behood")
    if found:
        return found
    for name in ("bekind", "behood"):
        cargo_bin = os.path.expanduser(f"~/.cargo/bin/{name}")
        if os.path.isfile(cargo_bin) and os.access(cargo_bin, os.X_OK):
            return cargo_bin
    return None


def _load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def analyze_question(question, nlp=None, binary=None):
    """Return [(text, kind)] for the question's noun phrases via behood.

    When `nlp`/`binary` are not supplied (one-shot mode) they are resolved
    here; serve mode resolves them once at startup and passes them in.
    """
    if nlp is None:
        nlp = _load_spacy()
    if binary is None:
        binary = _behood_bin()
    if nlp is None or binary is None:
        return []

    doc = nlp(question)

    entities = []

    # Temporal question words: "when", "what time", "how long" ask for a time.
    # Behood judges these as time-seeking; lint-ai uses the kind to filter.
    import re
    ql = question.lower()
    temporal_qw = re.search(r'\b(when|what time|how long|what date|which date|what day|which day)\b', ql)
    if temporal_qw:
        entities.append({"text": temporal_qw.group(0), "kind": "time"})

    # Build noun-phrase descriptors for behood's phrase layer.
    np_descriptors = []
    for i, chunk in enumerate(doc.noun_chunks):
        np_descriptors.append({
            "id": f"q:{i}",
            "text": chunk.text,
            "head_lemma": chunk.root.lemma_.lower(),
            "head_pos": chunk.root.pos_,
            "ner_label": chunk.root.ent_type_,
            "modifiers": [
                {"text": t.text, "pos": t.pos_, "dep": t.dep_}
                for t in chunk
                if t != chunk.root
            ],
        })

    # Also send PROPN tokens as personhood mentions so names get judged.
    mentions = []
    for i, tok in enumerate(doc):
        if tok.pos_ == "PROPN":
            mentions.append({
                "id": f"m:{i}",
                "text": tok.text,
                "ner_label": tok.ent_type_,
                "pos": tok.pos_,
                "head_lemma": tok.lemma_.lower(),
            })

    if not np_descriptors and not mentions:
        return []

    payload = {
        "strategy": "discourse",
        "mentions": mentions,
        "chunks": [],
        "np_mentions": np_descriptors,
        "context": {"speaker_names": []},
    }

    try:
        proc = subprocess.run(
            [binary],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    try:
        data = json.loads(proc.stdout)
    except Exception:
        return []

    # Map verdicts back to text.
    id_to_text = {}
    for d in np_descriptors:
        id_to_text[d["id"]] = d["text"]
    for d in mentions:
        id_to_text[d["id"]] = d["text"]

    for v in data.get("phrase_verdicts", []):
        if v.get("is_entity_mention"):
            text = id_to_text.get(v["id"], "")
            kind = v.get("kind", "thing")
            if text:
                entities.append({"text": text, "kind": kind})
    for v in data.get("verdicts", []):
        if v.get("is_person"):
            text = id_to_text.get(v["id"], "")
            if text and not any(e["text"] == text for e in entities):
                entities.append({"text": text, "kind": "person"})

    return entities


def serve():
    """Line-delimited JSON protocol: one {"question": ...} per stdin line,
    one {"entities": [...]} per stdout line. spaCy and the bekind binary are
    resolved once at startup so per-query cost is milliseconds, not seconds.
    Exits non-zero when the backend cannot be initialized, so the caller can
    fail over to the heuristic path without paying per-query spawn costs.
    """
    # Fail fast: the binary check is cheap; the spaCy load costs seconds.
    binary = _behood_bin()
    if binary is None:
        sys.stderr.write("behood_query --serve: bekind binary not found\n")
        return 3
    nlp = _load_spacy()
    if nlp is None:
        sys.stderr.write("behood_query --serve: spaCy model unavailable\n")
        return 3
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            question = payload.get("question", "")
        except Exception:
            question = ""
        try:
            entities = analyze_question(question, nlp=nlp, binary=binary)
        except Exception:
            entities = []
        sys.stdout.write(json.dumps({"entities": entities}) + "\n")
        sys.stdout.flush()
    return 0


# ---------------------------------------------------------------------------
# Unix-socket daemon mode: one shared daemon per user, warm for every
# process. Short-lived processes (hooks, CLIs) would otherwise pay the full
# interpreter + spaCy load on every invocation, since a --serve child dies
# with its parent.
# ---------------------------------------------------------------------------

def _serve_line(line, nlp, binary):
    """Answer one JSON-lines request; always returns a response line.

    `nlp`/`binary` were resolved once at daemon startup. When the backend
    is unavailable the daemon stays up and answers [] — fail-open with zero
    per-query cost — and recovers naturally via the idle timeout.
    """
    try:
        req = json.loads(line)
        question = req.get("question", "") if isinstance(req, dict) else ""
    except Exception:
        question = ""
    if not isinstance(question, str):
        question = ""
    if nlp is None or binary is None:
        return json.dumps({"entities": []})
    try:
        entities = analyze_question(question, nlp=nlp, binary=binary)
    except Exception:
        entities = []
    return json.dumps({"entities": entities})


def _daemonize():
    """Double-fork into the background; the intermediate parent exits at once.

    After this returns (in the grandchild), stdio is /dev/null and the
    process is reparented to init when the original parent exits, so the
    spawner never leaves a zombie behind.
    """
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    if devnull > 2:
        os.close(devnull)


class _AnotherDaemonServing(Exception):
    """Raised when a live daemon already serves the requested socket path."""


def _probe_socket(path, timeout=5):
    """True if a live daemon answers the ready handshake on `path`."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(path)
        line = s.makefile("rb").readline()
        data = json.loads(line.decode("utf-8"))
        return data.get("ready") is True and data.get("protocol") == PROTOCOL_VERSION
    except Exception:
        return False
    finally:
        s.close()


def _bind_socket(path):
    """Bind and listen on the unix socket.

    The bind is the singleflight election: concurrent starters race here
    and exactly one wins. A loser that finds a live daemon exits quietly
    via _AnotherDaemonServing; a stale socket file (dead daemon) is
    reclaimed.

    The handshake is sent immediately after listen(), before the model
    loads, so a live daemon — even mid-startup — answers a probe in
    milliseconds. That keeps the probe from mistaking a loading daemon
    for a stale socket and stealing its path.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        srv.bind(path)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        # EADDRINUSE: a rival may be mid-startup (bound, not yet
        # listening) or this is a stale file. Retry the probe briefly —
        # a live daemon answers in ms — before reclaiming.
        for _ in range(5):
            if _probe_socket(path, timeout=1):
                srv.close()
                raise _AnotherDaemonServing()
            time.sleep(0.1)
        try:
            os.unlink(path)
        except OSError:
            pass
        srv.bind(path)
    srv.listen(128)
    return srv


class _SocketState:
    def __init__(self, idle_timeout):
        self._lock = threading.Lock()
        self._active = 0
        self._idle_timeout = idle_timeout
        self._last_activity = time.monotonic()

    def conn_open(self):
        with self._lock:
            self._active += 1
            self._last_activity = time.monotonic()

    def conn_close(self):
        with self._lock:
            self._active -= 1
            self._last_activity = time.monotonic()

    def request(self):
        with self._lock:
            self._last_activity = time.monotonic()

    def should_exit(self):
        with self._lock:
            idle = time.monotonic() - self._last_activity
            return self._active == 0 and idle > self._idle_timeout


def _serve_conn(conn, ready, boxes, state):
    try:
        # Handshake first, before the model may have finished loading: the
        # client learns we're alive (and our protocol version) in ms, and
        # election probes never mistake a loading daemon for a stale socket.
        conn.sendall(
            json.dumps({"ready": True, "protocol": PROTOCOL_VERSION}).encode() + b"\n"
        )
        # The model may still be loading; the request already sent by the
        # client waits in the socket buffer meanwhile.
        ready.wait(timeout=180)
        nlp, binary = boxes.get("nlp"), boxes.get("binary")
        for raw in conn.makefile("rb"):
            if not raw.strip():
                continue
            try:
                resp = _serve_line(raw, nlp, binary)
            except Exception:
                resp = '{"entities": []}'
            try:
                conn.sendall(resp.encode("utf-8") + b"\n")
            except OSError:
                break
            state.request()
    except OSError:
        pass
    finally:
        state.conn_close()
        try:
            conn.close()
        except OSError:
            pass


def serve_socket(path, idle_timeout=SOCKET_IDLE_TIMEOUT_SECS):
    """Serve requests on a unix socket, daemonized.

    Binds (winning the singleflight election against concurrent starters),
    listens, then accepts immediately: each connection gets its handshake
    at once, while the heavy model load runs on a background thread.
    Connections made during the load queue and are served once ready
    instead of being refused. Exits after `idle_timeout` seconds with no
    connections.
    """
    if sys.platform != "win32":
        _daemonize()
    srv = _bind_socket(path)
    state = _SocketState(idle_timeout)
    ready = threading.Event()
    boxes = {}

    def loader():
        # Fail fast: the binary check is cheap; the spaCy load costs
        # seconds. When behood is unavailable there is no point paying
        # the model load — the daemon stays up and answers [] (fail-open),
        # and the idle timeout recycles it once the backend appears.
        binary = _behood_bin()
        boxes["binary"] = binary
        boxes["nlp"] = _load_spacy() if binary is not None else None
        ready.set()

    threading.Thread(target=loader, name="behood-loader", daemon=True).start()
    srv.settimeout(30)
    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            if state.should_exit():
                break
            continue
        except OSError:
            break
        state.conn_open()
        t = threading.Thread(
            target=_serve_conn, args=(conn, ready, boxes, state), daemon=True
        )
        t.start()
    srv.close()
    try:
        os.unlink(path)
    except OSError:
        pass


def main():
    if "--serve-socket" in sys.argv[1:]:
        idx = sys.argv.index("--serve-socket")
        if idx + 1 >= len(sys.argv):
            sys.stderr.write("behood_query --serve-socket: missing socket path\n")
            return 2
        path = sys.argv[idx + 1]
        idle_timeout = SOCKET_IDLE_TIMEOUT_SECS
        rest = sys.argv[idx + 2:]
        if rest[:1] == ["--idle-timeout"] and len(rest) > 1:
            try:
                idle_timeout = max(1, int(rest[1]))
            except ValueError:
                pass
        try:
            serve_socket(path, idle_timeout)
        except _AnotherDaemonServing:
            # A live daemon already serves this path; nothing to do.
            pass
        return 0
    if "--serve" in sys.argv[1:]:
        return serve()
    # Fail fast: the bekind binary check is cheap; spaCy load costs seconds.
    # When behood is unavailable there is no point paying the model load.
    binary = _behood_bin()
    if binary is None:
        print(json.dumps({"entities": []}))
        return 0
    if len(sys.argv) > 1:
        question = " ".join(a for a in sys.argv[1:] if a != "--serve")
    else:
        question = sys.stdin.read().strip()
    try:
        entities = analyze_question(question, binary=binary)
    except Exception:
        entities = []
    print(json.dumps({"entities": entities}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
