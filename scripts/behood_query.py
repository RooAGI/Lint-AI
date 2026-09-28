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

Output (JSON to stdout):
    {"entities": [{"text": "Jean", "kind": "person"}, ...]}

Fail-open: on any error, outputs {"entities": []} and exits 0.
"""

import json
import os
import shutil
import subprocess
import sys


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


def main():
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
