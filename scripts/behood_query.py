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


def analyze_question(question):
    """Return [(text, kind)] for the question's noun phrases via behood."""
    nlp = _load_spacy()
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


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = sys.stdin.read().strip()
    try:
        entities = analyze_question(question)
    except Exception:
        entities = []
    print(json.dumps({"entities": entities}))


if __name__ == "__main__":
    main()
