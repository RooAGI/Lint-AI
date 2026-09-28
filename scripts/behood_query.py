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
                                     # one {"entities": [...]} per stdout line;
                                     # or one {"scope_texts": [...]} per stdin
                                     # line, one {"scope_verdicts": [...]} per
                                     # stdout line

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


_INTERROGATIVE_LEMMAS = {"what", "which", "who", "whom", "whose"}


def _is_interrogative_only(chunk):
    """True if a noun chunk carries no content beyond an interrogative pronoun."""
    for t in chunk:
        if t.is_space or t.pos_ == "PUNCT":
            continue
        if t.lemma_.lower() not in _INTERROGATIVE_LEMMAS:
            return False
    return True


def _sought_nominal(doc):
    """Dependency-parse fallback for the nominal a question is about.

    Copular questions like "What is the user's weekend exercise routine?"
    yield only the interrogative from noun_chunks; the sought phrase is the
    subject/predicate nominal ("routine") of the root clause.
    """
    for dep in ("nsubj", "nsubjpass", "attr", "dobj"):
        for tok in doc:
            if (
                tok.dep_ == dep
                and tok.lemma_.lower() not in _INTERROGATIVE_LEMMAS
                and tok.pos_ not in {"PRON", "AUX", "VERB", "PUNCT", "PART", "ADP", "DET", "CCONJ", "SCONJ"}
            ):
                return tok
    return None


def _descriptor_for_token(tok, did):
    """Build a noun-phrase descriptor from a token's full subtree span."""
    doc = tok.doc
    start, end = tok.left_edge.i, tok.right_edge.i
    # Trim trailing punctuation (e.g. the question mark).
    while end > start and doc[end].pos_ == "PUNCT":
        end -= 1
    return {
        "id": did,
        "text": doc[start:end + 1].text,
        "head_lemma": tok.lemma_.lower(),
        "head_pos": tok.pos_,
        "ner_label": tok.ent_type_,
        "modifiers": [
            {"text": t.text, "pos": t.pos_, "dep": t.dep_}
            for t in tok.subtree
            if t != tok and t.pos_ != "PUNCT" and not t.is_space
        ],
    }


# Dependency labels that fill a nominal slot (subject/object/...). When the
# parser gets the relation right but mis-tags the POS (e.g. "cilantro" as
# ADV in "dislikes cilantro and always asks ..."), noun_chunks drops the
# token; recovering by dep is robust to that quirk.
_NOMINAL_DEPS = frozenset(
    {"nsubj", "nsubjpass", "dobj", "pobj", "iobj", "attr", "appos", "conj"}
)
# POS tags that can never head a recovered nominal descriptor.
_NON_NOMINAL_POS = frozenset(
    {"VERB", "AUX", "ADP", "CCONJ", "SCONJ", "PART", "PUNCT", "SYM", "X", "NUM"}
)


def _recover_unchunked_nominals(doc, descriptors):
    """Recover nominal-slot tokens the chunker dropped (parser POS quirk).

    Systematic, not per-question: any token filling a nominal dependency
    slot (dobj/pobj/nsubj/...) that no noun chunk covers becomes an
    additional descriptor. The dependency label is trusted over the POS
    tag for slot-filling: in "The user dislikes cilantro and always asks
    ...", spaCy tags "cilantro" ADV but its dep is dobj -- the relation
    is right, the tag is wrong, and noun_chunks misses it.
    """
    covered = set()
    for chunk in doc.noun_chunks:
        covered.update(range(chunk.start, chunk.end))
    seen_texts = {d["text"] for d in descriptors}
    recovered = []
    for tok in doc:
        if tok.i in covered:
            continue
        if tok.dep_ not in _NOMINAL_DEPS:
            continue
        if tok.pos_ in _NON_NOMINAL_POS:
            continue
        if not tok.is_alpha:
            continue
        if tok.text in seen_texts:
            continue
        d = _descriptor_for_token(tok, f"q:rec{len(descriptors) + len(recovered)}")
        if d["text"]:
            recovered.append(d)
            seen_texts.add(tok.text)
    return recovered


def question_np_descriptors(doc):
    """Noun-phrase descriptors for behood's phrase layer, with a fallback.

    When noun_chunks yields only interrogative content (e.g. just "What"),
    fall back to the dependency-parse nominal so the sought phrase still
    reaches behood.
    """
    descriptors = []
    chunks = list(doc.noun_chunks)
    for i, chunk in enumerate(chunks):
        descriptors.append({
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
    if not chunks or all(_is_interrogative_only(c) for c in chunks):
        nominal = _sought_nominal(doc)
        if nominal is not None:
            fb = _descriptor_for_token(nominal, f"q:fb{len(descriptors)}")
            if fb["text"] and not any(d["text"] == fb["text"] for d in descriptors):
                descriptors.append(fb)
    # Systematic recovery: nominal-slot tokens the chunker dropped.
    descriptors.extend(_recover_unchunked_nominals(doc, descriptors))
    return descriptors


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
    np_descriptors = question_np_descriptors(doc)

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


def _scope_verdicts_via_binary(texts, binary):
    """Query the bekind JSON bridge for scope verdicts. Fail-open: []."""
    if not texts:
        return []
    payload = {
        "scope_texts": [{"id": f"s:{i}", "text": t} for i, t in enumerate(texts)],
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
    return data.get("scope_verdicts", [])


def _kind_verdicts_via_binary(texts, binary, nlp):
    """Kind verdicts per text for definitional kind tags. Fail-open: [].

    Builds noun-phrase descriptors (with unchunked-nominal recovery) for
    each text and judges them in a single bekind call, so per-fact cost
    stays at one subprocess. Returns one dict per input text:
        {"id", "kinds": [{"text", "kind"}]}
    Unlike analyze_question, ALL descriptor verdicts contribute their
    kind -- not just entity mentions: bekind reports kind_of for rejected
    mentions too (e.g. a POS-mistagged "cilantro" still judges herb-kind),
    and the tag only needs the kind signal.
    """
    if not texts or nlp is None:
        return []
    all_descriptors = []
    per_text_ids = []
    for i, text in enumerate(texts):
        try:
            doc = nlp(text)
        except Exception:
            per_text_ids.append((i, []))
            continue
        ids = []
        for d in question_np_descriptors(doc):
            d = dict(d)
            d["id"] = f"k:{i}:{d['id']}"
            ids.append(d["id"])
            all_descriptors.append(d)
        per_text_ids.append((i, ids))
    if not all_descriptors:
        return [{"id": f"k:{i}", "kinds": []} for i, _ in per_text_ids]
    payload = {
        "strategy": "discourse",
        "mentions": [],
        "chunks": [],
        "np_mentions": all_descriptors,
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
    id_to_kind = {}
    for v in data.get("phrase_verdicts", []):
        id_to_kind[v.get("id", "")] = v.get("kind", "thing")
    id_to_text = {d["id"]: d["text"] for d in all_descriptors}
    out = []
    for i, ids in per_text_ids:
        kinds = [
            {"text": id_to_text.get(vid, ""), "kind": id_to_kind.get(vid, "thing")}
            for vid in ids
        ]
        out.append({"id": f"k:{i}", "kinds": kinds})
    return out


def analyze_scope(texts, binary=None):
    """Return bekind scope verdicts for raw text spans.

    Additive: does not change analyze_question's output shape. Each input
    text gets one verdict dict:
        {"id", "activity_phrase", "temporal_words", "habitual", "evidence"}
    Fail-open: on any error, returns [].
    """
    if binary is None:
        binary = _behood_bin()
    if binary is None:
        return []
    return _scope_verdicts_via_binary(texts, binary)


def serve():
    """Line-delimited JSON protocol.

    One {"question": ...} per stdin line → one {"entities": [...]} per stdout
    line; or one {"scope_texts": [{"id", "text"}, ...]} per stdin line →
    one {"scope_verdicts": [...]} per stdout line; or one {"kind_texts":
    [{"id", "text"}, ...]} per stdin line → one {"kind_verdicts": [...]}
    per stdout line. spaCy and the bekind
    binary are resolved once at startup so per-query cost is milliseconds,
    not seconds. Exits non-zero when the backend cannot be initialized, so
    the caller can fail over to the heuristic path without paying per-query
    spawn costs.
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
        except Exception:
            payload = {}
        if "scope_texts" in payload:
            try:
                texts = [
                    t.get("text", "")
                    for t in payload["scope_texts"]
                    if isinstance(t, dict)
                ]
                verdicts = _scope_verdicts_via_binary(texts, binary)
            except Exception:
                verdicts = []
            sys.stdout.write(json.dumps({"scope_verdicts": verdicts}) + "\n")
        elif "kind_texts" in payload:
            try:
                texts = [
                    t.get("text", "")
                    for t in payload["kind_texts"]
                    if isinstance(t, dict)
                ]
                verdicts = _kind_verdicts_via_binary(texts, binary, nlp)
            except Exception:
                verdicts = []
            sys.stdout.write(json.dumps({"kind_verdicts": verdicts}) + "\n")
        else:
            question = payload.get("question", "")
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
