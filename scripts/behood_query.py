#!/usr/bin/env python3
"""Query-time text parsing for behood judgments.

Pure spaCy parsing: one pass per text over the already-loaded model,
emitting the noun-phrase descriptors (`mentions`, `np_mentions`) that
bekind's JSON bridge judges. Judgment itself happens in the bekind
`--serve` daemon, owned directly by lint-ai's Rust code — this process
never spawns a subprocess and never touches the bekind binary.

This is the parse half of Luyi's design: "the behood provide people as
the source, then we have place and thing." spaCy parses the question;
bekind judges the descriptors; lint-ai uses the text for matching and
the kind for filtering.

Usage:
    python3 behood_query.py --serve   # one {"parse_texts": [...]} per
                                      # stdin line, one {"parsed": [...]}
                                      # per stdout line
    python3 behood_query.py --parse "Which city have both Jean and John visited?"
                                      # one-shot: prints {"parsed": [...]}

Output (JSON to stdout):
    {"parsed": [{"id": "p:0", "mentions": [...], "np_mentions": [...]}]}

Fail-open: on any error, outputs {"parsed": []} and exits 0.
"""

import json
import sys


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


def _doc_descriptors(doc):
    """Noun-phrase descriptors + PROPN personhood mentions for one parsed doc.

    Shared by analyze_question and analyze_query_semantics so both build
    identical bekind payloads from a single spaCy parse.
    """
    np_descriptors = question_np_descriptors(doc)
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
    return np_descriptors, mentions




def parse_texts(texts, nlp=None):
    """Parse raw texts into bekind-ready descriptors.

    Pure parsing: one spaCy pass per text over the already-loaded model.
    No judgment, no subprocess, no bekind binary. Returns a list of
    {"id", "mentions", "np_mentions"} with caller-assigned ids echoed;
    texts that fail to parse are skipped (fail-open: the caller treats a
    missing slot as "no descriptors").
    """
    if nlp is None:
        nlp = _load_spacy()
    if nlp is None:
        return []
    parsed = []
    for id_, text in texts:
        try:
            doc = nlp(text)
        except Exception:
            continue
        np_descriptors, mentions = _doc_descriptors(doc)
        parsed.append({"id": id_, "mentions": mentions, "np_mentions": np_descriptors})
    return parsed


def serve():
    """Line-delimited JSON protocol: pure spaCy parsing.

    One {"parse_texts": [{"id", "text"}, ...]} per stdin line →
    one {"parsed": [{"id", "mentions", "np_mentions"}, ...]} per stdout
    line. The descriptors are bekind's Request payload pieces; judgment
    happens in the bekind --serve daemon, owned directly by lint-ai's
    Rust code. This process never spawns a subprocess: spaCy is loaded
    once at startup, so per-request cost is milliseconds, not seconds.
    Exits non-zero when the backend cannot be initialized, so the caller
    can fail over without paying per-request spawn costs.
    """
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
        texts = [
            (t.get("id", f"p:{i}"), t.get("text", ""))
            for i, t in enumerate(payload.get("parse_texts", []))
            if isinstance(t, dict)
        ]
        try:
            parsed = parse_texts(texts, nlp=nlp)
        except Exception:
            parsed = []
        sys.stdout.write(json.dumps({"parsed": parsed}) + "\n")
        sys.stdout.flush()
    return 0


def main():
    if "--serve" in sys.argv[1:]:
        return serve()
    if "--parse" in sys.argv[1:]:
        texts = [a for a in sys.argv[1:] if a not in ("--serve", "--parse")]
        try:
            parsed = parse_texts([(f"p:{i}", t) for i, t in enumerate(texts)])
        except Exception:
            parsed = []
        print(json.dumps({"parsed": parsed}))
        return 0
    sys.stderr.write("behood_query: expected --serve or --parse\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
