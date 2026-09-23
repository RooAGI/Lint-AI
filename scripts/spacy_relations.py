#!/usr/bin/env python3
"""Dependency-parse relation extraction for dialogue turns.

Reads dialogue turns on stdin (JSON), writes (subject, predicate, object)
triples on stdout (JSON). The rules below are *grammatical*, not lexical:
they operate on dependency labels, so no verb phrase is ever enumerated.
The only lexical data is the predicate string itself
(``verb_lemma[_prt][_prep]``); mapping predicates to semantic families
happens on the Rust side.

Protocol mirrors scripts/spacy_ner.py:
  stdin:  {"model": "en_core_web_sm", "turns": [
             {"speaker": str, "text": str, "session_id": str,
              "turn_idx": int, "doc_id": str, "session_date": str|null}]}
  stdout: {"relations": [
             {"subject": str, "predicate": str, "object": str,
              "is_place": bool, "session_id": str, "turn_idx": int,
              "doc_id": str, "evidence": str, "confidence": float}]}
Errors go to stderr as {"error": ...} with a non-zero exit code.
"""

import json
import sys

ALLOWED_MODELS = {
    "en_core_web_sm",
    "en_core_web_md",
    "en_core_web_lg",
}

PLACE_LABELS = {"GPE", "LOC", "FAC"}

# Dependency labels that head a subordinate clause; their subtrees are cut
# out of object noun phrases ("a shelter I volunteer at" -> "a shelter").
CLAUSE_DEPS = {"relcl", "acl", "advcl", "ccomp", "xcomp", "parataxis"}

# Leading tokens stripped from an object noun phrase.
STRIP_FIRST = {
    "a", "an", "the", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
}

# Pronouns that can never be subjects or objects of an extracted triple.
BAD_PRONOUNS = {
    "it", "its", "this", "that", "these", "those",
    "he", "him", "his", "she", "her", "they", "them", "their",
    "you", "your", "yours",
}

SELF_PRONOUNS = {"i", "we"}


def fail(message, code):
    print(json.dumps({"error": message}), file=sys.stderr)
    return code


def noun_phrase(noun, doc):
    """(text, token_ids) of a noun's phrase, minus dets and sub-clauses."""
    if noun.pos_ == "PRON" or noun.text.lower() in BAD_PRONOUNS:
        return None, None
    drop = set()
    for child in noun.children:
        if child.dep_ in CLAUSE_DEPS:
            drop.update(t.i for t in child.subtree)
    toks = [t for t in noun.subtree if t.i not in drop and not t.is_punct]
    if not toks:
        return None, None
    toks.sort(key=lambda t: t.i)
    while toks and toks[0].text.lower() in STRIP_FIRST:
        toks.pop(0)
    while toks and toks[-1].pos_ == "ADP":
        toks.pop()
    if not toks:
        return None, None
    text = doc[toks[0].i : toks[-1].i + 1].text.strip()
    if not text or text.lower() in BAD_PRONOUNS:
        return None, None
    return text, {t.i for t in toks}


def resolve_name(child, speaker, conf):
    """(name, confidence) or (None, 0.0) for one subject token."""
    low = child.text.lower()
    if low in SELF_PRONOUNS:
        return speaker, conf
    if low in BAD_PRONOUNS:
        return None, 0.0
    if child.pos_ == "PROPN" or (
        child.text[:1].isupper() and child.pos_ in ("NOUN", "PROPN")
    ):
        return child.text, conf
    return None, 0.0


def resolve_subject(verb, sent, speaker):
    """(name, confidence) or (None, 0.0) when unattributable.

    General grammatical rules, in order:
    1. the verb's own nominal subject;
    2. a subject-less ROOT (fragment): the speaker;
    3. a subject-less sentence-initial verb (dropped subject): the speaker;
    4. subject inheritance from the nearest ancestor verb with a subject
       ("to give out food" under "we went");
    5. otherwise unattributable: skip.
    """
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return resolve_name(child, speaker, 1.0)
    if verb.dep_ == "ROOT":
        return speaker, 0.9
    verbs = [t for t in sent if t.pos_ in ("VERB", "AUX")]
    if verbs and verbs[0].i == verb.i:
        return speaker, 0.9
    seen = {verb.i}
    head = verb.head
    while head.i not in seen:
        seen.add(head.i)
        if head.pos_ in ("VERB", "AUX"):
            for child in head.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    return resolve_name(child, speaker, 0.85)
        if head.dep_ == "ROOT":
            break
        head = head.head
    return None, 0.0


def predicate_for(verb, low_lemma, prep=None):
    prt = next(
        (c.lemma_.lower() for c in verb.children if c.dep_ == "prt"), None
    )
    pred = low_lemma
    if prt:
        pred += "_" + prt
    if prep is not None:
        pred += "_" + prep.lower()
    return pred


def low_lemma(verb, doc_low):
    """Lemma from the lowercased parse (fixes sentence-initial caps)."""
    i = verb.i
    if i < len(doc_low) and doc_low[i].text == verb.text.lower():
        return doc_low[i].lemma_.lower()
    return verb.lemma_.lower()


def extract_doc(turn, doc, doc_low):
    speaker = turn.get("speaker", "")
    out = []
    for sent in doc.sents:
        for verb in sent:
            is_root = verb.dep_ == "ROOT"
            if verb.pos_ not in ("VERB", "AUX") and not is_root:
                continue
            subject, sconf = resolve_subject(verb, sent, speaker)
            if subject is None:
                continue
            lemma = low_lemma(verb, doc_low)
            for child in verb.children:
                if child.dep_ == "dobj":
                    obj, ids = noun_phrase(child, doc)
                    if obj:
                        out.append(
                            (subject, predicate_for(verb, lemma),
                             obj, ids, sconf)
                        )
                elif child.dep_ == "prep":
                    pobjs = [c for c in child.children if c.dep_ == "pobj"]
                    for pobj in pobjs:
                        obj, ids = noun_phrase(pobj, doc)
                        if obj:
                            out.append(
                                (subject,
                                 predicate_for(verb, lemma, child.lemma_),
                                 obj, ids, sconf)
                            )
                    # Stranded preposition ("the shelter I volunteer at"):
                    # the relative clause's antecedent is the real object.
                    if verb.dep_ == "relcl" and verb.head.pos_ in (
                        "NOUN", "PROPN",
                    ):
                        obj, ids = noun_phrase(verb.head, doc)
                        if obj:
                            out.append(
                                (subject,
                                 predicate_for(verb, lemma, child.lemma_),
                                 obj, ids, min(sconf, 0.85))
                            )
    return out


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        return fail(f"invalid_json: {exc}", 2)

    model = payload.get("model", "en_core_web_sm")
    if model not in ALLOWED_MODELS:
        return fail(
            f"spacy_model_not_allowed({model}); allowed={sorted(ALLOWED_MODELS)}",
            5,
        )
    turns = payload.get("turns", [])

    try:
        import spacy
    except Exception as exc:
        return fail(f"spacy_import_failed: {exc}", 3)

    try:
        nlp = spacy.load(model)
    except Exception as exc:
        return fail(f"spacy_model_load_failed({model}): {exc}", 4)

    texts = [t.get("text", "") or "" for t in turns]
    docs = list(nlp.pipe(texts))
    docs_low = list(nlp.pipe([t.lower() for t in texts]))

    relations = []
    for turn, doc, doc_low in zip(turns, docs, docs_low):
        try:
            triples = extract_doc(turn, doc, doc_low)
        except Exception as exc:
            print(
                json.dumps({"warning": f"turn_failed: {exc}"}), file=sys.stderr
            )
            continue
        seen = set()
        place_ids = {
            t.i for ent in doc.ents
            if ent.label_ in PLACE_LABELS for t in ent
        }
        for subject, predicate, obj, ids, conf in triples:
            key = (subject, predicate, obj)
            if key in seen:
                continue
            seen.add(key)
            relations.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "is_place": bool(ids & place_ids),
                    "session_id": turn.get("session_id", ""),
                    "turn_idx": turn.get("turn_idx", 0),
                    "doc_id": turn.get("doc_id", ""),
                    "session_date": turn.get("session_date"),
                    "evidence": f"{turn.get('speaker', '')}: {turn.get('text', '')}",
                    "confidence": conf,
                }
            )

    json.dump({"relations": relations}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
