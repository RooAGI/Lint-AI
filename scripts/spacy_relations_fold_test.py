#!/usr/bin/env python3
"""Deterministic tests for the noun-attached `of` fold in spacy_relations.py.

The rule: a noun-attached PP folds into its noun phrase
("the destruction of the city" is one NP headed by "destruction");
its tokens get no verb-frame context, so a noun-attached "of" never
emits (verb, of). Only a prep attaching directly to the verb marks a
verb frame ("dreamed of Paris" keeps (dream, of)).

Transparent partitive heads ("the first volume of the encyclopedia")
pass kind through from the of-complement: the teacher reads the
complement's NER/lexicon, not the head's.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import spacy_relations as sr  # noqa: E402

import spacy  # noqa: E402

_nlp = spacy.load("en_core_web_sm")

check_failed = False


def check(name, cond):
    global check_failed
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        check_failed = True


def parse(text):
    doc = _nlp(text)
    return [doc], [[(sent, i) for i, sent in enumerate(doc.sents)]]


def contexts(text):
    docs, sent_index = parse(text)
    return sr._verb_contexts(docs, sent_index)


def descrs(text):
    docs, sent_index = parse(text)
    turns = [{"speaker": "n", "text": text, "session_id": "s"}]
    return sr._np_mention_descriptors(docs, sent_index, turns)


def ctx_for(ctx, text, word):
    """All (verb, prep) contexts for tokens whose text matches `word`."""
    docs, _ = parse(text)
    want = {t.i for t in docs[0] if t.text.lower() == word.lower()}
    out = []
    for (turn_idx, tok_i), cs in ctx.items():
        if tok_i in want:
            out.extend((c["verb"], c["prep"]) for c in cs)
    return out


# 1. Noun-attached "of" is folded: no verb-frame context for the PP.
# "destruction" (dobj of witnessed) keeps its own frame.
ctx = contexts("He witnessed the destruction of the city.")
city_ctx = []
docs, _ = parse("He witnessed the destruction of the city.")
doc = docs[0]
city_toks = {t.i for t in doc if t.text.lower() in ("city", "the")
             and t.head.text.lower() in ("of", "city")}
for (ti, tok_i), cs in ctx.items():
    if tok_i in {t.i for t in doc if t.text.lower() == "city"}:
        city_ctx.extend((c["verb"], c["prep"]) for c in cs)
check("fold: (witness, of) not emitted for city", city_ctx == [])
destr_ctx = []
for (ti, tok_i), cs in ctx.items():
    if tok_i in {t.i for t in doc if t.text.lower() == "destruction"}:
        destr_ctx.extend((c["verb"], c["prep"]) for c in cs)
check("fold: head keeps its own frame",
      any(v == "witness" for v, p in destr_ctx))

# 2. Verb-attached "of" is a genuine verb frame: kept.
ctx = contexts("He dreamed of Paris.")
paris_ctx = []
docs, _ = parse("He dreamed of Paris.")
doc = docs[0]
for (ti, tok_i), cs in ctx.items():
    if tok_i in {t.i for t in doc if t.text.lower() == "paris"}:
        paris_ctx.extend((c["verb"], c["prep"]) for c in cs)
check("verb-attached of kept: (dream, of)",
      ("dream", "of") in paris_ctx)

# 3. "die of" style: verb-attached of on another verb.
ctx = contexts("He died of hunger.")
hunger_ctx = []
docs, _ = parse("He died of hunger.")
doc = docs[0]
for (ti, tok_i), cs in ctx.items():
    if tok_i in {t.i for t in doc if t.text.lower() == "hunger"}:
        hunger_ctx.extend((c["verb"], c["prep"]) for c in cs)
check("verb-attached of kept: (die, of)",
      ("die", "of") in hunger_ctx)

# 4. Transparent head: teacher reads the of-complement.
ds = descrs("He read the first volume of the encyclopedia.")
vol = [d for d in ds if d.get("head_lemma") == "encyclopedia"
       and d.get("transparent_of")]
check("transparent head passes kind through",
      len(vol) == 1 and "volume" in vol[0]["text"].lower())
check("transparent head not double-counted as volume-headed",
      not any(d.get("head_lemma") == "volume" for d in ds))

# 5. Ordinary heads are untouched: head kind wins, no marker.
ds = descrs("The king of Bavaria arrived.")
king = [d for d in ds if d.get("head_lemma") == "king"]
check("ordinary head keeps head lemma",
      len(king) == 1 and not king[0].get("transparent_of"))
check("ordinary head keeps head NER",
      king[0].get("ner_label") in ("PERSON", ""))

# 6. Adjective-attached "of" is outside this mechanism.
ds = descrs("He was afraid of the dark.")
dark = [d for d in ds if d.get("head_lemma") == "dark"]
check("adjective-attached of: no transparent marker",
      len(dark) == 1 and not dark[0].get("transparent_of"))

# 7. Fold also covers other noun-attached preps, e.g. "in".
ctx = contexts("He described the meeting in Paris.")
paris_ctx = []
docs, _ = parse("He described the meeting in Paris.")
doc = docs[0]
for (ti, tok_i), cs in ctx.items():
    if tok_i in {t.i for t in doc if t.text.lower() == "paris"}:
        paris_ctx.extend((c["verb"], c["prep"]) for c in cs)
check("fold: noun-attached in suppressed",
      ("describe", "in") not in paris_ctx)

sys.exit(1 if check_failed else 0)
