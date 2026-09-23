#!/usr/bin/env python3
"""Dependency-parse relation extraction for dialogue turns.

Reads dialogue turns on stdin (JSON), writes (subject, predicate, object)
triples on stdout (JSON). The rules below are *grammatical*, not lexical:
they operate on dependency labels, so no verb phrase is ever enumerated.
The only lexical data is the predicate string itself
(``verb_lemma[_prt][_prep]``); mapping predicates to semantic families
happens on the Rust side.

Pronoun handling is deterministic grammatical coreference -- never
name-to-gender guessing:

* ``I/we/me/us/myself`` resolve to the speaker (first person is
  grammatical, not gendered);
* ``you`` resolves to the other participant when the conversation has
  exactly two speakers, and is dropped otherwise;
* ``he/she/him/her`` resolve to the nearest preceding person mention
  that is not the speaker (a speaker referring to themselves says "I");
* ``they/them`` resolve to the nearest preceding *coordinated* person
  set ("Jon and Maria ... They ..."), emitting one triple per member,
  and are dropped when there is no coordination;
* a coordinated subject ("Jon and Maria visited Rome") credits every
  person conjunct, emitting one triple per member;
* ``it/this/that/these/those`` resolve to the nearest preceding
  non-person noun phrase ("I bought a car. I sold it.");
* reflexives (``himself/herself/themselves``) resolve to the clause's
  nominal subject when it is a person.

Antecedents are searched over the current and previous three sentences,
most recent first. A resolved triple carries a ``coref`` provenance note
(e.g. ``"she->Maria"``) and its confidence is capped at 0.8, so
downstream consumers can tell explicit mentions from resolved ones.

Protocol mirrors scripts/spacy_ner.py:
  stdin:  {"model": "en_core_web_sm", "turns": [
             {"speaker": str, "text": str, "session_id": str,
              "turn_idx": int, "doc_id": str, "session_date": str|null}]}
  stdout: {"relations": [
             {"subject": str, "predicate": str, "object": str,
              "is_place": bool, "session_id": str, "turn_idx": int,
              "doc_id": str, "evidence": str, "confidence": float,
              "coref": str|null}]}
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

# Entity labels that are never persons (a PROPN with one of these labels,
# e.g. "Rome"/GPE, must not become a "he"/"she" antecedent).
NON_PERSON_ENTS = {
    "GPE", "LOC", "FAC", "ORG", "PRODUCT", "EVENT", "WORK_OF_ART",
    "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY",
    "ORDINAL", "CARDINAL",
}

# Dependency labels that head a subordinate clause; their subtrees are cut
# out of object noun phrases ("a shelter I volunteer at" -> "a shelter").
CLAUSE_DEPS = {"relcl", "acl", "advcl", "ccomp", "xcomp", "parataxis"}

# Leading tokens stripped from an object noun phrase.
STRIP_FIRST = {
    "a", "an", "the", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
}

# First-person pronouns: always the speaker. Grammatical, not gendered.
SELF_PRONOUNS = {"i", "we", "me", "us", "myself"}

# Second person: the other participant, but only when the conversation has
# exactly two speakers (otherwise "you" is ambiguous).
YOU_PRONOUNS = {"you", "yourself", "yourselves"}

# Third-person singular: nearest preceding person mention that is not the
# speaker. No gender involved.
SINGULAR_PRONOUNS = {"he", "him", "his", "she", "her"}

# Third-person plural: nearest preceding coordinated person set.
PLURAL_PRONOUNS = {"they", "them"}

# Neuter demonstratives: nearest preceding non-person noun phrase.
NEUTER_PRONOUNS = {"it", "its", "this", "that", "these", "those"}

# Reflexives bind to the clause's nominal subject.
REFLEXIVE_PRONOUNS = {"himself", "herself", "itself", "themselves"}

# How far back (in sentences) an antecedent may be.
SENT_WINDOW = 3

# Confidence cap for any triple built on a resolved pronoun.
COREF_CONF = 0.8


def fail(message, code):
    print(json.dumps({"error": message}), file=sys.stderr)
    return code


def is_person_token(tok):
    """Whether a token can be a person mention (for antecedent search)."""
    if tok.ent_type_ == "PERSON":
        return True
    if tok.ent_type_ in NON_PERSON_ENTS:
        return False
    return tok.pos_ == "PROPN" or (
        tok.text[:1].isupper() and tok.pos_ in ("NOUN", "PROPN")
    )


def phrase_text(noun, doc):
    """(text, token_ids) of a noun's phrase, minus dets and sub-clauses.

    Returns (None, None) for pronouns -- those go through coreference.
    """
    if noun.pos_ == "PRON":
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
    # Leading interjections are discourse noise, not part of the phrase
    # ("Wow, nice setup" -> "nice setup").
    while toks and toks[0].pos_ == "INTJ":
        toks.pop(0)
    while toks and toks[-1].pos_ == "ADP":
        toks.pop()
    if not toks:
        return None, None
    text = doc[toks[0].i : toks[-1].i + 1].text.strip()
    if not text:
        return None, None
    return text, {t.i for t in toks}


class CorefCtx:
    """Conversation-level mention stacks for grammatical coreference."""

    def __init__(self, speakers):
        # Distinct speaker display names, in first-seen order.
        self.speakers = speakers
        # Person mentions in document order:
        # {"sent": int, "tok_i": int, "name": str, "tok": Token}.
        self.person_mentions = []
        # Non-person noun-phrase mentions in document order:
        # {"sent": int, "tok_i": int, "text": str, "ids": set[int]}.
        self.entity_mentions = []

    def other_speaker(self, speaker):
        """The other participant iff exactly two speakers exist."""
        if len(self.speakers) == 2:
            a, b = self.speakers
            if a.lower() == speaker.lower():
                return b
            if b.lower() == speaker.lower():
                return a
        return None

    def add_sentence(self, speaker, sent, doc, sent_idx):
        for tok in sent:
            low = tok.text.lower()
            if low in SELF_PRONOUNS:
                self.person_mentions.append(
                    {"sent": sent_idx, "tok_i": tok.i,
                     "name": speaker, "tok": tok}
                )
            elif low in YOU_PRONOUNS:
                other = self.other_speaker(speaker)
                if other is not None:
                    self.person_mentions.append(
                        {"sent": sent_idx, "tok_i": tok.i,
                         "name": other, "tok": tok}
                    )
            elif is_person_token(tok):
                nums = tok.morph.get("Number")
                self.person_mentions.append(
                    {"sent": sent_idx, "tok_i": tok.i,
                     "name": tok.text, "tok": tok,
                     "number": nums[0] if nums else None}
                )
        for chunk in doc.noun_chunks:
            if chunk.sent.start != sent.start:
                continue
            root = chunk.root
            if root.pos_ == "PRON":
                continue
            if is_person_token(root):
                continue
            if root.dep_ in ("npadvmod",):
                continue
            text, ids = phrase_text(root, doc)
            if text:
                nums = root.morph.get("Number")
                self.entity_mentions.append(
                    {"sent": sent_idx, "tok_i": root.i,
                     "text": text, "ids": ids,
                     "number": nums[0] if nums else None}
                )

    def _in_window(self, m, sent_idx, tok_i):
        return (
            m["sent"] >= sent_idx - SENT_WINDOW
            and (m["sent"] < sent_idx
                 or (m["sent"] == sent_idx and m["tok_i"] < tok_i))
        )

    def nearest_person(self, sent_idx, tok_i, speaker, number=None):
        """Nearest preceding person mention that is not the speaker.

        `number="Sing"` excludes plural mentions ("Turtles") as
        antecedents for he/she/him/her.
        """
        snorm = speaker.lower()
        cands = []
        for m in self.person_mentions:
            if not self._in_window(m, sent_idx, tok_i):
                continue
            if m["name"].lower() == snorm:
                continue
            if number == "Sing" and m.get("number") == "Plur":
                continue
            cands.append(m)
        return cands[-1] if cands else None

    def nearest_entity(self, sent_idx, tok_i, number):
        """Nearest preceding non-person noun phrase with number agreement."""
        cands = []
        for m in self.entity_mentions:
            if not self._in_window(m, sent_idx, tok_i):
                continue
            nums = m.get("number")
            if nums and number and nums != number:
                continue
            cands.append(m)
        return cands[-1] if cands else None


def conj_person_names(tok, speaker):
    """Person names coordinated with a mention token ("Jon and Maria")."""
    group = [tok]
    if tok.dep_ == "conj":
        group.append(tok.head)
        group.extend(
            c for c in tok.head.children
            if c.dep_ == "conj" and c.i != tok.i
        )
    group.extend(c for c in tok.children if c.dep_ == "conj")
    names = []
    for t in group:
        low = t.text.lower()
        if low in SELF_PRONOUNS:
            names.append(speaker)
        elif is_person_token(t):
            names.append(t.text)
    seen = set()
    out = []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def resolve_pronoun(tok, speaker, sent_idx, ctx, as_subject):
    """Resolve one pronoun token: [(text, ids, note)] or None.

    Purely grammatical: person mentions, coordination, and noun phrases
    ordered by recency. Gender is never consulted. Neuter pronouns never
    resolve as subjects -- this index is person-centric, so a thing
    subject ("aerial yoga") would corrupt the person list.
    """
    low = tok.text.lower()
    if low in SELF_PRONOUNS:
        return [(speaker, set(), None)]
    if low in YOU_PRONOUNS:
        # "you two / you all / you guys" is plural: every participant.
        plural = any(
            t.text.lower() in ("all", "both", "guys", "two", "three")
            for t in tok.subtree
        )
        if plural:
            if len(ctx.speakers) < 2:
                return None
            note = f"{tok.text}->({'+'.join(ctx.speakers)})"
            return [(s, set(), note) for s in ctx.speakers]
        other = ctx.other_speaker(speaker)
        if other is None:
            return None
        return [(other, set(), f"{tok.text}->{other}")]
    if low in SINGULAR_PRONOUNS:
        m = ctx.nearest_person(sent_idx, tok.i, speaker, number="Sing")
        if m is None:
            return None
        return [(m["name"], set(), f"{tok.text}->{m['name']}")]
    if low in PLURAL_PRONOUNS:
        m = ctx.nearest_person(sent_idx, tok.i, speaker)
        if m is None:
            return None
        names = conj_person_names(m["tok"], speaker)
        if len(names) < 2:
            # "they" with no coordinated antecedent is ambiguous: drop.
            return None
        note = f"{tok.text}->({'+'.join(names)})"
        return [(n, set(), note) for n in names]
    if low in REFLEXIVE_PRONOUNS:
        verb = tok.head if tok.head.pos_ in ("VERB", "AUX") else None
        if verb is not None:
            for child in verb.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    clow = child.text.lower()
                    if clow in SELF_PRONOUNS:
                        return [(speaker, set(),
                                 f"{tok.text}->{speaker}")]
                    if is_person_token(child):
                        return [(child.text, set(),
                                 f"{tok.text}->{child.text}")]
        return None
    if low in NEUTER_PRONOUNS:
        if as_subject:
            return None
        number = "Plur" if low in ("these", "those") else "Sing"
        m = ctx.nearest_entity(sent_idx, tok.i, number)
        if m is None:
            return None
        return [(m["text"], m["ids"], f"{tok.text}->{m['text']}")]
    return None


def noun_phrase(noun, doc, ctx, sent_idx, speaker):
    """[(text, token_ids, coref_note)] for an object noun; usually one."""
    if noun.pos_ == "PRON":
        refs = resolve_pronoun(noun, speaker, sent_idx, ctx,
                               as_subject=False)
        return refs if refs else []
    text, ids = phrase_text(noun, doc)
    if not text:
        return []
    return [(text, ids, None)]


def resolve_name(child, speaker, conf, ctx, sent_idx):
    """[(name, confidence, coref_note)] for one subject token.

    A coordinated subject ("Jon and Maria ...") credits every person
    conjunct. An explicit PROPN subject keeps the historical permissive
    rule; the stricter person check applies to antecedent search only.
    """
    if child.pos_ == "PRON":
        refs = resolve_pronoun(child, speaker, sent_idx, ctx,
                               as_subject=True)
        if not refs:
            return []
        return [
            (text, conf if note is None else min(conf, COREF_CONF), note)
            for text, _ids, note in refs
        ]
    if child.pos_ == "PROPN" or (
        child.text[:1].isupper() and child.pos_ in ("NOUN", "PROPN")
    ):
        names = conj_person_names(child, speaker)
        head_low = child.text.lower()
        head_name = speaker if head_low in SELF_PRONOUNS else child.text
        if head_name.lower() not in {n.lower() for n in names}:
            names = [head_name] + names
        return [(n, conf, None) for n in names]
    return []


def resolve_subject(verb, sent, speaker, ctx, sent_idx):
    """[(name, confidence, coref_note)] or [] when unattributable.

    General grammatical rules, in order:
    1. the verb's own nominal subject (pronouns via coreference);
    2. a subject-less ROOT (fragment): the speaker;
    3. a subject-less sentence-initial verb (dropped subject): the speaker;
    4. subject inheritance from the nearest ancestor verb with a subject
       ("to give out food" under "we went");
    5. otherwise unattributable: skip.
    """
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return resolve_name(child, speaker, 1.0, ctx, sent_idx)
    if verb.dep_ == "ROOT":
        return [(speaker, 0.9, None)]
    verbs = [t for t in sent if t.pos_ in ("VERB", "AUX")]
    if verbs and verbs[0].i == verb.i:
        return [(speaker, 0.9, None)]
    seen = {verb.i}
    head = verb.head
    while head.i not in seen:
        seen.add(head.i)
        if head.pos_ in ("VERB", "AUX"):
            for child in head.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    return resolve_name(child, speaker, 0.85, ctx, sent_idx)
        if head.dep_ == "ROOT":
            break
        head = head.head
    return []


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


def extract_doc(turn, doc, doc_low, ctx, sent_map):
    speaker = turn.get("speaker", "")
    out = []
    for sent in doc.sents:
        sent_idx = sent_map[sent.start]
        for verb in sent:
            is_root = verb.dep_ == "ROOT"
            if verb.pos_ not in ("VERB", "AUX") and not is_root:
                continue
            subjects = resolve_subject(verb, sent, speaker, ctx, sent_idx)
            if not subjects:
                continue
            lemma = low_lemma(verb, doc_low)
            for child in verb.children:
                if child.dep_ == "dobj":
                    objs = noun_phrase(child, doc, ctx, sent_idx, speaker)
                    for obj, ids, onote in objs:
                        for subject, sconf, snote in subjects:
                            notes = [n for n in (snote, onote) if n]
                            conf = sconf if not notes else min(sconf,
                                                              COREF_CONF)
                            out.append(
                                (subject, predicate_for(verb, lemma),
                                 obj, ids, conf,
                                 ";".join(notes) if notes else None)
                            )
                elif child.dep_ == "prep":
                    pobjs = [c for c in child.children if c.dep_ == "pobj"]
                    for pobj in pobjs:
                        objs = noun_phrase(pobj, doc, ctx, sent_idx, speaker)
                        for obj, ids, onote in objs:
                            for subject, sconf, snote in subjects:
                                notes = [n for n in (snote, onote) if n]
                                conf = sconf if not notes else min(sconf,
                                                                  COREF_CONF)
                                out.append(
                                    (subject,
                                     predicate_for(verb, lemma, child.lemma_),
                                     obj, ids, conf,
                                     ";".join(notes) if notes else None)
                                )
                    # Stranded preposition ("the shelter I volunteer at"):
                    # the relative clause's antecedent is the real object.
                    if verb.dep_ == "relcl" and verb.head.pos_ in (
                        "NOUN", "PROPN",
                    ):
                        objs = noun_phrase(verb.head, doc, ctx, sent_idx,
                                           speaker)
                        for obj, ids, onote in objs:
                            for subject, sconf, snote in subjects:
                                notes = [n for n in (snote, onote) if n]
                                conf = min(sconf, 0.85)
                                if notes:
                                    conf = min(conf, COREF_CONF)
                                out.append(
                                    (subject,
                                     predicate_for(verb, lemma, child.lemma_),
                                     obj, ids, conf,
                                     ";".join(notes) if notes else None)
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

    # Conversation-level coreference state: mention stacks are built over
    # all turns in order before any triple is extracted.
    speakers = []
    for t in turns:
        s = t.get("speaker", "")
        if s.lower() not in {x.lower() for x in speakers}:
            speakers.append(s)
    ctx = CorefCtx(speakers)
    sent_maps = []
    sent_idx = 0
    for turn, doc in zip(turns, docs):
        speaker = turn.get("speaker", "")
        sent_map = {}
        for sent in doc.sents:
            sent_map[sent.start] = sent_idx
            ctx.add_sentence(speaker, sent, doc, sent_idx)
            sent_idx += 1
        sent_maps.append(sent_map)

    relations = []
    for turn, doc, doc_low, sent_map in zip(turns, docs, docs_low,
                                            sent_maps):
        try:
            triples = extract_doc(turn, doc, doc_low, ctx, sent_map)
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
        for subject, predicate, obj, ids, conf, coref in triples:
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
                    "coref": coref,
                }
            )

    json.dump({"relations": relations}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
