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
* ``he/she/him/her`` resolve to the most *salient* preceding person
  mention that is not the speaker (a speaker referring to themselves
  says "I"). Personhood is grammatical, not guessed, and is decided by
  the compiled ``behood`` Rust lib (https://github.com/RooAGI/Behood;
  install with ``cargo install --git https://github.com/RooAGI/Behood``):
  NER PERSON,
  participant names, proper nouns repeated in the conversation, and
  names introduced by "named"/"met"/"called" ("a woman named Jean").
  A one-off capitalized word NER did not recognize ("Nature",
  "Turtles", "Shepherd") is not a person, so no pronoun resolves to
  it;
* ``they/them`` resolve to the nearest preceding *coordinated* person
  set ("Jon and Maria ... They ..."), emitting one triple per member,
  and are dropped when there is no coordination;
* a coordinated subject ("Jon and Maria visited Rome") credits every
  person conjunct, emitting one triple per member;
* ``it/this/that/these/those`` resolve to the most *salient* preceding
  non-person noun phrase ("I bought a car. I sold it."). Whether a
  chunk is a referring expression at all is decided by the compiled
  ``behood`` entityhood layer: pronouns, persons, interjections, bare
  exclamatory fragments ("Yay!"), clause-smuggled fragments, emoji,
  and bare negative formulas ("No prob.") are discourse noise, never
  antecedents;
* reflexives (``himself/herself/themselves``) resolve to the clause's
  nominal subject when it is a person.

Salience is deterministic centering-style scoring, not recency: each
candidate mention scores ``2 * role_weight + frequency - max(0,
sent_dist - 1)``, where role_weight is presentational attribute 4 >
subject 3 > direct object 2 > oblique 1 > predicative subject 0, and
frequency counts mentions of the same entity in the window. Remaining
ties break toward the earliest introduction. In copular clauses
salience flows to the predicate: only a presentational attribute
("That's a chill pic!") establishes a new topic and outranks even
subjects, while other copular nominals ("the skill is awesome",
"I'm a big fan of pottery") state the comment and score 0. Resolved
pronouns feed back into the mention stacks, so a topic gains salience
as it is referred to. An object pronoun never resolves to its own
clause's subject ("he saw him": disjoint reference).

Antecedents are searched over the current and previous three sentences
and ranked by salience. A resolved triple carries a ``coref`` provenance note
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
import os
import shutil
import subprocess
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

# Verbs whose object introduces a person by name ("a woman named Jean",
# "I met Jon", "they called her Priya"): discourse support for
# personhood when NER did not label the name PERSON.
INTRODUCTION_LEMMAS = {"name", "call", "meet"}

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

# Grammatical-role weights for salience scoring: presentational
# attributes ("That's a chill pic!") outrank subjects, which outrank
# direct objects, which outrank obliques. Other copular nominals --
# predicative subjects ("the skill is awesome") and non-presentational
# attributes ("I'm a big fan of pottery") -- state the comment and
# score 0: they anchor reference only when nothing better exists.
ROLE_W = {"psubj": 4, "subj": 3, "obj": 2, "obl": 1, "pred": 0}

# Linking verbs whose nominal subjects are predicative comments, not
# topics ("the skill is awesome"): demoted to oblique for non-persons.
COPULA_LEMMAS = {"be", "seem", "become"}

# Modifier relations that anchor a noun phrase to a referent ("my
# dog", "nice setup", "the birthday cakes"). A ROOT-headed chunk in a
# verbless sentence with none of these is a bare exclamation
# ("Yay!", "Dang", "Cheers"): discourse, not a referring NP.
ANCHORING_MODS = {"det", "poss", "amod", "compound", "nummod"}

# Determiners marking a bare negative NP that exhausts its sentence
# ("No prob.", "No way."): a discourse formula, not a referring
# expression, so it never enters the antecedent stack.
FORMULA_DETS = {"no", "nah", "nope"}


def _demonstrative_subject(head):
    """Whether the copula's subject presents new information."""
    return any(
        c.dep_ in ("nsubj", "nsubjpass")
        and c.text.lower() in ("that", "this", "it")
        for c in head.children
    )


def role_of(dep, head=None, is_person=False):
    """Grammatical-role salience class for a mention.

    In a copular clause salience flows to the predicate, not the
    subject -- but only a presentational attribute ("That's a chill
    pic!") establishes a new topic. Other copular nominals
    ("the skill is awesome", "I'm a big fan of pottery") state the
    comment and score 0. Persons are never demoted: on the animacy
    hierarchy a person ("John is happy", "The CEO is Maria") stays
    topic-worthy even in a copular clause.
    """
    head_lemma = head.lemma_ if head is not None else None
    copular = head_lemma in COPULA_LEMMAS
    if dep in ("nsubj", "nsubjpass", "csubj"):
        if copular and not is_person:
            return "pred"
        return "subj"
    if dep == "attr" and copular:
        if is_person:
            return "subj"
        # "That's a chill pic! Where did you find it?"
        if _demonstrative_subject(head):
            return "psubj"
        return "pred"
    if dep in ("dobj", "obj"):
        return "obj"
    return "obl"


def _is_neg_formula(chunk, sent):
    """A bare negative NP exhausting a verbless sentence ("No prob.")."""
    if chunk.start != sent.start:
        return False
    if any(not t.is_punct for t in sent if t.i >= chunk.end):
        return False
    return any(
        c.dep_ == "det" and c.text.lower() in FORMULA_DETS
        for c in chunk.root.children
    )

def _sentence_has_matrix_verb(sent):
    """Whether the sentence has a verb outside any subordinate clause.

    "No prob, always good to chat about those tranquil times" contains
    a verb ("chat") but only inside an xcomp clause: the matrix is
    verbless, so the sentence is a fragment. A verb reached through a
    clause relation (relcl/acl/advcl/ccomp/xcomp/parataxis) does not
    count.
    """
    for tok in sent:
        if tok.pos_ not in ("VERB", "AUX"):
            continue
        node = tok
        subordinate = False
        while True:
            if node.dep_ in CLAUSE_DEPS:
                subordinate = True
                break
            if node.head.i == node.i or node.dep_ == "ROOT":
                break
            node = node.head
        if not subordinate:
            return True
    return False


# Confidence cap for any triple built on a resolved pronoun.
COREF_CONF = 0.8


def fail(message, code):
    print(json.dumps({"error": message}), file=sys.stderr)
    return code


def _behood_bin():
    """Path to the compiled `behood` classifier, if available.

    The classifier lives in its own repo
    (https://github.com/RooAGI/Behood); install it with
    `cargo install --git https://github.com/RooAGI/Behood` (or
    `cargo install behood` once published). BEHOOD_BIN overrides PATH
    discovery. When no binary is found the pure-Python fallback in
    this script applies.
    """
    env = os.environ.get("BEHOOD_BIN")
    if env:
        return env
    return shutil.which("behood")


def _classify(descriptors, chunk_descriptors, speaker_names):
    """Classify mentions and chunks via the Rust behood lib.

    Returns (personhood_verdicts, entity_verdicts), each keyed by the
    caller-assigned string id, or (None, None) when the binary is
    missing or fails, in which case the pure-Python fallbacks apply.
    """
    binary = _behood_bin()
    if binary is None:
        return None, None
    payload = {
        "strategy": "discourse",
        "mentions": descriptors,
        "chunks": chunk_descriptors,
        "context": {"speaker_names": [s.lower() for s in speaker_names]},
    }
    try:
        proc = subprocess.run(
            [binary],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        print(json.dumps({"warning": f"behood_spawn_failed: {exc}"}),
              file=sys.stderr)
        return None, None
    if proc.returncode != 0:
        print(json.dumps({"warning": "behood_classifier_failed: "
                          f"{proc.stderr.strip()[:200]}"}), file=sys.stderr)
        return None, None
    try:
        data = json.loads(proc.stdout)
        verdicts = data["verdicts"]
        entity_verdicts = data.get("entity_verdicts", [])
    except Exception as exc:
        print(json.dumps({"warning": f"behood_bad_output: {exc}"}),
              file=sys.stderr)
        return None, None
    return ({v["id"]: v["is_person"] for v in verdicts},
            {v["id"]: v["is_entity"] for v in entity_verdicts})


def is_person_token(tok, ctx=None, sent_idx=None):
    """Whether a token can be a person mention (for antecedent search).

    The verdict comes from the compiled `behood` Rust lib
    (https://github.com/RooAGI/Behood; installed as the `behood`
    binary): NER PERSON is always a
    person, non-person NER
    labels never are, and an unrecognized proper noun counts only with
    discourse support (participant name, repetition, or introduction by
    a naming verb). When the classifier is unavailable, the
    pure-Python fallback below applies.

    Verdicts are keyed by the stable ``f"{sent_idx}:{tok.i}"`` id, never
    by object identity: spaCy token wrappers are not guaranteed stable
    across passes over the document.
    """
    if (ctx is not None and ctx.personhood_verdicts is not None
            and sent_idx is not None):
        verdict = ctx.personhood_verdicts.get(f"{sent_idx}:{tok.i}")
        if verdict is not None:
            return verdict
    return _python_is_person_token(tok, ctx)


def _python_is_person_token(tok, ctx=None):
    """Pure-Python personhood fallback (mirrors the Rust DiscourseEvidence).

    Used only when the compiled classifier cannot run.
    """
    if tok.ent_type_ == "PERSON":
        return True
    if tok.ent_type_ in NON_PERSON_ENTS:
        return False
    if not (tok.pos_ == "PROPN"
            or (tok.text[:1].isupper() and tok.pos_ in ("NOUN", "PROPN"))):
        return False
    if ctx is None:
        return True
    key = tok.text.lower()
    if key in ctx.speaker_names:
        return True
    if ctx.name_counts.get(key, 0) >= 2:
        return True
    return tok.head.lemma_.lower() in INTRODUCTION_LEMMAS


def fallback_name_counts(docs):
    """Conversation-wide counts of proper-noun fallback candidates.

    Only tokens that would reach the proper-noun fallback in
    is_person_token are counted: NER PERSON needs no discourse
    support, and NER non-person labels are never persons.
    """
    counts = {}
    for doc in docs:
        for tok in doc:
            if tok.ent_type_ == "PERSON" or tok.ent_type_ in NON_PERSON_ENTS:
                continue
            if tok.pos_ == "PROPN" or (
                tok.text[:1].isupper() and tok.pos_ in ("NOUN", "PROPN")
            ):
                key = tok.text.lower()
                counts[key] = counts.get(key, 0) + 1
    return counts


def _is_entity_chunk(chunk_id, chunk, sent, doc, ctx=None):
    """Whether a noun chunk is a referring expression (antecedent search).

    The verdict comes from the compiled `behood` Rust lib
    (https://github.com/RooAGI/Behood; installed as the `behood`
    binary): a chunk is rejected when the grammar shows it is discourse
    noise -- pronoun root, person root, adverbial/interjection role,
    bare exclamatory fragment, clause-smuggled fragment, no letters, or
    a bare negative formula. When the classifier is unavailable, the
    pure-Python fallback below applies.

    Verdicts are keyed by the stable ``f"{sent_idx}:{root.i}"`` chunk id.
    """
    if ctx is not None and ctx.entity_verdicts is not None:
        verdict = ctx.entity_verdicts.get(chunk_id)
        if verdict is not None:
            return verdict
    return _python_is_entity_chunk(chunk, sent, doc, ctx)


def _python_is_entity_chunk(chunk, sent, doc, ctx=None):
    """Pure-Python entityhood fallback (mirrors the Rust ReferringExpression).

    Used only when the compiled classifier cannot run.
    """
    root = chunk.root
    if root.pos_ == "PRON":
        return False
    if _python_is_person_token(root, ctx):
        return False
    if root.dep_ in ("npadvmod", "intj"):
        return False
    if root.dep_ == "ROOT" and not _sentence_has_matrix_verb(sent):
        if any(t.dep_ in CLAUSE_DEPS for t in root.subtree if t.i != root.i):
            return False
        if not any(c.dep_ in ANCHORING_MODS for c in root.children):
            return False
    text, _ = phrase_text(root, doc)
    if not text or not any(c.isalpha() for c in text):
        return False
    if _is_neg_formula(chunk, sent):
        return False
    return True


def _chunk_descriptors(docs, sent_index):
    """Plain-data noun-chunk descriptors for the behood entityhood layer.

    Every surface fact the Rust ReferringExpression rule needs, so the
    binary can decide antecedent eligibility without parser objects.
    """
    out = []
    for doc, per_turn in zip(docs, sent_index):
        for sent, s_idx in per_turn:
            has_mv = _sentence_has_matrix_verb(sent)
            for chunk in doc.noun_chunks:
                if chunk.sent.start != sent.start:
                    continue
                root = chunk.root
                text, _ = phrase_text(root, doc)
                out.append({
                    "id": f"{s_idx}:{root.i}",
                    "text": text or "",
                    "root_pos": root.pos_,
                    "root_dep": root.dep_,
                    "root_mention_id": f"{s_idx}:{root.i}",
                    "has_matrix_verb": has_mv,
                    "has_clause_material": any(
                        t.dep_ in CLAUSE_DEPS
                        for t in root.subtree if t.i != root.i),
                    "has_anchoring_mod": any(
                        c.dep_ in ANCHORING_MODS for c in root.children),
                    "is_neg_formula": _is_neg_formula(chunk, sent),
                })
    return out


def phrase_text(noun, doc):
    """(text, token_ids) of a noun's phrase, minus dets and sub-clauses.

    Clauses are dropped at any depth, not just as direct children: a
    kept modifier can itself head a clause ("No prob, always good to
    chat about those tranquil times" -- "chat" is xcomp of "good",
    amod of "prob"), and that clause is not part of the noun phrase.
    What survives is genuinely nominal: bare participial modifiers
    ("a broken window") stay, clause-headed ones go. Relative clauses
    were already dropped; this extends the same rule to clauses nested
    under kept modifiers.

    Returns (None, None) for pronouns -- those go through coreference.
    """
    if noun.pos_ == "PRON":
        return None, None
    drop = set()

    def visit(node):
        for child in node.children:
            if child.dep_ in CLAUSE_DEPS:
                drop.update(t.i for t in child.subtree)
            else:
                visit(child)

    visit(noun)
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
    """Conversation-level mention stacks for grammatical coreference.

    Mentions carry a grammatical role ("psubj"/"subj"/"obj"/"obl"/
    "pred") and a normalized key; antecedents are picked by salience
    (``2 * role_weight + frequency - max(0, sent_dist - 1)``), with
    remaining ties broken toward the earliest introduction.
    """

    def __init__(self, speakers, name_counts=None, verdicts=None,
                 entity_verdicts=None):
        # Distinct speaker display names, in first-seen order.
        self.speakers = speakers
        # Lowercased participant names: exempt from the discourse-support
        # requirement for proper-noun personhood.
        self.speaker_names = {s.lower() for s in speakers}
        # Conversation-wide counts of proper-noun fallback candidates,
        # for the repeated-mention personhood rule (Python fallback only;
        # the Rust classifier counts internally).
        self.name_counts = name_counts or {}
        # Personhood verdicts from the compiled `behood` lib, keyed
        # by stable "sent_idx:tok.i" id. None when the classifier was
        # unavailable.
        self.personhood_verdicts = verdicts
        # Entityhood verdicts from the compiled `behood` lib, keyed
        # by stable "sent_idx:root.i" chunk id. None when the
        # classifier was unavailable.
        self.entity_verdicts = entity_verdicts
        # Person mentions in document order:
        # {"sent", "tok_i", "key", "text", "role", "number"?}.
        self.person_mentions = []
        # Non-person noun-phrase mentions in document order:
        # {"sent", "tok_i", "key", "text", "role", "ids", "number"?}.
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
                     "key": speaker.lower(), "text": speaker,
                     "role": role_of(tok.dep_, tok.head, is_person=True),
                     "tok": tok}
                )
            elif low in YOU_PRONOUNS:
                other = self.other_speaker(speaker)
                if other is not None:
                    self.person_mentions.append(
                        {"sent": sent_idx, "tok_i": tok.i,
                         "key": other.lower(), "text": other,
                         "role": role_of(tok.dep_, tok.head, is_person=True),
                         "tok": tok}
                    )
            elif is_person_token(tok, self, sent_idx):
                nums = tok.morph.get("Number")
                self.person_mentions.append(
                    {"sent": sent_idx, "tok_i": tok.i,
                     "key": tok.text.lower(), "text": tok.text,
                     "role": role_of(tok.dep_, tok.head, is_person=True),
                     "number": nums[0] if nums else None,
                     "tok": tok}
                )
        for chunk in doc.noun_chunks:
            if chunk.sent.start != sent.start:
                continue
            root = chunk.root
            # Entityhood verdict from the compiled `behood` lib: the
            # grammar decides whether this chunk is a referring
            # expression at all (pronouns, persons, interjections,
            # bare fragments, emoji, and negative formulas are
            # discourse noise, never antecedents).
            if not _is_entity_chunk(f"{sent_idx}:{root.i}", chunk, sent,
                                    doc, self):
                continue
            text, ids = phrase_text(root, doc)
            if not text:
                continue
            nums = root.morph.get("Number")
            self.entity_mentions.append(
                {"sent": sent_idx, "tok_i": root.i,
                 "key": text.lower(), "text": text, "ids": ids,
                 "role": role_of(root.dep_, root.head),
                 "number": nums[0] if nums else None}
            )

    def _in_window(self, m, sent_idx, tok_i):
        return (
            m["sent"] >= sent_idx - SENT_WINDOW
            and (m["sent"] < sent_idx
                 or (m["sent"] == sent_idx and m["tok_i"] < tok_i))
        )

    def _pick(self, cands, cur_sent):
        """Highest-salience candidate.

        ``2 * role_weight + frequency - max(0, sent_dist - 1)``:
        presentational attributes outrank subjects, subjects outrank
        direct objects, obliques follow, and predicative subjects
        score 0. Frequency counts mentions of the same entity in the
        window; adjacent sentences carry no recency penalty, decay
        starts at distance 2. Remaining ties go to the
        earliest-introduced entity (topic continuity).
        """
        reps = {}
        for m in cands:
            r = reps.get(m["key"])
            pos = (m["sent"], m["tok_i"])
            if r is None:
                reps[m["key"]] = {"m": m, "freq": 1,
                                  "first": pos, "last": pos}
            else:
                r["freq"] += 1
                r["first"] = min(r["first"], pos)
                if pos > r["last"]:
                    r["last"] = pos
                    r["m"] = m
        best = None
        best_key = None
        for key, r in reps.items():
            m = r["m"]
            score = (2 * ROLE_W[m["role"]] + r["freq"]
                     - max(0, cur_sent - m["sent"] - 1))
            # Negated first-mention position: earlier introduction wins.
            tie = (score, -r["first"][0], -r["first"][1])
            if best_key is None or tie > best_key:
                best_key = tie
                best = m
        return best

    def nearest_person(self, sent_idx, tok_i, speaker):
        """Nearest preceding person mention that is not the speaker.

        Used only as the structural anchor for they/them coordination;
        singular pronouns use salience (best_person).
        """
        snorm = speaker.lower()
        cands = [
            m for m in self.person_mentions
            if self._in_window(m, sent_idx, tok_i)
            and m["key"] != snorm
        ]
        return cands[-1] if cands else None

    def best_person(self, sent_idx, tok_i, speaker, exclude=frozenset(),
                    number=None):
        """Most salient preceding person mention (not the speaker).

        `number="Sing"` excludes plural mentions ("Turtles") as
        antecedents for he/she/him/her.
        """
        snorm = speaker.lower()
        cands = []
        for m in self.person_mentions:
            if not self._in_window(m, sent_idx, tok_i):
                continue
            if m["key"] == snorm or m["key"] in exclude:
                continue
            if number == "Sing" and m.get("number") == "Plur":
                continue
            cands.append(m)
        if not cands:
            return None
        return self._pick(cands, sent_idx)

    def best_entity(self, sent_idx, tok_i, exclude=frozenset(),
                    number=None):
        """Most salient preceding non-person noun phrase."""
        cands = []
        for m in self.entity_mentions:
            if not self._in_window(m, sent_idx, tok_i):
                continue
            if m["key"] in exclude:
                continue
            nums = m.get("number")
            if nums and number and nums != number:
                continue
            cands.append(m)
        if not cands:
            return None
        return self._pick(cands, sent_idx)


def conj_person_names(tok, speaker, ctx=None, sent_idx=None):
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
        elif is_person_token(t, ctx, sent_idx):
            names.append(t.text)
    seen = set()
    out = []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def resolve_pronoun(tok, speaker, sent_idx, ctx, as_subject,
                    exclude_keys=frozenset()):
    """Resolve one pronoun token: [(text, ids, note)] or None.

    Purely grammatical: person mentions, coordination, and noun phrases
    ranked by salience. Gender is never consulted. Neuter pronouns never
    resolve as subjects -- this index is person-centric, so a thing
    subject ("aerial yoga") would corrupt the person list.

    `exclude_keys` holds entity keys the pronoun must not corefer with:
    an object pronoun is disjoint from its clause's subject
    ("he saw him"). A successful resolution feeds back into the mention
    stacks, so referred-to topics gain salience.
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
        m = ctx.best_person(sent_idx, tok.i, speaker,
                             exclude=exclude_keys, number="Sing")
        if m is None:
            return None
        ctx.person_mentions.append(
            {"sent": sent_idx, "tok_i": tok.i,
             "key": m["key"], "text": m["text"],
             "role": role_of(tok.dep_, tok.head, is_person=True),
             "tok": m["tok"]}
        )
        return [(m["text"], set(), f"{tok.text}->{m['text']}")]
    if low in PLURAL_PRONOUNS:
        m = ctx.nearest_person(sent_idx, tok.i, speaker)
        if m is None or m["key"] in exclude_keys:
            return None
        names = conj_person_names(m["tok"], speaker, ctx, sent_idx)
        if len(names) < 2:
            # "they" with no coordinated antecedent is ambiguous: drop.
            return None
        note = f"{tok.text}->({'+'.join(names)})"
        for n in names:
            ctx.person_mentions.append(
                {"sent": sent_idx, "tok_i": tok.i,
                 "key": n.lower(), "text": n,
                 "role": role_of(tok.dep_, tok.head, is_person=True),
                 "tok": m["tok"]}
            )
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
                    if is_person_token(child, ctx, sent_idx):
                        return [(child.text, set(),
                                 f"{tok.text}->{child.text}")]
        return None
    if low in NEUTER_PRONOUNS:
        if as_subject:
            return None
        number = "Plur" if low in ("these", "those") else "Sing"
        m = ctx.best_entity(sent_idx, tok.i, exclude=exclude_keys,
                             number=number)
        if m is None:
            return None
        ctx.entity_mentions.append(
            {"sent": sent_idx, "tok_i": tok.i,
             "key": m["key"], "text": m["text"], "ids": set(),
             "role": role_of(tok.dep_, tok.head),
             "number": m.get("number")}
        )
        return [(m["text"], m["ids"], f"{tok.text}->{m['text']}")]
    return None


def noun_phrase(noun, doc, ctx, sent_idx, speaker, exclude_keys=frozenset()):
    """[(text, token_ids, coref_note)] for an object noun; usually one."""
    if noun.pos_ == "PRON":
        refs = resolve_pronoun(noun, speaker, sent_idx, ctx,
                               as_subject=False, exclude_keys=exclude_keys)
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
        names = conj_person_names(child, speaker, ctx, sent_idx)
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
            # Disjoint reference: an object pronoun cannot corefer with
            # its own clause's subject ("he saw him").
            subj_keys = frozenset(s.lower() for s, _, _ in subjects)
            lemma = low_lemma(verb, doc_low)
            for child in verb.children:
                if child.dep_ == "dobj":
                    objs = noun_phrase(child, doc, ctx, sent_idx, speaker,
                                       exclude_keys=subj_keys)
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
                        objs = noun_phrase(pobj, doc, ctx, sent_idx, speaker,
                                           exclude_keys=subj_keys)
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
                                           speaker, exclude_keys=subj_keys)
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
    # Global sentence indexing, computed before classification: mention
    # and chunk ids are stable "sent_idx:tok.i" coordinates, never
    # object identity (spaCy token wrappers are not guaranteed stable
    # across passes over the document).
    sent_index = []
    sent_idx = 0
    for doc in docs:
        per_turn = []
        for sent in doc.sents:
            per_turn.append((sent, sent_idx))
            sent_idx += 1
        sent_index.append(per_turn)

    ctx = CorefCtx(speakers, fallback_name_counts(docs),
                   *_classify(
                       [{"id": f"{s_idx}:{tok.i}",
                         "text": tok.text,
                         "ner_label": tok.ent_type_,
                         "pos": tok.pos_,
                         "head_lemma": tok.head.lemma_}
                        for doc, per_turn in zip(docs, sent_index)
                        for sent, s_idx in per_turn
                        for tok in sent],
                       _chunk_descriptors(docs, sent_index),
                       speakers))
    sent_maps = []
    for turn, doc, per_turn in zip(turns, docs, sent_index):
        speaker = turn.get("speaker", "")
        sent_map = {}
        for sent, s_idx in per_turn:
            sent_map[sent.start] = s_idx
            ctx.add_sentence(speaker, sent, doc, s_idx)
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
