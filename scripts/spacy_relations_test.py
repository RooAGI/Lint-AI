#!/usr/bin/env python3
"""Regression tests for scripts/spacy_relations.py coreference rules.

Runs the extractor as a subprocess on small scripted conversations and
asserts the emitted triples. Grammatical coreference only: no gender is
ever consulted.
"""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("spacy_relations.py")


def extract(turns):
    payload = {"model": "en_core_web_sm", "turns": turns}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"extractor failed: {proc.stderr}"
    return json.loads(proc.stdout)["relations"]


def T(speaker, text, idx=0):
    return {
        "speaker": speaker,
        "text": text,
        "session_id": "s",
        "turn_idx": idx,
        "doc_id": "d",
        "session_date": None,
    }


def triples(rels):
    return {
        (r["subject"], r["predicate"], r["object"]): r for r in rels
    }


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        check.failed = True


check.failed = False

# 1. she -> Maria (nearest non-speaker person mention); me -> speaker.
rels = triples(extract([T("Jon", "I met Maria yesterday. She invited me to Paris. ")]))
r = rels.get(("Maria", "invite_to", "Paris"))
check("she->Maria subject", r is not None and r["coref"] == "She->Maria"
      and r["confidence"] <= 0.8)
r = rels.get(("Maria", "invite", "Jon"))
check("me->Jon object", r is not None and "She->Maria" in (r["coref"] or ""))
check("explicit triple kept", ("Jon", "meet", "Maria") in rels)

# 2. they -> coordinated set; it -> Rome (place flag survives coref).
rels = triples(extract([T("Gina", "Jon and Maria visited Rome. They loved it. ")]))
for subj in ("Jon", "Maria"):
    r = rels.get((subj, "love", "Rome"))
    check(f"they->({subj}) love Rome",
          r is not None and r["coref"] is not None
          and "They->" in r["coref"] and "it->Rome" in r["coref"]
          and r["is_place"] is True)
check("coordinated subject credits both",
      ("Jon", "visit", "Rome") in rels and ("Maria", "visit", "Rome") in rels)

# 3. it -> nearest non-person noun phrase.
rels = triples(extract([T("Jon", "I bought a car. I sold it yesterday. ")]))
r = rels.get(("Jon", "sell", "car"))
check("it->car", r is not None and r["coref"] == "it->car")

# 4. you -> the other participant (exactly two speakers).
rels = triples(extract([
    T("Jon", "Do you like Rome? ", 0),
    T("Gina", "I love it. ", 1),
]))
r = rels.get(("Gina", "like", "Rome"))
check("you->Gina", r is not None and r["coref"] == "you->Gina")
r = rels.get(("Gina", "love", "Rome"))
check("cross-turn it->Rome", r is not None and r["coref"] == "it->Rome")

# 5. you with three speakers is ambiguous: dropped.
rels = triples(extract([
    T("Jon", "Do you like Rome? ", 0),
    T("Gina", "I do. ", 1),
    T("Maria", "I do. ", 2),
]))
check("you dropped with 3 speakers",
      not any(r["subject"] in ("Gina", "Maria") and r["predicate"] == "like"
              for r in rels))

# 6. Pleonastic / antecedent-less pronouns: no triples.
rels = extract([T("Jon", "It is raining. ")])
check("pleonastic it dropped", rels == [])
rels = extract([T("Jon", "He left. ")])
check("antecedent-less he dropped", rels == [])

# 7. Speaker-skip: "Maria and I ... She ..." -> Maria, not the speaker.
rels = triples(extract([T("Jon", "Maria and I visited Rome. She loved it. ")]))
r = rels.get(("Maria", "love", "Rome"))
check("she skips speaker", r is not None and r["coref"] is not None
      and r["coref"].startswith("She->Maria"))

# 8. Reflexive binds to the clause subject.
rels = triples(extract([T("Jon", "Jon hurt himself. ")]))
r = rels.get(("Jon", "hurt", "Jon"))
check("himself->Jon", r is not None and r["coref"] == "himself->Jon")

# 9. No gender guessing: a place PROPN never becomes a she/he antecedent.
rels = triples(extract([T("Jon", "I visited Rome. She was beautiful. ")]))
check("Rome not a she-antecedent",
      not any(r["subject"] == "Rome" for r in rels.values()))

# 10. Old behavior preserved (same output as the pre-coref script).
rels = triples(extract([T("Jon", "I took a trip to Rome. ")]))
r = rels.get(("Jon", "take", "trip to Rome"))
check("explicit triple unchanged",
      r is not None and r["coref"] is None and r["is_place"] is True)

# 11. Neuter pronouns never resolve as subjects (person-centric index).
rels = extract([T("Jon", "Aerial yoga is great. It keeps me fit. ")])
check("neuter subject dropped",
      not any(r["subject"] == "aerial yoga" for r in rels))

# 12. Number agreement: singular "it" skips a plural antecedent.
rels = triples(extract([T("Jon", "I met the students. I photographed it. ")]))
check("it skips plural antecedent",
      not any(p == "photograph" for (_, p, _) in rels))
rels = triples(extract([T("Jon", "I baked a cake. I photographed it. ")]))
r = rels.get(("Jon", "photograph", "cake"))
check("it takes singular antecedent", r is not None and r["coref"] == "it->cake")

# 13. Singular he/she/him/her skip plural mentions ("Turtles").
rels = extract([T("Nate", "Turtles bring me joy. I saw him at the pond. ")])
check("him skips plural Turtles",
      not any(r["subject"] == "Nate" and r["predicate"] == "see"
              for r in rels))

# 14. Plural "you two" fans out to every participant.
rels = triples(extract([
    T("Jon", "You two should visit Rome. ", 0),
    T("Gina", "We will. ", 1),
]))
check("you two -> both",
      ("Jon", "visit", "Rome") in rels and ("Gina", "visit", "Rome") in rels
      and rels[("Jon", "visit", "Rome")]["coref"] == "You->(Jon+Gina)")

# 15. Leading interjections are stripped from mention text.
rels = triples(extract([T("Dave", "Wow, nice setup! I love it. ")]))
r = rels.get(("Dave", "love", "nice setup"))
check("it->nice setup (no Wow)",
      r is not None and r["coref"] == "it->nice setup")

# 16. Salience beats recency for persons: She -> Maria, not nearer Luke.
rels = triples(extract([T("Jon", "I met Maria and Luke. She baked a cake. ")]))
r = rels.get(("Maria", "bake", "cake"))
check("She->Maria over nearer Luke",
      r is not None and r["coref"] == "She->Maria"
      and ("Luke", "bake", "cake") not in rels)

# 17. Clause subject outranks nearer object ("Maria told Luke she ...").
rels = triples(extract([T("Jon", "Maria told Luke she baked a cake. ")]))
r = rels.get(("Maria", "bake", "cake"))
check("She->Maria (subject beats object)",
      r is not None and r["coref"] == "she->Maria")

# 18. Topic continuity: "this" -> pottery, not the nearer copular
# subject "skill" (demoted: predicative subjects are the comment).
rels = triples(extract([T("Melanie", "I love pottery. "
                                  "The creativity and skill is awesome. "
                                  "Making it is calming. "
                                  "Look at this! ")]))
r = rels.get(("Melanie", "look_at", "pottery"))
check("this->pottery (topic over copular subject)",
      r is not None and r["coref"] == "this->pottery")

# 19. "include this" -> the intro, not the nearer "movie script".
rels = triples(extract([T("Joanna", "I just finished with the intro to my "
                                  "next movie script, and I decided to "
                                  "include this at the beginning. ")]))
r = rels.get(("Joanna", "include", "intro"))
check("this->the intro (first-mentioned wins)",
      r is not None and r["coref"] == "this->intro")

# 20. "No prob." is a discourse formula: "it" has no antecedent.
rels = extract([T("Nate", "No prob. I made it with coconut milk. ")])
check("No prob not an antecedent",
      not any(r["predicate"] == "make" for r in rels))

# 21. Disjoint reference: "he saw him" -> "him" cannot be "he".
rels = extract([T("Jon", "Maria said he saw him at the park. ")])
check("him disjoint from clause subject",
      not any(r["predicate"] == "see" for r in rels))

# 22. Presented attribute beats an older subject in context:
# "That's a chill pic! Where did you find it?" -> it->chill pic,
# not the subject of the previous sentence.
rels = triples(extract([
    T("Maria", "Nature's beauty reminds me to slow down and enjoy the small stuff. ", idx=0),
    T("John", "That's a chill pic! Where did you find it? ", idx=1),
]))
r = rels.get(("Maria", "find", "chill pic"))
check("it->chill pic over older subject",
      r is not None and r["coref"] == "you->Maria;it->chill pic")

# 23. Nominal-chunk hygiene (conv-49): a fragment smuggling a clause
# ("No prob, always good to chat about those tranquil times") is
# discourse, not an antecedent; "it" falls through to the real entity.
rels = triples(extract([
    T("Evan", "Yeah, it's like a little slice of paradise. I always feel so peaceful and serene when I'm there. ", idx=0),
    T("Sam", "Wow, it really seems like a peaceful retreat. Thanks for showing me! ", idx=1),
    T("Evan", "No prob, always good to chat about those tranquil times. Take it easy! ", idx=2),
]))
r = rels.get(("Evan", "take", "peaceful retreat"))
check("fragment with smuggled clause not an antecedent",
      r is not None and r["coref"] == "it->peaceful retreat"
      and not any("chat" in (x.get("coref") or "") for x in rels.values()))

# 24. Nominal-chunk hygiene (conv-44): "Oh man, sorry to hear that" --
# "that" must not resolve into the fragment.
rels = extract([T("Audrey", "Oh man, sorry to hear that, Melanie. I hope you feel better. ")])
check("that not resolved into fragment",
      not any("sorry" in (r.get("coref") or "") for r in rels)
      and not any(r["predicate"] == "hear" for r in rels))

# 25. A relative clause modifying the head is genuinely nominal: the
# phrase ("the car I saw" -> "the car") stays a valid antecedent.
rels = triples(extract([T("Jon", "I bought the car I saw yesterday. I sold it today. ")]))
r = rels.get(("Jon", "sell", "car"))
check("relcl head kept as antecedent",
      r is not None and r["coref"] == "it->car")

# 26. Bare participial modifiers are nominal: "a broken window" stays.
rels = triples(extract([T("Jon", "I saw a broken window. I photographed it yesterday. ")]))
r = rels.get(("Jon", "photograph", "broken window"))
check("participial modifier kept",
      r is not None and r["coref"] == "it->broken window")

# 27. A bare exclamatory NP ("nice setup") is a complete NP on its own:
# kept as an antecedent.
rels = triples(extract([T("Jon", "Wow, nice setup. ", idx=0),
                        T("Maria", "Where did you buy it? ", idx=1)]))
r = rels.get(("Jon", "buy", "nice setup"))
check("exclamatory NP kept as antecedent",
      r is not None and "it->nice setup" in (r["coref"] or ""))

sys.exit(1 if check.failed else 0)
