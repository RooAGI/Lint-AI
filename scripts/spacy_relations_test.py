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

sys.exit(1 if check.failed else 0)
