#!/usr/bin/env python3
"""Trace harness for the three-part query template.

Usage:
    python3 trace_harness.py "<question>" <conv_id> [k]

Template:
    1. Query keywords carried: tokenized -> filtered -> focus -> exact query
    2. Behood's role: index-time tags, query-time matching, whether tags used
    3. Expansion: ranked sessions -> gold comparison -> verified reason

Requires the benchmark server running with /trace endpoint.
"""
import json
import re
import sys
import urllib.parse
import urllib.request

SERVER = "http://127.0.0.1:8099"
LOCOMO = "/home/hatch/workspace/lint-ai-benchmark-logs/locomo10.json"


def evidence_to_sessions(conv_id, evidence):
    sessions = set()
    for ev in evidence:
        m = re.match(r"D(\d+):\d+", str(ev))
        if m:
            sessions.add(f"{conv_id}::session_{m.group(1)}")
    return sessions


def get_session_content(conv_id, session_short, locomo_data):
    """Get first few turns of a session for verification."""
    for c in locomo_data:
        if c["sample_id"] == conv_id:
            conv_data = c.get("conversation", {})
            sess = conv_data.get(session_short, {})
            turns = []
            if isinstance(sess, dict):
                # dict of turn_id -> turn
                for tid in sorted(sess.keys())[:4]:
                    t = sess[tid]
                    if isinstance(t, dict):
                        turns.append(f"{t.get('speaker','?')}: {t.get('text','')[:100]}")
            elif isinstance(sess, list):
                for t in sess[:4]:
                    if isinstance(t, dict):
                        turns.append(f"{t.get('speaker','?')}: {t.get('text','')[:100]}")
            return turns
    return []


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    question = sys.argv[1]
    conv = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    with open(LOCOMO) as f:
        locomo = json.load(f)

    # Find gold
    gold = set()
    gold_answer = ""
    for c in locomo:
        if c["sample_id"] == conv:
            for qa in c.get("qa", []):
                if qa["question"] == question:
                    gold = evidence_to_sessions(conv, qa.get("evidence", []))
                    gold_answer = qa.get("answer", "")
                    break

    # Get trace from server
    params = urllib.parse.urlencode({"conv": conv, "q": question, "k": k})
    with urllib.request.urlopen(f"{SERVER}/trace?{params}", timeout=120) as resp:
        t = json.load(resp)

    focus = t["focus"]
    structured = t["structured"]
    results = t["results"]

    print(f"Trace for {conv} question:")
    print(f'"{question}"')
    print()
    print("1. Query keywords carried:")
    print(f"   Tokenized: {', '.join(focus.get('tokenized', []))}")
    # Filtered = focus + constraint (stopwords/question words removed)
    filtered = focus["focus_terms"] + focus["constraint_terms"]
    print(f"   Filtered: {', '.join(filtered)}")
    print(f"   Focus: {focus['question_word']} -> {focus['focus_terms']}")
    if structured:
        persons = structured.get("persons", [])
        family = structured.get("family", "")
        directed = structured.get("directed")
        if directed:
            print(f"   Exact query: directed {directed['speaker']} -> {directed['recipient']} -> {directed['topic_keywords']}")
        elif len(persons) == 2:
            print(f"   Exact query: persons={persons}, seeking {family}")
        elif len(persons) == 1:
            print(f"   Exact query: persons={persons}, family={family}")
        else:
            print(f"   Exact query: family={family}, no persons")
    else:
        print("   Exact query: analyzer declined (lexical fallback)")

    print()
    print("2. Behood's role:")
    # Collect Behood phrases from top results and gold sessions
    print("   Index time (Behood noun-phrase tags):")
    # Show phrases from top-3 results
    for r in results[:3]:
        phrases = r.get("behood_phrases", [])
        # Filter to person/place (most relevant for identity questions)
        relevant = [p for p in phrases if p["kind"] in ("person", "place")]
        if relevant:
            tags = ", ".join(f"{p['text']} → {p['kind']}" for p in relevant[:6])
            print(f"   - {r['session_id']}: {tags}")
    # Query time: check if query persons appear in indexed phrases
    if structured and structured.get("persons"):
        print("   Query time:")
        query_persons = [p.lower() for p in structured["persons"]]
        # Check all indexed phrases across top results
        all_phrase_texts = set()
        for r in results[:k]:
            for p in r.get("behood_phrases", []):
                if p["kind"] == "person":
                    all_phrase_texts.add(p["text"].lower())
        for qp in structured["persons"]:
            if qp.lower() in all_phrase_texts:
                print(f"   - '{qp}' found in indexed persons")
            else:
                print(f"   - '{qp}' NOT found in indexed persons")

    print()
    print("3. Expansion:")
    top_k = results[:k]
    for i, r in enumerate(top_k):
        mark = " <-- GOLD" if r["session_id"] in gold else ""
        print(f"   {i+1}. {r['session_id']} score={r['score']:.2f}{mark}")
    # Find first gold beyond top-k
    for i, r in enumerate(results[k:], start=k+1):
        if r["session_id"] in gold:
            print(f"   {i}. {r['session_id']} score={r['score']:.2f} <-- GOLD (rank {i})")
            break
    print()
    if gold:
        print(f"   Gold sessions: {sorted(gold)}")
        print(f"   Gold answer: {gold_answer[:120]}")
    hit = any(r["session_id"] in gold for r in top_k)
    print(f"   Verdict: {'HIT' if hit else 'MISS'}")

    if not hit and gold:
        print()
        print("   The truth:")
        # Show gold session content
        for gs in sorted(gold)[:2]:
            short = gs.split("::")[-1]
            turns = get_session_content(conv, short, locomo)
            print(f"   {short}:")
            for turn in turns[:2]:
                print(f"     {turn}")


if __name__ == "__main__":
    main()
