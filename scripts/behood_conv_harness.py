#!/usr/bin/env python3
"""Behood conversation metadata harness.

Analyzes a full conversation through the spaCy relations extractor
(same as index time) and summarizes:
  - Persons (by relation count, speakers vs noise)
  - Places
  - Activities (subject->verb->object)
  - Things

Usage:
    python3 behood_conv_harness.py <conv_id> [locomo.json]

No server rebuild needed — calls scripts/spacy_relations.py directly.
"""
import json
import subprocess
import sys
from collections import Counter

LOCOMO = "/home/hatch/workspace/lint-ai-benchmark-logs/locomo10.json"
SPACY_SCRIPT = "/home/hatch/workspace/wt-snapshot-incr/scripts/spacy_relations.py"


def load_conversation(conv_id, locomo_path):
    with open(locomo_path) as f:
        data = json.load(f)
    for c in data:
        if c["sample_id"] == conv_id:
            return c
    raise ValueError(f"Conversation {conv_id} not found")


def build_turns(conv):
    turns = []
    conv_data = conv.get("conversation", {})
    for key in sorted(conv_data.keys()):
        if not key.startswith("session_") or "_date_time" in key:
            continue
        sess = conv_data[key]
        if not isinstance(sess, list):
            continue
        for idx, t in enumerate(sess):
            if not isinstance(t, dict):
                continue
            turns.append({
                "speaker": t.get("speaker", ""),
                "text": t.get("text", ""),
                "session_id": f"{conv['sample_id']}::{key}",
                "turn_idx": idx,
            })
    return turns


def run_extractor(turns):
    payload = {"turns": turns, "model": "en_core_web_sm"}
    proc = subprocess.run(
        ["python3", SPACY_SCRIPT],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        print(f"Extractor failed: {proc.stderr[:500]}", file=sys.stderr)
        sys.exit(1)
    return json.loads(proc.stdout)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    conv_id = sys.argv[1]
    locomo_path = sys.argv[2] if len(sys.argv) > 2 else LOCOMO

    conv = load_conversation(conv_id, locomo_path)
    conv_data = conv.get("conversation", {})
    speakers = [conv_data.get("speaker_a", ""), conv_data.get("speaker_b", "")]
    speakers = [s for s in speakers if s]

    turns = build_turns(conv)
    print(f"Analyzing {conv_id}: {len(turns)} turns...", file=sys.stderr)
    out = run_extractor(turns)

    relations = out.get("relations", [])
    key_phrases = out.get("key_phrases", [])

    # Aggregate persons by relation count
    person_counts = Counter()
    place_counts = Counter()
    thing_counts = Counter()
    activities = Counter()
    for rel in relations:
        subj = rel.get("subject", "")
        obj = rel.get("object", "")
        verb = rel.get("predicate", "")  # note: 'predicate' not 'verb'
        obj_kind = rel.get("object_kind", "")
        is_activity = rel.get("is_activity", False)
        # Persons: subject is capitalized (name) or explicitly marked
        if subj and subj[0].isupper() and len(subj.split()) <= 3:
            person_counts[subj] += 1
        if obj_kind == "place":
            place_counts[obj] += 1
        elif obj_kind == "thing":
            if obj:
                thing_counts[obj] += 1
        # Activities: is_activity flag set
        if is_activity and subj and verb and obj:
            activities[f"{subj}→{verb}→{obj}"] += 1

    # Also count from key phrases
    for kp in key_phrases:
        kind = kp.get("kind", "")
        text = kp.get("text", "")
        if kind == "person":
            person_counts[text] += 1
        elif kind == "place":
            place_counts[text] += 1
        elif kind == "thing":
            thing_counts[text] += 1

    print(f"\nHere's the whole {conv_id} through the harness:")

    # Persons — only speakers; parser-artifact subjects (Nature, Hiking,
    # etc.) are not persons and are not displayed (Luyi 2026-09-26).
    speaker_names = {s.lower() for s in speakers}
    real = [(n, c) for n, c in person_counts.most_common()
            if n.lower() in speaker_names]
    print(f"Persons ({len(real)}):")
    if real:
        print("  " + ", ".join(f"{n}: {c}" for n, c in real))

    # Places
    print(f"\nPlaces ({len(place_counts)}):")
    top_places = place_counts.most_common(10)
    print("  " + ", ".join(f"{n} ({c})" for n, c in top_places))

    # Activities
    print(f"\nActivities ({len(activities)} unique, top):")
    top_acts = activities.most_common(10)
    print("  " + ", ".join(f"{a}" for a, c in top_acts))

    # Things
    print(f"\nThings ({len(thing_counts)} unique, top):")
    top_things = thing_counts.most_common(10)
    print("  " + ", ".join(f"{n} ({c})" for n, c in top_things))

    # Summary
    print(f"\nWhat this tells you: The conversation is about {speakers[0]} and {speakers[1]}.")


if __name__ == "__main__":
    main()
