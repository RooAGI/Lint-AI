#!/usr/bin/env python3
import json
import sys

ALLOWED_MODELS = {
    "en_core_web_sm",
    "en_core_web_md",
    "en_core_web_lg",
}


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        print(json.dumps({"error": f"invalid_json: {exc}"}), file=sys.stderr)
        return 2

    model = payload.get("model", "en_core_web_sm")
    if model not in ALLOWED_MODELS:
        print(
            json.dumps(
                {
                    "error": (
                        f"spacy_model_not_allowed({model}); "
                        f"allowed={sorted(ALLOWED_MODELS)}"
                    )
                }
            ),
            file=sys.stderr,
        )
        return 5
    docs = payload.get("documents", [])

    try:
        import spacy
    except Exception as exc:
        print(json.dumps({"error": f"spacy_import_failed: {exc}"}), file=sys.stderr)
        return 3

    try:
        nlp = spacy.load(model)
    except Exception as exc:
        print(json.dumps({"error": f"spacy_model_load_failed({model}): {exc}"}), file=sys.stderr)
        return 4

    out_docs = []
    for doc in docs:
        doc_id = doc.get("id", "")
        text = doc.get("text", "")
        parsed = nlp(text)
        entities = []
        for ent in parsed.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "score": None,
                }
            )
        out_docs.append({"id": doc_id, "entities": entities})

    json.dump({"documents": out_docs}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
