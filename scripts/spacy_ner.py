#!/usr/bin/env python3
import json
import sys

ALLOWED_MODELS = {
    "en_core_web_sm",
    "en_core_web_md",
    "en_core_web_lg",
    "ko_core_news_sm",
    "zh_core_web_sm",
    "es_core_news_sm",
}


def fail(msg, code):
    print(json.dumps({"error": msg}), file=sys.stderr)
    return code


def run_payload(payload, get_nlp):
    model = payload.get("model", "en_core_web_sm")
    docs = payload.get("documents", [])
    nlp = get_nlp(model)

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

    return {"documents": out_docs}


def main() -> int:
    """One-shot (stdin JSON -> stdout JSON) or --serve: one JSON payload
    per stdin line, one JSON result per stdout line, model kept loaded.
    """
    serve = "--serve" in sys.argv[1:]
    try:
        import spacy
    except Exception as exc:
        return fail(f"spacy_import_failed: {exc}", 3)

    loaded = {}

    def get_nlp(model):
        if model not in ALLOWED_MODELS:
            raise RuntimeError(
                f"spacy_model_not_allowed({model}); "
                f"allowed={sorted(ALLOWED_MODELS)}"
            )
        if model not in loaded:
            try:
                loaded[model] = spacy.load(model)
            except Exception as exc:
                raise RuntimeError(
                    f"spacy_model_load_failed({model}): {exc}")
        return loaded[model]

    def handle(payload):
        try:
            return run_payload(payload, get_nlp)
        except (ValueError, RuntimeError) as exc:
            return {"error": str(exc)}

    if serve:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                out = {"error": f"invalid_json: {exc}"}
            else:
                out = handle(payload)
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
        return 0

    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        return fail(f"invalid_json: {exc}", 2)
    out = handle(payload)
    if "error" in out:
        # Mirror the one-shot exit codes for the known failure classes.
        msg = out["error"]
        if msg.startswith("spacy_model_not_allowed"):
            return fail(msg, 5)
        if msg.startswith("spacy_model_load_failed"):
            return fail(msg, 4)
        return fail(msg, 1)
    json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
