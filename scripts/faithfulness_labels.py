"""The owner's faithfulness labels (#82): build the review page, apply its verdicts, check them.

    faithfulness_labels.py                    check data/faithfulness_labels.v1.jsonl
    faithfulness_labels.py --page             write the page, to open as a file from build/
    faithfulness_labels.py --page --artifact  the page without <html> and <head>, for claude.ai
    faithfulness_labels.py --apply FILE       write the labels file from the page's export

The answers are chosen by `rag.faithfulness.select` from F's two answer runs. The page shows
each answer blind, as A or B, sentence by sentence, with the full text of every node it cites;
the owner marks it supported or not, and the first sentence that isn't. The check passes while
the labels file doesn't exist yet, and once it does, every chosen answer must be labelled.
"""

import argparse
import json
import sys
from pathlib import Path

from rag.corpus import CorpusError, load_manifest
from rag.faithfulness import (
    answers,
    assign,
    build_id,
    by_id,
    quote_spans,
    select,
    sentences,
    validate_labels,
)
from rag.parse import ParseError, lead_in, parse_corpus
from rag.questions import validate
from rag.review import DOCUMENT, context, link

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
QUESTIONS = ROOT / "data" / "questions.v1.jsonl"
LABELS = ROOT / "data" / "faithfulness_labels.v1.jsonl"
# F's two dev runs, both with the same prompt and retrieval setup (hybrid, paragraphs, top 5).
RUNS = (
    ROOT / "runs" / "generation" / "2026-11-03T21-03-51Z-claude-opus-5-5.json",
    ROOT / "runs" / "generation" / "2026-11-03T20-59-15Z-claude-haiku-5-5.json",
)
TEMPLATE = Path(__file__).with_name("faithfulness_review.html")
OUT = ROOT / "build" / "faithfulness-review" / "faithfulness-review.html"
VERDICTS = ("supported", "unsupported")


def load() -> tuple:
    """The parsed corpus, the corpus documents, the questions, and the runs by file name."""
    docs = load_manifest(MANIFEST)
    tree = parse_corpus(docs, ROOT)
    if tree.errors:
        raise ParseError("; ".join(tree.errors))
    questions, errors = validate(QUESTIONS.read_text(encoding="utf-8"), tree)
    if errors:
        raise ValueError(f"{QUESTIONS.name} is not valid; run `make questions`")
    logs = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in RUNS}
    return tree, docs, questions, logs


def page_data(tree, docs, questions, logs) -> dict:
    """What the page shows. Model names and run files stay out of it: answers are A and B.

    Each cited node's text comes in pieces, [text, quoted], with the answer's quotes found by
    `quote_spans` the way the answer check found them.
    """
    question_of = {q.id: q for q in questions}
    doc_of = {doc.id: doc for doc in docs}
    chosen = select(questions, logs)
    letters = assign(chosen, list(logs))
    entries = {run: by_id(log) for run, log in logs.items()}
    items, nodes = [], {}
    for qid in chosen:
        q = question_of[qid]
        for letter, run in letters[qid].items():
            verdict = entries[run][qid]["verdict"]
            quotes: dict[str, list[str]] = {}
            for citation in verdict["citations"]:
                quotes.setdefault(citation["node_id"], []).append(citation["quote"])
            cited = []
            for node_id, node_quotes in quotes.items():
                text = tree.full_text(node_id)
                spans, missing = quote_spans(text, node_quotes)
                cited.append(
                    {"node_id": node_id, "pieces": _pieces(text, spans), "missing": missing}
                )
                node = tree[node_id]
                doc = doc_of[node.doc_id]
                nodes[node_id] = {
                    "path": list(node.heading_path),
                    "lead_in": lead_in(tree, node, context(tree, node)),
                    "link": link(doc),
                    "is_note": not doc.celex,
                }
            items.append(
                {
                    "key": f"{qid}-{letter}",
                    "question_id": qid,
                    "letter": letter,
                    "question": q.question,
                    "type": q.type,
                    "sentences": sentences(verdict["answer"]),
                    "cited": cited,
                }
            )
    return {"build": build_id(chosen, letters, logs), "items": items, "nodes": nodes}


def _pieces(text: str, spans: list[list[int]]) -> list[list]:
    """`text` cut into [piece, quoted] at the spans, which may overlap."""
    pieces, last = [], 0
    for start, end in spans:
        if end <= last:
            continue
        start = max(start, last)
        if start > last:
            pieces.append([text[last:start], False])
        pieces.append([text[start:end], True])
        last = end
    if last < len(text):
        pieces.append([text[last:], False])
    return pieces


def write_page(artifact: bool) -> int:
    data = page_data(*load())
    text = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", text)
    if not artifact:
        page = DOCUMENT.replace("{page}", page)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: {len(data['items'])} answers, {len(text) // 1024} KB of data")
    return 0


def apply(path: Path) -> int:
    """Turn the page's exported labels into the labels file, refusing an unfinished export or
    one made on a page built from other answers."""
    _, _, questions, logs = load()
    chosen = select(questions, logs)
    letters = assign(chosen, list(logs))
    texts = answers(chosen, logs)
    export = json.loads(path.read_text(encoding="utf-8"))
    build = build_id(chosen, letters, logs)
    if export.get("build") != build:
        print(
            f"error: {path.name} was saved from a page built from other answers "
            f"({export.get('build')}, not {build}); rebuild the page and label again",
            file=sys.stderr,
        )
        return 1
    reviews = export["reviews"]
    keys = {f"{qid}-{letter}" for qid in chosen for letter in letters[qid]}
    problems = [f"{key}: not one of the chosen answers" for key in sorted(set(reviews) - keys)]
    rows = []
    for qid in chosen:
        for letter, run in letters[qid].items():
            key = f"{qid}-{letter}"
            review = reviews.get(key) or {}
            verdict, index = review.get("verdict"), review.get("sentence")
            parts = sentences(texts[qid, run])
            unsupported = verdict == "unsupported"
            if verdict not in VERDICTS:
                problems.append(f"{key}: no label")
                continue
            if unsupported and not (type(index) is int and 0 <= index < len(parts)):
                problems.append(f"{key}: not supported, but no first unsupported sentence")
                continue
            rows.append(
                {
                    "question_id": qid,
                    "run": run,
                    "supported": not unsupported,
                    "first_unsupported": index if unsupported else None,
                    "sentence": parts[index] if unsupported else None,
                    "note": str(review.get("note") or "").strip(),
                }
            )
    if problems:
        print("\n".join(f"error: {p}" for p in problems), file=sys.stderr)
        return 1
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _, errors = validate_labels(text, texts)
    if errors:
        print("\n".join(f"error: {e}" for e in errors), file=sys.stderr)
        return 1
    LABELS.write_text(text, encoding="utf-8")
    supported = sum(row["supported"] for row in rows)
    print(f"wrote {LABELS.relative_to(ROOT)}: {len(rows)} answers, {supported} supported")
    return 0


def check() -> int:
    path = LABELS.relative_to(ROOT)
    if not LABELS.exists():
        print(f"{path}: not labelled yet (#82)")
        return 0
    _, _, questions, logs = load()
    text = LABELS.read_text(encoding="utf-8")
    labels, errors = validate_labels(text, answers(select(questions, logs), logs))
    for error in errors:
        print(f"{path}: {error}", file=sys.stderr)
    if errors:
        return 1
    supported = sum(label.supported for label in labels)
    print(f"{path}: {len(labels)} answers labelled, {supported} supported")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--page", action="store_true", help="write the review page")
    parser.add_argument("--artifact", action="store_true", help="with --page: for claude.ai")
    parser.add_argument("--apply", type=Path, metavar="FILE", help="verdicts exported by the page")
    args = parser.parse_args()
    try:
        if args.page:
            return write_page(args.artifact)
        if args.apply:
            return apply(args.apply)
        return check()
    except (CorpusError, ParseError, ValueError, OSError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
