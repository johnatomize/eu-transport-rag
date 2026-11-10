"""The owner's hand grades of test answers (#88): build the grading page, apply its grades, check.

    hand_grades.py               check runs/grades/hand-grades.v1.jsonl
    hand_grades.py --page        write the page, to open as a file from build/
    hand_grades.py --apply FILE  write the grades file from the page's export

The answers are chosen by `rag.hand_grades.select` from the finalists' three test runs. The page
shows each question's answers blind, as A, B and C, with the gold answer and its passages, and
the quotes each answer cites; the owner grades each one correct, partly correct or wrong. The
check passes while the grades file doesn't exist yet, and once it does, every chosen answer must
be graded.
"""

import argparse
import json
import sys
from pathlib import Path

from rag.corpus import CorpusError, load_manifest
from rag.faithfulness import by_id
from rag.hand_grades import (
    CORRECT,
    GRADED_RUNS,
    GRADES,
    assign,
    build_id,
    expected,
    load_grades,
    select,
    validate_grades,
)
from rag.parse import ParseError, lead_in, parse_corpus
from rag.questions import validate
from rag.review import DOCUMENT, context, link

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
QUESTIONS = ROOT / "data" / "questions.v1.jsonl"
GRADES_FILE = ROOT / "runs" / "grades" / "hand-grades.v1.jsonl"
RUNS = tuple(ROOT / "runs" / "generation" / name for name in GRADED_RUNS)
TEMPLATE = Path(__file__).with_name("hand_grades_review.html")
OUT = ROOT / "build" / "hand-grades" / "hand-grades.html"


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
    """What the page shows. Run files and retrieval setups stay out of it: answers are letters."""
    question_of = {q.id: q for q in questions}
    doc_of = {doc.id: doc for doc in docs}
    letters = assign(select(questions, logs))
    nodes = {}

    def source(node_id: str, text: bool) -> str:
        node = tree[node_id]
        doc = doc_of[node.doc_id]
        entry = nodes.setdefault(
            node_id, {"path": list(node.heading_path), "link": link(doc), "is_note": not doc.celex}
        )
        if text and "text" not in entry:
            entry["lead_in"] = lead_in(tree, node, context(tree, node))
            entry["text"] = tree.full_text(node_id)
        return node_id

    items = []
    for qid, by_letter in letters.items():
        q = question_of[qid]
        gold = [source(node_id, text=True) for node_id in q.gold_node_ids]
        for letter, run in by_letter.items():
            verdict = by_id(logs[run])[qid]["verdict"]
            quotes: dict[str, list[str]] = {}
            for citation in verdict["citations"]:
                quotes.setdefault(source(citation["node_id"], text=False), []).append(
                    citation["quote"]
                )
            items.append(
                {
                    "key": f"{qid}-{letter}",
                    "question_id": qid,
                    "letter": letter,
                    "answers": len(by_letter),
                    "question": q.question,
                    "type": q.type,
                    "gold": q.answer,
                    "gold_nodes": gold,
                    "answer": verdict["answer"],
                    "cited": [{"node_id": n, "quotes": qs} for n, qs in quotes.items()],
                }
            )
    return {"build": build_id(letters, logs), "items": items, "nodes": nodes}


def write_page() -> int:
    data = page_data(*load())
    text = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(DOCUMENT.replace("{page}", page), encoding="utf-8")
    questions = len({item["question_id"] for item in data["items"]})
    print(f"{OUT.relative_to(ROOT)}: {len(data['items'])} answers to {questions} questions")
    return 0


def apply(path: Path) -> int:
    """Turn the page's exported grades into the grades file, refusing an unfinished export or
    one made on a page built from other answers."""
    _, _, questions, logs = load()
    letters = assign(select(questions, logs))
    export = json.loads(path.read_text(encoding="utf-8"))
    build = build_id(letters, logs)
    if export.get("build") != build:
        print(
            f"error: {path.name} was saved from a page built from other answers "
            f"({export.get('build')}, not {build}); rebuild the page and grade again",
            file=sys.stderr,
        )
        return 1
    reviews = export.get("reviews") or {}
    keys = {f"{qid}-{letter}" for qid in letters for letter in letters[qid]}
    problems = [f"{key}: not one of the answers to grade" for key in sorted(set(reviews) - keys)]
    rows = []
    for qid, by_letter in letters.items():
        for letter, run in by_letter.items():
            key = f"{qid}-{letter}"
            review = reviews.get(key) or {}
            grade, note = review.get("grade"), str(review.get("note") or "").strip()
            if grade not in GRADES:
                problems.append(f"{key}: no grade")
            elif grade != "correct" and not note:
                problems.append(f"{key}: graded {grade.replace('_', ' ')}, but no note says why")
            else:
                rows.append({"question_id": qid, "run": run, "grade": grade, "note": note})
    if problems:
        print("\n".join(f"error: {p}" for p in problems), file=sys.stderr)
        return 1
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _, errors = validate_grades(text, expected(select(questions, logs)))
    if errors:
        print("\n".join(f"error: {e}" for e in errors), file=sys.stderr)
        return 1
    GRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    GRADES_FILE.write_text(text, encoding="utf-8")
    counts = {g: sum(row["grade"] == g for row in rows) for g in GRADES}
    print(
        f"wrote {GRADES_FILE.relative_to(ROOT)}: {len(rows)} answers, "
        + ", ".join(f"{n} {g.replace('_', ' ')}" for g, n in counts.items())
    )
    return 0


def check() -> int:
    path = GRADES_FILE.relative_to(ROOT)
    if not GRADES_FILE.exists():
        print(f"{path}: not graded yet (#88)")
        return 0
    _, _, questions, logs = load()
    try:
        by_run = load_grades(GRADES_FILE.read_text(encoding="utf-8"), questions, logs)
    except ValueError as e:
        print(f"{path}: {e}", file=sys.stderr)
        return 1
    grades = [grade for run in by_run.values() for grade in run.values()]
    correct = sum(grade == CORRECT for grade in grades)
    print(f"{path}: {len(grades)} answers graded, {correct} correct")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--page", action="store_true", help="write the grading page")
    parser.add_argument("--apply", type=Path, metavar="FILE", help="grades exported by the page")
    args = parser.parse_args()
    try:
        if args.page:
            return write_page()
        if args.apply:
            return apply(args.apply)
        return check()
    except (CorpusError, ParseError, ValueError, OSError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
