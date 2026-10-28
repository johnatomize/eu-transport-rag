"""Validate data/questions.v1.jsonl against the question schema and the parsed corpus.

Prints every problem, then the number of questions by type, split and phrasing. Exits with
status 1 if the file is missing or any line is invalid. `make questions` and `make check` run it.
"""

import sys
from collections import Counter
from pathlib import Path

from rag.corpus import CorpusError, load_manifest
from rag.parse import ParseError, parse_corpus
from rag.questions import PHRASINGS, SPLITS, TYPES, Question, validate

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
QUESTIONS = ROOT / "data" / "questions.v1.jsonl"


def counts(questions: list[Question]) -> list[str]:
    """A table of question types against split and phrasing, with totals."""
    columns = [(split, phrasing) for split in SPLITS for phrasing in PHRASINGS]
    cells = Counter((q.type, q.split, q.phrasing) for q in questions)

    def line(name: str, values: list) -> str:
        return f"{name:14}" + "".join(f"{v:>18}" for v in values)

    lines = [line("", [f"{s} {p}" for s, p in columns] + ["total"])]
    for kind in TYPES:
        row = [cells[kind, s, p] for s, p in columns]
        lines.append(line(kind, row + [sum(row)]))
    totals = [sum(cells[kind, s, p] for kind in TYPES) for s, p in columns]
    lines.append(line("total", totals + [sum(totals)]))
    return lines


def main() -> int:
    path = QUESTIONS.relative_to(ROOT)
    if not QUESTIONS.exists():
        print(f"error: {path} is missing", file=sys.stderr)
        return 1
    try:
        tree = parse_corpus(load_manifest(MANIFEST), ROOT)
    except (CorpusError, ParseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if tree.errors:
        print("\n".join(f"error: {e}" for e in tree.errors), file=sys.stderr)
        return 1
    questions, errors = validate(QUESTIONS.read_text(encoding="utf-8"), tree)
    for error in errors:
        print(f"{path}: {error}", file=sys.stderr)
    print("\n".join(counts(questions)))
    if errors:
        print(f"{path}: {len(errors)} problems", file=sys.stderr)
        return 1
    checked = sum(q.checked for q in questions)
    print(f"{path}: {len(questions)} questions, all valid; {checked} checked by the owner")
    return 0


if __name__ == "__main__":
    sys.exit(main())
