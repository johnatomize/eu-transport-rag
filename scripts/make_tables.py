"""Build the results tables from the committed run logs, without running retrieval.

    make_tables.py           write results/retrieval.md (`make tables`)
    make_tables.py --check   fail if results/retrieval.md differs from what would be written

Refuses a run log made from another version of the questions or the corpus manifest, or whose
chunks differ from what the parser and chunking code give now. That check cuts the corpus
again, but nothing is retrieved.
"""

import argparse
import sys
from pathlib import Path

from rag.corpus import CorpusError, load_manifest
from rag.parse import ParseError, parse_corpus
from rag.questions import validate
from rag.runlog import StaleRunError, check_chunks, load
from rag.tables import retrieval

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
QUESTIONS = ROOT / "data" / "questions.v1.jsonl"
RUNS = ROOT / "runs" / "retrieval"
OUT = ROOT / "results" / "retrieval.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args(argv)

    try:
        tree = parse_corpus(load_manifest(MANIFEST), ROOT)
        runs = [load(path, ROOT) for path in sorted(RUNS.glob("*.json"), key=lambda p: p.stem)]
        check_chunks(runs, tree)
    except (CorpusError, ParseError, StaleRunError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    questions, errors = validate(QUESTIONS.read_text(encoding="utf-8"), tree)
    if errors:
        print(f"error: {QUESTIONS.name} is not valid; run `make questions`", file=sys.stderr)
        return 1
    try:
        table = retrieval(runs, questions, tree)
    except ValueError as e:
        print(f"error: {e}; run `make retrieval-runs`", file=sys.stderr)
        return 1
    path = OUT.relative_to(ROOT)
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != table:
            print(f"error: {path} doesn't match the run logs; run `make tables`", file=sys.stderr)
            return 1
        print(f"{path} matches the {len(runs)} run logs")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(table, encoding="utf-8")
    print(f"wrote {path} from {len(runs)} run logs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
