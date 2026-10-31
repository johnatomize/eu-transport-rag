"""Run every retrieval configuration on the dev split and write a run log for each.

    retrieval_runs.py     write runs/retrieval/<name>.json (`make retrieval-runs`)

Configurations are listed in RUNS; later checkpoints add theirs. A log whose rankings haven't
changed is left as it is, so a rerun doesn't touch the committed logs and tables. Run logs are
committed, and `make tables` builds the results tables from them alone. Format: `rag.runlog`.
"""

import json
import sys
from pathlib import Path

from rag.chunk import STRATEGIES
from rag.corpus import CorpusError, load_manifest
from rag.parse import ParseError, parse_corpus
from rag.questions import validate
from rag.retrieve import BM25
from rag.runlog import dumps, make_run, same_results

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
QUESTIONS = ROOT / "data" / "questions.v1.jsonl"
OUT = ROOT / "runs" / "retrieval"

# Checkpoint D: BM25 with its defaults over each chunking strategy.
RUNS = [
    (f"bm25-{strategy}", strategy, {"retriever": "bm25", "k1": 1.5, "b": 0.75})
    for strategy in STRATEGIES
]


def main() -> int:
    try:
        tree = parse_corpus(load_manifest(MANIFEST), ROOT)
    except (CorpusError, ParseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    questions, errors = validate(QUESTIONS.read_text(encoding="utf-8"), tree)
    if errors:
        print(f"error: {QUESTIONS.name} is not valid; run `make questions`", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for name, strategy, config in RUNS:
        chunks = STRATEGIES[strategy](tree)
        retriever = BM25(chunks, k1=config["k1"], b=config["b"])
        run = make_run(name, {**config, "strategy": strategy}, retriever, chunks, questions, ROOT)
        path = OUT / f"{name}.json"
        where = path.relative_to(ROOT)
        if path.exists() and same_results(json.loads(path.read_text(encoding="utf-8")), run):
            print(f"{name:20} unchanged, {where} kept")
            continue
        path.write_text(dumps(run), encoding="utf-8")
        print(f"{name:20} {len(run['questions'])} questions → {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
