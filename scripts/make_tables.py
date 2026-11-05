"""Build the results tables from the committed run logs, without retrieving or calling a model.

    make_tables.py           write the results tables (`make tables`)
    make_tables.py --check   fail if any differs from what would be written

The tables are results/retrieval.md, results/answers.md, and results/faithfulness_judge.md,
which compares each calibration run of the faithfulness judge with the owner's labels.

Refuses a retrieval run log made from another version of the questions or the corpus manifest,
or whose chunks differ from what the parser and chunking code give now; and an answer run log
that shows a chunk the corpus no longer gives. That check cuts the corpus again, but nothing is
retrieved. The answer table scores the test split too, once its run exists: it is checkpoint
G's final table (design §C).
"""

import argparse
import json
import sys
from pathlib import Path

from rag.corpus import CorpusError, load_manifest
from rag.faithfulness import by_id, validate_labels
from rag.parse import ParseError, parse_corpus
from rag.questions import validate
from rag.runlog import StaleRunError, check_chunks, load
from rag.tables import answers, judge_calibration, retrieval

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
QUESTIONS = ROOT / "data" / "questions.v1.jsonl"
RUNS = ROOT / "runs" / "retrieval"
ANSWER_RUNS = ROOT / "runs" / "generation"
OUT = ROOT / "results" / "retrieval.md"
ANSWERS_OUT = ROOT / "results" / "answers.md"
JUDGE_RUNS = ROOT / "runs" / "judge"
LABELS = ROOT / "data" / "faithfulness_labels.v1.jsonl"
JUDGE_OUT = ROOT / "results" / "faithfulness_judge.md"


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
    answer_logs = []
    for path in sorted(ANSWER_RUNS.glob("*.json")):
        try:
            answer_logs.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError) as e:
            print(f"error: can't read {path.relative_to(ROOT)}: {e}", file=sys.stderr)
            return 1
    try:
        tables = {OUT: retrieval(runs, questions, tree)}
    except ValueError as e:
        print(f"error: {e}; run `make retrieval-runs`", file=sys.stderr)
        return 1
    try:
        tables[ANSWERS_OUT] = answers(answer_logs, questions, tree)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    try:
        judge_logs = calibration_logs()
        tables[JUDGE_OUT] = judge_table(judge_logs, dict(answer_logs))
    except (OSError, ValueError, KeyError) as e:
        print(f"error: the faithfulness judge's table: {e}", file=sys.stderr)
        return 1
    # unfinished runs are listed too
    counts = {OUT: len(runs), ANSWERS_OUT: len(answer_logs), JUDGE_OUT: len(judge_logs)}
    if args.check:
        stale = [p for p, t in tables.items() if not p.exists() or p.read_text("utf-8") != t]
        for p in stale:
            path = p.relative_to(ROOT)
            print(f"error: {path} doesn't match the run logs; run `make tables`", file=sys.stderr)
        if stale:
            return 1
        for p in tables:
            print(f"{p.relative_to(ROOT)} matches the {counts[p]} run logs")
        return 0
    for p, table in tables.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(table, encoding="utf-8")
        print(f"wrote {p.relative_to(ROOT)} from {counts[p]} run logs")
    return 0


def calibration_logs() -> list[tuple[str, dict]]:
    """The judge's calibration run logs, by file name, which starts with the time."""
    logs = []
    for path in sorted(JUDGE_RUNS.glob("calibration-*.json")):
        log = json.loads(path.read_text(encoding="utf-8"))
        if log["config"]["kind"] != "calibration":
            raise ValueError(f"{path.name} is named as a calibration run but isn't one")
        logs.append((path.name, log))
    return logs


def judge_table(judge_logs: list[tuple[str, dict]], answer_logs: dict[str, dict]) -> str:
    """The judge's calibration table, from its runs and the owner's labels."""
    if not LABELS.exists():
        if judge_logs:
            raise ValueError(f"calibration runs but no {LABELS.name}")
        return judge_calibration([], [], {})
    text = LABELS.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    texts = {
        (r["question_id"], r["run"]): by_id(answer_logs[r["run"]])[r["question_id"]]["verdict"][
            "answer"
        ]
        for r in rows
    }
    labels, errors = validate_labels(text, texts)
    if errors:
        raise ValueError(f"{LABELS.name} is not valid; run `make faithfulness-labels`")
    return judge_calibration(judge_logs, labels, texts)


if __name__ == "__main__":
    sys.exit(main())
