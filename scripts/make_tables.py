"""Build the results tables from the committed run logs, without retrieving or calling a model.

    make_tables.py           write the results tables (`make tables`)
    make_tables.py --check   fail if any differs from what would be written

The tables are results/retrieval.md; results/answers.md, with the share of each run's answers
the chosen faithfulness judge (judge_model in config/generation.yaml, with the judge's current
instructions) found supported, from its whole runs in runs/judge/; and
results/faithfulness_judge.md, which compares each calibration run of the judge with the
owner's labels.

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
from rag.faithfulness import LABELLED_RUNS, select, validate_labels
from rag.faithfulness import answers as labelled_answers
from rag.generation_config import GenerationConfigError, load_generation_config
from rag.judge import prompt_version
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
CONFIG = ROOT / "config" / "generation.yaml"
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
        judge = (load_generation_config(CONFIG).judge_model, prompt_version())
        judge_runs = judge_logs("run")
        tables[ANSWERS_OUT] = answers(answer_logs, questions, tree, None, judge_runs, judge)
    except (OSError, ValueError, KeyError, GenerationConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    try:
        calibrations = judge_logs("calibration")
        tables[JUDGE_OUT] = judge_table(calibrations, dict(answer_logs), questions)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: the faithfulness judge's table: {e}", file=sys.stderr)
        return 1
    # unfinished runs are listed too
    counts = {
        OUT: len(runs),
        ANSWERS_OUT: len(answer_logs) + len(judge_runs),
        JUDGE_OUT: len(calibrations),
    }
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


def judge_logs(kind: str) -> list[tuple[str, dict]]:
    """The judge's run logs of one kind, "calibration" or "run", by file name, which starts with
    the kind and then the time."""
    logs = []
    for path in sorted(JUDGE_RUNS.glob(f"{kind}-*.json")):
        try:
            log = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ValueError(f"can't read {path.relative_to(ROOT)}: {e}") from None
        if log["config"]["kind"] != kind:
            raise ValueError(f"{path.name} is named as a {kind} run but isn't one")
        logs.append((path.name, log))
    return logs


def judge_table(
    judge_logs: list[tuple[str, dict]], answer_logs: dict[str, dict], questions: list
) -> str:
    """The judge's calibration table, from its runs and the owner's labels.

    The labels must label exactly the answers chosen for labelling, as `make check` requires.
    """
    if not LABELS.exists():
        if judge_logs:
            raise ValueError(f"calibration runs but no {LABELS.name}")
        return judge_calibration([], [], {})
    labelled_logs = {name: answer_logs[name] for name in LABELLED_RUNS}
    texts = labelled_answers(select(questions, labelled_logs), labelled_logs)
    labels, errors = validate_labels(LABELS.read_text(encoding="utf-8"), texts)
    if errors:
        raise ValueError(f"{LABELS.name} is not valid; run `make faithfulness-labels`")
    return judge_calibration(judge_logs, labels, texts)


if __name__ == "__main__":
    sys.exit(main())
