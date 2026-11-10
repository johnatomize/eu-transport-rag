"""The owner's hand grades of test answers without key facts (#88): which answers, and the file.

Correctness of an answer to a question with key facts is matched by code (`rag.keyfacts`). The
answerable test questions without key facts are graded by the owner instead, on the finalists'
test run only (#75; "2026-11-04: what checkpoint G measures" in docs/DECISIONS.md). The rules,
recorded in docs/DECISIONS.md ("the owner's hand grades of test answers"):

- **Which answers.** Every answerable test question without key facts, in each of the three
  finalists' test runs, whose reply is an answer that passed the check. A refusal, "could not
  answer reliably" or a missing reply is not correct without a grade, as it is for questions
  with key facts, so it isn't shown.
- **Blind.** The page calls a question's answers A, B and C, in an order drawn with a fixed seed
  per question, and never names the finalist; `assign` maps the letters back to run logs.
- **The grades file**, `runs/grades/hand-grades.v1.jsonl`, beside the run logs, one answer per
  line:

      {"question_id": "q026", "run": "2026-11-09T07-49-38Z-claude-opus-5-5.json",
       "grade": "correct", "note": ""}

  `grade` is `correct`, `partly_correct` or `wrong`, against the question's gold answer; `note`
  is the owner's own words, and says what is missing or wrong whenever the grade isn't correct.
"""

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag.faithfulness import by_id
from rag.questions import Question

# The finalists' test runs in runs/generation/ (#86): the base, article chunks (top 3), and
# Haiku 5.5 reordering search's top 20.
GRADED_RUNS = (
    "2026-11-09T07-49-38Z-claude-opus-5-5.json",
    "2026-11-09T07-49-41Z-claude-opus-5-5.json",
    "2026-11-09T07-49-44Z-claude-opus-5-5.json",
)
SEED = "hand-grades-v1"
LETTERS = ("A", "B", "C")
CORRECT, PARTLY, WRONG = "correct", "partly_correct", "wrong"
GRADES = (CORRECT, PARTLY, WRONG)

_FIELDS = {"question_id": str, "run": str, "grade": str, "note": str}


@dataclass(frozen=True)
class Grade:
    question_id: str
    run: str
    grade: str
    note: str


def _answered(entry: Mapping | None) -> bool:
    """Whether a run log entry is an answer that passed the check (not a refusal)."""
    return entry is not None and entry["verdict"]["ok"] and entry["verdict"]["answerable"]


def graded_questions(questions: Sequence[Question]) -> list[Question]:
    """The answerable test questions without key facts, in question order."""
    return [q for q in questions if q.split == "test" and q.gold_node_ids and not q.key_facts]


def select(questions: Sequence[Question], logs: Mapping[str, Mapping]) -> dict[str, list[str]]:
    """The runs whose answer to each graded question gets a grade, {question_id: [run, ...]}.

    Runs are listed in file-name order; a question no run answered is left out.
    """
    entries = {run: by_id(log) for run, log in logs.items()}
    out = {}
    for q in graded_questions(questions):
        runs = [run for run in sorted(logs) if _answered(entries[run].get(q.id))]
        if runs:
            out[q.id] = runs
    return out


def assign(chosen: Mapping[str, Sequence[str]], seed: str = SEED) -> dict[str, dict[str, str]]:
    """Each question's answers as letters, {question_id: {"A": run, ...}}, in a seeded order."""
    out = {}
    for qid, runs in chosen.items():
        if len(runs) > len(LETTERS):
            raise ValueError(f"{qid}: at most {len(LETTERS)} answers, not {len(runs)}")
        order = sorted(runs)
        random.Random(f"{seed}:{qid}:order").shuffle(order)
        out[qid] = dict(zip(LETTERS, order, strict=False))
    return out


def build_id(letters: Mapping[str, Mapping[str, str]], logs: Mapping[str, Mapping]) -> str:
    """A short fingerprint of what the page shows: the questions, the letters and the answers.

    The page keeps grades under it and puts it in its export, so grades made on a page built
    from other answers are never applied to these.
    """
    record = []
    for qid in sorted(letters):
        shown = []
        for letter, run in sorted(letters[qid].items()):
            verdict = by_id(logs[run])[qid]["verdict"]
            shown.append([letter, verdict["answer"], verdict["citations"]])
        record.append([qid, shown])
    digest = hashlib.sha256(json.dumps(record, ensure_ascii=False).encode()).hexdigest()
    return digest[:12]


def expected(chosen: Mapping[str, Sequence[str]]) -> set[tuple[str, str]]:
    """Every (question_id, run) that needs a grade."""
    return {(qid, run) for qid, runs in chosen.items() for run in runs}


def validate_grades(text: str, wanted: set[tuple[str, str]]) -> tuple[list[Grade], list[str]]:
    """Check a grades file against the answers that should be graded.

    Returns the valid grades and one message per problem, naming the line. Every wanted answer
    must be graded once, and nothing else; a grade other than correct needs a note.
    """
    grades, errors = [], []
    seen: dict[tuple[str, str], int] = {}
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as e:
            errors.append(f"line {number}: not JSON: {e}")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {number}: not a JSON object")
            continue
        problems = [f"{name}: missing" for name in _FIELDS if name not in row]
        problems += [
            f"{name}: {row[name]!r} is the wrong type"
            for name, kind in _FIELDS.items()
            if name in row and not isinstance(row[name], kind)
        ]
        problems += [f"{name}: not a field" for name in row if name not in _FIELDS]
        key = (row.get("question_id"), row.get("run"))
        if not problems:
            if row["grade"] not in GRADES:
                problems.append(f"grade: {row['grade']!r} is not one of {', '.join(GRADES)}")
            elif row["grade"] != CORRECT and not row["note"].strip():
                problems.append("note: says nothing, but the answer isn't graded correct")
            if key not in wanted:
                problems.append("not one of the answers to grade")
            elif key in seen:
                problems.append(f"already graded on line {seen[key]}")
            else:
                seen[key] = number
        errors += [f"line {number} ({key[0]}, {key[1]}): {p}" for p in problems]
        if not problems:
            grades.append(Grade(**row))
    missing = wanted - set(seen)
    if missing and not errors:
        errors.append(f"{len(missing)} of {len(wanted)} answers not graded")
    return grades, errors


def load_grades(
    text: str, questions: Sequence[Question], logs: Mapping[str, Mapping]
) -> dict[str, dict[str, str]]:
    """The grades file's grades by run and question ID, {run: {question_id: grade}}.

    `logs` are the answer run logs by file name; each of GRADED_RUNS must be among them, and
    gets an entry even if none of its answers needed a grade. Raises ValueError naming every
    problem when the file doesn't grade exactly the answers chosen for grading.
    """
    absent = [name for name in GRADED_RUNS if name not in logs]
    if absent:
        raise ValueError(f"the graded runs {', '.join(absent)} aren't among the answer run logs")
    graded_logs = {name: logs[name] for name in GRADED_RUNS}
    grades, errors = validate_grades(text, expected(select(questions, graded_logs)))
    if errors:
        raise ValueError("the hand grades are not valid: " + "; ".join(errors))
    out: dict[str, dict[str, str]] = {name: {} for name in GRADED_RUNS}
    for g in grades:
        out[g.run][g.question_id] = g.grade
    return out
