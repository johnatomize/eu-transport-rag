import importlib.util
import json
from pathlib import Path

import pytest

from rag.hand_grades import (
    GRADED_RUNS,
    LETTERS,
    assign,
    build_id,
    expected,
    graded_questions,
    select,
    validate_grades,
)
from rag.questions import Question, validate

ROOT = Path(__file__).parents[1]


def question(qid: str, split: str = "test", facts=(), gold=("561/2006:art4(c)",)):
    return Question(
        qid, "?", "Gold.", tuple(facts), tuple(gold), "definition", "verbatim", split, True
    )


def entry(qid: str, ok: bool = True, answerable: bool = True, answer: str = "Yes."):
    return {
        "id": qid,
        "verdict": {"ok": ok, "answerable": answerable, "answer": answer, "citations": []},
    }


def log(*entries):
    return {"answers": list(entries)}


# --- which answers -----------------------------------------------------------------------------


def test_only_answerable_test_questions_without_key_facts_are_graded():
    qs = [
        question("q1"),
        question("q2", split="dev"),
        question("q3", facts=("9 hours",)),
        question("q4", gold=()),  # unanswerable
    ]
    assert [q.id for q in graded_questions(qs)] == ["q1"]


def test_refusals_and_failed_answers_are_left_off_the_sheet():
    qs = [question("q1"), question("q2"), question("q3")]
    logs = {
        "a.json": log(entry("q1"), entry("q2", answerable=False), entry("q3", ok=False)),
        "b.json": log(entry("q1"), entry("q2"), entry("q3", ok=False)),
    }
    assert select(qs, logs) == {"q1": ["a.json", "b.json"], "q2": ["b.json"]}


def test_assign_gives_each_question_its_runs_as_letters_in_a_seeded_order():
    chosen = {f"q{i:03}": ["x", "y", "z"] for i in range(20)}
    letters = assign(chosen)
    assert all(
        sorted(v) == list(LETTERS) and sorted(v.values()) == ["x", "y", "z"]
        for v in letters.values()
    )
    assert {v["A"] for v in letters.values()} == {"x", "y", "z"}  # not always the same finalist
    assert assign({"q001": ["z", "x", "y"]}) == assign({"q001": ["x", "y", "z"]})
    assert assign({"q1": ["only"]}) == {"q1": {"A": "only"}}
    with pytest.raises(ValueError):
        assign({"q1": ["a", "b", "c", "d"]})


def test_build_id_changes_with_the_answers_or_the_letters():
    logs = {"a.json": log(entry("q1")), "b.json": log(entry("q1", answer="Maybe."))}
    letters = {"q1": {"A": "a.json", "B": "b.json"}}
    base = build_id(letters, logs)
    assert build_id(letters, logs) == base
    assert build_id({"q1": {"A": "b.json", "B": "a.json"}}, logs) != base
    changed = {
        "a.json": log(entry("q1", answer="No.")),
        "b.json": log(entry("q1", answer="Maybe.")),
    }
    assert build_id(letters, changed) != base


def test_the_real_sheet_has_25_answers_to_9_questions(tree):
    questions, errors = validate((ROOT / "data/questions.v1.jsonl").read_text("utf-8"), tree)
    assert not errors
    assert len(graded_questions(questions)) == 10
    logs = {
        name: json.loads((ROOT / "runs/generation" / name).read_text("utf-8"))
        for name in GRADED_RUNS
    }
    assert all(log["config"]["split"] == "test" for log in logs.values())
    chosen = select(questions, logs)
    assert len(chosen) == 9 and len(expected(chosen)) == 25
    assert "q070" not in chosen  # refused by every finalist
    assert len(chosen["q030"]) == 1  # answered only with Haiku's reordering


# --- the grades file ---------------------------------------------------------------------------

WANTED = {("q1", "a.json"), ("q1", "b.json")}


def row(run="a.json", grade="correct", note="", **extra):
    return json.dumps({"question_id": "q1", "run": run, "grade": grade, "note": note, **extra})


def test_a_complete_grades_file_passes():
    grades, errors = validate_grades(
        row() + "\n" + row("b.json", "wrong", "Says no.") + "\n", WANTED
    )
    assert not errors
    assert [g.grade for g in grades] == ["correct", "wrong"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (row(), "1 of 2 answers not graded"),
        (row() + "\n" + row("b.json", "partly_correct"), "no note"),
        (row() + "\n" + row("b.json", "fine"), "not one of"),
        (row() + "\n" + row("c.json"), "not one of the answers to grade"),
        (row() + "\n" + row(), "already graded on line 1"),
        (row(extra=1), "extra: not a field"),
        (row() + "\n{", "not JSON"),
        (json.dumps({"question_id": "q1", "run": "a.json", "grade": "correct"}), "note: missing"),
    ],
)
def test_grades_file_problems_are_named(text, message):
    _, errors = validate_grades(text, WANTED)
    assert any(message.replace("no note", "says nothing") in e for e in errors), errors


# --- applying the page's export ----------------------------------------------------------------


def _script():
    spec = importlib.util.spec_from_file_location(
        "hand_grades_script", ROOT / "scripts/hand_grades.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_writes_the_grades_file_and_refuses_an_unfinished_or_stale_export(
    tmp_path, monkeypatch
):
    script = _script()
    qs = [question("q1")]
    logs = {"a.json": log(entry("q1")), "b.json": log(entry("q1"))}
    monkeypatch.setattr(script, "load", lambda: (None, None, qs, logs))
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "GRADES_FILE", tmp_path / "grades.jsonl")
    letters = assign(select(qs, logs))
    build = build_id(letters, logs)
    export = tmp_path / "export.json"

    def save(reviews, build=build):
        export.write_text(json.dumps({"build": build, "reviews": reviews}))

    save({"q1-A": {"grade": "correct"}})
    assert script.apply(export) == 1  # B not graded
    save({"q1-A": {"grade": "correct"}, "q1-B": {"grade": "wrong", "note": ""}})
    assert script.apply(export) == 1  # wrong without a note
    save({"q1-A": {"grade": "correct"}, "q1-B": {"grade": "correct"}}, build="other")
    assert script.apply(export) == 1  # made on another page
    assert not (tmp_path / "grades.jsonl").exists()

    save(
        {
            "q1-A": {"grade": "correct", "note": " "},
            "q1-B": {"grade": "wrong", "note": " Says no. "},
        }
    )
    assert script.apply(export) == 0
    rows = [json.loads(line) for line in (tmp_path / "grades.jsonl").read_text().splitlines()]
    assert {(r["run"], r["grade"], r["note"]) for r in rows} == {
        (letters["q1"]["A"], "correct", ""),
        (letters["q1"]["B"], "wrong", "Says no."),
    }
