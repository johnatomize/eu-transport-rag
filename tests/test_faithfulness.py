import json
from pathlib import Path

import pytest

from rag.faithfulness import QUOTAS, answers, assign, select, sentences, validate_labels
from rag.questions import Question, validate

ROOT = Path(__file__).parents[1]
F_RUNS = (
    "2026-11-03T21-03-51Z-claude-opus-5-5.json",
    "2026-11-03T20-59-15Z-claude-haiku-5-5.json",
)


def question(qid: str, kind: str = "single_fact", split: str = "dev", gold=("561/2006:art6(3)",)):
    return Question(qid, "?", "", (), tuple(gold), kind, "paraphrased", split, True)


def entry(qid: str, answer: str = "It is 90 hours.", ok: bool = True, answerable: bool = True):
    return {"id": qid, "verdict": {"ok": ok, "answerable": answerable, "answer": answer}}


def log(*entries):
    return {"answers": list(entries)}


# --- sentences ---------------------------------------------------------------------------------


def test_sentences_split_after_full_stops_and_lines():
    text = "You may drive 90 hours. Within that, 56 hours a week!\n- A list item.\nDone?"
    assert sentences(text) == [
        "You may drive 90 hours.",
        "Within that, 56 hours a week!",
        "- A list item.",
        "Done?",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "See Art. 6 of the Regulation.",
        "Write the time (e.g. 20:30 for 8:30 pm).",
        "Under Regulation (EU) No. 165/2014 it applies.",
        "Vehicles over 3.5 tonnes are covered.",
    ],
)
def test_sentences_keep_abbreviations_and_decimals_whole(text):
    assert sentences(text) == [text]


def test_sentences_keep_a_numbered_item_with_its_text():
    assert sentences("Order:\n1. Under the dial.\n2. Between the dials.") == [
        "Order:",
        "1. Under the dial.",
        "2. Between the dials.",
    ]


def test_sentences_split_before_a_quoted_or_bracketed_start():
    assert sentences('It says so. "Driving time" means this.') == [
        "It says so.",
        '"Driving time" means this.',
    ]


# --- select and assign -------------------------------------------------------------------------


def test_select_takes_only_dev_questions_every_run_answered():
    questions = [
        question("q001"),
        question("q002"),  # refused by one run
        question("q003"),  # failed the check in one run
        question("q004", split="test"),
        question("q005", gold=()),  # unanswerable
        question("q006"),
    ]
    runs = {
        "a": log(*(entry(q) for q in ("q001", "q002", "q003", "q004", "q005", "q006"))),
        "b": log(
            entry("q001"),
            entry("q002", answerable=False),
            entry("q003", ok=False),
            entry("q004"),
            entry("q005"),
            entry("q006"),
        ),
    }
    assert select(questions, runs, {"single_fact": 2}, always=()) == ["q001", "q006"]


def test_select_includes_always_and_draws_the_rest_by_seed():
    questions = [question(f"q{i:03}") for i in range(1, 11)]
    runs = {"a": log(*(entry(q.id) for q in questions))}
    first = select(questions, runs, {"single_fact": 4}, always=("q007",))
    assert "q007" in first and len(first) == 4
    assert select(questions, runs, {"single_fact": 4}, always=("q007",)) == first
    assert select(questions, runs, {"single_fact": 4}, always=("q007",), seed="other") != first


def test_select_refuses_a_quota_that_cannot_be_filled():
    runs = {"a": log(entry("q001"))}
    with pytest.raises(ValueError, match="quota of 2"):
        select([question("q001")], runs, {"single_fact": 2}, always=())


def test_select_refuses_an_always_question_not_answered_by_every_run():
    runs = {"a": log(entry("q001", answerable=False))}
    with pytest.raises(ValueError, match="q001"):
        select([question("q001")], runs, {"single_fact": 1}, always=("q001",))


def test_assign_gives_each_question_both_runs_in_a_seeded_order():
    letters = assign([f"q{i:03}" for i in range(1, 21)], ["opus", "haiku"])
    assert all(sorted(v.values()) == ["haiku", "opus"] for v in letters.values())
    assert {v["A"] for v in letters.values()} == {"haiku", "opus"}  # not always the same model
    assert assign(["q001"], ["haiku", "opus"]) == assign(["q001"], ["opus", "haiku"])


def test_the_real_selection_has_15_questions_and_q001(tree):
    questions, errors = validate((ROOT / "data/questions.v1.jsonl").read_text("utf-8"), tree)
    assert not errors
    by_id = {q.id: q for q in questions}
    logs = {
        name: json.loads((ROOT / "runs/generation" / name).read_text("utf-8")) for name in F_RUNS
    }
    chosen = select(questions, logs)
    assert len(chosen) == sum(QUOTAS.values()) == 15
    assert "q001" in chosen
    assert {k: sum(by_id[q].type == k for q in chosen) for k in QUOTAS} == QUOTAS


# --- validate_labels ---------------------------------------------------------------------------

EXPECTED = answers(
    ["q001"],
    {"opus": log(entry("q001", "It is 90 hours. So 34 remain.")), "haiku": log(entry("q001"))},
)


def row(run="opus", supported=True, index=None, sentence=None, **extra):
    fields = {
        "question_id": "q001",
        "run": run,
        "supported": supported,
        "first_unsupported": index,
        "sentence": sentence,
        "note": "",
    }
    return json.dumps({**fields, **extra})


def test_validate_labels_accepts_a_complete_file():
    text = row("opus", False, 1, "So 34 remain.") + "\n" + row("haiku") + "\n"
    labels, errors = validate_labels(text, EXPECTED)
    assert errors == []
    assert [label.supported for label in labels] == [False, True]


def test_validate_labels_reports_answers_not_labelled():
    _, errors = validate_labels(row("haiku"), EXPECTED)
    assert errors == ["1 of 2 chosen answers not labelled"]


@pytest.mark.parametrize(
    ("line", "problem"),
    [
        (row("other"), "not one of the answers chosen"),
        (row("opus", True, 0, "It is 90 hours."), "has no first unsupported sentence"),
        (row("opus", False), "needs a sentence number from 0 to 1"),
        (row("opus", False, 2, "x"), "needs a sentence number from 0 to 1"),
        (row("opus", False, 1, "So 35 remain."), "is not sentence 1"),
        (row("opus", 1), "supported: 1 is the wrong type"),
        (row("opus", False, True, "So 34 remain."), "first_unsupported: True is the wrong type"),
        (row("opus", extra=1), "extra: not a field"),
        ("[1]", "not a JSON object"),
        ("{", "not JSON"),
    ],
)
def test_validate_labels_rejects_bad_rows(line, problem):
    _, errors = validate_labels(line + "\n" + row("haiku"), EXPECTED)
    assert any(problem in e for e in errors), errors


def test_validate_labels_rejects_a_second_label_for_one_answer():
    text = "\n".join([row("haiku"), row("opus"), row("haiku")])
    _, errors = validate_labels(text, EXPECTED)
    assert any("already labelled on line 1" in e for e in errors), errors
