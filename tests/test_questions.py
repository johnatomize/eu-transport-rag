import json
from itertools import product
from pathlib import Path

import pytest

from rag.questions import SPLITS, TYPES, Question, coverage, unchecked, validate

GOOD = {
    "id": "q001",
    "question": "How many hours can I drive in two weeks in a row?",
    "answer": "At most 90 hours in any two consecutive weeks.",
    "key_facts": ["90 hours"],
    "gold_node_ids": ["561/2006:art6(3)"],
    "type": "single_fact",
    "phrasing": "paraphrased",
    "split": "dev",
    "checked": False,
    "notes": "",
}
UNANSWERABLE = {**GOOD, "type": "unanswerable", "answer": "", "key_facts": [], "gold_node_ids": []}
DROP = object()
ROOT = Path(__file__).parents[1]
TEST_SPLIT = ROOT / "tests" / "snapshots" / "test_split.txt"
LAST_SPLIT_ID = "q100"  # the last question present when the split was frozen


def row(base: dict = GOOD, **changes) -> str:
    """`base` with some fields changed, or removed when set to DROP, as one JSON line."""
    merged = {**base, **changes}
    return json.dumps({k: v for k, v in merged.items() if v is not DROP})


def errors(tree, *lines: str) -> list[str]:
    return validate("\n".join(lines), tree)[1]


def test_a_valid_row_becomes_a_question(tree):
    questions, problems = validate(row(), tree)
    assert problems == []
    assert questions == [
        Question(
            id="q001",
            question=GOOD["question"],
            answer=GOOD["answer"],
            key_facts=("90 hours",),
            gold_node_ids=("561/2006:art6(3)",),
            type="single_fact",
            phrasing="paraphrased",
            split="dev",
            checked=False,
        )
    ]


def test_notes_is_optional_and_blank_lines_are_skipped(tree):
    text = "\n" + row(notes=DROP) + "\n\n" + row(id="q002", gold_node_ids=["561/2006:art6"])
    questions, problems = validate(text, tree)
    assert problems == []
    assert [q.id for q in questions] == ["q001", "q002"]


def test_lines_are_split_on_newlines_only_and_a_bom_is_ignored(tree):
    # U+2028 is legal inside a JSON string, and str.splitlines() would split on it.
    text = "\ufeff" + row(answer="At most 90 hours\u2028in any two consecutive weeks.")
    questions, problems = validate(text, tree)
    assert problems == []
    assert "\u2028" in questions[0].answer


@pytest.mark.parametrize(
    "line, message",
    [
        ("{not json", "line 1: not valid JSON (Expecting property name enclosed in double quotes)"),
        ('["q001"]', "line 1: not a JSON object"),
        (row(answer=DROP), "line 1 (q001): answer: missing"),
        (row(gold_node_id=["561/2006:art6(3)"]), "line 1 (q001): gold_node_id: unknown field"),
        (row(checked="no"), "line 1 (q001): checked: must be true or false"),
        (row(key_facts="90 hours"), "line 1 (q001): key_facts: must be a list of strings"),
        (row(key_facts=[90]), "line 1 (q001): key_facts: must be a list of strings"),
        (row(key_facts=[" "]), "line 1 (q001): key_facts: an empty fact"),
        (row(key_facts=["56 hours"]), "line 1 (q001): key_facts: '56 hours' is not in the answer"),
        (row()[:-1] + ', "split": "test"}', "line 1: split: given twice"),
        (row(id="Q1"), "line 1 (Q1): id: 'Q1' is not q followed by three digits"),
        (row(question=" "), "line 1 (q001): question: empty"),
        (
            row(type="fact"),
            "line 1 (q001): type: 'fact' is not one of single_fact, definition, multi_article, "
            "exception, cross_act, unanswerable, version_trap",
        ),
        (
            row(phrasing="literal"),
            "line 1 (q001): phrasing: 'literal' is not one of paraphrased, verbatim",
        ),
        (row(split="train"), "line 1 (q001): split: 'train' is not one of dev, test"),
        (
            row(gold_node_ids=["561/2006:art6(9)"]),
            "line 1 (q001): gold_node_ids: '561/2006:art6(9)' is not a node in the parsed corpus",
        ),
        (
            row(gold_node_ids=["561/2006:art6(3)", "561/2006:art6(3)"]),
            "line 1 (q001): gold_node_ids: '561/2006:art6(3)' is listed twice",
        ),
        (
            row(gold_node_ids=["561/2006:chII"]),
            "line 1 (q001): gold_node_ids: '561/2006:chII' is a whole chapter; name the nodes",
        ),
        (
            row(gold_node_ids=["guidance-7"]),
            "line 1 (q001): gold_node_ids: 'guidance-7' is a whole note; name the nodes",
        ),
        (
            row(gold_node_ids=["561/2006:art6", "561/2006:art6(3)"]),
            "line 1 (q001): gold_node_ids: '561/2006:art6(3)' is inside '561/2006:art6', "
            "also listed",
        ),
    ],
)
def test_a_malformed_row_is_rejected_with_a_clear_message(tree, line, message):
    assert errors(tree, line) == [message]


def test_a_duplicate_id_names_the_first_line(tree):
    assert errors(tree, row(), row()) == ["line 2 (q001): id: already used on line 1"]


@pytest.mark.parametrize("node_id", ["561/2006:art26", "561/2006:art26(4)", "165/2014:art45(2)"])
def test_a_gold_node_in_an_amending_article_is_rejected(tree, node_id):
    article = node_id.split("(")[0]
    assert errors(tree, row(gold_node_ids=[node_id])) == [
        f"line 1 (q001): gold_node_ids: {node_id!r} is in an amending article ({article}), "
        "whose wording has since been replaced"
    ]


def test_an_unanswerable_question_has_no_answer_facts_or_gold_nodes(tree):
    assert errors(tree, row(UNANSWERABLE)) == []
    assert errors(
        tree,
        row(UNANSWERABLE, answer="Yes, 9.", key_facts=["9"], gold_node_ids=GOOD["gold_node_ids"]),
    ) == [
        f"line 1 (q001): {name}: must be empty for an unanswerable question"
        for name in ("answer", "key_facts", "gold_node_ids")
    ]


def test_every_other_question_has_an_answer_and_gold_nodes(tree):
    assert errors(tree, row(answer="", key_facts=[], gold_node_ids=[])) == [
        "line 1 (q001): answer: empty; only an unanswerable question has no answer",
        "line 1 (q001): gold_node_ids: empty; only an unanswerable question has none",
    ]


def test_a_multi_article_question_needs_two_articles_or_annexes(tree):
    one = row(type="multi_article", gold_node_ids=["561/2006:art6(1)", "561/2006:art6(3)"])
    # A guidance page is in neither, so it doesn't count as the second one.
    page = row(type="multi_article", gold_node_ids=["561/2006:art6(3)", "guidance-7:p3"])
    two = row(type="multi_article", gold_node_ids=["561/2006:art6(3)", "561/2006:art7"])
    annex = row(type="multi_article", gold_node_ids=["165/2014:art34(5)", "165/2014:anxI.III.e.2"])
    message = (
        "line 1 (q001): gold_node_ids: a multi_article question needs two articles or annexes "
        "or more"
    )
    assert errors(tree, one) == [message]
    assert errors(tree, page) == [message]
    assert errors(tree, two) == []
    assert errors(tree, annex) == []


def test_a_cross_act_question_needs_two_acts(tree):
    # A guidance note is not an act, even when it interprets one.
    note = row(type="cross_act", gold_node_ids=["561/2006:art6(3)", "guidance-7:p3"])
    two = row(type="cross_act", gold_node_ids=["561/2006:art6(2)", "2002/15:art4(a)"])
    assert errors(tree, note) == [
        "line 1 (q001): gold_node_ids: a cross_act question needs two acts or more"
    ]
    assert errors(tree, two) == []


def test_a_version_trap_records_the_old_answer_in_its_notes(tree):
    assert errors(tree, row(type="version_trap", notes=DROP)) == [
        "line 1 (q001): notes: a version_trap question needs what changed and the old answer"
    ]
    assert errors(tree, row(type="version_trap", notes="Before 2020/1054: …")) == []


def test_every_type_must_appear_in_both_splits():
    every = [
        Question(f"q{i:03d}", "?", "", (), (), kind, "paraphrased", split, False)
        for i, (kind, split) in enumerate(product(TYPES, SPLITS))
    ]
    assert coverage(every) == []
    without = [q for q in every if not (q.type == "version_trap" and q.split == "test")]
    assert coverage(without) == ["type 'version_trap' has no test questions"]


def test_every_question_must_be_checked_by_the_owner():
    questions = [
        Question("q001", "?", "", (), (), "unanswerable", "paraphrased", "dev", True),
        Question("q002", "?", "", (), (), "unanswerable", "paraphrased", "dev", False),
    ]
    assert unchecked(questions) == ["q002: not checked by the owner yet"]
    assert unchecked(questions[:1]) == []


def test_the_test_split_is_frozen():
    # The split was assigned once and frozen (docs/DECISIONS.md): later questions go to dev,
    # and a test question is corrected in place, never moved. A failure means a test question
    # was moved, deleted or added.
    text = (ROOT / "data" / "questions.v1.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.split("\n") if line.strip()]
    test = [r["id"] for r in rows if r["split"] == "test"]
    assert test == TEST_SPLIT.read_text(encoding="utf-8").splitlines()
    assert all(r["split"] == "dev" for r in rows if r["id"] > LAST_SPLIT_ID)
