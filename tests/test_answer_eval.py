import json
from pathlib import Path

import pytest

from rag.answer_eval import (
    ANSWERED,
    GENERATION,
    KEY_FACTS,
    NO_REPLY,
    OWNER,
    REFUSED,
    RETRIEVAL,
    UNRELIABLE,
    chunk_nodes,
    compare,
    evaluate,
    score,
    sign_test,
)
from rag.parse import Node, Tree
from rag.questions import Question, validate
from rag.retrieval_eval import UnseenSplitError, wilson

ROOT = Path(__file__).parents[1]


def small_tree() -> Tree:
    """Act a: Art. 1 with paragraphs (1) and (2), (2) with point (a); Art. 2 with paragraph (1)."""
    tree = Tree()
    tree.add(Node("a", "a", "act", None, "Act", ("Act",), None))
    for article, paragraphs in (("art1", ("(1)", "(2)")), ("art2", ("(1)",))):
        tree.add(Node(f"a:{article}", "a", "article", article[3:], None, (), "a"))
        for label in paragraphs:
            tree.add(Node(f"a:{article}{label}", "a", "paragraph", label, None, (), f"a:{article}"))
    tree.add(Node("a:art1(2)(a)", "a", "point", "(a)", None, (), "a:art1(2)"))
    return tree


TREE = small_tree()
# Paragraph chunks, and one article chunk that holds all of Art. 1.
NODES_OF = {
    "p:1(1)": ("a:art1(1)",),
    "p:1(2)": ("a:art1(2)", "a:art1(2)(a)"),
    "p:2(1)": ("a:art2(1)",),
    "art:1": ("a:art1(1)", "a:art1(2)", "a:art1(2)(a)"),
}


def question(gold=("a:art1(1)",), qid="q001", key_facts=("9 hours",), **fields) -> Question:
    base = dict(
        id=qid,
        question="?",
        answer="",
        key_facts=tuple(key_facts) if gold else (),
        gold_node_ids=tuple(gold),
        type="single_fact" if gold else "unanswerable",
        phrasing="paraphrased",
        split="dev",
        checked=True,
    )
    return Question(**{**base, **fields})


def entry(
    qid="q001",
    retrieved=("p:1(1)",),
    answer="At most nine hours a day.",
    cited=("a:art1(1)",),
    ok=True,
    answerable=True,
    cost=0.02,
    input_tokens=2000,
    output_tokens=400,
    seconds=5.0,
) -> dict:
    """A run log entry as `rag.generate.answer_all` writes it, cut to what scoring reads."""
    if not answerable:
        answer, cited = "The documents don't say.", ()
    return {
        "id": qid,
        "retrieved": list(retrieved),
        "verdict": {
            "ok": ok,
            "answerable": answerable if ok else None,
            "answer": answer if ok else None,
            "problems": [] if ok else ["a quote isn't in the cited node"],
            "citations": [{"node_id": n, "quote": "…"} for n in cited] if ok else [],
        },
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "cost_usd": cost,
        "seconds": seconds,
    }


def log(entries, failed=(), split="dev") -> dict:
    return {
        "config": {"split": split, "retrieval": {"strategy": "paragraph"}},
        "answers": list(entries),
        "failed": [{"id": qid, "error": "Throttled: gave up"} for qid in failed],
    }


# --- Correctness -----------------------------------------------------------------------------


def test_an_answer_stating_every_key_fact_is_correct():
    s = score(question(), entry(answer="Nine hours."), TREE, NODES_OF)
    assert s.outcome == ANSWERED and s.correct is True and s.missing_facts == ()
    assert s.trace is None


def test_an_answer_missing_a_key_fact_is_wrong_and_names_it():
    q = question(key_facts=("9 hours", "10 hours"))
    s = score(q, entry(answer="Nine hours."), TREE, NODES_OF)
    assert s.correct is False and s.missing_facts == ("10 hours",)
    assert s.trace == GENERATION  # the gold paragraph was shown


def test_a_refusal_an_unreliable_answer_and_no_reply_are_not_correct():
    q = question()
    refused = score(q, entry(answerable=False), TREE, NODES_OF)
    unreliable = score(q, entry(ok=False), TREE, NODES_OF)
    no_reply = score(q, None, TREE, NODES_OF)
    assert (refused.outcome, unreliable.outcome, no_reply.outcome) == (
        REFUSED,
        UNRELIABLE,
        NO_REPLY,
    )
    for s in (refused, unreliable, no_reply):
        assert s.correct is False and s.missing_facts == ("9 hours",)
        assert s.citation_recall == 0.0 and s.citation_precision is None
    assert no_reply.trace is None  # the log doesn't say what search found
    assert no_reply.cost_usd is None and no_reply.seconds is None


def test_questions_without_key_facts_get_no_correctness():
    s = score(question(key_facts=()), entry(answer="Anything."), TREE, NODES_OF)
    assert s.correct is None and s.trace is None
    # Refused, it is still traced: something went wrong whatever the gold answer says.
    s = score(question(key_facts=()), entry(answerable=False), TREE, NODES_OF)
    assert s.correct is None and s.trace == GENERATION


def test_the_owners_grade_scores_a_question_without_key_facts():
    q = question(key_facts=())
    for grade, correct in (("correct", True), ("partly_correct", False), ("wrong", False)):
        s = score(q, entry(answer="Anything."), TREE, NODES_OF, {"q001": grade})
        assert s.correct is correct and s.correct_by == OWNER and s.right is correct
    # Graded wrong, it is traced like an answer missing a key fact.
    assert score(q, entry(), TREE, NODES_OF, {"q001": "wrong"}).trace == GENERATION
    # A refusal isn't shown to the owner and isn't correct.
    refused = score(q, entry(answerable=False), TREE, NODES_OF, {})
    assert refused.correct is False and refused.correct_by == OWNER
    # Key facts still decide a question that has them, whatever the grades say.
    keyed = score(question(), entry(answer="Nine hours."), TREE, NODES_OF, {})
    assert keyed.correct is True and keyed.correct_by == KEY_FACTS


def test_an_answer_without_a_grade_or_a_grade_without_an_answer_is_refused():
    q = question(key_facts=())
    with pytest.raises(ValueError, match="don't grade it"):
        score(q, entry(), TREE, NODES_OF, {})
    with pytest.raises(ValueError, match="not answered"):
        score(q, entry(answerable=False), TREE, NODES_OF, {"q001": "correct"})


def test_a_graded_run_scores_its_questions_without_key_facts():
    ev = evaluate(RUN, QUESTIONS, TREE, NODES_OF, hand_grades={"q003": "correct"})
    q003 = next(s for s in ev.scores if s.question_id == "q003")
    assert q003.correct is True
    assert ev.groups[0].over["correct"] == 6  # the five with key facts, and q003
    with pytest.raises(ValueError, match="q001"):  # q001 has key facts
        evaluate(RUN, QUESTIONS, TREE, NODES_OF, hand_grades={"q003": "correct", "q001": "wrong"})


def test_unanswerable_questions_are_never_scored_for_correctness_or_citations():
    q = question(gold=())
    refused = score(q, entry(answerable=False), TREE, NODES_OF)
    assert refused.correct is None and refused.trace is None
    assert refused.citation_precision is None and refused.citation_recall is None
    answered = score(q, entry(), TREE, NODES_OF)
    assert answered.outcome == ANSWERED and answered.trace == GENERATION


# --- Citations ---------------------------------------------------------------------------------


def test_citing_a_node_inside_the_gold_node_finds_it():
    # Gold is the paragraph; the answer cites its point (a), whose words lie inside it.
    q = question(gold=("a:art1(2)",))
    s = score(q, entry(retrieved=("p:1(2)",), cited=("a:art1(2)(a)",)), TREE, NODES_OF)
    assert s.citation_precision == 1.0 and s.citation_recall == 1.0


def test_citing_the_node_above_the_gold_node_does_not_find_it():
    # Gold is point (a); citing paragraph (2) quotes its own words, the lead-in, not point (a).
    q = question(gold=("a:art1(2)(a)",))
    s = score(q, entry(retrieved=("p:1(2)",), cited=("a:art1(2)",)), TREE, NODES_OF)
    assert s.citation_precision == 0.0 and s.citation_recall == 0.0


def test_precision_and_recall_with_two_gold_nodes_and_extra_citations():
    q = question(gold=("a:art1(1)", "a:art2(1)"))
    cited = ("a:art1(1)", "a:art1(2)", "a:art1(1)")  # the repeated node counts once
    s = score(q, entry(retrieved=("art:1", "p:2(1)"), cited=cited), TREE, NODES_OF)
    assert s.cited == ("a:art1(1)", "a:art1(2)")
    assert s.citation_precision == 0.5
    assert s.citation_recall == 0.5


def test_an_unknown_cited_node_is_refused():
    with pytest.raises(ValueError, match="not in the corpus"):
        score(question(), entry(cited=("a:art9",)), TREE, NODES_OF)


# --- Tracing ---------------------------------------------------------------------------------


def test_a_wrong_answer_without_the_gold_node_shown_is_a_retrieval_miss():
    s = score(question(), entry(retrieved=("p:2(1)",), answerable=False), TREE, NODES_OF)
    assert s.gold_retrieved == () and s.trace == RETRIEVAL


def test_an_article_chunk_shows_its_gold_paragraph():
    q = question(gold=("a:art1(2)",))
    s = score(q, entry(retrieved=("art:1",), answerable=False), TREE, NODES_OF)
    assert s.gold_retrieved == ("a:art1(2)",) and s.trace == GENERATION


def test_one_of_two_gold_nodes_shown_makes_it_a_generation_error():
    q = question(gold=("a:art1(1)", "a:art2(1)"))
    s = score(q, entry(retrieved=("p:2(1)",), ok=False), TREE, NODES_OF)
    assert s.gold_retrieved == ("a:art2(1)",) and s.trace == GENERATION


def test_a_chunk_the_corpus_no_longer_gives_is_refused():
    with pytest.raises(ValueError, match="another parse"):
        score(question(), entry(retrieved=("p:9(9)",)), TREE, NODES_OF)


def test_a_chunk_map_from_another_parse_is_refused():
    stale = {**NODES_OF, "p:1(1)": ("a:art1(1)-renamed",)}
    with pytest.raises(ValueError, match="chunk map made from another parse"):
        score(question(), entry(), TREE, stale)


# --- A run -------------------------------------------------------------------------------------


QUESTIONS = [
    question(qid="q001"),  # answered, correct
    question(qid="q002", phrasing="verbatim"),  # answered, wrong
    question(qid="q003", key_facts=()),  # answered, no key facts
    question(qid="q004"),  # refused, gold not shown: retrieval miss
    question(qid="q005"),  # failed the check
    question(qid="q006"),  # no reply
    question(gold=(), qid="q007"),  # unanswerable, refused
    question(gold=(), qid="q008"),  # unanswerable, answered
    question(qid="q100", split="test"),
]
RUN = log(
    [
        entry("q001", answer="9 hours", cost=0.01, seconds=2.0),
        entry("q002", answer="10 hours", cited=("a:art1(1)", "a:art1(2)"), cost=0.03),
        entry("q003", cost=0.02),
        entry("q004", retrieved=("p:2(1)",), answerable=False, cost=0.01),
        entry("q005", ok=False, cost=0.02),
        entry("q007", answerable=False, cost=0.01),
        entry("q008", cost=0.02, seconds=8.0),
    ],
    failed=["q006"],
)


def test_a_run_is_scored_overall_and_by_group():
    ev = evaluate(RUN, QUESTIONS, TREE, NODES_OF)
    assert [s.question_id for s in ev.scores] == [f"q00{i}" for i in range(1, 9)]
    all_ = ev.groups[0]
    assert all_.group == "all" and all_.n == 8
    assert all_.counts == {
        ANSWERED: 4,
        REFUSED: 2,
        UNRELIABLE: 1,
        NO_REPLY: 1,
        RETRIEVAL: 1,  # q004; q006 got no reply and isn't traced
        GENERATION: 3,  # q002, q005 and q008
    }
    m, over = all_.metrics, all_.over
    # Correctness over the five answerable questions with key facts: only q001.
    assert (m["correct"].value, over["correct"]) == (0.2, 5)
    assert (m["correct"].low, m["correct"].high) == wilson(1, 5)
    assert (m["refusal_accuracy"].value, over["refusal_accuracy"]) == (0.5, 2)
    assert (m["false_refusals"].value, over["false_refusals"]) == (1 / 6, 6)
    assert (m["unreliable"].value, over["unreliable"]) == (1 / 8, 8)
    assert (m["no_reply"].value, over["no_reply"]) == (1 / 8, 8)
    # Precision over the three answerable questions answered: 1, 0.5 and 1.
    assert m["citation_precision"].value == pytest.approx(2.5 / 3)
    assert over["citation_precision"] == 3
    # Recall over all six answerable questions: q001, q002 and q003 found theirs.
    assert (m["citation_recall"].value, over["citation_recall"]) == (0.5, 6)
    # Cost and time over the seven replies; the failed call has none.
    assert m["cost_usd"].value == pytest.approx(0.12 / 7) and over["cost_usd"] == 7
    assert m["seconds"].value == pytest.approx(35 / 7)
    assert m["cost_usd"].low <= m["cost_usd"].value <= m["cost_usd"].high

    groups = {g.group: g for g in ev.groups}
    assert set(groups) == {"all", "paraphrased", "verbatim", "single_fact", "unanswerable"}
    verbatim = groups["verbatim"]
    assert verbatim.question_ids == ("q002",) and verbatim.metrics["correct"].value == 0.0
    # A group without unanswerable questions has no refusal accuracy.
    assert "refusal_accuracy" not in verbatim.metrics
    assert "correct" not in groups["unanswerable"].metrics


def test_a_run_sent_as_a_batch_has_cost_and_tokens_but_no_seconds():
    batch = {**RUN, "answers": [{**e, "seconds": None} for e in RUN["answers"]]}
    m = evaluate(batch, QUESTIONS, TREE, NODES_OF).groups[0].metrics
    assert "seconds" not in m
    direct = evaluate(RUN, QUESTIONS, TREE, NODES_OF).groups[0].metrics
    for metric in ("cost_usd", "input_tokens", "output_tokens"):
        assert m[metric] == direct[metric]


def test_scoring_is_reproducible():
    assert evaluate(RUN, QUESTIONS, TREE, NODES_OF) == evaluate(RUN, QUESTIONS, TREE, NODES_OF)


def test_an_unfinished_run_is_refused():
    unfinished = log([e for e in RUN["answers"] if e["id"] != "q008"], failed=["q006"])
    with pytest.raises(ValueError, match="nothing for 1 dev questions"):
        evaluate(unfinished, QUESTIONS, TREE, NODES_OF)


def test_a_run_with_a_question_twice_or_from_another_split_is_refused():
    twice = log([*RUN["answers"], entry("q001")], failed=["q006"])
    with pytest.raises(ValueError, match="twice"):
        evaluate(twice, QUESTIONS, TREE, NODES_OF)
    both = log(RUN["answers"], failed=["q006", "q001"])
    with pytest.raises(ValueError, match="both answered and failed"):
        evaluate(both, QUESTIONS, TREE, NODES_OF)
    extra = log([*RUN["answers"], entry("q100")], failed=["q006"])
    with pytest.raises(ValueError, match="not in the dev split: q100"):
        evaluate(extra, QUESTIONS, TREE, NODES_OF)


def test_the_test_split_needs_the_flag():
    test_run = log([entry("q100")], split="test")
    with pytest.raises(UnseenSplitError):
        evaluate(test_run, QUESTIONS, TREE, NODES_OF)
    ev = evaluate(test_run, QUESTIONS, TREE, NODES_OF, allow_test=True)
    assert [s.question_id for s in ev.scores] == ["q100"]


def test_a_typed_question_log_is_not_scored():
    with pytest.raises(ValueError, match="'typed'"):
        evaluate(log([], split="typed"), QUESTIONS, TREE, NODES_OF)


# --- F's real dev runs ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_questions(tree):
    questions, errors = validate((ROOT / "data" / "questions.v1.jsonl").read_text(), tree)
    assert not errors
    return questions


@pytest.fixture(scope="module")
def paragraph_nodes(tree):
    return chunk_nodes(tree, "paragraph")


@pytest.mark.parametrize(
    ("name", "answered", "refused", "unreliable"),
    [
        ("2026-11-03T20-59-15Z-claude-haiku-5-5.json", 39, 20, 1),
        ("2026-11-03T21-03-51Z-claude-opus-5-5.json", 42, 18, 0),
    ],
)
def test_f_dev_runs_give_the_counts_in_progress(
    tree, real_questions, paragraph_nodes, name, answered, refused, unreliable
):
    run = json.loads((ROOT / "runs" / "generation" / name).read_text(encoding="utf-8"))
    ev = evaluate(run, real_questions, tree, paragraph_nodes)
    counts = ev.groups[0].counts
    assert (counts[ANSWERED], counts[REFUSED], counts[UNRELIABLE]) == (
        answered,
        refused,
        unreliable,
    )
    assert counts[NO_REPLY] == 0
    # The ten unanswerable dev questions were all refused, by both models.
    assert ev.groups[0].metrics["refusal_accuracy"].value == 1.0
    # The seven answerable questions refused by both because search found none of their gold
    # passages are retrieval misses.
    misses = {s.question_id for s in ev.scores if s.trace == RETRIEVAL}
    assert misses == {"q024", "q025", "q029", "q036", "q048", "q063", "q064"}


# --- Paired comparison ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wins", "losses", "p"),
    [
        (8, 0, 2 / 2**8),  # E's hybrid against BM25 over paragraph chunks: 0.0078
        (0, 8, 2 / 2**8),
        (4, 1, 0.375),  # E's article chunks
        (5, 2, 0.453125),  # E's windows
        (3, 3, 1.0),
        (0, 0, 1.0),
        (10, 2, 2 * (1 + 12 + 66) / 2**12),
    ],
)
def test_the_sign_test_gives_known_binomial_values(wins, losses, p):
    assert sign_test(wins, losses) == pytest.approx(p)


def test_the_sign_test_refuses_negative_counts():
    with pytest.raises(ValueError):
        sign_test(-1, 3)


def test_right_is_correct_or_an_unanswerable_question_refused():
    assert score(question(), entry(answer="Nine hours."), TREE, NODES_OF).right is True
    assert score(question(), entry(answer="Ten hours."), TREE, NODES_OF).right is False
    assert score(question(key_facts=()), entry(), TREE, NODES_OF).right is None
    unanswerable = question(gold=())
    assert score(unanswerable, entry(answerable=False), TREE, NODES_OF).right is True
    assert score(unanswerable, entry(), TREE, NODES_OF).right is False
    # A call that got no reply says nothing about the setup, answerable or not.
    assert score(unanswerable, None, TREE, NODES_OF).right is None
    assert score(question(), None, TREE, NODES_OF).right is None


def test_two_runs_are_compared_question_by_question():
    # RUN gets right q001 and q007; this one gets right q001, q004, q005 and q008, and still
    # answers q003, which has no key facts and is left out.
    other = log(
        [
            entry("q001", answer="9 hours"),
            entry("q002", answer="10 hours"),
            entry("q003"),
            entry("q004"),
            entry("q005"),
            entry("q006", answer="Ten hours."),
            entry("q007"),
            entry("q008", answerable=False),
        ]
    )
    a = evaluate(other, QUESTIONS, TREE, NODES_OF)
    b = evaluate(RUN, QUESTIONS, TREE, NODES_OF)
    c = compare(a, b)
    # q006 got no reply in RUN, so it is left out with q003.
    assert (c.wins, c.losses, c.both_right, c.both_wrong) == (3, 1, 1, 1)
    assert c.n == 6 and c.p == pytest.approx(sign_test(3, 1)) == pytest.approx(0.625)
    flipped = compare(b, a)
    assert (flipped.wins, flipped.losses, flipped.p) == (1, 3, c.p)


def test_runs_on_different_splits_are_not_compared():
    dev = evaluate(RUN, QUESTIONS, TREE, NODES_OF)
    test = evaluate(log([entry("q100")], split="test"), QUESTIONS, TREE, NODES_OF, allow_test=True)
    with pytest.raises(ValueError, match="can't pair a dev run with a test run"):
        compare(dev, test)
