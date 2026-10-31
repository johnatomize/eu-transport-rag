import math

import pytest

from rag.parse import Node, Tree
from rag.questions import Question
from rag.retrieval_eval import (
    KS,
    UnseenSplitError,
    bootstrap,
    evaluate,
    score,
    wilson,
)


def small_tree() -> Tree:
    """Act a: Art. 1 with paragraphs (1) and (2), Art. 2 with paragraph (1)."""
    tree = Tree()
    tree.add(Node("a", "a", "act", None, "Act", ("Act",), None))
    for article, paragraphs in (("art1", ("(1)", "(2)")), ("art2", ("(1)",))):
        tree.add(Node(f"a:{article}", "a", "article", article[3:], None, (), "a"))
        for label in paragraphs:
            tree.add(Node(f"a:{article}{label}", "a", "paragraph", label, None, (), f"a:{article}"))
    return tree


TREE = small_tree()


def question(gold: tuple[str, ...], qid: str = "q001", **fields) -> Question:
    base = dict(
        id=qid,
        question="?",
        answer="",
        key_facts=(),
        gold_node_ids=gold,
        type="single_fact",
        phrasing="paraphrased",
        split="dev",
        checked=True,
    )
    return Question(**{**base, **fields})


# --- Wilson interval against published values -------------------------------------------------

# Newcombe (1998), "Two-sided confidence intervals for the single proportion: comparison of seven
# methods", Statistics in Medicine 17:857-872, Table II, method 3 (Wilson score, no continuity
# correction), 95%.
NEWCOMBE = [
    (81, 263, 0.2553, 0.3662),
    (15, 148, 0.0624, 0.1605),
    (0, 20, 0.0, 0.1611),
    (1, 29, 0.0061, 0.1718),
]


@pytest.mark.parametrize(("successes", "n", "low", "high"), NEWCOMBE)
def test_wilson_matches_newcombe(successes, n, low, high):
    got = wilson(successes, n)
    assert got == pytest.approx((low, high), abs=5e-5)


def test_wilson_is_symmetric_and_bounded():
    low, high = wilson(20, 20)
    assert high == 1.0 and low == pytest.approx(1 - wilson(0, 20)[1])


# --- Bootstrap ------------------------------------------------------------------------------


def test_bootstrap_is_reproducible_with_a_seed():
    values = [0.13, 0.57, 0.91, 0.02, 0.44, 0.78, 0.36, 0.65, 0.29, 0.83]
    assert bootstrap(values, seed=7) == bootstrap(values, seed=7)
    assert bootstrap(values, seed=7) != bootstrap(values, seed=8)
    # Pinned, so that a change to the resampling shows up as a change in every interval.
    assert bootstrap(values, seed=7) == pytest.approx((0.316, 0.668))


def test_bootstrap_brackets_the_mean_and_collapses_on_constant_data():
    values = [i % 3 / 2 for i in range(60)]
    low, high = bootstrap(values)
    mean = sum(values) / len(values)
    assert low < mean < high
    assert bootstrap([0.4] * 30) == pytest.approx((0.4, 0.4))


def test_bootstrap_agrees_with_wilson_on_a_proportion():
    """For 0/1 data the bootstrap and Wilson intervals should be close at n = 60."""
    values = [1.0] * 36 + [0.0] * 24
    low, high = bootstrap(values)
    w_low, w_high = wilson(36, 60)
    assert abs(low - w_low) < 0.03 and abs(high - w_high) < 0.03


# --- Scoring one ranking, worked by hand ---------------------------------------------------


def test_an_article_chunk_finds_a_gold_paragraph():
    # An article chunk holds the article's paragraphs, so its node IDs are theirs.
    s = score(question(("a:art1(2)",)), [("a:art2(1)",), ("a:art1(1)", "a:art1(2)")], TREE)
    assert s.found_at == {"a:art1(2)": 2}
    assert [s.recall[k] for k in KS] == [0, 1, 1, 1]
    assert s.mrr == 0.5
    assert s.ndcg == pytest.approx(1 / math.log2(3))


def test_a_paragraph_chunk_finds_part_of_a_gold_article():
    s = score(question(("a:art1",)), [("a:art1(2)",)], TREE)
    assert s.found_at == {"a:art1": 1}
    assert s.recall[1] == 1 and s.mrr == 1.0 and s.ndcg == 1.0


def test_a_neighbouring_paragraph_finds_nothing():
    s = score(question(("a:art1(1)",)), [("a:art1(2)",), ("a:art2(1)",)], TREE)
    assert s.found_at == {"a:art1(1)": None}
    assert all(s.recall[k] == 0 for k in KS)
    assert s.mrr == 0.0 and s.ndcg == 0.0


def test_two_gold_nodes_with_only_one_found():
    s = score(question(("a:art1(1)", "a:art2(1)")), [("a:art2",), ("a:art2(1)",)], TREE)
    # A chunk finds a gold node only by holding it or a descendant, never by holding an
    # ancestor: "a:art2" alone doesn't find gold "a:art2(1)", the chunk at rank 2 does.
    assert s.found_at == {"a:art1(1)": None, "a:art2(1)": 2}
    assert all(s.recall[k] == 0 for k in KS)
    assert s.fractional_recall == {1: 0.0, 3: 0.5, 5: 0.5, 10: 0.5}
    assert s.mrr == 0.5
    ideal = 1 + 1 / math.log2(3)
    assert s.ndcg == pytest.approx((1 / math.log2(3)) / ideal)


def test_a_gold_node_found_twice_counts_once_in_ndcg():
    # Overlapping windows: ranks 1 and 2 both hold the one gold node.
    s = score(question(("a:art1(1)",)), [("a:art1(1)",), ("a:art1(1)", "a:art1(2)")], TREE)
    assert s.ndcg == 1.0
    s = score(question(("a:art1(1)",)), [("a:art2(1)",), ("a:art1(1)",), ("a:art1(1)",)], TREE)
    assert s.ndcg == pytest.approx(1 / math.log2(3))


def test_one_chunk_finding_two_gold_nodes_scores_at_most_1():
    s = score(question(("a:art1(1)", "a:art1(2)")), [("a:art1(1)", "a:art1(2)")], TREE)
    assert s.recall[1] == 1
    assert s.ndcg == 1.0


def test_ranks_beyond_10_count_for_nothing():
    ranking = [("a:art2(1)",)] * 10 + [("a:art1(1)",)]
    s = score(question(("a:art1(1)",)), ranking, TREE)
    assert s.found_at == {"a:art1(1)": None}
    assert s.recall[10] == 0 and s.mrr == 0.0 and s.ndcg == 0.0


def test_an_unanswerable_question_cannot_be_scored():
    with pytest.raises(ValueError, match="q001"):
        score(question(()), [("a:art1(1)",)], TREE)


# --- A question set ------------------------------------------------------------------------


QUESTIONS = [
    question(("a:art1(1)",), "q001"),
    question(("a:art2(1)",), "q002", phrasing="verbatim", type="definition"),
    question((), "q003", type="unanswerable"),
    question(("a:art1(2)",), "q004", split="test"),
]
RANKINGS = {"q001": [("a:art1(1)",)], "q002": [("a:art1(1)",), ("a:art2(1)",)]}


def test_evaluate_scores_dev_by_group_and_counts_unanswerable():
    result = evaluate(QUESTIONS, RANKINGS, TREE)
    assert [s.question_id for s in result.scores] == ["q001", "q002"]
    assert result.unanswerable == 1
    groups = {g.group: g for g in result.groups}
    assert list(groups) == ["all", "paraphrased", "verbatim", "single_fact", "definition"]
    assert groups["all"].n == 2
    assert groups["all"].metrics["recall@1"].value == 0.5
    assert groups["all"].metrics["recall@1"].low == pytest.approx(wilson(1, 2)[0])
    assert groups["all"].metrics["mrr@10"].value == 0.75
    assert groups["verbatim"].metrics["recall@3"].value == 1.0


def test_evaluate_is_reproducible():
    assert evaluate(QUESTIONS, RANKINGS, TREE) == evaluate(QUESTIONS, RANKINGS, TREE)


def test_the_test_split_needs_the_flag():
    with pytest.raises(UnseenSplitError, match="checkpoint G"):
        evaluate(QUESTIONS, RANKINGS, TREE, splits=("dev", "test"))
    result = evaluate(
        QUESTIONS, {**RANKINGS, "q004": [("a:art1",)]}, TREE, splits=("test",), allow_test=True
    )
    assert [s.question_id for s in result.scores] == ["q004"]


def test_a_node_the_tree_lacks_is_an_error():
    with pytest.raises(ValueError, match="q001.*a:art9"):
        score(question(("a:art1(1)",)), [("a:art9",)], TREE)


def test_unknown_or_empty_splits_are_errors():
    with pytest.raises(ValueError, match="unknown split Dev"):
        evaluate(QUESTIONS, RANKINGS, TREE, splits=("Dev",))
    with pytest.raises(ValueError, match="no answerable questions"):
        evaluate([question((), "q003", type="unanswerable")], {}, TREE)
    assert evaluate(QUESTIONS, RANKINGS, TREE, splits="dev").splits == ("dev",)


def test_a_missing_ranking_is_an_error():
    with pytest.raises(ValueError, match="q002"):
        evaluate(QUESTIONS, {"q001": []}, TREE)
