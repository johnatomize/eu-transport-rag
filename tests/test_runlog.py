import json
from pathlib import Path

import pytest

from rag.chunk import Chunk
from rag.parse import Node, Tree
from rag.questions import Question
from rag.retrieve import BM25, Hit
from rag.runlog import (
    INPUTS,
    StaleRunError,
    check_chunks,
    chunks_fingerprint,
    dumps,
    load,
    make_run,
    rankings,
    same_results,
)
from rag.tables import retrieval


def small_tree() -> Tree:
    tree = Tree()
    tree.add(Node("a", "a", "act", None, "Act", ("Act",), None))
    for article in ("art1", "art2"):
        tree.add(Node(f"a:{article}", "a", "article", article[3:], None, (), "a"))
    return tree


def question(qid: str, gold: tuple[str, ...], text: str, **fields) -> Question:
    base = dict(
        id=qid, question=text, answer="", key_facts=(), gold_node_ids=gold, type="single_fact",
        phrasing="paraphrased", split="dev", checked=True,
    )  # fmt: skip
    return Question(**{**base, **fields})


TREE = small_tree()
CHUNKS = [
    Chunk("t:a:art1", "t", "a", "Act › Article 1", "daily driving time nine hours", ("a:art1",)),
    Chunk("t:a:art2", "t", "a", "Act › Article 2", "weekly rest period 45 hours", ("a:art2",)),
]
QUESTIONS = [
    question("q001", ("a:art1",), "daily driving time"),
    question("q002", ("a:art2",), "weekly rest", phrasing="verbatim"),
    question("q003", (), "ferry crossings", type="unanswerable"),
    question("q004", ("a:art1",), "nine hours", split="test"),
]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for path in INPUTS:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(f"contents of {path}\n", encoding="utf-8")
    return tmp_path


def run(root: Path, name: str = "bm25-t") -> dict:
    config = {"retriever": "bm25", "k1": 1.5, "b": 0.75, "strategy": "t"}
    return make_run(name, config, BM25(CHUNKS), CHUNKS, QUESTIONS, root)


def test_a_run_holds_every_dev_question_and_the_chunks_it_ranked(root):
    log = run(root)
    assert [q["id"] for q in log["questions"]] == ["q001", "q002", "q003"]
    assert log["questions"][0]["ranking"][0][0] == "t:a:art1"
    assert log["chunks"] == {"t:a:art1": ["a:art1"], "t:a:art2": ["a:art2"]}
    assert log["words"] == {"t:a:art1": 9, "t:a:art2": 9}
    assert log["config"] == {
        "retriever": "bm25", "k1": 1.5, "b": 0.75, "strategy": "t", "chunks": 2,
        "mean_chunk_words": 9.0, "chunks_sha256": chunks_fingerprint(CHUNKS), "split": "dev",
        "depth": 50,
    }  # fmt: skip
    assert set(log["inputs"]) == set(INPUTS)
    assert rankings(log)["q002"] == [("a:art2",), ("a:art1",)]


def test_two_runs_differ_only_in_when_and_from_which_commit(root):
    first, second = run(root), run(root)
    assert same_results(first, second)
    second["questions"][0]["ranking"].reverse()
    assert not same_results(first, second)


def test_the_chunk_fingerprint_changes_with_any_chunk():
    changed = [CHUNKS[0], Chunk(*[getattr(CHUNKS[1], f) for f in ("id", "strategy", "doc_id",
               "prefix")], "weekly rest period 24 hours", ("a:art2",))]  # fmt: skip
    assert chunks_fingerprint(CHUNKS) == chunks_fingerprint(list(CHUNKS))
    assert chunks_fingerprint(changed) != chunks_fingerprint(CHUNKS)


def test_a_log_from_other_chunking_code_is_refused(tree, root):
    from rag.chunk import STRATEGIES

    chunks = STRATEGIES["article"](tree)
    config = {"retriever": "bm25", "strategy": "article"}
    log = make_run("bm25-article", config, BM25(chunks), chunks, QUESTIONS, root)
    check_chunks([log], tree)
    log["config"]["chunks_sha256"] = "0" * 64
    with pytest.raises(StaleRunError, match="bm25-article: its article chunks differ"):
        check_chunks([log], tree)
    log["config"]["strategy"] = "sentence"
    with pytest.raises(StaleRunError, match="no chunking strategy 'sentence'"):
        check_chunks([log], tree)


def test_a_run_without_chunks_is_an_error(root):
    with pytest.raises(ValueError, match="no chunks"):
        make_run("x", {}, BM25(CHUNKS), [], QUESTIONS, root)


def test_runs_are_on_the_dev_split_only(root):
    config = {"retriever": "bm25", "strategy": "t"}
    with pytest.raises(ValueError, match="dev split only"):
        make_run("x", config, BM25(CHUNKS), CHUNKS, QUESTIONS, root, split="test")


def test_a_log_reads_back_as_written_one_line_per_question(root):
    log = run(root)
    text = dumps(log)
    assert json.loads(text) == log
    assert sum(line.startswith('  {"id": "q') for line in text.splitlines()) == 3


def test_a_stale_log_is_refused(root):
    path = root / "log.json"
    path.write_text(dumps(run(root)), encoding="utf-8")
    assert load(path, root)["run"]["name"] == "bm25-t"
    (root / "data" / "questions.v1.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(StaleRunError, match="data/questions.v1.jsonl"):
        load(path, root)


def test_the_table_scores_each_run_and_lists_what_every_run_misses(root):
    good = run(root, "good")
    bad = run(root, "bad")
    for q in bad["questions"]:  # the worst ranking: every gold node at the bottom
        q["ranking"].reverse()
    table = retrieval([good, bad], QUESTIONS, TREE)
    assert "2 questions with gold nodes, and 1 unanswerable questions left out" in table
    # Both chunks are 9 words long and both are in every top 5: 18 words.
    assert "| good | 9 | 18 | 1.00 (0.34–1.00) |" in table
    assert "| bad | 9 | 18 | 0.00 (0.00–0.66) |" in table
    assert "| good | bm25 (k1 1.5, b 0.75) | t | 2 | 50 | – | unknown |" in table
    assert "## Questions no run finds in the top 10 (0)" in table
    assert retrieval([good, bad], QUESTIONS, TREE) == table


def test_the_table_leaves_out_a_phrasing_with_no_questions(root):
    paraphrased = [q for q in QUESTIONS if q.phrasing == "paraphrased"]
    table = retrieval([run(root)], paraphrased, TREE)
    assert "## Paraphrased questions (n = 1)" in table
    assert "Verbatim questions" not in table


def test_the_table_needs_runs_from_one_split(root):
    with pytest.raises(ValueError, match="more than one run log named bm25-t"):
        retrieval([run(root), run(root)], QUESTIONS, TREE)
    other = run(root)
    other["config"]["split"] = "test"
    with pytest.raises(ValueError, match="more than one split"):
        retrieval([run(root), other], QUESTIONS, TREE)
    with pytest.raises(ValueError, match="no run logs"):
        retrieval([], QUESTIONS, TREE)


class Expanded:
    """A retriever whose second hit was brought in by a cross-reference from the first."""

    def search(self, query: str, k: int) -> list[Hit]:
        return [Hit(CHUNKS[0], 2.0), Hit(CHUNKS[1], 2.0, via=CHUNKS[0].id)][:k]


def test_a_log_marks_the_chunks_expansion_added_and_the_table_counts_them(root):
    config = {"retriever": "bm25", "strategy": "t", "expansion": "references"}
    log = make_run("bm25-t-refs", config, Expanded(), CHUNKS, QUESTIONS, root)
    assert log["questions"][0]["ranking"] == [["t:a:art1", 2.0], ["t:a:art2", 2.0, "t:a:art1"]]
    assert json.loads(dumps(log)) == log
    assert rankings(log)["q001"] == [("a:art1",), ("a:art2",)]
    table = retrieval([log], QUESTIONS, TREE)
    # One of the two places in each top 5 was placed by a reference.
    assert "| bm25-t-refs | bm25 (expansion references) | t | 2 | 50 | 50% | unknown |" in table


def test_reranked_runs_are_compared_question_by_question(root):
    base = run(root, "hybrid-paragraph")
    reranked = run(root, "hybrid-paragraph-rerank20")
    reranked["config"]["reranker"] = "some-reranker"
    reranked["questions"][0]["ranking"] = []  # q001's gold node is no longer found
    table = retrieval([base, reranked], QUESTIONS, TREE)
    assert "## Reranking compared" in table
    section = table.split("## Reranking compared")[1].split("\n## ")[0]
    assert (
        "| hybrid-paragraph-rerank20 | 0.50 (0.09–0.91) | 0 / 1 | 1.00 | – | q001 | – | – | – | – |"
        in section
    )
    assert "## Reranking compared" not in retrieval([base], QUESTIONS, TREE)


def test_the_cost_of_reranking_comes_from_the_scoring_logs(root):
    local = {
        "config": {"reranker": "qwen", "instruction_name": "needed"},
        "result": {"questions": 60, "pairs": 1200, "seconds": 150.0},
    }
    hosted = {
        "config": {"reranker": "jev"},
        "paid": {"cost_usd": 0.03},
        "answers": [{"passages": 20, "seconds": 1.0}, {"passages": 20, "seconds": 2.0}],
    }
    table = retrieval([run(root)], QUESTIONS, TREE, [("a.json", local), ("b.json", hosted)])
    assert "| qwen | needed | `a.json` | 60 | 20 | 2.50 | 0.00000 | 0.00 |" in table
    assert "| jev | – | `b.json` | 2 | 20 | 1.50 | 0.01500 | 15.00 |" in table
