import pytest

from rag.chunk import STRATEGIES, Chunk
from rag.retrieve import BM25, Retriever, tokens


def chunk(chunk_id: str, body: str) -> Chunk:
    return Chunk(chunk_id, "test", "a", f"Act › {chunk_id}", body, (chunk_id,))


CHUNKS = [
    chunk("art6(1)", "The daily driving time shall not exceed nine hours."),
    chunk(
        "art6(3)",
        "The total accumulated driving time during any two consecutive weeks "
        "shall not exceed 90 hours.",
    ),
    chunk(
        "art7",
        "After a driving period of 4.5 hours a driver shall take an uninterrupted "
        "break of not less than 45 minutes.",
    ),
    chunk("art4(e)", "'Regular weekly rest period' means any period of rest of at least 45 hours."),
    chunk("anx", "A vehicle with a maximum permissible mass exceeding 3,5 tonnes."),
]


def test_the_tokeniser_keeps_references_and_decimals():
    assert tokens("Regulation (EC) No 561/2006") == [
        "regul",
        "(ec)",
        "no",
        "561/2006",
        "561",
        "2006",
    ]
    assert tokens("see Article 6(3)") == ["see", "articl", "6(3)", "6", "(3)"]
    assert "13(1)(r)" in tokens("Article 13(1)(r) applies")
    assert "4.5" in tokens("after 4.5 hours")
    assert tokens("3,5 tonnes") == tokens("3.5 tonnes") == ["3.5", "tonn"]


def test_the_tokeniser_tells_similar_references_apart():
    assert tokens("Article 8a") == ["articl", "8a"]
    assert tokens("Annex I and Annex II") == ["annex", "annex-i", "annex", "annex-ii"]
    assert tokens("points (a) and (b)") == ["point", "(a)", "(b)"]
    assert tokens("consolidated on 31.12.2024") == ["consolid", "31.12.2024"]
    assert tokens("signed 15/03/2006") == ["sign", "15/03/2006"]
    assert tokens("the driver’s card") == ["driver", "card"]


def test_the_tokeniser_stems_and_drops_stop_words():
    assert tokens("How long may I drive?") == tokens("long driving")
    assert tokens("the and of") == []


def test_an_exact_keyword_query_ranks_the_right_chunk_first():
    index = BM25(CHUNKS)
    assert index.search("two consecutive weeks", 1)[0].chunk.id == "art6(3)"
    assert index.search("uninterrupted break", 1)[0].chunk.id == "art7"
    assert index.search("4,5 hours driving", 1)[0].chunk.id == "art7"
    assert index.search("maximum permissible mass 3.5 tonnes", 1)[0].chunk.id == "anx"


def test_ties_are_broken_by_chunk_order_and_runs_are_identical():
    index = BM25(CHUNKS)
    hits = index.search("nothing matches this", 5)
    assert [h.chunk.id for h in hits] == [c.id for c in CHUNKS]
    assert all(h.score == 0 for h in hits)
    assert index.search("driving hours", 5) == BM25(CHUNKS).search("driving hours", 5)
    scores = [h.score for h in index.search("driving hours", 5)]
    assert scores == sorted(scores, reverse=True)


def test_k_larger_than_the_chunks_returns_every_chunk():
    assert len(BM25(CHUNKS).search("driving", 50)) == len(CHUNKS)


def test_k_must_be_at_least_1():
    with pytest.raises(ValueError, match="k must be at least 1"):
        BM25(CHUNKS).search("driving", 0)


def test_chunks_without_tokens_score_0():
    empty = Chunk("x", "test", "a", "", "the and of", ("x",))
    assert BM25([empty]).search("driving", 1)[0].score == 0


def test_bm25_is_a_retriever():
    retriever: Retriever = BM25(CHUNKS)
    assert retriever.search("rest", 2)


def test_an_empty_index_is_an_error():
    with pytest.raises(ValueError):
        BM25([])


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_a_verbatim_question_finds_its_article_in_the_corpus(tree, strategy):
    index = BM25(STRATEGIES[strategy](tree))
    hits = index.search("total accumulated driving time during any two consecutive weeks", 3)
    assert any("561/2006:art6(3)" in h.chunk.node_ids for h in hits)
