import pytest

from rag.chunk import STRATEGIES, Chunk
from rag.retrieve import BM25, RRF, Hit, Retriever, tokens


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


# --- Hybrid: reciprocal rank fusion ----------------------------------------------------------

BY_ID = {c.id: c for c in CHUNKS}


class Fixed:
    """A retriever that returns a fixed ranking, whatever the query; scores count down to 1."""

    def __init__(self, *ids: str, scores: tuple[float, ...] = ()):
        self.ids = ids
        self.scores = scores or tuple(float(len(ids) - i) for i in range(len(ids)))
        self.asked: list[int] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.asked.append(k)
        return [Hit(BY_ID[i], s) for i, s in zip(self.ids[:k], self.scores, strict=False)]


def ids(hits: list[Hit]) -> list[str]:
    return [h.chunk.id for h in hits]


def test_a_chunk_ranked_well_by_both_beats_one_ranked_first_by_only_one():
    fused = RRF(CHUNKS, [Fixed("art7", "art6(3)"), Fixed("anx", "art6(3)")])
    hits = fused.search("q", 3)
    assert hits[0].chunk.id == "art6(3)"
    assert hits[0].score == pytest.approx(2 / 62)
    assert set(ids(hits[1:])) == {"art7", "anx"}  # each found by only one, still there


def test_fused_ties_are_broken_by_chunk_order():
    # art6(1) and art7 are each first in one ranking: equal scores, chunk order decides.
    fused = RRF(CHUNKS, [Fixed("art7"), Fixed("art6(1)")])
    hits = fused.search("q", 2)
    assert ids(hits) == ["art6(1)", "art7"]
    assert hits[0].score == hits[1].score
    assert fused.search("q", 2) == RRF(CHUNKS, [Fixed("art6(1)"), Fixed("art7")]).search("q", 2)


def test_each_ranking_is_read_to_the_depth_or_to_k_if_deeper():
    a, b = Fixed(*BY_ID), Fixed(*reversed(BY_ID))
    RRF(CHUNKS, [a, b], depth=2).search("q", 1)
    RRF(CHUNKS, [a, b], depth=2).search("q", 4)
    assert a.asked == b.asked == [2, 4]
    # Beyond the depth a chunk gets nothing from that ranking: anx is last in a, first in b.
    # art6(1) is first in a and anx first in b; each is fifth in the other, beyond the depth,
    # so both score 1/61 and chunk order puts art6(1) first.
    hits = RRF(CHUNKS, [a, b], depth=2).search("q", 2)
    assert ids(hits) == ["art6(1)", "anx"]
    assert [h.score for h in hits] == [1 / 61, 1 / 61]


def test_chunks_a_retriever_did_not_match_get_no_credit():
    # BM25 pads its top k with score-0 chunks in chunk order; they must not count as ranked.
    keyword = Fixed("art7", "art6(1)", "art6(3)", scores=(3.0, 0.0, 0.0))
    meaning = Fixed("anx", "art6(1)")
    hits = RRF(CHUNKS, [keyword, meaning]).search("q", 3)
    assert ids(hits) == ["art7", "anx", "art6(1)"]
    assert [h.score for h in hits] == [1 / 61, 1 / 61, 1 / 62]


def test_a_chunk_returned_twice_by_one_retriever_counts_once():
    hits = RRF(CHUNKS, [Fixed("art7", "art7"), Fixed("anx")]).search("q", 2)
    assert ids(hits) == ["art7", "anx"]
    assert hits[0].score == hits[1].score == 1 / 61


def test_fewer_matches_than_k_are_followed_by_the_rest_in_chunk_order():
    keyword = Fixed("art7", "art6(1)", "art6(3)", scores=(3.0, 0.0, 0.0))
    hits = RRF(CHUNKS, [keyword]).search("q", 3)
    assert ids(hits) == ["art7", "art6(1)", "art6(3)"]
    assert [h.score for h in hits] == [1 / 61, 0.0, 0.0]


def test_fusing_a_ranking_with_itself_keeps_its_order():
    bm25 = BM25(CHUNKS)
    fused = RRF(CHUNKS, [bm25, bm25])
    for query in ["driving hours", "weekly rest", "3,5 tonnes"]:
        assert ids(fused.search(query, 5)) == ids(bm25.search(query, 5))


def test_rrf_is_a_retriever_and_refuses_bad_settings():
    retriever: Retriever = RRF(CHUNKS, [BM25(CHUNKS)])
    assert retriever.search("rest", 2)
    with pytest.raises(ValueError, match="k must be at least 1"):
        retriever.search("rest", 0)
    with pytest.raises(ValueError, match="no retrievers"):
        RRF(CHUNKS, [])
    with pytest.raises(ValueError, match="depth at least 1"):
        RRF(CHUNKS, [BM25(CHUNKS)], depth=0)
    with pytest.raises(ValueError, match="no chunks"):
        RRF([], [BM25(CHUNKS)])
    with pytest.raises(ValueError, match="two chunks share an ID"):
        RRF([*CHUNKS, CHUNKS[0]], [BM25(CHUNKS)])


def test_a_chunk_from_another_chunking_is_an_error():
    stranger = Chunk("other", "test", "a", "", "text", ("other",))

    class Stranger:
        def search(self, query: str, k: int) -> list[Hit]:
            return [Hit(stranger, 1.0)]

    with pytest.raises(ValueError, match="not one of the chunks"):
        RRF(CHUNKS, [Stranger()]).search("q", 1)
