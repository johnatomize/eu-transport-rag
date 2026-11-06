import pytest

from rag.chunk import STRATEGIES, Chunk
from rag.parse import Node, Ref, Tree
from rag.retrieve import (
    BM25,
    RRF,
    Expansion,
    Hit,
    Rerank,
    Retriever,
    reference_targets,
    tokens,
)


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


# --- Cross-reference expansion ---------------------------------------------------------------


def ref_tree() -> Tree:
    """Act a: Chapter I with Articles 4 (points (e), (f)), 6 (paragraphs (1), (2)) and 7; act b:
    Article 3 with point (b). The references are set by hand, as `resolve_refs` would."""
    tree = Tree()

    def add(node_id: str, kind: str, parent: str | None, *refs: Ref) -> None:
        doc = node_id.split(":")[0]
        node = Node(node_id, doc, kind, None, None, (), parent)
        node.refs = list(refs)
        tree.add(node)

    add("a", "act", None)
    add("a:ch1", "chapter", "a")
    add("a:art4", "article", "a:ch1")
    add("a:art4(e)", "point", "a:art4")
    add("a:art4(f)", "point", "a:art4")
    add("a:art6", "article", "a:ch1")
    add(
        "a:art6(1)",
        "paragraph",
        "a:art6",
        Ref("point (e) of Article 4", "a", "a:art4(e)"),
        Ref("Article 3 of Directive 2003/59/EC", "Directive 2003/59/EC", None),  # outside
        Ref("Article 99", "a", None),  # unresolved
        Ref("Chapter I", "a", "a:ch1"),  # too broad for one passage
    )
    add(
        "a:art6(2)",
        "paragraph",
        "a:art6",
        Ref("point (b) of Article 3 of b", "b", "b:art3(b)"),
        Ref("paragraph 1", "a", "a:art6(1)"),
    )
    add(
        "a:art7",
        "article",
        "a:ch1",
        Ref("point (e) of Article 4", "a", "a:art4(e)"),
        Ref("Article 4", "a", "a:art4"),
    )
    add("b", "act", None)
    add("b:art3", "article", "b")
    add("b:art3(b)", "point", "b:art3", Ref("Article 7 of a", "a", "a:art7"))
    return tree


REF_TREE = ref_tree()


def chunks_of(strategy: str, *groups: tuple[str, ...]) -> list[Chunk]:
    return [Chunk(f"{strategy}:{g[0]}", strategy, g[0][0], "", " ".join(g), g) for g in groups]


ARTICLES = chunks_of(
    "art",
    ("a:art4", "a:art4(e)", "a:art4(f)"),
    ("a:art6", "a:art6(1)", "a:art6(2)"),
    ("a:art7",),
    ("b:art3", "b:art3(b)"),
)
PARAGRAPHS = chunks_of(
    "par",
    ("a:art4(e)",),
    ("a:art4(f)",),
    ("a:art6(1)",),
    ("a:art6(2)",),
    ("a:art7",),
    ("b:art3(b)",),
)


class Ranked:
    """A retriever over `chunks` that returns a fixed ranking; scores count down to 1."""

    def __init__(self, chunks: list[Chunk], *ids: str, scores: tuple[float, ...] = ()):
        self.by_id = {c.id: c for c in chunks}
        self.ids = ids
        self.scores = scores or tuple(float(len(ids) - i) for i in range(len(ids)))

    def search(self, query: str, k: int) -> list[Hit]:
        return [Hit(self.by_id[i], s) for i, s in zip(self.ids[:k], self.scores, strict=False)]


def test_a_chunk_refers_to_the_chunks_that_hold_what_it_cites():
    targets = reference_targets(ARTICLES, REF_TREE)
    # Article 6 cites point (e) of Article 4 and point (b) of b's Article 3: their article
    # chunks. The act outside the corpus, the unresolved reference, the chapter and paragraph 1,
    # which the chunk holds itself, bring in nothing.
    assert targets["art:a:art6"] == ("art:a:art4", "art:b:art3")
    # Point (e) and the whole of Article 4 are in the same chunk: it is added once.
    assert targets["art:a:art7"] == ("art:a:art4",)
    assert targets["art:b:art3"] == ("art:a:art7",)  # across acts, too
    assert targets["art:a:art4"] == ()


def test_with_paragraph_chunks_a_reference_points_to_the_paragraph():
    targets = reference_targets(PARAGRAPHS, REF_TREE)
    assert targets["par:a:art6(1)"] == ("par:a:art4(e)",)
    assert targets["par:a:art6(2)"] == ("par:b:art3(b)", "par:a:art6(1)")
    # "Article 4" is held by both its paragraph chunks, one node each; the earlier wins, and
    # it is the one point (e) already brought in.
    assert targets["par:a:art7"] == ("par:a:art4(e)",)


def test_references_in_an_articles_lead_in_count_for_the_chunks_that_carry_it():
    tree = ref_tree()
    article = tree["a:art6"]
    article.segments = [(0, "Subject to Article 7:")]
    article.refs = [Ref("Article 7", "a", "a:art7")]
    tree["a:art6(1)"].position, tree["a:art6(2)"].position = 1, 2
    targets = reference_targets(PARAGRAPHS, tree)
    assert targets["par:a:art6(1)"] == ("par:a:art4(e)", "par:a:art7")
    assert targets["par:a:art6(2)"] == ("par:b:art3(b)", "par:a:art6(1)", "par:a:art7")
    assert reference_targets(PARAGRAPHS, REF_TREE)["par:a:art6(1)"] == ("par:a:art4(e)",)


def test_a_reference_points_to_the_chunk_that_holds_most_of_it():
    groups = [
        ("a:art6(2)", "a:art7"),
        ("a:art7", "a:art4(e)"),
        ("a:art4(e)", "a:art4(f)", "a:art4"),
    ]
    windows = [Chunk(f"w{i}", "win", "a", "", "", g) for i, g in enumerate(groups, 1)]
    targets = reference_targets(windows, REF_TREE)
    # Article 7, in w1, cites point (e) of Article 4, held by w2 and w3 equally: the earlier
    # wins. It cites the whole of Article 4 too: w3 holds three of its nodes, w2 one.
    assert targets["w1"] == ("w2", "w3")
    # w2 holds part of Article 4, so what Article 7 cites there brings in nothing.
    assert targets["w2"] == ()


def test_expansion_puts_referenced_chunks_right_after_the_chunk_that_cites_them():
    base = Ranked(ARTICLES, "art:a:art7", "art:a:art6", "art:b:art3", "art:a:art4")
    hits = Expansion(base, ARTICLES, REF_TREE).search("q", 3)
    assert ids(hits) == ["art:a:art7", "art:a:art4", "art:a:art6"]  # inside k: b:art3 drops
    assert [h.via for h in hits] == [None, "art:a:art7", None]
    assert [h.score for h in hits] == [4.0, 4.0, 3.0]  # an added chunk takes its citer's score


def test_nothing_is_added_twice_and_a_longer_list_starts_with_the_shorter():
    base = Ranked(ARTICLES, "art:a:art7", "art:a:art6", "art:b:art3", "art:a:art4")
    expansion = Expansion(base, ARTICLES, REF_TREE)
    hits = expansion.search("q", 50)
    # Article 4 is cited by 7 and 6 and retrieved itself, and b:art3 is cited and retrieved,
    # yet each is in the list once.
    assert ids(hits) == ["art:a:art7", "art:a:art4", "art:a:art6", "art:b:art3"]
    assert hits[3].via == "art:a:art6"
    for k in range(1, 5):
        assert expansion.search("q", k) == hits[:k]


def test_the_references_of_a_chunk_that_matched_nothing_are_not_followed():
    base = Ranked(ARTICLES, "art:a:art6", "art:a:art7", scores=(2.0, 0.0))
    hits = Expansion(base, ARTICLES, REF_TREE).search("q", 4)
    assert ids(hits) == ["art:a:art6", "art:a:art4", "art:b:art3", "art:a:art7"]
    base = Ranked(ARTICLES, "art:a:art7", "art:a:art6", scores=(0.0, 0.0))
    assert ids(Expansion(base, ARTICLES, REF_TREE).search("q", 2)) == ["art:a:art7", "art:a:art6"]


def test_expansion_wraps_any_retriever_and_refuses_strangers():
    for inner in (BM25(ARTICLES), RRF(ARTICLES, [BM25(ARTICLES)])):
        retriever: Retriever = Expansion(inner, ARTICLES, REF_TREE)
        assert len(retriever.search("a:art7", 3)) == 3
    with pytest.raises(ValueError, match="k must be at least 1"):
        Expansion(BM25(ARTICLES), ARTICLES, REF_TREE).search("q", 0)
    with pytest.raises(ValueError, match="two chunks share an ID"):
        Expansion(BM25(ARTICLES), [*ARTICLES, ARTICLES[0]], REF_TREE)
    with pytest.raises(ValueError, match="not one of the chunks"):
        Expansion(BM25(PARAGRAPHS), ARTICLES, REF_TREE).search("a:art7", 1)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_in_the_corpus_references_point_to_other_chunks(tree, strategy):
    chunks = STRATEGIES[strategy](tree)
    targets = reference_targets(chunks, tree)
    known = {c.id for c in chunks}
    assert all(t in known and t != c for c, ts in targets.items() for t in ts)
    assert sum(map(len, targets.values())) > 100


def test_article_6_5_brings_in_the_working_time_directive(tree):
    # 561/2006 Art. 6(5) says to record "periods of availability, as defined in Article 3(b)
    # of Directive 2002/15/EC", a cross-act reference, and cites point (e) of its own Art. 4.
    targets = reference_targets(STRATEGIES["paragraph"](tree), tree)
    assert "paragraph:2002/15:art3(b)" in targets["paragraph:561/2006:art6(5)"]
    assert "paragraph:561/2006:art4(e)" in targets["paragraph:561/2006:art6(5)"]


# --- Reranking -----------------------------------------------------------------------------


class SameForAll:
    """A retriever that returns the same ranking for every query, scores falling with rank."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.asked: list[int] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.asked.append(k)
        return [Hit(c, 1.0 - i / 100) for i, c in enumerate(self.chunks[:k])]


class ByWord:
    """A stand-in reranker: a text's score is how often it holds the query's first word."""

    def __init__(self):
        self.read: list[list[str]] = []

    def scores(self, query: str, texts):
        self.read.append(list(texts))
        word = query.split()[0].lower()
        return [float(t.lower().count(word)) for t in texts]


def test_reranking_reorders_only_the_top_n():
    inner = SameForAll(CHUNKS)
    reranked = Rerank(inner, ByWord(), depth=3).search("rest", 5)
    # None of the top 3 says "rest"; the fourth does, but it is beyond N, so the order stays.
    assert [h.chunk.id for h in reranked] == [c.id for c in CHUNKS]
    reranked = Rerank(inner, ByWord(), depth=3).search("break", 5)
    assert [h.chunk.id for h in reranked] == ["art7", "art6(1)", "art6(3)", "art4(e)", "anx"]


def test_hits_beyond_n_keep_their_order_and_scores():
    inner = SameForAll(CHUNKS)
    reranked = Rerank(inner, ByWord(), depth=2).search("driving", 5)
    assert [h.score for h in reranked[2:]] == [h.score for h in inner.search("driving", 5)[2:]]


def test_equal_scores_keep_the_wrapped_order():
    inner = SameForAll(CHUNKS)
    reranked = Rerank(inner, ByWord(), depth=5).search("nothing", 5)
    assert [h.chunk.id for h in reranked] == [c.id for c in CHUNKS]
    assert all(h.score == 0.0 for h in reranked)


def test_the_same_input_gives_the_same_order():
    reranker = Rerank(SameForAll(CHUNKS), ByWord(), depth=5)
    assert reranker.search("driving", 5) == reranker.search("driving", 5)


def test_the_reranker_reads_the_top_n_even_for_a_smaller_k():
    inner, scorer = SameForAll(CHUNKS), ByWord()
    hits = Rerank(inner, scorer, depth=4).search("rest", 1)
    assert inner.asked == [4]
    assert scorer.read == [[c.text for c in CHUNKS[:4]]]
    assert [h.chunk.id for h in hits] == ["art4(e)"]


def test_reranking_keeps_what_brought_a_chunk_in():
    class Via:
        def search(self, query, k):
            return [Hit(CHUNKS[0], 1.0), Hit(CHUNKS[2], 1.0, via="art6(1)")][:k]

    hits = Rerank(Via(), ByWord(), depth=2).search("break", 2)
    assert (hits[0].chunk.id, hits[0].via) == ("art7", "art6(1)")


@pytest.mark.parametrize("scores", [[1.0], [1.0, float("nan")]])
def test_a_reranker_that_gives_the_wrong_scores_is_refused(scores):
    class Broken:
        def scores(self, query, texts):
            return scores

    with pytest.raises(ValueError):
        Rerank(SameForAll(CHUNKS), Broken(), depth=2).search("rest", 2)


def test_a_reranking_depth_below_one_is_refused():
    with pytest.raises(ValueError):
        Rerank(SameForAll(CHUNKS), ByWord(), depth=0)
