"""Vector search in memory; the database backend is tested in test_vector_db.py (`make test-db`)."""

import array
import datetime as dt

import pytest

from rag.chunk import Chunk
from rag.embed import EmbeddingCache, MissingVectorsError, setting_name
from rag.embed_estimate import EmbeddingConfig
from rag.retrieve import BM25, RRF
from rag.vector import VectorIndexError, VectorSearch, cosine_distance, literal

CONFIG = EmbeddingConfig(
    model_id="amazon.titan-embed-text-v2:0",
    region="eu-north-1",
    dimensions=4,
    normalize=True,
    max_input_tokens=100,
    price_usd_per_million_input_tokens=0.021,
    price_read_on=dt.date(2026, 10, 31),
)


def chunk(chunk_id: str, body: str, strategy: str = "test") -> Chunk:
    return Chunk(chunk_id, strategy, "a", f"Act › {chunk_id}", body, (chunk_id,))


# Small whole numbers, so every sum is exact and both backends give the same distances bit for
# bit. "c" and "e" share a vector, and "d" points the same way as "a", twice as long.
VECTORS = {
    "a": (1.0, 0.0, 0.0, 0.0),
    "b": (0.0, 1.0, 0.0, 0.0),
    "c": (1.0, 1.0, 0.0, 0.0),
    "d": (2.0, 0.0, 0.0, 0.0),
    "e": (1.0, 1.0, 0.0, 0.0),
    "f": (-1.0, 0.0, 1.0, 0.0),
}
QUERIES = {
    "along a": (3.0, 0.0, 0.0, 0.0),
    "between a and b": (1.0, 1.0, 0.0, 0.0),
    "away from a": (-2.0, 0.0, 0.0, 0.0),
    "mostly b": (1.0, 2.0, 0.0, 1.0),
}
CHUNKS = [chunk(name, f"text {name}") for name in VECTORS]


def make_cache(tmp_path, vectors=VECTORS, queries=QUERIES, chunks=CHUNKS, config=CONFIG):
    cache = EmbeddingCache(tmp_path / "cache.jsonl", config)
    by_id = {c.id: c for c in chunks}
    for name, vector in vectors.items():
        cache.put(by_id[name].text, vector, 1)
    for query, vector in queries.items():
        cache.put(query, vector, 1)
    return cache


def ids(hits):
    return [h.chunk.id for h in hits]


def test_ranks_by_cosine_similarity_with_ties_in_chunk_order(tmp_path):
    search = VectorSearch(CHUNKS, make_cache(tmp_path))
    hits = search.search("along a", 6)
    # a and d point the same way as the question: similarity 1, in chunk order.
    assert ids(hits) == ["a", "d", "c", "e", "b", "f"]
    assert [h.score for h in hits[:2]] == [1.0, 1.0]
    assert hits[2].score == pytest.approx(2**-0.5)
    assert hits[2].score == hits[3].score  # c and e share a vector
    assert hits[4].score == 0.0
    assert hits[5].score == pytest.approx(-(2**-0.5))
    assert ids(search.search("between a and b", 2)) == ["c", "e"]
    assert ids(search.search("away from a", 1)) == ["f"]


def test_returns_at_most_k_and_at_most_every_chunk(tmp_path):
    search = VectorSearch(CHUNKS, make_cache(tmp_path))
    assert len(search.search("mostly b", 3)) == 3
    assert len(search.search("mostly b", 50)) == len(CHUNKS)
    with pytest.raises(ValueError, match="at least 1"):
        search.search("mostly b", 0)


def test_the_distance_is_pgvectors():
    assert cosine_distance((1.0, 0.0), (2.0, 0.0), 1.0, 4.0) == 0.0
    assert cosine_distance((1.0, 0.0), (-1.0, 0.0), 1.0, 1.0) == 2.0
    assert cosine_distance((1.0, 0.0), (0.0, 1.0), 1.0, 1.0) == 1.0
    # Rounding can push the similarity just past 1; pgvector clamps it, and so do we.
    a = (0.1, 0.2, 0.3)
    a2 = 0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3
    assert cosine_distance(a, a, a2 * (1 - 1e-15), a2) == 0.0


def test_a_question_without_a_cached_vector_is_refused(tmp_path):
    search = VectorSearch(CHUNKS, make_cache(tmp_path))
    with pytest.raises(MissingVectorsError, match="make embed"):
        search.search("How long may I drive in a day?", 5)


def test_chunks_without_cached_vectors_are_refused(tmp_path):
    vectors = {k: v for k, v in VECTORS.items() if k != "b"}
    with pytest.raises(MissingVectorsError, match="1 of 6"):
        VectorSearch(CHUNKS, make_cache(tmp_path, vectors=vectors))


def test_vectors_of_length_zero_are_refused(tmp_path):
    with pytest.raises(VectorIndexError, match="length 0"):
        VectorSearch(CHUNKS, make_cache(tmp_path, vectors={**VECTORS, "b": (0.0,) * 4}))
    search = VectorSearch(CHUNKS, make_cache(tmp_path, queries={"nothing": (0.0,) * 4}))
    with pytest.raises(VectorIndexError, match="length 0"):
        search.search("nothing", 3)


def test_chunks_must_be_one_strategy_with_distinct_ids(tmp_path):
    cache = make_cache(tmp_path)
    with pytest.raises(VectorIndexError, match="no chunks"):
        VectorSearch([], cache)
    with pytest.raises(VectorIndexError, match="more than one strategy"):
        VectorSearch([*CHUNKS, chunk("g", "text a", strategy="other")], cache)
    with pytest.raises(VectorIndexError, match="share an ID"):
        VectorSearch([*CHUNKS, CHUNKS[0]], cache)


def test_works_inside_hybrid_search(tmp_path):
    cache = make_cache(tmp_path, queries={"text b": (0.0, 1.0, 0.0, 0.0)})
    hybrid = RRF(CHUNKS, [BM25(CHUNKS), VectorSearch(CHUNKS, cache)])
    assert ids(hybrid.search("text b", 1)) == ["b"]  # first by both


def test_the_literal_keeps_float32_values_exactly():
    value = 0.1  # not a float32; its float32 rounding is what the cache holds
    f32 = array.array("f", [value])[0]
    assert literal([f32, 1.0, -2.5e-08]) == f"[{f32!r},1.0,-2.5e-08]"
    assert float(literal([f32])[1:-1]) == f32


def test_the_setting_name_is_the_cache_files_stem():
    assert setting_name(CONFIG) == "amazon.titan-embed-text-v2_0-4-normalized"
