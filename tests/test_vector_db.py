"""Vector search in Postgres with pgvector, against the in-memory search (`make test-db`).

Each test works in a schema of its own, dropped afterwards, so the real index is never touched.
"""

import dataclasses
import random
import uuid
from pathlib import Path

import psycopg
import pytest
from test_vector import CHUNKS, CONFIG, QUERIES, VECTORS, chunk, ids, make_cache

from rag.db import connect
from rag.vector import (
    SEARCH,
    TABLE,
    PgVectorSearch,
    VectorIndexError,
    VectorSearch,
    _parse,
    literal,
    load_index,
)

pytestmark = pytest.mark.db

ROOT = Path(__file__).parents[1]


@pytest.fixture
def conn():
    with connect(ROOT / ".env") as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")  # in public, outside the schema
        schema = f"test_{uuid.uuid4().hex[:12]}"
        conn.execute(f"CREATE SCHEMA {schema}")
        conn.execute(f"SET search_path TO {schema}, public")
        try:
            yield conn
        finally:
            conn.execute(f"DROP SCHEMA {schema} CASCADE")


def same_hits(a, b):
    return [(h.chunk.id, h.score) for h in a] == [(h.chunk.id, h.score) for h in b]


def test_exact_vectors_give_the_same_hits_as_in_memory(conn, tmp_path):
    cache = make_cache(tmp_path)
    assert load_index(conn, CHUNKS, cache) == len(CHUNKS)
    db, memory = PgVectorSearch(conn, CHUNKS, cache), VectorSearch(CHUNKS, cache)
    for query in QUERIES:
        for k in (1, 2, 3, 4, 6, 50):
            assert same_hits(db.search(query, k), memory.search(query, k)), (query, k)
    # Ties in chunk order: a and d, then c and e, are equally close to "along a".
    assert ids(db.search("along a", 4)) == ["a", "d", "c", "e"]


def test_real_sized_vectors_give_the_same_ranking_to_float32_precision(conn, tmp_path):
    # 300 chunks of 1024 dimensions, normalised, as Titan returns them; the cache rounds them to
    # float32. pgvector sums in float32 and the in-memory search in float64 (see rag.vector).
    rng = random.Random(47)
    config = dataclasses.replace(CONFIG, dimensions=1024)

    def unit():
        v = [rng.gauss(0, 1) for _ in range(1024)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]

    chunks = [chunk(f"c{i}", f"text {i}") for i in range(300)]
    queries = {f"question {i}": unit() for i in range(20)}
    vectors = {c.id: unit() for c in chunks}
    cache = make_cache(tmp_path, vectors, queries, chunks, config)
    load_index(conn, chunks, cache)
    db, memory = PgVectorSearch(conn, chunks, cache), VectorSearch(chunks, cache)
    for query in queries:
        expected = {h.chunk.id: h.score for h in memory.search(query, len(chunks))}
        got = db.search(query, len(chunks))
        for h in got:
            assert h.score == pytest.approx(expected[h.chunk.id], abs=1e-6)
        mem_order = list(expected)
        for position, h in enumerate(got):
            if h.chunk.id != mem_order[position]:  # a swap is allowed only within 1e-6
                assert abs(expected[h.chunk.id] - expected[mem_order[position]]) < 1e-6


def rows(conn):
    return conn.execute(
        f"SELECT model, strategy, position, chunk_id, text, embedding::text FROM {TABLE} "
        "ORDER BY model, strategy, position"
    ).fetchall()


def test_loading_again_gives_the_same_rows_and_leaves_other_strategies(conn, tmp_path):
    cache = make_cache(tmp_path)
    other = [dataclasses.replace(c, strategy="other") for c in CHUNKS[:2]]
    load_index(conn, CHUNKS, cache)
    load_index(conn, other, cache)  # same texts, so the same cached vectors
    first = rows(conn)
    assert len(first) == len(CHUNKS) + 2
    load_index(conn, CHUNKS, cache)
    assert rows(conn) == first
    # The stored vectors are exactly the cached ones.
    for *_, text, embedding in first:
        assert _parse(embedding) == cache.get(text).vector


def test_a_failed_load_changes_nothing(conn, tmp_path):
    cache = make_cache(tmp_path)
    load_index(conn, CHUNKS, cache)
    before = rows(conn)
    changed = [*CHUNKS[:-1], chunk("f", "text never embedded")]
    with pytest.raises(Exception, match="no cached vector"):
        load_index(conn, changed, cache)
    assert rows(conn) == before


def test_a_load_that_fails_partway_changes_nothing(conn, tmp_path, monkeypatch):
    cache = make_cache(tmp_path)
    load_index(conn, CHUNKS, cache)
    before = rows(conn)
    real = literal
    sent = []

    def bad_after_three(vector):
        sent.append(vector)
        return "[not a vector]" if len(sent) > 3 else real(vector)

    monkeypatch.setattr("rag.vector.literal", bad_after_three)
    with pytest.raises(psycopg.Error):
        load_index(conn, CHUNKS, cache)  # the DELETE has run, and COPY fails on the fourth row
    assert rows(conn) == before


def test_the_table_is_looked_up_in_the_first_schema_only(conn, tmp_path):
    # Another schema later in the search path, as public is for the real index, has a table.
    other = f"test_{uuid.uuid4().hex[:12]}"
    current = conn.execute("SELECT current_schema()").fetchone()[0]
    conn.execute(f"CREATE SCHEMA {other}")
    try:
        conn.execute(f"SET search_path TO {other}, public")
        load_index(conn, CHUNKS, make_cache(tmp_path))
        conn.execute(f"SET search_path TO {current}, {other}, public")
        with pytest.raises(VectorIndexError, match="no chunk_vectors table"):
            PgVectorSearch(conn, CHUNKS, make_cache(tmp_path))
    finally:
        conn.execute(f"DROP SCHEMA {other} CASCADE")


def test_a_reload_after_the_search_started_is_caught(conn, tmp_path):
    cache = make_cache(tmp_path)
    load_index(conn, CHUNKS, cache)
    search = PgVectorSearch(conn, CHUNKS, cache)
    load_index(conn, CHUNKS[::-1], cache)  # the same texts in another order
    with pytest.raises(VectorIndexError, match="reloaded"):
        search.search("along a", 3)


def test_a_missing_or_stale_index_is_refused(conn, tmp_path):
    cache = make_cache(tmp_path)
    with pytest.raises(VectorIndexError, match="no chunk_vectors table"):
        PgVectorSearch(conn, CHUNKS, cache)
    load_index(conn, CHUNKS[:-1], cache)
    with pytest.raises(VectorIndexError, match="5 rows, 6 chunks"):
        PgVectorSearch(conn, CHUNKS, cache)
    load_index(conn, CHUNKS, cache)
    edited = [*CHUNKS[:-1], dataclasses.replace(CHUNKS[-1], body="new text")]
    with pytest.raises(VectorIndexError, match="out of date"):
        PgVectorSearch(conn, edited, cache)
    # The same texts, but a vector changed in the cache since loading.
    changed = make_cache(tmp_path, vectors={**VECTORS, "b": (0.0, 2.0, 1.0, 0.0)})
    with pytest.raises(VectorIndexError, match="differ from the embedding cache"):
        PgVectorSearch(conn, CHUNKS, changed)
    with pytest.raises(VectorIndexError, match="out of date"):
        PgVectorSearch(
            conn, CHUNKS, make_cache(tmp_path, config=dataclasses.replace(CONFIG, normalize=False))
        )


def plan_nodes(plan):
    yield plan
    for child in plan.get("Plans", []):
        yield from plan_nodes(child)


def test_search_is_exact_with_no_approximate_index(conn, tmp_path):
    cache = make_cache(tmp_path)
    load_index(conn, CHUNKS, cache)
    params = {"query": literal(QUERIES["along a"]), "model": "m", "strategy": "test", "k": 3}
    # Even with sequential scans discouraged, no index can order by distance.
    conn.execute("SET enable_seqscan = off")
    plan = conn.execute("EXPLAIN (FORMAT JSON) " + SEARCH, params).fetchone()[0][0]["Plan"]
    nodes = list(plan_nodes(plan))
    indexes = {n["Index Name"] for n in nodes if "Index Name" in n}
    methods = {
        conn.execute(
            "SELECT am.amname FROM pg_class c JOIN pg_am am ON c.relam = am.oid "
            "WHERE c.oid = to_regclass(%s)",
            (name,),
        ).fetchone()[0]
        for name in indexes
    }
    assert methods <= {"btree"}, methods  # at most the primary key, to pick the strategy
    assert any(n["Node Type"] in ("Sort", "Incremental Sort") for n in nodes)


def test_no_approximate_index_can_be_built(conn, tmp_path):
    load_index(conn, CHUNKS, make_cache(tmp_path))
    for method in ("hnsw", "ivfflat"):
        with pytest.raises(psycopg.Error, match="dimensions"):
            conn.execute(f"CREATE INDEX ON {TABLE} USING {method} (embedding vector_cosine_ops)")
