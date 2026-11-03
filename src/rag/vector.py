"""Vector search: rank chunks by cosine similarity to the question's vector (design §E).

Two backends implement `Retriever`, and give the same rankings to float32 precision (design §I;
see Precision below):

- `VectorSearch` keeps the vectors in memory, and needs nothing but the embedding cache.
- `PgVectorSearch` searches the `chunk_vectors` table in the local Postgres + pgvector database,
  loaded by `load_index` (`make vector-index`).

Both take every vector, the question's included, from the embedding cache, so neither runs the
model: a question with no cached vector raises `MissingVectorsError`. The question is looked up as
the model was given it, after the query prefix (`query_input`).

**Distance.** Both compute pgvector's cosine distance, 1 − s, where the similarity
s = a·b / sqrt(|a|² |b|²) is clamped to [−1, 1]. A hit's score is s, read back as 1 − distance,
so it is the same number in both backends. Chunks are ranked by distance, ties in chunk order.
`RRF` reads a score of exactly 0 as "matched nothing" and gives it no credit. A chunk at right
angles to the question, similarity 0, means as little; real embeddings of related legal text
land well above 0, and RRF reads only the top 50 of about 400 chunks.

**Exact.** The database compares the question with every chunk; there is no approximate index
(HNSW or IVFFlat), so local and deployed results are the same. The `embedding` column has no
fixed size, which also means no such index can be built on it.

**Precision.** pgvector sums float32 products in an order its compiler picks, and the in-memory
search sums in float64, so on real vectors the two distances agree to about 1e-6 and two chunks
closer than that may swap places. Vectors are stored as float32, so a difference that small says
nothing about meaning. When every sum is exact in float32, as with the small whole numbers the
tests use, the two distances are bit for bit the same, ties included.
"""

import array
import heapq
import math
from collections.abc import Sequence

import psycopg

from rag.chunk import Chunk
from rag.embed import EmbeddingCache, setting_name, vectors
from rag.embed_estimate import query_input
from rag.retrieve import Hit, check_k

TABLE = "chunk_vectors"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    model text NOT NULL,          -- the model setting, as named by embed.setting_name
    strategy text NOT NULL,       -- the chunking strategy
    position integer NOT NULL,    -- the chunk's place in the strategy's document order
    chunk_id text NOT NULL,
    text text NOT NULL,           -- what was embedded: the chunk's prefix and body
    embedding vector NOT NULL,    -- no fixed size, so no approximate index can be built on it
    PRIMARY KEY (model, strategy, position),
    UNIQUE (model, strategy, chunk_id)
)
"""

SEARCH = f"""
SELECT position, chunk_id, embedding <=> %(query)s::vector AS distance
FROM {TABLE}
WHERE model = %(model)s AND strategy = %(strategy)s
ORDER BY distance, position
LIMIT %(k)s
"""


class VectorIndexError(Exception):
    pass


def _norm2(vector: Sequence[float]) -> float:
    n = math.sumprod(vector, vector)
    if not math.isfinite(n):
        raise VectorIndexError("a vector holds a value that is not a finite number")
    if not n > 0:
        raise VectorIndexError("a vector of length 0 has no direction to compare")
    return n


def cosine_distance(a: Sequence[float], b: Sequence[float], a2: float, b2: float) -> float:
    """pgvector's `<=>`, given the squared lengths `a2` and `b2`."""
    similarity = math.sumprod(a, b) / math.sqrt(a2 * b2)
    return 1.0 - min(1.0, max(-1.0, similarity))


def literal(vector: Sequence[float]) -> str:
    """A vector in pgvector's text form. `repr` is exact, so float32 values survive the trip."""
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def _check_chunks(chunks: Sequence[Chunk]) -> str:
    """The chunks' common strategy; refuses an empty list, mixed strategies or repeated IDs."""
    if not chunks:
        raise VectorIndexError("no chunks to search")
    strategies = {c.strategy for c in chunks}
    if len(strategies) != 1:
        raise VectorIndexError(f"chunks of more than one strategy: {sorted(strategies)}")
    if len({c.id for c in chunks}) != len(chunks):
        raise VectorIndexError("two chunks share an ID")
    return strategies.pop()


class VectorSearch:
    """Exact cosine search over vectors held in memory."""

    def __init__(self, chunks: Sequence[Chunk], cache: EmbeddingCache):
        _check_chunks(chunks)
        self.chunks = list(chunks)
        self.cache = cache
        self._vectors = vectors([c.text for c in self.chunks], cache)
        self._norms = [_norm2(v) for v in self._vectors]

    def search(self, query: str, k: int) -> list[Hit]:
        check_k(k)
        q = vectors([query_input(query, self.cache.config)], self.cache)[0]
        q2 = _norm2(q)
        distances = [
            cosine_distance(q, v, q2, n) for v, n in zip(self._vectors, self._norms, strict=True)
        ]
        order = heapq.nsmallest(k, range(len(self.chunks)), key=lambda i: (distances[i], i))
        return [Hit(self.chunks[i], 1.0 - distances[i]) for i in order]


def _parse(text: str) -> tuple[float, ...]:
    """A vector in pgvector's text form, as the float32 values the cache holds.

    pgvector prints the shortest decimal that reads back as the same float32, so each value is
    rounded to float32 again.
    """
    return tuple(array.array("f", (float(v) for v in text[1:-1].split(","))))


def _table_exists(conn: psycopg.Connection) -> bool:
    """Whether the table exists in the first schema of the search path, where it is created."""
    return (
        conn.execute(
            "SELECT to_regclass(format('%%I.%%I', current_schema(), %s::text))", (TABLE,)
        ).fetchone()[0]
        is not None
    )


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(SCHEMA)


def load_index(conn: psycopg.Connection, chunks: Sequence[Chunk], cache: EmbeddingCache) -> int:
    """Replace the rows of these chunks' strategy and the cache's model setting, in one step.

    Every chunk needs a cached vector, or nothing changes. Loading the same chunks and vectors
    again gives the same rows. Returns the number of rows written.
    """
    strategy = _check_chunks(chunks)
    found = vectors([c.text for c in chunks], cache)
    for v in found:
        _norm2(v)
    model = setting_name(cache.config)
    with conn.transaction():
        ensure_schema(conn)
        conn.execute(f"DELETE FROM {TABLE} WHERE model = %s AND strategy = %s", (model, strategy))
        with conn.cursor() as cur:
            columns = "model, strategy, position, chunk_id, text, embedding"
            with cur.copy(f"COPY {TABLE} ({columns}) FROM STDIN") as copy:
                for position, (chunk, vector) in enumerate(zip(chunks, found, strict=True)):
                    copy.write_row(
                        (model, strategy, position, chunk.id, chunk.text, literal(vector))
                    )
    return len(chunks)


class PgVectorSearch:
    """Exact cosine search in Postgres with pgvector, over rows written by `load_index`.

    Refuses to start unless the table holds exactly these chunks, in this order, with this text
    and the cache's vectors, for the cache's model setting: a stale index would otherwise rank
    chunks or vectors that have changed. Each search checks the IDs again, in case the table was
    reloaded since.
    """

    def __init__(self, conn: psycopg.Connection, chunks: Sequence[Chunk], cache: EmbeddingCache):
        self.strategy = _check_chunks(chunks)
        self.conn = conn
        self.chunks = list(chunks)
        self.cache = cache
        self.model = setting_name(cache.config)
        if not _table_exists(conn):
            raise VectorIndexError(f"there is no {TABLE} table yet; run `make vector-index`")
        rows = conn.execute(
            f"SELECT position, chunk_id, text, embedding::text FROM {TABLE} "
            "WHERE model = %s AND strategy = %s ORDER BY position",
            (self.model, self.strategy),
        ).fetchall()
        expected = [(i, c.id, c.text) for i, c in enumerate(self.chunks)]
        if [row[:3] for row in rows] != expected:
            raise VectorIndexError(
                f"the {self.strategy} chunks in {TABLE} for {self.model} are missing or out of "
                f"date ({len(rows)} rows, {len(expected)} chunks); run `make vector-index`"
            )
        cached = vectors([c.text for c in self.chunks], cache)
        if [_parse(row[3]) for row in rows] != cached:
            raise VectorIndexError(
                f"the {self.strategy} vectors in {TABLE} for {self.model} differ from the "
                "embedding cache; run `make vector-index`"
            )

    def search(self, query: str, k: int) -> list[Hit]:
        check_k(k)
        q = vectors([query_input(query, self.cache.config)], self.cache)[0]
        _norm2(q)
        rows = self.conn.execute(
            SEARCH, {"query": literal(q), "model": self.model, "strategy": self.strategy, "k": k}
        ).fetchall()
        hits = []
        for position, chunk_id, distance in rows:
            if position >= len(self.chunks) or self.chunks[position].id != chunk_id:
                raise VectorIndexError(
                    f"{TABLE} was reloaded with other {self.strategy} chunks; search again"
                )
            hits.append(Hit(self.chunks[position], 1.0 - distance))
        return hits
