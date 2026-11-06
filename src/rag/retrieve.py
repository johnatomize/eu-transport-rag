"""Retrieval: return the chunks that best match a question, best first.

Every retriever implements `Retriever`, so that the evaluation, the service and the comparison
of backends don't care which one they get. This module has the keyword baseline, BM25 (design
§D); hybrid search, which merges the rankings of other retrievers by reciprocal rank fusion
(design §E); cross-reference expansion, which wraps any retriever and brings in the chunks that
its hits refer to (design §E, problem 2); and reranking, which wraps any retriever and lets a
model reorder its best hits (design §G). Vector search, also from E, is in `rag.vector`, beside
its database backend, and the reranking model and its score cache are in `rag.rerank`.

BM25 is the Okapi variant with the usual defaults, k1 = 1.5 and b = 0.75, and the idf that
stays positive, log(1 + (N - df + 0.5) / (df + 0.5)), as in Lucene. It is written here rather
than taken from a library: it is about 30 lines, and owning it keeps ties and rounding under our
control, so that two runs give identical rankings.
"""

import itertools
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from snowballstemmer import stemmer

from rag.chunk import Chunk
from rag.parse import Tree


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    # The ID of the chunk whose cross-reference brought this one in, for expansion's hits.
    via: str | None = None


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Hit]:
        """The `k` best chunks for `query`, best first; ties in chunk order.

        A score of exactly 0 means the chunk matched nothing: BM25 fills its top k with such
        chunks when fewer than k hold a query term.
        """
        ...


class Scorer(Protocol):
    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        """How well each text answers `query`, one number each, in order; higher is better."""
        ...


def check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, not {k}")


# --- Tokens --------------------------------------------------------------------------------

# Legal text hinges on references and numbers, so these are tokens of their own, tried in order:
_TOKEN = re.compile(
    r"""
      (?P<number>\d+(?:[./]\d+){2,})                     # 31.12.2024, 15/03/2006, item 1.5.2
    | (?P<act>\d{1,4}/\d{2,4}(?:/[a-z]{2,3})?)            # 561/2006, 2002/15/ec
    | (?P<ref>\d+[a-z]?(?:\(\d+[a-z]?\)|\([a-z]{1,4}\))+)  # 6(3), 8(6a), 4(e), 13(1)(r)
    | (?P<label>\(\d+[a-z]?\)|\([a-z]{1,4}\))              # (3), (a), (iv), as printed on a point
    | (?P<annex>annex\s+[ivx]+\b)                         # annex i, annex ii
    | (?P<decimal>\d+[.,]\d+)                             # 4.5, the act's 4,5
    | (?P<article>\d+[a-z]\b)                             # 8a, 23a
    | (?P<word>[^\W\d_]+|\d+)                             # words and plain numbers
    """,
    re.VERBOSE,
)
_PART = re.compile(r"\d+[a-z]?|\(\d+[a-z]?\)|\([a-z]{1,4}\)")
# Words too common to tell chunks apart, and the "s" left by an apostrophe. Numbers, references
# and labels are never dropped.
STOP_WORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in into is it its may
    my of on or s shall that the their there these this to was were what when where which who
    will with""".split()
)
_STEM = stemmer("english")


def tokens(text: str) -> list[str]:
    """Lowercased tokens, with English words stemmed and stop words dropped.

    References and numbers stay whole:
    - An article reference is one token, with its article and labels as tokens too, so
      "Article 6(3)" matches "6(3)", a heading "Article 6" and the label "(3)". A point's
      printed label, such as "(a)", is a token of its own.
    - An act number ("561/2006") is one token, with its numbers as tokens too. "Article 8a"
      gives "8a", not "8"; "Annex I" gives "annex" and "annex-i".
    - A decimal comma becomes a point, so "4,5 tonnes" in the act matches "4.5 tonnes" in a
      question. Dates and item numbers with more than two parts ("31.12.2024") are one token.
    """
    out = []
    for match in _TOKEN.finditer(text.lower()):
        token = match.group()
        kind = match.lastgroup
        if kind == "word":
            if token not in STOP_WORDS:
                out.append(token if token.isdigit() else _STEM.stemWord(token))
        elif kind == "decimal":
            out.append(token.replace(",", "."))
        elif kind == "annex":
            out.extend(["annex", "annex-" + token.split()[-1]])
        elif kind == "ref":
            out.append(token)
            out.extend(_PART.findall(token))
        elif kind == "act":
            out.append(token)
            out.extend(re.findall(r"\d+|[a-z]+", token))
        else:
            out.append(token)
    return out


# --- BM25 ----------------------------------------------------------------------------------


class BM25:
    """A BM25 index over a list of chunks, built once and searched any number of times."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("no chunks to index")
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self._counts = [Counter(tokens(chunk.text)) for chunk in self.chunks]
        self._lengths = [sum(counts.values()) for counts in self._counts]
        # At least 1, so that chunks without a single token don't divide by zero.
        self._average = max(sum(self._lengths) / len(self._lengths), 1.0)
        df = Counter(token for counts in self._counts for token in counts)
        n = len(self.chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> list[float]:
        """The score of every chunk for `query`, in chunk order."""
        terms = Counter(t for t in tokens(query) if t in self._idf)
        scores = []
        for counts, length in zip(self._counts, self._lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self._average)
            score = 0.0
            for term, times in terms.items():
                tf = counts.get(term, 0)
                if tf:
                    score += times * self._idf[term] * tf * (self.k1 + 1) / (tf + norm)
            scores.append(score)
        return scores

    def search(self, query: str, k: int) -> list[Hit]:
        check_k(k)
        scores = self.scores(query)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [Hit(self.chunks[i], scores[i]) for i in order[:k]]


# --- Hybrid: reciprocal rank fusion -------------------------------------------------------

RRF_K = 60  # the constant from Cormack, Clarke and Büttcher (2009), the usual default
RRF_DEPTH = 50  # how far down each ranking is read, when k is smaller


class RRF:
    """Merge the rankings of several retrievers over the same chunks by reciprocal rank fusion.

    Each retriever's top `depth` chunks get 1 / (`rrf_k` + rank) from it, ranks counting from 1,
    and a chunk's score is the sum over the rankings that hold it. Only ranks count, not scores,
    so BM25 scores and cosine similarities need no common scale. A chunk ranked well by both
    beats one ranked first by only one: 2 / 62 against 1 / 61. A chunk found by only one still
    takes part. A hit with a score of 0 matched nothing and gets no credit, and a chunk that a
    retriever returns twice counts once. Ties are broken by chunk order, as in BM25; when fewer
    than k chunks get any credit, the rest of the top k follows in chunk order with score 0, as
    BM25 does.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        retrievers: Sequence[Retriever],
        rrf_k: int = RRF_K,
        depth: int = RRF_DEPTH,
    ):
        if not chunks:
            raise ValueError("no chunks to fuse")
        if not retrievers:
            raise ValueError("no retrievers to fuse")
        if rrf_k < 0 or depth < 1:
            raise ValueError(f"rrf_k must be at least 0 and depth at least 1, not {rrf_k}, {depth}")
        self.chunks = list(chunks)
        self.retrievers = list(retrievers)
        self.rrf_k, self.depth = rrf_k, depth
        self._position = {chunk.id: i for i, chunk in enumerate(self.chunks)}
        if len(self._position) != len(self.chunks):
            raise ValueError("two chunks share an ID")

    def search(self, query: str, k: int) -> list[Hit]:
        check_k(k)
        depth = max(self.depth, k)
        parts: dict[str, list[float]] = {}
        for retriever in self.retrievers:
            seen = set()
            for rank, hit in enumerate(retriever.search(query, depth), 1):
                chunk_id = hit.chunk.id
                if chunk_id not in self._position:
                    raise ValueError(f"a retriever returned {chunk_id}, not one of the chunks")
                if hit.score == 0 or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                parts.setdefault(chunk_id, []).append(1 / (self.rrf_k + rank))
        # fsum, so that the same ranks give exactly the same score whatever their order.
        scores = {chunk_id: math.fsum(p) for chunk_id, p in parts.items()}
        order = sorted(scores, key=lambda c: (-scores[c], self._position[c]))[:k]
        hits = [Hit(self.chunks[self._position[c]], scores[c]) for c in order]
        rest = (Hit(chunk, 0.0) for chunk in self.chunks if chunk.id not in scores)
        hits += itertools.islice(rest, k - len(hits))
        return hits


# --- Cross-reference expansion -----------------------------------------------------------


def _too_broad(node_id: str, tree: Tree) -> bool:
    """Whether a reference target is too broad to stand for one passage: a chapter, or an
    annex made of parts (165/2014 Annex I alone is about 4,000 words)."""
    node = tree[node_id]
    if node.kind == "chapter":
        return True
    return node.kind == "annex" and any(tree[c].kind == "annex_part" for c in node.children)


_UNITS = frozenset({"article", "annex_part", "annex", "page"})


def _lead_in_nodes(node_id: str, tree: Tree) -> list[str]:
    """The ancestors, up to the node's unit, whose text before it is its lead-in.

    Chunking puts that text in the prefix of a chunk starting at the node (`rag.parse.lead_in`),
    so the chunk's references include theirs.
    """
    out = []
    current = tree[node_id]
    while current.kind not in _UNITS and current.parent is not None:
        parent = tree[current.parent]
        if any(position < current.position for position, _ in parent.segments):
            out.append(parent.id)
        current = parent
    return out


def reference_targets(chunks: Sequence[Chunk], tree: Tree) -> dict[str, tuple[str, ...]]:
    """For each chunk, the chunks its cross-references point to, in the order its text cites them.

    The references of a chunk are those of every node in its `node_ids` and of the nodes whose
    text before its first node is the lead-in in its prefix. References carry no position, so
    a window, which holds part of a node, may cite something from the part it doesn't hold, and
    an article's references in its closing text count for every chunk that carries its lead-in.

    Each reference becomes the one chunk that holds the most of the cited node, counted in nodes
    of its subtree, the earliest on a tie: the article's chunk, the paragraph's chunk, or the
    window that covers most of it. Skipped: references outside the corpus or unresolved (no
    target), targets too broad for one passage, targets in no chunk (such as amending articles),
    and targets the chunk already holds. A reference resolved to the nearest ancestor of a node
    the parser lacks (`Ref.partial`) is followed to that ancestor.
    """
    # held[node][i]: how many of chunk i's node IDs lie in the node's subtree.
    held: dict[str, Counter[int]] = {}
    for i, chunk in enumerate(chunks):
        for node_id in chunk.node_ids:
            current: str | None = node_id
            while current is not None:
                held.setdefault(current, Counter())[i] += 1
                current = tree[current].parent
    targets: dict[str, tuple[str, ...]] = {}
    for i, chunk in enumerate(chunks):
        found: list[str] = []
        lead = _lead_in_nodes(chunk.node_ids[0], tree) if chunk.node_ids else []
        for node_id in [*chunk.node_ids, *lead]:
            for ref in tree[node_id].refs:
                target = ref.target
                if target is None or target not in held or _too_broad(target, tree):
                    continue
                if held[target][i]:
                    continue  # the chunk holds (part of) it already
                best = min(held[target].items(), key=lambda item: (-item[1], item[0]))[0]
                if chunks[best].id not in found:
                    found.append(chunks[best].id)
        targets[chunk.id] = tuple(found)
    return targets


class Expansion:
    """Wrap a retriever so that the chunks its hits refer to come right after them, inside k.

    The owner's choice (#49, docs/DECISIONS.md): a referenced chunk takes one of the k places,
    so an expanded top 5 is still 5 chunks, and the extra context is only what longer chunks
    add. Going down the wrapped retriever's ranking, each hit is followed by the chunks its
    cross-references point to (`reference_targets`), unless they are in the list already,
    until k chunks are in it. References are followed one step: an added chunk's own references
    are not. A hit with a score of 0 matched nothing, so its references are not followed either.

    An added chunk takes the score of the chunk that cites it, so `Hit.via` names that chunk and,
    over a search's scores, scores never rise down the list. The first k of a longer list are the
    list for k.

    With `follow_zero`, a hit's references are followed whatever its score: over a reranker
    (`Rerank`, #99), whose scores are log-odds or fused ranks, 0 doesn't mean that nothing
    matched, and scores past the reranked hits are on another scale, so they can rise.
    """

    def __init__(
        self, retriever: Retriever, chunks: Sequence[Chunk], tree: Tree, follow_zero: bool = False
    ):
        self.retriever = retriever
        self.follow_zero = follow_zero
        self._by_id = {chunk.id: chunk for chunk in chunks}
        if len(self._by_id) != len(chunks):
            raise ValueError("two chunks share an ID")
        self.references = reference_targets(chunks, tree)

    def search(self, query: str, k: int) -> list[Hit]:
        check_k(k)
        hits: list[Hit] = []
        seen: set[str] = set()
        # The wrapped retriever's top k are enough: each of them ends up in the list, unless k
        # chunks are in it first.
        for hit in self.retriever.search(query, k):
            chunk_id = hit.chunk.id
            if chunk_id not in self._by_id:
                raise ValueError(f"the retriever returned {chunk_id}, not one of the chunks")
            if chunk_id in seen:
                continue
            hits.append(hit)
            seen.add(chunk_id)
            if hit.score != 0 or self.follow_zero:
                for target in self.references[chunk_id]:
                    if target not in seen:
                        hits.append(Hit(self._by_id[target], hit.score, via=chunk_id))
                        seen.add(target)
            if len(hits) >= k:
                break
        return hits[:k]


# --- Reranking -----------------------------------------------------------------------------


class Rerank:
    """Wrap a retriever so that a reranker reorders its top `depth` hits (design §G).

    A reranker reads the question together with each passage and scores the pair, rather than
    comparing two things made separately, as search does; it is too slow to read every chunk,
    so it reads only the wrapped retriever's best `depth`. Those are sorted by the scorer's
    score, best first, and hits beyond `depth` follow in the wrapped retriever's order, so the
    order changes only within the top `depth`. Equal scores keep the wrapped retriever's order,
    so the same scores always give the same list.

    A reordered hit's score is the scorer's (for `rag.rerank`, the model's log-odds that the
    passage answers the question, which can be 0 or below without meaning that it matched
    nothing); a hit beyond `depth` keeps the wrapped retriever's score, on that scale. `Hit.via`
    is kept, so a wrapped `Expansion` still says what brought a chunk in.

    With `blend` (#99), the top `depth` are not sorted by the scorer alone but by reciprocal rank
    fusion of two rankings, as hybrid search fuses its two: the wrapped retriever's and the
    scorer's, each hit getting 1 / (`blend` + rank) from each, ranks counting from 1. A passage
    the search placed high then needs the reranker to rank it low too before it drops. The hit's
    score is the fused score; ties keep the wrapped retriever's order.
    """

    def __init__(self, retriever: Retriever, scorer: Scorer, depth: int, blend: int | None = None):
        if depth < 1:
            raise ValueError(f"the reranking depth must be at least 1, not {depth}")
        if blend is not None and blend < 0:
            raise ValueError(f"the blend constant must be at least 0, not {blend}")
        self.retriever = retriever
        self.scorer = scorer
        self.depth = depth
        self.blend = blend

    def search(self, query: str, k: int) -> list[Hit]:
        check_k(k)
        hits = self.retriever.search(query, max(k, self.depth))
        top, rest = hits[: self.depth], hits[self.depth :]
        scores = self.scorer.scores(query, [hit.chunk.text for hit in top])
        if len(scores) != len(top):
            raise ValueError(f"the reranker gave {len(scores)} scores for {len(top)} passages")
        if not all(math.isfinite(s) for s in scores):
            raise ValueError("the reranker gave a score that is not a finite number")
        order = sorted(range(len(top)), key=lambda i: -scores[i])  # stable: ties keep their order
        if self.blend is not None:
            by_reranker = {i: rank for rank, i in enumerate(order, 1)}
            scores = [
                math.fsum((1 / (self.blend + i + 1), 1 / (self.blend + by_reranker[i])))
                for i in range(len(top))
            ]
            order = sorted(range(len(top)), key=lambda i: -scores[i])
        reranked = [Hit(top[i].chunk, scores[i], top[i].via) for i in order]
        return (reranked + rest)[:k]
