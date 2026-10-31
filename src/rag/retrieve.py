"""Retrieval: return the chunks that best match a question, best first.

Every retriever implements `Retriever`, so that the evaluation, the service and the comparison
of backends don't care which one they get. This module has the keyword baseline, BM25 (design
§D); E adds vector and hybrid search behind the same interface.

BM25 is the Okapi variant with the usual defaults, k1 = 1.5 and b = 0.75, and the idf that
stays positive, log(1 + (N - df + 0.5) / (df + 0.5)), as in Lucene. It is written here rather
than taken from a library: it is about 30 lines, and owning it keeps ties and rounding under our
control, so that two runs give identical rankings.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from snowballstemmer import stemmer

from rag.chunk import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Hit]:
        """The `k` best chunks for `query`, best first; ties in chunk order."""
        ...


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
        if k < 1:
            raise ValueError(f"k must be at least 1, not {k}")
        scores = self.scores(query)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [Hit(self.chunks[i], scores[i]) for i in order[:k]]
