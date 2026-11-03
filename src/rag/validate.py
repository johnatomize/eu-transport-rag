"""The check every answer passes before it is shown, made by code alone, without a model.

`check_reply` takes the model's raw reply and the chunks the retriever gave it, and returns a
`Verdict`. An answer passes only when all of these hold:

- the reply fits the answer format (`parse_answer`); a reply that doesn't fails the check
  instead of crashing;
- an answer the passages don't answer (`answerable: false`) cites nothing, and an answer they
  do answer cites at least one node;
- every cited node ID is one the prompt marked as a `<node>`: the ID of a paragraph, point or
  page whose text is in one of the retrieved chunks. An ID from the corpus that wasn't
  retrieved fails, and so does a chunk's or an article's ID when only its paragraphs were
  shown;
- every quote is a continuous piece of that node's text as the retrieved chunks hold it. A
  window that holds only part of a node offers only that part.

Before a quote is looked for, both it and the text are put through `normalise`, which drops
invisible characters and changes only spacing and the typography of quote marks and dashes;
nothing else (case, digits, "3,5" against "3.5", ellipses) is forgiven. A match must start and
end on a word's edge, so a quote can't cut a number short. A passing citation carries the
quote as the corpus writes it, which is what a reader is shown, never the model's copy.

An answer that fails is shown as "could not answer reliably" (`prompt.unreliable_message`), and
the reasons stay in `Verdict.problems` for the run log. Choices and reasons: "the check of every
answer" in docs/DECISIONS.md.
"""

import itertools
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from rag.chunk import Chunk
from rag.prompt import Answer, AnswerFormatError, parse_answer

# The typography a model may write differently from the act without changing a word: curly
# and straight quote marks, and the dashes. Each maps to one character, so offsets survive.
# Named rather than written as glyphs, because several of them look alike.
_PLAIN = {
    "'": (
        "LEFT SINGLE QUOTATION MARK",
        "RIGHT SINGLE QUOTATION MARK",
        "SINGLE LOW-9 QUOTATION MARK",
        "SINGLE HIGH-REVERSED-9 QUOTATION MARK",
        "PRIME",
    ),
    '"': (
        "LEFT DOUBLE QUOTATION MARK",
        "RIGHT DOUBLE QUOTATION MARK",
        "DOUBLE LOW-9 QUOTATION MARK",
        "DOUBLE PRIME",
    ),
    "-": (
        "HYPHEN",
        "NON-BREAKING HYPHEN",
        "FIGURE DASH",
        "EN DASH",
        "EM DASH",
        "MINUS SIGN",
    ),
}
_TYPOGRAPHY = {unicodedata.lookup(name): plain for plain, names in _PLAIN.items() for name in names}
# Characters a reader never sees, which are dropped from both sides before matching: the soft
# hyphen, zero-width spaces and joiners, direction marks and the byte order mark, as in
# `prompt._GAP`. The corpus holds none of them today.
_INVISIBLE = re.compile(r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")


@dataclass(frozen=True)
class CheckedCitation:
    node_id: str
    quote: str  # the quote as the corpus writes it, not as the model copied it


@dataclass(frozen=True)
class Verdict:
    """What the check found. `ok` answers may be shown; any other is "could not answer
    reliably", with `problems` saying why, one line each."""

    answer: Answer | None  # None when the reply didn't fit the answer format
    citations: tuple[CheckedCitation, ...]  # every citation, checked; empty unless ok
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.answer is not None and not self.problems


def normalise(text: str) -> tuple[str, list[int]]:
    """`text` with invisible characters dropped, every run of whitespace as one space and the
    typography made plain.

    Also returns, for each character of the result, where it starts in `text`, plus the length
    of `text` at the end, so a match in the result can be read back from `text`.
    """
    out: list[str] = []
    starts: list[int] = []
    for i, char in enumerate(text):
        if _INVISIBLE.match(char):
            continue
        if char.isspace():
            if out and out[-1] == " ":
                continue
            char = " "
        out.append(_TYPOGRAPHY.get(char, char))
        starts.append(i)
    starts.append(len(text))
    return "".join(out), starts


def quotable(chunks: Iterable[Chunk]) -> dict[str, list[str]]:
    """The text of each node the prompt marked, as the chunks hold it.

    A node's lines that follow each other in a chunk are one piece of text, joined by line
    breaks, as in its `<node>` element; a node can have several pieces, from several chunks or
    from text before and after its own points.
    """
    texts: dict[str, list[str]] = {}
    for chunk in chunks:
        if not chunk.line_owners:
            raise ValueError(f"{chunk.id}: no line_owners, so its nodes are unknown")
        lines = zip(chunk.line_owners, chunk.body.split("\n"), strict=True)
        for owner, run in itertools.groupby(lines, key=lambda pair: pair[0]):
            if owner is not None:
                piece = "\n".join(line for _, line in run)
                if piece not in texts.setdefault(owner, []):
                    texts[owner].append(piece)
    return texts


def find_quote(quote: str, pieces: Iterable[str]) -> str | None:
    """The corpus text `quote` stands for, from the first piece that holds it, or None.

    The quote is put in Unicode's composed form (NFC) first, as the corpus is. A match must
    start and end on a word's edge: "shall not exceed 5" is not a quote of "shall not exceed 56
    hours".
    """
    wanted, _ = normalise(unicodedata.normalize("NFC", quote))
    wanted = wanted.strip()
    if not wanted:
        return None
    for piece in pieces:
        flat, starts = normalise(piece)
        at = flat.find(wanted)
        while at >= 0:
            end = at + len(wanted)
            cut_start = wanted[0].isalnum() and at > 0 and flat[at - 1].isalnum()
            cut_end = wanted[-1].isalnum() and end < len(flat) and flat[end].isalnum()
            if not cut_start and not cut_end:
                return piece[starts[at] : starts[end - 1] + 1]
            at = flat.find(wanted, at + 1)
    return None


def check_answer(answer: Answer, chunks: Iterable[Chunk]) -> Verdict:
    """Check a parsed answer's citations against the chunks the prompt was built from."""
    texts = quotable(chunks)
    problems: list[str] = []
    if not answer.answerable and answer.citations:
        n = len(answer.citations)
        problems.append(
            f"says the passages don't answer it, but gives {n} citation{'s' if n > 1 else ''}"
        )
    if answer.answerable and not answer.citations:
        problems.append("gives an answer without citations")
    checked: list[CheckedCitation] = []
    for i, citation in enumerate(answer.citations, 1):
        where = f"citation {i} ({citation.node_id})"
        if citation.node_id not in texts:
            problems.append(f"{where}: not a node of the retrieved passages")
            continue
        found = find_quote(citation.quote, texts[citation.node_id])
        if found is None:
            problems.append(f"{where}: the quote is not in that node's text: {citation.quote!r}")
            continue
        if (cited := CheckedCitation(citation.node_id, found)) not in checked:
            checked.append(cited)  # the same citation twice is shown once
    if problems:
        return Verdict(answer, (), tuple(problems))
    return Verdict(answer, tuple(checked), ())


def check_reply(reply: str, chunks: Iterable[Chunk]) -> Verdict:
    """Read the model's raw reply and check it; a reply that isn't in the format fails."""
    try:
        answer = parse_answer(reply)
    except AnswerFormatError as e:
        return Verdict(None, (), (f"not in the answer format: {e}",))
    return check_answer(answer, chunks)
