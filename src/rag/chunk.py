"""Cut the parsed corpus into chunks, the units that retrieval indexes and answers cite.

Three strategies, each a function `(Tree) -> list[Chunk]` listed in `STRATEGIES` by name:

- `article`: one chunk per article and per top-level annex part (165/2014 Annex I alone is
  about 4,000 words), and one per guidance page.
- `paragraph`: an article or top-level annex part splits into its children, each with the text
  that introduces it in its prefix; one without children stays whole, and its own text after
  the first child becomes a chunk with its own ID. Guidance pages stay whole.
- `window-200-50`: windows of 200 words overlapping by 50, over each document's text in reading
  order. They cross article and page boundaries, never documents.

Every chunk's prefix starts with the heading path of its unit, so that "1. The daily driving
time…" keeps its act and article. Articles that amend another act, and an act's own title text,
are left out. Chunks come out in document order. Choices and reasons: "chunking strategies" in
docs/DECISIONS.md.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from functools import partial

from rag.parse import Line, Node, Tree, heading_line, lead_in


@dataclass(frozen=True)
class Chunk:
    id: str  # "paragraph:561/2006:art6(3)"
    strategy: str  # "article", "paragraph" or "window-200-50"
    doc_id: str
    prefix: str  # the unit's heading path, then any lead-in; not counted as the chunk's text
    body: str  # the document text the chunk holds
    # Every node whose own text is in `body`, at least in part, in document order. A chunk
    # finds a gold node when this holds the node or a descendant of it (docs/DECISIONS.md).
    node_ids: tuple[str, ...]
    published: str | None = None  # the note's date, for chunks of guidance notes
    # The node whose own text each line of `body` is, or None for a heading line, so that a
    # prompt can mark which node every line belongs to (docs/DECISIONS.md, "the prompt").
    line_owners: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        if self.line_owners and len(self.line_owners) != self.body.count("\n") + 1:
            raise ValueError(f"{self.id}: line_owners doesn't match the lines of the body")

    @property
    def text(self) -> str:
        """What is indexed, and in E embedded: the prefix, then the body."""
        return f"{self.prefix}\n{self.body}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "strategy": self.strategy,
            "doc_id": self.doc_id,
            "prefix": self.prefix,
            "body": self.body,
            "node_ids": list(self.node_ids),
            "published": self.published,
            "line_owners": list(self.line_owners),
        }


def units(tree: Tree) -> Iterator[Node]:
    """The articles, top-level annex parts and guidance pages, in document order.

    An annex without parts is a unit itself. Amending articles are left out (design problem
    1); they stay in the tree.
    """
    for node in tree:
        parent = tree[node.parent] if node.parent is not None else None
        if node.kind == "article" and not node.amending:
            yield node
        elif node.kind == "annex_part" and parent.kind == "annex":
            yield node
        elif node.kind == "annex" and not any(tree[c].kind == "annex_part" for c in node.children):
            yield node
        elif node.kind == "page":
            yield node
        elif node.kind == "annex" and node.segments:
            # Text above an annex's parts belongs to no unit; nothing has any so far.
            raise ValueError(f"{node.id}: annex text above its parts is not chunked")


def _chunk(
    tree: Tree, strategy: str, chunk_id: str, start: Node, around: Node, lines: list[Line]
) -> Chunk:
    """A chunk of `lines`, with the heading path of `start` and its lead-in inside `around`."""
    prefix = " › ".join(start.heading_path)
    lead = lead_in(tree, start, around)
    owners = [line.owner for line in lines if line.owner is not None]
    return Chunk(
        id=f"{strategy}:{chunk_id}",
        strategy=strategy,
        doc_id=start.doc_id,
        prefix=f"{prefix}\n{lead}" if lead else prefix,
        body="\n".join(line.text for line in lines),
        node_ids=tuple(dict.fromkeys(owners)),
        published=tree[start.doc_id].published,
        line_owners=tuple(line.owner for line in lines),
    )


def article_chunks(tree: Tree) -> list[Chunk]:
    """One chunk per article, top-level annex part and guidance page."""
    return [
        _chunk(tree, "article", unit.id, unit, unit, tree.lines(unit.id)) for unit in units(tree)
    ]


def _end(tree: Tree, node_id: str) -> int:
    """The position of the last piece of text in a node's subtree."""
    return max(max((p for p, _ in n.segments), default=n.position) for n in tree.subtree(node_id))


def paragraph_chunks(tree: Tree) -> list[Chunk]:
    """One chunk per child of an article or top-level annex part, with its lead-in.

    A unit without children, and every guidance page, is one chunk. A unit's own text before a
    child is that child's lead-in, in its prefix; its text after the last child (closing text,
    as in 165/2014 Art. 10) is a chunk with the unit's ID.
    """
    chunks = []
    for unit in units(tree):
        if not unit.children:
            chunks.append(_chunk(tree, "paragraph", unit.id, unit, unit, tree.lines(unit.id)))
            continue
        for child_id in unit.children:
            lines = tree.lines(child_id, label=True)
            chunks.append(_chunk(tree, "paragraph", child_id, tree[child_id], unit, lines))
        last = _end(tree, unit.children[-1])
        closing = [Line(unit.id, unit.id, text) for p, text in unit.segments if p > last]
        if closing:
            chunks.append(_chunk(tree, "paragraph", unit.id, unit, unit, closing))
    return chunks


def window_chunks(tree: Tree, size: int = 200, overlap: int = 50) -> list[Chunk]:
    """Windows of `size` words, each starting `size - overlap` words after the one before.

    Windows run over each document's units in order, each unit's heading on a line of its own,
    and never cross into the next document; the last window of a document may be shorter. A
    window's prefix is the heading path of the node it starts in, and that node's lead-in
    inside its unit.
    """
    if not 0 <= overlap < size:
        raise ValueError(f"overlap {overlap} must be at least 0 and less than size {size}")
    strategy = f"window-{size}-{overlap}"
    # Each word, as a one-word line, with the unit it is in and whether it starts a line.
    by_doc: dict[str, list[tuple[Line, Node, bool]]] = {}
    for unit in units(tree):
        words = by_doc.setdefault(unit.doc_id, [])
        heading = heading_line(unit)
        lines = [Line(None, unit.id, heading)] if heading else []
        for line in lines + tree.lines(unit.id):
            for i, word in enumerate(line.text.split()):
                words.append((Line(line.owner, line.node, word), unit, i == 0))
    chunks = []
    for doc_id, words in by_doc.items():
        starts = range(0, max(len(words) - overlap, 1), size - overlap)
        for number, start in enumerate(starts, 1):
            window = words[start : start + size]
            first, unit, _ = window[0]
            rest = (("\n" if new else " ") + word.text for word, _, new in window[1:])
            body = first.text + "".join(rest)
            lines = [Line(word.owner, word.node, "") for word, _, _ in window]
            chunk = _chunk(tree, strategy, f"{doc_id}:w{number}", tree[first.node], unit, lines)
            owners = [first.owner] + [word.owner for word, _, new in window[1:] if new]
            chunks.append(replace(chunk, body=body, line_owners=tuple(owners)))
    return chunks


STRATEGIES: dict[str, Callable[[Tree], list[Chunk]]] = {
    "article": article_chunks,
    "paragraph": paragraph_chunks,
    "window-200-50": partial(window_chunks, size=200, overlap=50),
}
