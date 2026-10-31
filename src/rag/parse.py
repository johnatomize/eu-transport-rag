"""Parse the acts in the corpus into a tree of nodes with stable IDs.

An act becomes act → chapter → article → paragraph → point, plus annex → annex part → point.
Every node has an ID such as `561/2006:art6(3)`: the document ID from the manifest, then the
node's path. Numbered and lettered enumerations are in brackets, the way the acts cite
themselves (`art8(6)(a)`, `art4(e)`). In annexes, headings and decimal numbers are joined with
dots (`anxI.III.c.4`, `anxI.III.a.3.1`) and lettered points are in brackets (`anxI.I(a)`).
Subparagraphs and indents (`—`) are not nodes; they stay in their parent's text.

Two readers produce the same `Node`s: one for consolidated XHTML (561/2006, 165/2014) and one
for the HTML 4 of 2002/15, which is a flat run of paragraphs whose structure has to be rebuilt
from the labels. Amendment markers (`▼M3`, `►M4 … ◄`) are removed from the text and kept as
`Node.amendments`.

A Commission guidance note (PDF) becomes a note with one node per page, `guidance-7:p3`, so
that answers can cite a page; its text comes from `rag.pdftext`.

Cross-references are resolved after every document has been parsed, since they can point into
another act.

Anything the readers don't understand goes into `Tree.unparsed`, and a duplicate node ID goes
into `Tree.errors`; nothing is dropped silently. See docs/DECISIONS.md for the ID scheme.
"""

import html
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from pdfminer.psexceptions import PSException

from rag.corpus import Document
from rag.pdftext import read_pdf

_NS = "{http://www.w3.org/1999/xhtml}"

KINDS = ("act", "chapter", "article", "paragraph", "point", "annex", "annex_part", "note", "page")


class ParseError(Exception):
    """A document that can't be parsed at all. The message names the document."""


@dataclass(frozen=True)
class Amendment:
    marker: str  # "M3", or "C1" for a corrigendum
    celex: str  # the amending act or corrigendum, such as "32020R1054"
    action: str | None  # "REPLACED", "INSERTED" or "DELETED", when the text says


@dataclass(frozen=True)
class Ref:
    text: str  # the reference as written, such as "point (e) of Article 4"
    act: str  # a document ID in the corpus, or the act's name if it is outside the corpus
    target: str | None  # the node ID, or None if outside the corpus or unresolved
    # True when the cited node isn't in the tree and `target` is its nearest ancestor; the
    # report lists these, since one can hide a gap in the parse.
    partial: bool = False


@dataclass
class Node:
    id: str
    doc_id: str
    kind: str
    label: str | None  # "6", "(2a)", "(a)", "III", "p3"; part of the ID, not repeated in the text
    # "Article 2 — Definitions"; only acts, chapters, articles, annexes, notes and pages
    heading: str | None
    heading_path: tuple[str, ...]
    parent: str | None
    children: list[str] = field(default_factory=list)
    # The node's own text, each piece with its position in the document, so that a subtree can
    # be rendered in the original order (a paragraph's closing subparagraphs come after its
    # points).
    segments: list[tuple[int, str]] = field(default_factory=list)
    amendments: list[Amendment] = field(default_factory=list)
    refs: list[Ref] = field(default_factory=list)
    # True for an article that amends another act; its text holds wording that was later
    # superseded (design problem 1), so chunking decides whether to index it.
    amending: bool = False
    # The act an amending article amends, which its references point into by default.
    amends: str | None = None
    # For a guidance note: the provisions it says it interprets, rewritten into the form the
    # reference scanner reads ("Article 8(2) and (5) of Regulation (EC) No 561/2006").
    legal_basis: str | None = None
    # For a guidance note: the act in the corpus it interprets, which its references without an
    # act point into; None if its legal basis names no act in the corpus, or more than one.
    interprets: str | None = None
    # For a guidance note: when its content was written, from the manifest ("2007-12-18", or
    # "1993-02" when only the month is known), so that older guidance can be told apart.
    published: str | None = None
    # Where the node starts in the document, in the same numbering as its segments.
    position: int = 0

    @property
    def text(self) -> str:
        return "\n".join(text for _, text in self.segments)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "kind": self.kind,
            "label": self.label,
            "heading": self.heading,
            "heading_path": list(self.heading_path),
            "parent": self.parent,
            "children": self.children,
            "text": self.text,
            "amendments": [vars(a) for a in self.amendments],
            "refs": [vars(r) for r in self.refs],
            "amending": self.amending,
            "amends": self.amends,
            "legal_basis": self.legal_basis,
            "interprets": self.interprets,
            "published": self.published,
        }


class Tree:
    """Every node of the parsed corpus by ID, in document order."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.unparsed: list[str] = []
        self.errors: list[str] = []

    def __contains__(self, node_id: object) -> bool:
        return node_id in self.nodes

    def __getitem__(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes.values())

    def add(self, node: Node) -> Node:
        if node.id in self.nodes:
            self.errors.append(f"{node.doc_id}: duplicate node ID {node.id}")
            return self.nodes[node.id]
        self.nodes[node.id] = node
        if node.parent is not None:
            self.nodes[node.parent].children.append(node.id)
        return node

    def subtree(self, node_id: str) -> Iterator[Node]:
        node = self.nodes[node_id]
        yield node
        for child in node.children:
            yield from self.subtree(child)

    def full_text(self, node_id: str) -> str:
        """The text of a node and everything below it, in document order.

        Paragraphs and points start with their label, and the headings of chapters, articles,
        annexes and annex parts below the node are on lines of their own.
        """
        return "\n".join(line.text for line in self.lines(node_id))

    def lines(self, node_id: str, label: bool = False) -> list["Line"]:
        """`full_text` line by line, each line with the node it belongs to.

        With `label`, the node's own first line starts with its printed label too.
        """
        pieces: list[tuple[int, Line]] = []
        for node in self.subtree(node_id):
            below = node.id != node_id
            heading = heading_line(node)
            if below and heading and node.kind != "act":
                pieces.append((node.position, Line(None, node.id, heading)))
            printed = ""
            if (below or label) and node.kind in {"paragraph", "point"}:
                printed = printed_label(node)
            if printed and not node.segments:
                pieces.append((node.position, Line(node.id, node.id, printed)))
            for i, (position, text) in enumerate(node.segments):
                text = f"{printed} {text}" if printed and i == 0 else text
                pieces.append((position, Line(node.id, node.id, text)))
        return [line for _, line in sorted(pieces, key=lambda piece: piece[0])]


@dataclass(frozen=True)
class Line:
    """A line of text from `Tree.lines`."""

    owner: str | None  # the node whose own text this is; None for a heading
    node: str  # the node the line belongs to: the owner, or the node a heading is the heading of
    text: str


def heading_line(node: Node) -> str | None:
    """A node's heading as printed on its own line: "Article 6", "III. Construction…"."""
    return node.heading_path[-1] if node.kind == "annex_part" else node.heading


def printed_label(node: Node) -> str:
    """The label as printed: "3." for a paragraph and "3.1." for an annex point; else "(a)"."""
    if node.kind == "paragraph":
        return node.label.strip("()") + "."
    return node.label if node.label.startswith("(") else f"{node.label}."


def lead_in(tree: Tree, node: Node, around: Node | None) -> str:
    """The text that introduces a node inside `around`, outermost first.

    `around` is an ancestor such as the node's article, annex part or guidance page. A point
    such as "(ii) loading and unloading;" means little without the text that comes before it in
    the paragraphs above it and in the article itself. The review page (#17) shows it above a
    gold node, and chunking (#33) puts it in a chunk's prefix.
    """
    if around is None:
        return ""
    pieces: list[str] = []
    current = node
    while current is not around:
        parent = tree[current.parent]
        before = [text for position, text in parent.segments if position < current.position]
        if before and parent is not around and parent.kind in {"paragraph", "point"}:
            before[0] = f"{printed_label(parent)} {before[0]}"
        pieces[:0] = before
        current = parent
    return "\n".join(pieces)


def parse_document(doc: Document, root: Path, tree: Tree | None = None) -> Tree:
    """Parse one document into `tree` (a new tree if none). References are resolved separately."""
    tree = tree if tree is not None else Tree()
    data = (root / doc.file).read_bytes()
    if doc.format == "xhtml":
        _XhtmlReader(doc, tree).read(data)
    elif doc.format == "html":
        _HtmlReader(doc, tree).read(data)
    elif doc.format == "pdf":
        _read_note(doc, tree, data)
    else:
        raise ParseError(f"{doc.id}: no parser for format {doc.format}")
    _propagate_amendments(tree, doc.id)
    return tree


def parse_corpus(docs: list[Document], root: Path) -> Tree:
    """Parse every document in the manifest and resolve the cross-references between them.

    References can point only into acts; guidance notes cite acts, not each other.
    """
    tree = Tree()
    for doc in docs:
        parse_document(doc, root, tree)
    resolve_refs(tree, {doc.id for doc in docs if doc.kind == "act"})
    return tree


# --- Text ---------------------------------------------------------------------------------

_FOOTNOTE = "\x00"
_DASHES = {"—", "–", "-", "•"}


def _clean(text: str) -> str:
    """Normalise a piece of text: no markers or footnote calls, collapsed whitespace, NFC."""
    text = re.sub(rf"\s*\(\s*{_FOOTNOTE}\s*\)", "", text)
    text = text.replace(_FOOTNOTE, "").replace("◄", "")
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def _sentence_case(text: str) -> str:
    """'CREWS, DRIVING TIMES' → 'Crews, driving times'. Mixed-case text is left alone."""
    text = re.sub(r"\s+", " ", text).strip()
    if text.upper() != text:
        return text
    return text[:1] + text[1:].lower()


def _short_title(doc: Document) -> str:
    """'Regulation (EC) No 561/2006 (driving times…)' → 'Regulation (EC) No 561/2006'."""
    return re.sub(r"\s*\([^()]*\)$", "", doc.title)


class _Builder:
    """Creates nodes in document order and tracks the amendment marker in effect."""

    def __init__(self, doc: Document, tree: Tree):
        self.doc = doc
        self.tree = tree
        self.position = 0
        self.marker: Amendment | None = None  # None while the base text (▼B) is in effect

    def node(
        self,
        node_id: str,
        kind: str,
        label: str | None,
        parent: Node | None,
        heading: str | None = None,
    ) -> Node:
        path = parent.heading_path if parent else ()
        if heading:
            path = (*path, heading)
        node = Node(
            id=node_id,
            doc_id=self.doc.id,
            kind=kind,
            label=label,
            heading=heading,
            heading_path=path,
            parent=parent.id if parent else None,
        )
        self.position += 1
        node.position = self.position
        return self.tree.add(node)

    def text(self, node: Node, text: str, inline: list[Amendment] | None = None) -> None:
        text = _clean(text)
        if not text:
            return
        self.position += 1
        node.segments.append((self.position, text))
        for amendment in ([self.marker] if self.marker else []) + (inline or []):
            if amendment not in node.amendments:
                node.amendments.append(amendment)

    def unparsed(self, where: Node, what: str) -> None:
        self.tree.unparsed.append(f"{where.id}: {what}")


def _amendment(a: ET.Element) -> Amendment | None:
    """The amendment an `▼M3` or `►M3` link stands for; None for the base text (`B`)."""
    marker = re.sub(r"[▼►\s]", "", "".join(a.itertext()))
    if not marker or marker == "B":
        return None
    celex = a.get("href", "").rsplit("/", 1)[-1].replace("%28", "(").replace("%29", ")")
    title = a.get("title", "")
    action = title.split(": ", 1)[1].strip() if ": " in title else None
    return Amendment(marker=marker, celex=celex, action=action)


def _propagate_amendments(tree: Tree, doc_id: str) -> None:
    """Every node also lists the amendments anywhere below it (an article lists its points')."""
    for node in reversed(list(tree)):
        if node.doc_id != doc_id or node.parent is None:
            continue
        parent = tree[node.parent]
        for amendment in node.amendments:
            if amendment not in parent.amendments:
                parent.amendments.append(amendment)


# --- Consolidated XHTML (561/2006, 165/2014) ---------------------------------------------

_PARAGRAPH_LABEL = re.compile(r"(\d+[a-z]*)\.")
_POINT_LABEL = re.compile(r"\(([0-9a-z]+)\)|(\d+)\.")
_ACT = re.compile(
    r"(?:(?:Council\s+|Commission\s+)?(?:Implementing\s+|Delegated\s+)?"
    r"(?:Regulation|Directive|Decision)\s+(?:\((?:EC|EU|EEC|Euratom)\)\s+)?(?:No\s+)?"
    r"(?P<number>\d{1,4}/\d{2,4})(?:/(?:EC|EEC|EU|Euratom))?"
    # The international agreement on drivers' hours, as 2002/15 Art. 4 and guidance-8 name it.
    r"|AETR\s+Agreement|European\s+Agreement\s+concerning\s+the\s+Work\s+of\s+Crews\b[^().;]{0,120}\(AETR\))"
)
_AMENDED = re.compile(rf"(?P<act>{_ACT.pattern})\s+is\s+(?:hereby\s+)?amended\s+as\s+follows")

# Front matter and back matter that is deliberately not parsed: the document reference,
# disclaimer and amendment tables, and footnotes, which are Official Journal references.
_SKIPPED_CLASSES = {
    "reference",
    "disclaimer",
    "hd-modifiers",
    "footnote",
    "arrow",
    "separator",
    "separator-short",
    "separator-annex",
    "title-division-1",
    "title-division-2",
    "title-annex-1",
}


def _tag(el: ET.Element) -> str:
    return el.tag.removeprefix(_NS)


def _classes(el: ET.Element) -> set[str]:
    return set(el.get("class", "").split())


def _plain(el: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _rows(table: ET.Element) -> list[ET.Element]:
    """A table's own rows, not those of tables nested in its cells."""
    return table.findall(f"{_NS}tr") + table.findall(f"{_NS}tbody/{_NS}tr")


def _is_blank(el: ET.Element) -> bool:
    return (
        _tag(el) in {"p", "br", "hr", "col"} and not _plain(el) and el.find(f".//{_NS}img") is None
    )


class _XhtmlReader:
    def __init__(self, doc: Document, tree: Tree):
        self.doc = doc
        self.b = _Builder(doc, tree)

    def read(self, data: bytes) -> None:
        try:
            root = ET.fromstring(data)
        except ET.ParseError as e:
            raise ParseError(f"{self.doc.id}: not well-formed XHTML: {e}") from e
        container = next(
            (el for el in root.iter(f"{_NS}div") if "eli-container" in _classes(el)), None
        )
        if container is None:
            raise ParseError(f"{self.doc.id}: no eli-container element; not a consolidated text")
        self.act = self.b.node(self.doc.id, "act", None, None, heading=_short_title(self.doc))
        self._division(container, self.act)

    # Chapters, articles and annexes.
    def _division(self, el: ET.Element, parent: Node) -> None:
        for child in el:
            el_id = child.get("id", "")
            classes = _classes(child)
            if "modref" in classes:
                self._modref(child)
            elif el_id.startswith("tit_"):
                self.b.text(self.act, " ".join(_plain(p) for p in child))
            elif el_id == "enc_1":
                self._division(child, parent)
            elif m := re.fullmatch(r"cpt_([IVXLC]+)", el_id):
                titles = [_plain(p) for p in child if "title-division-2" in _classes(p)]
                heading = f"Chapter {m[1]}" + (f" — {_sentence_case(titles[0])}" if titles else "")
                chapter = self.b.node(f"{self.doc.id}:ch{m[1]}", "chapter", m[1], parent, heading)
                self._division(child, chapter)
            elif m := re.fullmatch(r"art_(\w+)", el_id):
                self._article(child, m[1], parent)
            elif m := re.fullmatch(r"anx_(\w+)", el_id):
                _AnnexReader(self, m[1]).read(child)
            elif el_id == "fnp_1" or classes & _SKIPPED_CLASSES or _is_blank(child):
                continue  # the closing formula ("This Regulation shall be binding…"), and layout
            else:
                self.b.unparsed(parent, f"{_tag(child)}.{child.get('class')}: {_plain(child)[:60]}")

    def _modref(self, el: ET.Element) -> None:
        a = el.find(f"{_NS}a")
        if a is not None:
            self.b.marker = _amendment(a)

    def _article(self, el: ET.Element, label: str, parent: Node) -> None:
        # The article's own subtitle; quoted amending text can carry subtitles of its own.
        titles = [_plain(p) for c in el if "eli-title" in _classes(c) for p in c]
        heading = f"Article {label}" + (f" — {titles[0]}" if titles else "")
        article = self.b.node(f"{self.doc.id}:art{label}", "article", label, parent, heading)
        body = [c for c in el if not _classes(c) & {"title-article-norm", "eli-title", "modref"}]
        if body and (m := _AMENDED.search(_plain(body[0]))):
            article.amending = True
            article.amends = m["act"]
        paragraph = None  # text after paragraph N, outside its element, belongs to it
        for child in el:
            classes = _classes(child)
            if classes & {"title-article-norm", "eli-title"}:
                continue
            if "modref" in classes:
                self._modref(child)
            elif _tag(child) == "div" and "norm" in classes and (no := self._no_parag(child)):
                m = _PARAGRAPH_LABEL.fullmatch(no)
                if not m:
                    self.b.unparsed(article, f"paragraph label {no!r}")
                    continue
                paragraph = self.b.node(f"{article.id}({m[1]})", "paragraph", f"({m[1]})", article)
                for part in child:
                    if "no-parag" not in _classes(part):
                        self._content(part, paragraph, flatten=False, start=True)
            else:
                self._block(child, paragraph or article, flatten=False)

    @staticmethod
    def _no_parag(el: ET.Element) -> str | None:
        span = next((c for c in el if "no-parag" in _classes(c)), None)
        return _plain(span) if span is not None else None

    def _content(self, el: ET.Element, owner: Node, flatten: bool, start: bool = False) -> None:
        """Read a container whose inline text and block children all belong to `owner`."""
        if start and not any(self._is_block(c) for c in el):
            self._inline(el, owner)
            return
        run: list[ET.Element | str] = [el.text or ""]
        for child in el:
            if self._is_block(child):
                self._flush(run, owner)
                run = []
                self._block(child, owner, flatten)
                run.append(child.tail or "")
            else:
                run.append(child)
        self._flush(run, owner)

    def _is_block(self, el: ET.Element) -> bool:
        return _tag(el) in {"p", "div", "table"}

    def _flush(self, run: list, owner: Node) -> None:
        pieces, inline = [], []
        for item in run:
            if isinstance(item, str):
                pieces.append(item)
            else:
                text, amendments = self._inline_text(item, owner)
                pieces.append(text + (item.tail or ""))
                inline += amendments
        self.b.text(owner, "".join(pieces), inline)

    def _block(self, el: ET.Element, owner: Node, flatten: bool, prefix: str = "") -> None:
        classes = _classes(el)
        tag = _tag(el)
        if "modref" in classes:
            self._modref(el)
        elif "grid-container" in classes:
            self._list_item(el, owner, flatten)
        elif tag == "table":
            self._table(el, owner, flatten)
        elif tag == "p":
            self._inline(el, owner, prefix)
        elif tag == "div" and "norm" in classes and self._no_parag(el) and flatten:
            # A paragraph quoted inside an amending article: kept as text, with its label.
            self._content(el, owner, flatten)
        elif tag == "div" and not ("norm" in classes and self._no_parag(el)):
            self._content(el, owner, flatten)
        elif _is_blank(el):
            pass
        else:
            self.b.unparsed(owner, f"{tag}.{el.get('class')}: {_plain(el)[:60]}")

    def _list_item(self, el: ET.Element, owner: Node, flatten: bool) -> None:
        cells = [c for c in el if _tag(c) == "div"]
        if len(cells) != 2:
            self.b.unparsed(owner, f"list item with {len(cells)} cells: {_plain(el)[:60]}")
            return
        label = _plain(cells[0])
        m = _POINT_LABEL.fullmatch(label)
        if flatten or label in _DASHES or not m:
            # An indent, or a point inside quoted amending text: text of `owner`, with its label.
            if not (flatten or label in _DASHES):
                self.b.unparsed(owner, f"list label {label!r}")
            self._labelled_text(cells[1], owner, label, flatten)
            return
        part = f"({m[1] or m[2]})"
        point = self.b.node(f"{owner.id}{part}", "point", part, owner)
        article = self._article_of(point)
        self._content(cells[1], point, flatten=article.amending, start=True)

    def _labelled_text(self, cell: ET.Element, owner: Node, label: str, flatten: bool) -> None:
        """Read a cell into `owner`, with `label` in front of its first piece of text."""
        before = len(owner.segments)
        self._content(cell, owner, flatten, start=True)
        if len(owner.segments) > before:
            position, text = owner.segments[before]
            owner.segments[before] = (position, f"{label} {text}")

    def _article_of(self, node: Node) -> Node:
        tree = self.b.tree
        while node.kind not in {"article", "annex", "act"}:
            node = tree[node.parent]
        return node

    def _table(self, el: ET.Element, owner: Node, flatten: bool = False) -> None:
        for row in _rows(el):
            cells = [c for c in row if _tag(c) == "td"]
            if any(
                c.find(f".//{_NS}div") is not None or c.find(f".//{_NS}table") is not None
                for c in cells
            ):
                # Layout tables that hold lists or tables: read the cells as content.
                for cell in cells:
                    self._content(cell, owner, flatten, start=True)
                continue
            texts, amendments = [], []
            for cell in cells:
                text, found = self._inline_text(cell, owner)
                texts.append(text)
                amendments += found
            self.b.text(owner, " | ".join(t.strip() for t in texts if _clean(t)), amendments)

    def _inline(self, el: ET.Element, owner: Node, prefix: str = "") -> None:
        text, amendments = self._inline_text(el, owner)
        self.b.text(owner, f"{prefix} {text}" if prefix else text, amendments)

    def _inline_text(
        self, el: ET.Element, owner: Node, record: bool = True
    ) -> tuple[str, list[Amendment]]:
        """The text of an element, without markers and with footnote calls marked for removal.

        With `record`, images are listed as unparsed under `owner` and markers take effect;
        without it, the text can be looked at with no side effects.
        """
        amendments: list[Amendment] = []

        def walk(e: ET.Element) -> str:
            tag = _tag(e)
            if tag == "a" and e.get("href", "").startswith("#E"):
                return _FOOTNOTE
            if tag == "a" and re.match(r"\s*[►▼]", "".join(e.itertext())):
                if amendment := _amendment(e):
                    amendments.append(amendment)
                return ""
            if tag == "img":
                # Mostly the tachograph pictograms ("under the sign [image]: driving time").
                if record:
                    self.b.unparsed(owner, "image, left as [image] in the text")
                return " [image] "
            if tag == "br":
                return " "
            if "modref" in _classes(e):
                if record:
                    self._modref(e)
                return " "
            parts = [e.text or ""]
            for child in e:
                parts.append(walk(child))
                parts.append(child.tail or "")
            return "".join(parts)

        return walk(el), amendments


# --- Annexes of consolidated XHTML ---------------------------------------------------------

_DECIMAL = re.compile(r"(\d+(?:\.\d+)*)\.")
_LETTER = re.compile(r"\(([a-z]+)\)")
_HEADING_LABEL = re.compile(r"(?:([IVXLC]+)\.|\(([a-z]+)\)|(\d+(?:\.\d+)*)\.)\s+(.*)")
_LABELLED_TEXT = re.compile(r"(\d+(?:\.\d+)*\.|\([a-z]+\))\s+(.*)", re.S)


@dataclass
class _Open:
    node: Node
    level: int  # heading level; points sit below every heading
    numeric: bool  # a decimal label, which is a path relative to the nearest other heading
    scoped: bool  # a point inside a list item or table cell, closed when that element ends


class _AnnexReader:
    """Annexes are headings and points in document order, placed by their labels.

    Headings (`title-gr-seq-level-N`) open parts. A point is recognised from a list item's
    label, a layout table's label cell, or a label at the start of a paragraph wrapped in a
    bare `div`. Text without a label belongs to the innermost open part or point.
    """

    POINT_LEVEL = 100

    def __init__(self, reader: _XhtmlReader, label: str):
        self.r = reader
        self.b = reader.b
        doc_id = reader.doc.id
        annex = self.b.node(
            f"{doc_id}:anx{label}", "annex", label, reader.act, heading=f"Annex {label}"
        )
        self.annex = annex
        self.stack = [_Open(annex, 1, numeric=False, scoped=False)]

    @property
    def top(self) -> Node:
        return self.stack[-1].node

    def read(self, el: ET.Element) -> None:
        self._children(el)

    def _children(self, el: ET.Element) -> None:
        for child in el:
            self._element(child)

    def _element(self, el: ET.Element) -> None:
        classes = _classes(el)
        tag = _tag(el)
        level = next(
            (int(c.rsplit("-", 1)[1]) for c in classes if c.startswith("title-gr-seq-level-")),
            None,
        )
        if "modref" in classes:
            self.r._modref(el)
        elif level == 1:
            title = _sentence_case(_plain(el))
            self.annex.heading = f"Annex {self.annex.label} — {title}"
            self.annex.heading_path = (*self.annex.heading_path[:-1], self.annex.heading)
        elif level is not None:
            self._heading(level, _plain(el))
        elif classes & _SKIPPED_CLASSES or _is_blank(el):
            pass
        elif "grid-container" in classes:
            self._list_item(el)
        elif tag == "table":
            self._table(el)
        elif tag == "div" and not classes and self._labelled_div(el):
            pass
        elif tag == "div":
            self._children(el)
        elif tag == "p":
            self.r._inline(el, self.top)
        else:
            self.b.unparsed(self.top, f"{tag}.{el.get('class')}: {_plain(el)[:60]}")

    def _heading(self, level: int, text: str) -> None:
        while len(self.stack) > 1 and self.stack[-1].level >= level:
            self.stack.pop()
        m = _HEADING_LABEL.fullmatch(text)
        if not m:
            # An unlabelled heading, such as the title of a form: text of the enclosing part.
            self.b.text(self.top, text)
            return
        roman, letter, decimal, title = m.groups()
        if decimal:
            node_id = f"{self._base().id}.{decimal}"
            label = decimal
        else:
            label = roman or letter
            node_id = f"{self.top.id}.{label}"
        node = self.b.node(node_id, "annex_part", label, self.top, heading=None)
        node.heading = _sentence_case(title)
        node.heading_path = (*self.top.heading_path, f"{text.split()[0]} {node.heading}")
        self.stack.append(_Open(node, level, numeric=bool(decimal), scoped=False))

    def _base(self) -> Node:
        """The nearest open heading without a decimal label; decimal paths are relative to it."""
        return next(o.node for o in reversed(self.stack) if not o.numeric and o.level < 100)

    def _point(self, label: str, scoped: bool) -> Node | None:
        """Open a point for `label` ("3.1." or "(a)"), closing the points it follows."""
        if m := _DECIMAL.fullmatch(label):
            path = m[1]
            base = self._base()
            node_id = f"{base.id}.{path}"
            # The parent is the deepest open node whose ID this one extends ("3" for "3.1").
            while not (
                self.top is base or self.stack[-1].scoped or node_id.startswith(self.top.id + ".")
            ):
                self.stack.pop()
            if self.top is not base and not node_id.startswith(self.top.id + "."):
                node_id = f"{self.top.id}.{path}"  # inside a list item: keep ID and parent in step
            point_label = path
        elif m := _LETTER.fullmatch(label):
            while (
                self.stack[-1].level == self.POINT_LEVEL
                and not self.stack[-1].scoped
                and _LETTER.fullmatch(self.top.label or "")
            ):
                self.stack.pop()  # a sibling letter
            node_id = f"{self.top.id}({m[1]})"
            point_label = f"({m[1]})"
        else:
            return None
        node = self.b.node(node_id, "point", point_label, self.top)
        self.stack.append(_Open(node, self.POINT_LEVEL, numeric=False, scoped=scoped))
        return node

    def _close_scoped(self, node: Node) -> None:
        if any(o.node is node for o in self.stack):
            while self.stack.pop().node is not node:
                pass

    def _list_item(self, el: ET.Element) -> None:
        cells = [c for c in el if _tag(c) == "div"]
        if len(cells) != 2:
            self.b.unparsed(self.top, f"list item with {len(cells)} cells: {_plain(el)[:60]}")
            return
        label = _plain(cells[0])
        if label in _DASHES:
            self.r._labelled_text(cells[1], self.top, label, flatten=False)
            return
        node = self._point(label, scoped=True)
        if node is None:
            self.b.unparsed(self.top, f"list label {label!r}")
            self.r._labelled_text(cells[1], self.top, label, flatten=False)
            return
        self._cell(cells[1])
        self._close_scoped(node)

    def _cell(self, cell: ET.Element) -> None:
        """A list item's or table cell's content: its own text, then any blocks inside it."""
        if not any(_tag(c) in {"p", "div", "table"} for c in cell):
            self.r._inline(cell, self.top)
            return
        self.b.text(self.top, cell.text or "")
        for child in cell:
            self._element(child)
            self.b.text(self.top, child.tail or "")

    def _table(self, el: ET.Element) -> None:
        rows = _rows(el)
        cells = [c for c in rows[0] if _tag(c) == "td"] if len(rows) == 1 else []
        if len(cells) == 2 and (
            _DECIMAL.fullmatch(_plain(cells[0])) or _LETTER.fullmatch(_plain(cells[0]))
        ):
            node = self._point(_plain(cells[0]), scoped=True)
            self._cell(cells[1])
            self._close_scoped(node)
            return
        self.r._table(el, self.top)

    def _labelled_div(self, el: ET.Element) -> bool:
        """A bare `div` whose first paragraph starts with a label: a point, open until the next."""
        first = next(iter(el), None)
        if first is None or _tag(first) != "p":
            return False
        probe, _ = self.r._inline_text(first, self.top, record=False)
        m = _LABELLED_TEXT.fullmatch(_clean(probe))
        if not m:
            return False
        node = self._point(m[1], scoped=False)
        if node is None:
            return False
        # Read again for real, so that images and markers are recorded under the new point.
        text, amendments = self.r._inline_text(first, node)
        self.b.text(node, _LABELLED_TEXT.fullmatch(_clean(text))[2], amendments)
        for child in list(el)[1:]:
            self._element(child)
        return True


# --- HTML 4 (2002/15) ------------------------------------------------------------------------


class _Paragraphs(HTMLParser):
    """Collects the text of every `<p>` element."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._current: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self._flush()
            self._current = []

    def handle_endtag(self, tag):
        if tag == "p":
            self._flush()

    def handle_data(self, data):
        if self._current is not None:
            self._current.append(data)

    def _flush(self) -> None:
        if self._current is not None:
            self.paragraphs.append(re.sub(r"\s+", " ", "".join(self._current)).strip())
        self._current = None


_ROMAN = [
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
]  # fmt: skip
_HTML_LABEL = re.compile(r"(?:(\d+)\.|\(([a-z]+)\)|(-))\s+(.*)", re.S)

# Text the flat HTML can't place: it follows point 3(a)(2) but closes point 3(a) as a whole.
# The file is pinned by sha256, so this can't drift silently.
_HTML_PLACEMENT = {"The break times referred to in Article 5": "art3(a)"}


def _successor(kind: str, value: str) -> str | None:
    if kind == "num":
        return str(int(value) + 1)
    if kind == "letter":
        return chr(ord(value) + 1) if len(value) == 1 and value < "z" else None
    if kind == "roman":
        i = _ROMAN.index(value)
        return _ROMAN[i + 1] if i + 1 < len(_ROMAN) else None
    return None


_FIRST = {"num": "1", "letter": "a", "roman": "i"}


@dataclass
class _Level:
    kind: str  # num, letter, roman or indent
    value: str
    node: Node | None  # None for an indent, which isn't a node


class _HtmlReader:
    """The base act of 2002/15: a flat run of paragraphs, structured by their labels."""

    def __init__(self, doc: Document, tree: Tree):
        self.doc = doc
        self.b = _Builder(doc, tree)

    def read(self, data: bytes) -> None:
        parser = _Paragraphs()
        parser.feed(data.decode("utf-8", errors="strict"))
        paragraphs = [html.unescape(p) for p in parser.paragraphs]
        try:
            start = paragraphs.index("Article 1")
        except ValueError as e:
            raise ParseError(f"{self.doc.id}: no 'Article 1'; not the expected HTML") from e
        end = next((i for i, p in enumerate(paragraphs) if p.startswith("Done at")), None)
        if end is None:
            raise ParseError(f"{self.doc.id}: no 'Done at …' after the articles")
        footnotes = {m[1] for p in paragraphs[end:] if (m := re.match(r"\((\d+)\) ", p))}

        act = self.b.node(self.doc.id, "act", None, None, heading=_short_title(self.doc))
        # The title is the block of paragraphs just before the preamble ("THE EUROPEAN
        # PARLIAMENT AND THE COUNCIL …"); EUR-Lex puts a banner and a summary line above it.
        preamble = next((i for i, p in enumerate(paragraphs) if p.startswith("THE ")), start)
        title: list[str] = []
        for p in reversed(paragraphs[:preamble]):
            if not p and title:
                break
            if p:
                title.insert(0, p)
        self.b.text(act, " ".join(title))
        article: Node | None = None
        stack: list[_Level] = []
        expect_heading = False
        for text in paragraphs[start:end]:
            text = re.sub(
                r"(?<=[A-Za-z])\((\d+)\)", lambda m: "" if m[1] in footnotes else m[0], text
            )
            if not text:
                continue
            if m := re.fullmatch(r"Article (\d+[a-z]*)", text):
                article = self.b.node(
                    f"{self.doc.id}:art{m[1]}", "article", m[1], act, heading=f"Article {m[1]}"
                )
                stack = []
                expect_heading = True
                continue
            if expect_heading:
                expect_heading = False
                article.heading = f"Article {article.label} — {text}"
                article.heading_path = (*act.heading_path, article.heading)
                continue
            placed = next((v for k, v in _HTML_PLACEMENT.items() if text.startswith(k)), None)
            if placed:
                target = f"{self.doc.id}:{placed}"
                if target not in self.b.tree:
                    raise ParseError(f"{self.doc.id}: no {target} for {text[:40]!r}")
                self.b.text(self.b.tree[target], text)
                continue
            m = _HTML_LABEL.fullmatch(text)
            if not m:
                self.b.text(self._owner(stack, article), text)
                continue
            num, bracket, dash, rest = m.groups()
            if dash:
                self._place(stack, ["indent"], "-")  # an indent always fits
                self.b.text(self._owner(stack, article), f"— {rest}")
                continue
            kinds = ["num"] if num else ["letter", "roman"] if bracket in _ROMAN else ["letter"]
            value = num or bracket
            level = self._place(stack, kinds, value)
            if level is None:
                self.b.unparsed(article, f"label out of sequence: {text[:60]}")
                self.b.text(self._owner(stack, article), text)
                continue
            parent = self._owner(stack[:-1], article)
            kind = "paragraph" if level.kind == "num" and parent is article else "point"
            level.node = self.b.node(f"{parent.id}({value})", kind, f"({value})", parent)
            self.b.text(level.node, rest)

    @staticmethod
    def _owner(stack: list[_Level], article: Node) -> Node:
        """The innermost open node; indents aren't nodes, so their text goes to the one above."""
        return next((level.node for level in reversed(stack) if level.node), article)

    @staticmethod
    def _place(stack: list[_Level], kinds: list[str], value: str) -> _Level | None:
        """Put a label on the stack: the next sibling of an open level, or a new first child.

        Returns the new level, or None if the label fits neither. `(i)` after `(h)` is a letter,
        and `(i)` anywhere else starts a list of roman numerals.
        """
        for i in range(len(stack) - 1, -1, -1):
            level = stack[i]
            if level.kind in kinds and (
                level.kind == "indent" or _successor(level.kind, level.value) == value
            ):
                del stack[i:]
                stack.append(_Level(level.kind, value, None))
                return stack[-1]
        for kind in kinds:
            if kind == "indent" or _FIRST[kind] == value:
                stack.append(_Level(kind, value, None))
                return stack[-1]
        return None


# --- Guidance notes (PDF) -------------------------------------------------------------------

# The field that names the provisions a note interprets, and the fields that can follow it.
_BASIS = re.compile(r"(?:^|(?<=\s))(?:Articles?|Legal basis)\s*:\s*")
_NEXT_FIELD = re.compile(
    r"(?:Issue(?: raised)?|Approach to be followed|Clarification|Comment|Practical implementation)"
    r"\s*:"
)


def _read_note(doc: Document, tree: Tree, data: bytes) -> None:
    """A note node, and under it one node per page holding that page's text."""
    try:
        pages = read_pdf(data)
    except PSException as e:  # pdfminer's errors for a damaged or unreadable file
        raise ParseError(f"{doc.id}: not a readable PDF: {e}") from e
    if not pages:
        raise ParseError(f"{doc.id}: PDF has no pages")
    b = _Builder(doc, tree)
    note = b.node(doc.id, "note", None, None, heading=doc.title)
    note.published = doc.published
    for page in pages:
        label = f"p{page.number}"
        node = b.node(f"{doc.id}:{label}", "page", label, note, heading=f"Page {page.number}")
        for text in page.segments:
            b.text(node, text)
        for what in page.unparsed:
            b.unparsed(node, what)
    note.legal_basis = _legal_basis(pages[0].segments)


def _legal_basis(segments: list[str]) -> str | None:
    """The field naming the provisions a note interprets, rewritten for the reference scanner.

    The notes write it loosely: "Article: 8 (2) and (5) of Regulation (EC) No 561/2006",
    "Articles: 34(1) and (3) of Regulation (EU) No 165/2014, 15(7) of Regulation (EEC) No
    3821/85", or "Articles: 1, 2, 4(a), 4(c)" with the act named only in the note's header.
    These become "Article 8 (2) and (5) of Regulation (EC) No 561/2006", "Article 34(1) and (3)
    of … 165/2014, Article 15(7) of …" and "Article 1, 2, 4(a), 4(c) of Regulation (EC) No
    561/2006". The page text itself is not changed, but references found in the legal basis
    carry this rewritten wording.
    """
    for i, segment in enumerate(segments):
        m = _BASIS.search(segment)
        if m is None:
            continue
        end = _NEXT_FIELD.search(segment, m.end())
        text = segment[m.end() : end.start() if end else len(segment)].strip()
        text = re.sub(r"\b(Regulation|Directive)\(", r"\1 (", text)
        # A bare number after an act ("…/85 and 13 (d)", "…/2014, 15(7)") is another article,
        # unless it is the number of another act ("… 561/2006 and 2002/15/EC").
        text = re.sub(
            r"(/\d{2,4}(?:/EC|/EEC|/EU)?,?\s+(?:and\s+)?)(?=\d+[a-z]?(?![\d/]))",
            r"\1Article ",
            text,
        )
        if text[:1].isdigit():
            text = f"Article {text}"
        if not _ACT.search(text):
            # Only the header before the field says which act the note is about.
            header = " ".join([*segments[:i], segment[: m.start()]])
            named: dict[str, str] = {}  # act number → how it is first written
            for a in _ACT.finditer(header):
                named.setdefault(a["number"] or a.group(0), a.group(0))
            if len(named) == 1:
                text = f"{text.rstrip('.')} of {re.sub(r'\s+', ' ', *named.values())}"
        return text
    return None


def _interpreted_act(note: Node, acts: set[str]) -> str | None:
    named = {_act_of(m, acts) for m in _ACT.finditer(note.legal_basis or "")}
    in_corpus = named & acts
    return in_corpus.pop() if len(in_corpus) == 1 else None


# --- Cross-references ---------------------------------------------------------------------

# Lower case, or capitalised at the start of a sentence ("Points (a) and (b) shall not apply").
_ORD = r"(?i:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)"
# A bracketed label; the guidance notes often put a space before it ("Article 13 (1) (d)").
_BR = r"(?:[ ]?\([0-9a-z]+\))"  # [ ], not a bare space: _ELEMENT is verbose
_ART_ITEM = rf"(?:\d+[a-z]*{_BR}*|{_BR}+)"
_PARA_ITEM = rf"(?:\d+[a-z]*{_BR}*)"
_POINT_ITEM = rf"(?:{_BR}+|\d+(?:\.\d+)*)"  # "(b)(iii)", or "4.1" in an annex
_ROMAN_ITEM = r"(?:[IVXLC]+[A-Z]?)"


def _items(item: str) -> str:
    return rf"{item}(?:(?:\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or|to)\s+){item})*"


_ELEMENT = re.compile(
    rf"""
    (?P<sub>{_ORD}\s+(?:sub)?paragraphs?\b(?!\s+\d)|{_ORD}\s+indents?\b
         |[Ss]ubparagraphs?\b(?!\s+\d))
  | [Pp]oints?\s+(?P<points>{_items(_POINT_ITEM)})(?![\w(])
  | [Pp]aragraphs?\s+(?P<paras>{_items(_PARA_ITEM)})(?![\w(])
  | Articles?\s+(?P<arts>{_items(_ART_ITEM)})(?![\w(])
  | Chapters?\s+(?P<chapters>{_items(_ROMAN_ITEM)})\b
  | Annex(?:es)?\s+(?P<annexes>{_items(_ROMAN_ITEM)})\b
  | Parts?\s+(?P<parts>{_items(_ROMAN_ITEM)})\b
    """,
    re.X,
)
_START = re.compile(
    rf"\b(?:{_ORD}\s+)?(?:[Pp]oints?|[Pp]aragraphs?|[Ss]ubparagraphs?|indents?|Articles?|Chapters?"
    r"|Annex(?:es)?|Parts?)\b"
)
_OF = re.compile(r"\s*,?\s+(?:of|in)\s+(?:the\s+)?")
# Between two references that share the act named after the second ("Annex I and Annex IB of
# Regulation (EEC) No 3821/85", "Article 2, points 2 and 3, respectively, of …").
_JOINED = re.compile(r"\s*(?:,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+)")
_THERETO = re.compile(r"\s+thereto\b")
_THIS = re.compile(r"this\s+(?P<this>Regulation|Directive|Article|paragraph|Annex)\b")
_SUFFIX = re.compile(
    rf"(?:\s*,\s*respectively\s*,)?\s*,?\s+(?:of|to|in|under)\s+(?:the\s+(?:repealed\s+)?)?"
    rf"(?:{_THIS.pattern}|{_ACT.pattern})"
)


_ITEM = re.compile(rf"(?:(\bto)\s+)?(\d+(?:\.\d+)+|\d+[a-z]*{_BR}*|{_BR}+|{_ROMAN_ITEM})")


def _split(items: str) -> list[tuple[str, bool]]:
    """'6, 8 and 9 to 11' → [('6', False), ('8', False), ('9', False), ('11', True)].

    The flag marks the end of a range that starts at the item before it.
    """
    return [(m[2].replace(" ", ""), m[1] is not None) for m in _ITEM.finditer(items)]


def _act_of(match: re.Match, acts: set[str]) -> str:
    number = match["number"]
    return number if number in acts else re.sub(r"\s+", " ", match.group(0))


def resolve_refs(tree: Tree, acts: set[str]) -> None:
    """Find the cross-references in every node's own text and resolve them to node IDs.

    A guidance note's legal basis is scanned as part of its first page.
    """
    for note in tree:
        if note.kind == "note":
            note.interprets = _interpreted_act(note, acts)
    for node in tree:
        node.refs = []
        context = _Context.of(tree, node, acts)
        texts = [text for _, text in node.segments]
        if node.kind == "page" and node.label == "p1" and tree[node.parent].legal_basis:
            texts.append(tree[node.parent].legal_basis)
        for text in texts:
            for ref in _scan(text, context, tree, acts):
                if ref not in node.refs:
                    node.refs.append(ref)


@dataclass
class _Context:
    doc_id: str
    node: Node
    article: Node | None
    paragraph: Node | None
    annex: Node | None
    default_act: str  # where a reference without an act points: the amended act, or this one

    @classmethod
    def of(cls, tree: Tree, node: Node, acts: set[str]) -> "_Context":
        article = paragraph = annex = None
        current: Node | None = node
        while current is not None:
            if current.kind == "paragraph" and paragraph is None:
                paragraph = current
            if current.kind == "article":
                article = current
            if current.kind == "annex":
                annex = current
            current = tree[current.parent] if current.parent else None
        default = node.doc_id
        if node.kind in {"note", "page"}:
            note = node if node.kind == "note" else tree[node.parent]
            default = note.interprets or node.doc_id
        if article is not None and article.amends:
            m = _ACT.search(article.amends)
            default = _act_of(m, acts) if m else article.amends
        return cls(node.doc_id, node, article, paragraph, annex, default)


@dataclass
class _Chain:
    """A reference as found in the text: its elements, outermost last, and any act named."""

    start: int
    end: int
    elements: list[re.Match]
    act: str | None  # named after the reference
    this: str | None  # "Regulation" for "of this Regulation", and so on


def _chains(text: str, acts: set[str]) -> list[_Chain]:
    chains: list[_Chain] = []
    end = 0
    for start in _START.finditer(text):
        if start.start() < end:
            continue
        elements: list[re.Match] = []
        pos = start.start()
        while em := _ELEMENT.match(text, pos):
            elements.append(em)
            pos = em.end()
            om = _OF.match(text, pos)
            if om and _ELEMENT.match(text, om.end()):
                pos = om.end()
                continue
            break
        located = [e for e in elements if e.lastgroup != "sub"]
        if not located:
            continue
        act = this = None
        if sm := _SUFFIX.match(text, pos):
            if sm["this"]:
                this = sm["this"]
            else:
                act = _act_of(_ACT.search(sm.group(0)), acts)
            pos = sm.end()
        elif tm := _THERETO.match(text, pos):
            # "… Regulation (EEC) No 3821/85, including in Annex IB thereto"
            named = list(_ACT.finditer(text, 0, start.start()))
            if named:
                act = _act_of(named[-1], acts)
            pos = tm.end()
        end = pos
        chains.append(_Chain(start.start(), pos, located, act, this))
    # A reference joined to the next one by "and" or a comma shares the act named after it.
    for chain, following in zip(reversed(chains[:-1]), reversed(chains[1:]), strict=True):
        if chain.act is None and chain.this is None:
            between = text[chain.end : following.start]
            if _JOINED.fullmatch(between):
                chain.act, chain.this = following.act, following.this
    return chains


def _scan(text: str, ctx: _Context, tree: Tree, acts: set[str]) -> Iterator[Ref]:
    for chain in _chains(text, acts):
        written = text[chain.start : chain.end].strip()
        act = chain.act
        if act is None:
            # In an amending article, "this Regulation" is usually quoted wording of the
            # amended act, so it points there like any reference without an act.
            own = chain.this in {"Regulation", "Directive"} and not (
                ctx.article is not None and ctx.article.amending
            )
            act = ctx.doc_id if own else ctx.default_act
            if ctx.node.kind == "page" and _names_other_act(text, chain.end, act, acts):
                # A guidance note's default act is a guess; "Article 14(4) a of Regulation
                # (EEC) No 3821/85" means another act, so it stays unresolved rather than wrong.
                yield Ref(written, ctx.doc_id, None)
                continue
        if act not in acts:
            yield Ref(written, act, None)
            continue
        targets = _targets(chain.elements, act, ctx, tree, relative_ok=act == ctx.doc_id)
        if not targets:
            yield Ref(written, act, None)
        for target, partial in targets:
            yield Ref(written, act, target, partial)


_SENTENCE_END = re.compile(r"(?<!\be\.g)(?<!\bi\.e)\.(?:\s|$)|;")


def _names_other_act(text: str, start: int, act: str, acts: set[str]) -> bool:
    """Whether the first act named in the rest of the sentence from `start` isn't `act`."""
    end = _SENTENCE_END.search(text, start)
    named = _ACT.search(text, start, end.start() if end else len(text))
    return named is not None and _act_of(named, acts) != act


def _annex_ids(base: str, item: str) -> list[str]:
    """Where a point cited inside an annex can be: '(f)(3)' → base.f.3, '(b)' → base.b or base(b).

    A letter is a part heading ('(f) Maximum tolerances') or a lettered point ('(b) the two
    ends of the link'); the first is joined with a dot and the second in brackets.
    """
    ids = [f"{base}." + ".".join(re.findall(r"[0-9a-z]+(?:\.\d+)*", item))]
    if _LETTER.fullmatch(item):
        ids.append(f"{base}{item}")
    return ids


def _annex_targets(
    elements: list[re.Match], annex: str, scopes: list[str], tree: Tree
) -> list[str]:
    """References into an annex, such as 'point 4 of Part VI' or 'point (c)(4)'.

    `elements` run from the outermost in. An annex cites its parts and points relative to where
    the citing text is, so a path with no part is looked up under each of `scopes`, nearest
    first.
    """
    for element in elements:
        if element.lastgroup == "parts":
            scopes = [f"{annex}.{i}" for i, _ in _split(element["parts"])]
        elif element.lastgroup == "points":
            found = []
            for item, is_range_end in _split(element["points"]):
                match = next((i for s in scopes for i in _annex_ids(s, item) if i in tree), None)
                found.append((match, is_range_end))
            scopes = _ranged(found, tree)
        else:
            return []
    return scopes


def _scopes(node: Node, tree: Tree) -> list[str]:
    """A node's ID and its ancestors' IDs up to the act, nearest first."""
    ids = []
    current: Node | None = node
    while current is not None and current.kind != "act":
        ids.append(current.id)
        current = tree[current.parent] if current.parent else None
    return ids


def _targets(
    elements: list[re.Match], act: str, ctx: _Context, tree: Tree, relative_ok: bool
) -> list[tuple[str, bool]]:
    """Node IDs for a chain such as [points (b)(iii), Article 34(5)], outermost last.

    Each ID comes with a flag that is true when the cited node isn't in the tree and the
    reference was resolved to its nearest ancestor instead.
    """
    outer = elements[-1]
    kind = outer.lastgroup
    inner = list(reversed(elements[:-1]))
    if kind == "annexes":
        annexes = [f"{act}:anx{i}" for i, _ in _split(outer["annexes"])]
        if inner:
            found = [t for a in annexes if a in tree for t in _annex_targets(inner, a, [a], tree)]
            return [(t, False) for t in found]
        return [(a, False) for a in annexes if a in tree]
    if kind == "chapters":
        return [(c, False) for i, _ in _split(outer["chapters"]) if (c := f"{act}:ch{i}") in tree]
    if kind == "arts":
        bases = _article_items(outer["arts"], act, tree)
    elif not relative_ok:
        return []
    elif ctx.annex is not None and ctx.article is None:
        found = _annex_targets(
            list(reversed(elements)), ctx.annex.id, _scopes(ctx.node, tree), tree
        )
        return [(t, False) for t in found]
    elif ctx.article is None or kind == "parts":
        return []
    elif kind == "paras":
        bases = _ranged(
            [(f"{ctx.article.id}({p}){b}", r) for p, b, r in _paragraph_items(outer["paras"])], tree
        )
    else:  # points, relative to the paragraph, or else the article
        scopes = [n.id for n in (ctx.paragraph, ctx.article) if n is not None]
        found = []
        for item, is_range_end in _split(outer["points"]):
            part = item if item.startswith("(") else f"({item})"
            match = next((s + part for s in scopes if s + part in tree), None)
            found.append((match, is_range_end))
        bases = _ranged(found, tree)
        if not inner:
            return [(b, False) for b in bases]
    for element in inner:
        if element.lastgroup == "paras":
            parts = [(f"({p}){b}", r) for p, b, r in _paragraph_items(element["paras"])]
        elif element.lastgroup == "points":
            parts = [
                (i if i.startswith("(") else f"({i})", r) for i, r in _split(element["points"])
            ]
        else:
            continue
        bases = [t for base in bases for t in _ranged([(base + p, r) for p, r in parts], tree)]
    out = []
    for base in bases:
        resolved = _deepest(base, tree)
        if resolved and resolved not in [t for t, _ in out]:
            out.append((resolved, resolved != base))
    return out


def _paragraph_items(items: str) -> list[tuple[str, str, bool]]:
    """'1(a) and 2' → [('1', '(a)', False), ('2', '', False)]; the flag ends a range."""
    out = []
    for item, is_range_end in _split(items):
        m = re.fullmatch(r"(\d+[a-z]*)((?:\([0-9a-z]+\))*)", item)
        if m:
            out.append((m[1], m[2], is_range_end))
    return out


def _article_items(items: str, act: str, tree: Tree) -> list[str]:
    """'6(1) and (2)' → [act:art6(1), act:art6(2)]; '6 to 9' → every article from 6 to 9."""
    out: list[tuple[str, bool]] = []
    last_article = None
    for item, is_range_end in _split(items):
        m = re.fullmatch(r"(\d+[a-z]*)?((?:\([0-9a-z]+\))*)", item)
        if not m:
            continue
        number, brackets = m.groups()
        if number is None:  # "(2)" in "Article 6(1) and (2)"
            if last_article is not None:
                out.append((f"{act}:art{last_article}{brackets}", is_range_end))
            continue
        out.append((f"{act}:art{number}{brackets}", is_range_end))
        last_article = number
    return _ranged(out, tree)


def _ranged(items: list[tuple[str | None, bool]], tree: Tree) -> list[str]:
    """Expand ranges: an item flagged as a range's end brings in everything since the last one.

    "Articles 6 to 9" includes 8a, and "points (a) to (d)" includes (b) and (c). A range whose
    ends aren't both in the tree keeps only its ends. None stands for an item with no node; it
    is dropped, and a range starting there keeps only its end.
    """
    out: list[str] = []
    previous = None
    for node_id, is_range_end in items:
        if node_id is not None:
            if is_range_end and previous is not None:
                out.extend(_between(previous, node_id, tree))
            else:
                out.append(node_id)
        previous = node_id
    return out


def _between(first: str, last: str, tree: Tree) -> list[str]:
    """The nodes after `first` up to and including `last`: articles in document order, and
    anything else among its siblings."""
    if first not in tree or last not in tree or tree[first].kind != tree[last].kind:
        return [last]
    node = tree[first]
    if node.kind == "article":
        siblings = [n.id for n in tree if n.doc_id == node.doc_id and n.kind == "article"]
    elif node.parent == tree[last].parent and node.parent is not None:
        siblings = tree[node.parent].children
    else:
        return [last]
    i, j = siblings.index(first), siblings.index(last)
    return siblings[i + 1 : j + 1] if i < j else [last]


def _deepest(node_id: str, tree: Tree) -> str | None:
    """The node, or its nearest existing ancestor ID; None unless at least the article exists."""
    candidate = node_id
    while candidate not in tree:
        shorter = re.sub(r"\([0-9a-z]+\)$", "", candidate)
        if shorter == candidate:
            return None
        candidate = shorter
    return candidate
