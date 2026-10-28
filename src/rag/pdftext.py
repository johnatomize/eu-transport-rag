"""Extract the text of a guidance-note PDF, page by page.

pdfminer.six groups characters into text boxes, which become the page's segments. Cleaning is
deterministic, so the same file always gives the same text:

- Footnote markers are dropped: a digit much smaller than the text around it on its line
  ("165/2014¹" would otherwise read `165/20141`). The footnote's own number at the foot of the
  page goes the same way; its text stays.
- A text box in the top or bottom margin that repeats on every page of a note (a running
  header or footer) is kept on page 1 and dropped from the others. A box in the margin holding
  only the page's own number is dropped.
- Whitespace is collapsed, and a word split over two lines by a hyphen is joined with the
  hyphen kept ("pre-signed").

Text inside figures (the labels of a diagram) is read like any other text. Images, and text
laid out by a table or form, are not understood; each page lists them in `Page.unparsed` so that
nothing is lost silently. See docs/DECISIONS.md.
"""

import io
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from statistics import median

from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LAParams,
    LTAnno,
    LTChar,
    LTCurve,
    LTFigure,
    LTImage,
    LTPage,
    LTTextContainer,
    LTTextLine,
)

# A digit at most this fraction of its line's usual character size is a footnote marker. The
# markers in the corpus are 6.5–8 pt next to 10–12 pt text; ordinary digits are full size.
_MARKER_RATIO = 0.75
# A page drawn with more lines, curves and rectangles than this is a table or a form: the most
# any other page in the corpus has is 31 (underlines, boxes and the timeline diagrams), the form
# in guidance-9 has 126 and the nation-code table in guidance-10 has 686.
_TABLE_SHAPES = 100
# Running headers and footers, and page numbers, are only looked for in boxes that reach into
# the top or bottom margin, given as a fraction of the page's height.
_MARGIN = 0.15

_LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"})


@dataclass
class Page:
    number: int  # 1-based position in the PDF
    segments: list[str] = field(default_factory=list)  # one per text box, in reading order
    unparsed: list[str] = field(default_factory=list)


def read_pdf(data: bytes) -> list[Page]:
    """The cleaned text of every page."""
    pages: list[Page] = []
    raw: list[list[_Box]] = []
    laparams = LAParams(all_texts=True)  # also group the text inside figures into boxes
    for number, layout in enumerate(extract_pages(io.BytesIO(data), laparams=laparams), 1):
        page = Page(number)
        raw.append(list(_boxes(layout)))
        images = sum(1 for e in _walk(layout) if isinstance(e, LTImage))
        if images:
            page.unparsed.append(f"{_count(images, 'image')}, not extracted")
        shapes = sum(1 for e in _walk(layout) if isinstance(e, LTCurve))
        if shapes > _TABLE_SHAPES:
            page.unparsed.append(f"drawn as a diagram, table or form ({shapes} lines and boxes)")
        pages.append(page)
    running = _running(raw)
    for page, boxes in zip(pages, raw, strict=True):
        for box in boxes:
            if box.margin and (
                box.lines == (str(page.number),) or page.number > 1 and box in running
            ):
                continue
            page.segments.append(_join(box.lines))
    return pages


@dataclass(frozen=True)
class _Box:
    lines: tuple[str, ...]
    margin: bool  # reaches into the top or bottom margin


def _boxes(layout: LTPage) -> Iterator[_Box]:
    top, bottom = layout.height * (1 - _MARGIN), layout.height * _MARGIN
    for element in _walk(layout):
        if isinstance(element, LTTextContainer):
            lines = tuple(line for line in map(_line, _lines(element)) if line)
            if lines:
                yield _Box(lines, element.y1 > top or element.y0 < bottom)


def _walk(element) -> Iterator:
    """Every element on a page in reading order, looking inside figures but not text boxes."""
    for child in element:
        yield child
        if isinstance(child, LTFigure):
            yield from _walk(child)


def _running(pages: list[list[_Box]]) -> set[_Box]:
    """Boxes in the margin that appear on every page of a note of two or more pages."""
    if len(pages) < 2:
        return set()
    return set.intersection(*({box for box in boxes if box.margin} for boxes in pages))


def _lines(container: LTTextContainer) -> Iterator[LTTextLine]:
    if isinstance(container, LTTextLine):
        yield container
    else:
        yield from (line for line in container if isinstance(line, LTTextLine))


def _line(line: LTTextLine) -> str:
    chars = [c for c in line if isinstance(c, LTChar) and c.get_text().strip()]
    usual = median(c.size for c in chars) if chars else 0
    pieces = []
    for item in line:
        if isinstance(item, LTChar):
            text = item.get_text()
            if text.isdigit() and item.size <= _MARKER_RATIO * usual:
                continue
            pieces.append(text)
        elif isinstance(item, LTAnno):
            pieces.append(item.get_text())
    return _clean("".join(pieces))


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text.translate(_LIGATURES))
    return re.sub(r"\s+", " ", text).strip()


def _join(lines: tuple[str, ...]) -> str:
    """Lines of one text box as one string; "pre-" + "signed" → "pre-signed"."""
    out = lines[0]
    for line in lines[1:]:
        if re.search(r"[A-Za-z]-$", out) and line[:1].islower():
            out += line
        else:
            out += " " + line
    return out


def _count(n: int, name: str) -> str:
    return f"{n} {name}" if n == 1 else f"{n} {name}s"
