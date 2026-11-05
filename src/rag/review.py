"""Helpers shared by the owner's review pages: the gold answers (#17) and faithfulness (#82)."""

from rag.corpus import Document
from rag.parse import Node, Tree

_CONTEXT = {"article", "annex", "page"}
# claude.ai wraps an artifact in its own document; a file opened in a browser needs one.
DOCUMENT = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
    "<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>"
    "</head><body>\n{page}\n</body></html>\n"
)


def link(doc: Document) -> str:
    """Where the owner reads the document: EUR-Lex for an act, the Commission's PDF for a note."""
    if doc.celex:
        return f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{doc.celex}"
    return doc.url


def context(tree: Tree, node: Node) -> Node | None:
    """The article, annex or guidance page around a node, or None if the node is one itself."""
    current = node
    while current.kind not in _CONTEXT:
        if current.parent is None:
            return None
        current = tree[current.parent]
    return None if current is node else current
