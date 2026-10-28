"""Build the page the owner uses to check every gold answer against the text (#17).

    gold_review.py    write build/gold-review/gold-answer-review.html

The page shows each question of data/questions.v1.jsonl next to the full text of its gold nodes,
and the article or guidance page around them. It is published as a private claude.ai artifact,
where the owner records a verdict per question; the verdicts are read back from there, or
copied out of the page as JSON, and applied to the questions file by hand. The page template is
scripts/gold_review.html. Rebuild and republish the page after editing the questions file.
"""

import json
import sys
from pathlib import Path

from rag.corpus import CorpusError, Document, load_manifest
from rag.parse import Node, ParseError, Tree, parse_corpus

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
QUESTIONS = ROOT / "data" / "questions.v1.jsonl"
TEMPLATE = Path(__file__).with_suffix(".html")
OUT = ROOT / "build" / "gold-review" / "gold-answer-review.html"
_CONTEXT = {"article", "annex", "page"}


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


def page_data(questions: list[dict], tree: Tree, docs: list[Document]) -> dict:
    by_id = {doc.id: doc for doc in docs}
    nodes, contexts = {}, {}
    for question in questions:
        for node_id in question["gold_node_ids"]:
            node = tree[node_id]
            doc = by_id[node.doc_id]
            around = context(tree, node)
            if around is not None:
                contexts[around.id] = {
                    "path": list(around.heading_path),
                    "text": tree.full_text(around.id),
                }
            nodes[node_id] = {
                "kind": node.kind,
                "path": list(node.heading_path),
                "text": tree.full_text(node_id),
                "context_id": around.id if around is not None else None,
                "doc": doc.title,
                "source": doc.celex or "",
                "link": link(doc),
                "published": node.published or doc.published,
                "amended": sorted({a.celex for a in node.amendments}),
            }
    return {"questions": questions, "nodes": nodes, "contexts": contexts}


def main() -> int:
    try:
        docs = load_manifest(MANIFEST)
        tree = parse_corpus(docs, ROOT)
    except (CorpusError, ParseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if tree.errors:
        print("\n".join(f"error: {e}" for e in tree.errors), file=sys.stderr)
        return 1
    lines = QUESTIONS.read_text(encoding="utf-8").split("\n")
    questions = [json.loads(line) for line in lines if line.strip()]
    missing = sorted({i for q in questions for i in q["gold_node_ids"] if i not in tree})
    if missing:
        print(f"error: gold nodes not in the corpus: {', '.join(missing)}", file=sys.stderr)
        return 1
    # "<" is escaped so that no string in the data can close the page's <script> element.
    data = json.dumps(page_data(questions, tree, docs), ensure_ascii=False).replace("<", "\\u003c")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", data), "utf-8")
    print(f"{OUT.relative_to(ROOT)}: {len(questions)} questions, {len(data) // 1024} KB of data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
