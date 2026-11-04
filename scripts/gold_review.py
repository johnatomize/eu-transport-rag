"""Build the page the owner uses to check every gold answer against the text (#17).

    gold_review.py              write build/gold-review/gold-answer-review.html, to open as a file
    gold_review.py --artifact   write the page without <html> and <head>, for a claude.ai artifact

The page shows each question of data/questions.v1.jsonl next to the full text of its gold nodes,
and the article or guidance page around them. The owner records a verdict per question. Opened
as a file, the page keeps verdicts in the browser and saves them to a JSON file; published as a
private claude.ai artifact, it keeps them in the artifact. Either way, Claude applies them to the
questions file, and questions already marked checked there show as correct. The page template is
scripts/gold_review.html. Rebuild and republish the page after editing the questions file.
"""

import argparse
import json
import sys
from pathlib import Path

from rag.corpus import CorpusError, Document, load_manifest
from rag.parse import ParseError, Tree, lead_in, parse_corpus
from rag.review import DOCUMENT, context, link

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
QUESTIONS = ROOT / "data" / "questions.v1.jsonl"
TEMPLATE = Path(__file__).with_suffix(".html")
OUT = ROOT / "build" / "gold-review" / "gold-answer-review.html"


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
                "lead_in": lead_in(tree, node, around),
                "context_id": around.id if around is not None else None,
                "doc": doc.title,
                "source": doc.celex or "",
                "link": link(doc),
                "published": node.published or doc.published,
                "amended": sorted({a.celex for a in node.amendments}),
            }
    return {"questions": questions, "nodes": nodes, "contexts": contexts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", action="store_true", help="write the page for claude.ai")
    args = parser.parse_args()
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
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", data)
    if not args.artifact:
        page = DOCUMENT.replace("{page}", page)
    OUT.write_text(page, encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: {len(questions)} questions, {len(data) // 1024} KB of data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
