"""Parse every document in corpus/manifest.yaml into its node tree and print a report.

    parse_corpus.py               print node counts, references and anything unparsed (`make parse`)
    parse_corpus.py --out FILE    also write every node as one JSON line, for inspection
    parse_corpus.py --snapshot    rewrite tests/snapshots/node_ids.txt after a deliberate ID change

Exits with status 1 on a duplicate node ID, without writing the snapshot.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from rag.corpus import CorpusError, load_manifest
from rag.parse import KINDS, ParseError, Tree, parse_corpus

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
SNAPSHOT = ROOT / "tests" / "snapshots" / "node_ids.txt"


def _count(n: int, kind: str) -> str:
    name = kind.replace("_", " ")
    return f"{n} {name}" if n == 1 else f"{n} {name}es" if name == "annex" else f"{n} {name}s"


def report(tree: Tree, docs: list[str], acts: set[str]) -> list[str]:
    """Counts and references per document, then everything that needs a look.

    A reference is outside the corpus when it names an act that isn't in it. A guidance note's
    reference that names no act, in a note that interprets no single act, is unresolved.
    """
    lines = []
    for doc in docs:
        nodes = [n for n in tree if n.doc_id == doc]
        kinds = Counter(n.kind for n in nodes)
        counts = ", ".join(_count(kinds[k], k) for k in KINDS if kinds[k])
        refs = [r for n in nodes for r in n.refs]
        resolved = sum(1 for r in refs if r.target)
        outside = sum(1 for r in refs if r.act not in acts and r.act != doc)
        unresolved = len(refs) - resolved - outside
        amending = [n.id for n in nodes if n.amending]
        lines.append(f"{doc:16} {counts}")
        lines.append(
            f"{'':16} references: {resolved} resolved, {outside} to acts outside the corpus, "
            f"{unresolved} unresolved"
        )
        if amending:
            lines.append(f"{'':16} amending articles: {', '.join(amending)}")
        if nodes and nodes[0].kind == "note":
            lines.append(f"{'':16} interprets: {nodes[0].interprets or '-'}")
    unresolved = [
        f"  {n.id}: {r.text!r} ({r.act})"
        for n in tree
        for r in n.refs
        if r.target is None and (r.act in acts or r.act == n.doc_id)
    ]
    partial = [f"  {n.id}: {r.text!r} → {r.target}" for n in tree for r in n.refs if r.partial]
    for title, items in (
        ("unresolved references", unresolved),
        ("references to a node the parser has no node for, resolved to its parent", partial),
        ("unparsed", [f"  {u}" for u in tree.unparsed]),
        ("errors", [f"  {e}" for e in tree.errors]),
    ):
        if items:
            lines.append(f"{title} ({len(items)}):")
            lines.extend(items)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, metavar="FILE", help="write the nodes as JSON lines")
    parser.add_argument("--snapshot", action="store_true", help="rewrite the node ID snapshot")
    args = parser.parse_args(argv)

    try:
        docs = load_manifest(MANIFEST)
        tree = parse_corpus(docs, ROOT)
    except (CorpusError, ParseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    acts = {doc.id for doc in docs if doc.kind == "act"}
    print("\n".join(report(tree, [doc.id for doc in docs], acts)))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for node in tree:
                f.write(json.dumps(node.to_dict(), ensure_ascii=False) + "\n")
        print(f"wrote {len(tree.nodes)} nodes to {args.out}")
    if tree.errors:
        return 1
    if args.snapshot:
        SNAPSHOT.write_text("".join(f"{n.id}\n" for n in tree), encoding="utf-8")
        print(f"wrote {len(tree.nodes)} node IDs to {SNAPSHOT.relative_to(ROOT)}")
    print(f"all {len(docs)} documents parsed: {len(tree.nodes)} nodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
