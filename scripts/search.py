"""Print the best chunks for a typed question, for poking at retrieval by hand (`make search`).

search.py "How long may I drive in a day?"                     BM25 over paragraph chunks
search.py --strategy window-200-50 -k 10 "daily rest on a ferry"
search.py --refs "Who counts as a self-employed driver?"         with cross-reference expansion
"""

import argparse
import sys
from pathlib import Path

from rag.chunk import STRATEGIES
from rag.corpus import CorpusError, load_manifest
from rag.parse import ParseError, parse_corpus
from rag.retrieve import BM25, Expansion, Retriever

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--strategy", choices=STRATEGIES, default="paragraph")
    parser.add_argument("-k", type=int, default=5, help="how many chunks to show (default 5)")
    parser.add_argument("--refs", action="store_true", help="add the chunks that hits refer to")
    args = parser.parse_args(argv)

    try:
        tree = parse_corpus(load_manifest(MANIFEST), ROOT)
    except (CorpusError, ParseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    chunks = STRATEGIES[args.strategy](tree)
    index: Retriever = BM25(chunks)
    if args.refs:
        index = Expansion(index, chunks, tree)
    for rank, hit in enumerate(index.search(args.question, args.k), 1):
        body = " ".join(hit.chunk.body.split())
        via = f"  (referred to by {hit.via})" if hit.via else ""
        print(f"{rank}. {hit.score:6.2f}  {hit.chunk.id}{via}")
        print(f"   {(hit.chunk.prefix.splitlines() or [''])[0]}")
        print(f"   {body[:300]}{'…' if len(body) > 300 else ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
