"""Cut the parsed corpus into chunks with every strategy and report their sizes (`make chunks`).

    chunk_corpus.py               print the chunk count and length in words for each strategy
    chunk_corpus.py --out DIR     also write DIR/chunks-<strategy>.jsonl, for reading by hand

Length is the words of the indexed text: prefix and body.
"""

import argparse
import json
import sys
from pathlib import Path

from rag.chunk import STRATEGIES, Chunk
from rag.corpus import CorpusError, load_manifest
from rag.parse import ParseError, parse_corpus

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"


def report(name: str, chunks: list[Chunk]) -> str:
    if not chunks:
        return f"{name:15}    0 chunks"
    lengths = [len(chunk.text.split()) for chunk in chunks]
    mean = sum(lengths) / len(lengths)
    return (
        f"{name:15} {len(chunks):4} chunks, words min {min(lengths)}, "
        f"mean {mean:.0f}, max {max(lengths)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, metavar="DIR", help="write the chunks as JSON lines")
    args = parser.parse_args(argv)

    try:
        tree = parse_corpus(load_manifest(MANIFEST), ROOT)
    except (CorpusError, ParseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for name, strategy in STRATEGIES.items():
        chunks = strategy(tree)
        print(report(name, chunks))
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            path = args.out / f"chunks-{name}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            print(f"{'':15} wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
