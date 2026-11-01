"""Load chunks and their cached vectors into the local pgvector database (`make vector-index`).

vector_index.py                            every chunking strategy
vector_index.py --strategy paragraph       one strategy

Each strategy's rows for the model in config/embedding.yaml are replaced in one step, so running
it again gives the same rows. Vectors come from the cache in embeddings/ only; nothing is sent to
AWS. A strategy with any chunk not cached yet is refused and left as it was: `make embed` says
what sending the rest would cost.
"""

import argparse
import sys
from pathlib import Path

import psycopg

from rag.chunk import STRATEGIES
from rag.corpus import CorpusError, load_manifest
from rag.db import DatabaseError, connect
from rag.embed import EmbeddingCache, EmbedError, setting_name
from rag.embed_estimate import ConfigError, load_config
from rag.parse import ParseError, parse_corpus
from rag.vector import TABLE, VectorIndexError, load_index

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "embedding.yaml"
CACHE = ROOT / "embeddings"
MANIFEST = ROOT / "corpus" / "manifest.yaml"
ENV = ROOT / ".env"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--strategy", choices=STRATEGIES, action="append", help="default: every strategy"
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(CONFIG)
        cache = EmbeddingCache.load(CACHE, config)
        tree = parse_corpus(load_manifest(MANIFEST), ROOT)
        conn = connect(ENV)
    except (ConfigError, EmbedError, CorpusError, ParseError, DatabaseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    failed = False
    with conn:
        for name in args.strategy or STRATEGIES:
            try:
                rows = load_index(conn, STRATEGIES[name](tree), cache)
            except (EmbedError, VectorIndexError, psycopg.Error) as e:
                print(f"error: {name}: {e}", file=sys.stderr)
                failed = True
                continue
            print(f"{name}: {rows} chunks in {TABLE} for {setting_name(config)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
