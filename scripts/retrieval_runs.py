"""Run every retrieval configuration on the dev split and write a run log for each.

    retrieval_runs.py     write runs/retrieval/<name>.json (`make retrieval-runs`)

Configurations are listed by `runs`; later checkpoints add theirs. A log whose rankings haven't
changed is left as it is, so a rerun doesn't touch the committed logs and tables. Run logs are
committed, and `make tables` builds the results tables from them alone. Format: `rag.runlog`.
"""

import json
import sys
from pathlib import Path

from rag.chunk import STRATEGIES, Chunk
from rag.corpus import CorpusError, load_manifest
from rag.embed import EmbeddingCache, EmbedError, setting_name
from rag.embed_estimate import ConfigError, load_config
from rag.parse import ParseError, Tree, parse_corpus
from rag.questions import validate
from rag.retrieve import BM25, RRF, RRF_DEPTH, RRF_K, Expansion, Retriever
from rag.runlog import dumps, make_run, same_results
from rag.vector import VectorSearch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
QUESTIONS = ROOT / "data" / "questions.v1.jsonl"
OUT = ROOT / "runs" / "retrieval"
EMBEDDING_CONFIG = ROOT / "config" / "embedding.yaml"
EMBEDDINGS = ROOT / "embeddings"

# Checkpoint D: BM25 with its defaults over each chunking strategy. Checkpoint E: vector search
# with the configured embedding model, and hybrid search fusing the two by RRF; then each of the
# three again with cross-reference expansion. The model setting (name, revision, device, size) is
# in the vector and hybrid configs, so a new model makes new logs rather than silently new rows.
BM25_CONFIG = {"retriever": "bm25", "k1": 1.5, "b": 0.75}


def configs(model: str) -> list[tuple[str, dict]]:
    """Each run's name prefix and retriever config, before the strategy and expansion."""
    vector = {"retriever": "vector", "model": model}
    hybrid = {**BM25_CONFIG, "retriever": "hybrid", "model": model}
    hybrid |= {"rrf_k": RRF_K, "rrf_depth": RRF_DEPTH}
    return [("bm25", BM25_CONFIG), ("vector", vector), ("hybrid", hybrid)]


def runs(model: str) -> list[tuple[str, str, dict]]:
    plain = [(f"{n}-{s}", s, c) for n, c in configs(model) for s in STRATEGIES]
    refs = [(f"{n}-{s}-refs", s, {**c, "expansion": "references"}) for n, c in configs(model)
            for s in STRATEGIES]  # fmt: skip
    return plain + refs


def retriever_for(
    config: dict, chunks: list[Chunk], tree: Tree, cache: dict, vectors: EmbeddingCache
) -> Retriever:
    """The retriever a configuration describes; each index is built once per strategy."""
    strategy = chunks[0].strategy

    def bm25() -> Retriever:
        key = ("bm25", strategy, config["k1"], config["b"])
        if key not in cache:
            cache[key] = BM25(chunks, k1=config["k1"], b=config["b"])
        return cache[key]

    def vector() -> Retriever:
        key = ("vector", strategy, config["model"])
        if key not in cache:
            cache[key] = VectorSearch(chunks, vectors)
        return cache[key]

    kind = config["retriever"]
    if kind == "bm25":
        retriever = bm25()
    elif kind == "vector":
        retriever = vector()
    elif kind == "hybrid":
        retriever = RRF(chunks, [bm25(), vector()], config["rrf_k"], config["rrf_depth"])
    else:
        raise ValueError(f"unknown retriever {kind}")
    if config.get("expansion") == "references":
        retriever = Expansion(retriever, chunks, tree)
    return retriever


def main() -> int:
    try:
        tree = parse_corpus(load_manifest(MANIFEST), ROOT)
    except (CorpusError, ParseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    questions, errors = validate(QUESTIONS.read_text(encoding="utf-8"), tree)
    if errors:
        print(f"error: {QUESTIONS.name} is not valid; run `make questions`", file=sys.stderr)
        return 1
    try:
        embedding = load_config(EMBEDDING_CONFIG)
        vectors = EmbeddingCache.load(EMBEDDINGS, embedding)
    except (ConfigError, EmbedError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    chunked: dict[str, list[Chunk]] = {}
    indexes: dict = {}
    for name, strategy, config in runs(setting_name(embedding)):
        if strategy not in chunked:
            chunked[strategy] = STRATEGIES[strategy](tree)
        chunks = chunked[strategy]
        try:
            retriever = retriever_for(config, chunks, tree, indexes, vectors)
            config = {**config, "strategy": strategy}
            run = make_run(name, config, retriever, chunks, questions, ROOT)
        except EmbedError as e:
            print(f"error: {name}: {e}", file=sys.stderr)
            return 1
        path = OUT / f"{name}.json"
        where = path.relative_to(ROOT)
        if path.exists() and same_results(json.loads(path.read_text(encoding="utf-8")), run):
            print(f"{name:25} unchanged, {where} kept")
            continue
        path.write_text(dumps(run), encoding="utf-8")
        print(f"{name:25} {len(run['questions'])} questions → {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
