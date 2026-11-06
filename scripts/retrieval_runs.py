"""Run every retrieval configuration on the dev split and write a run log for each.

    retrieval_runs.py                 write runs/retrieval/<name>.json (`make retrieval-runs`)
    retrieval_runs.py --scores        report the reranker scores missing (`make rerank`)
    retrieval_runs.py --scores --run  score them with the local reranker (`make rerank RUN=1`)

Configurations are listed by `runs`; later checkpoints add theirs. A log whose rankings haven't
changed is left as it is, so a rerun doesn't touch the committed logs and tables. Run logs are
committed, and `make tables` builds the results tables from them alone. Format: `rag.runlog`.

Writing the logs never runs a model: vectors come from the embedding cache and reranker scores
from the score cache (`rag.rerank`), and a missing one stops the run before any log is written.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from rag.chunk import STRATEGIES, Chunk
from rag.corpus import CorpusError, load_manifest
from rag.embed import EmbeddingCache, EmbedError, setting_name
from rag.embed_estimate import ConfigError, load_config
from rag.parse import ParseError, Tree, parse_corpus
from rag.questions import validate
from rag.rerank import (
    CachedScorer,
    LocalReranker,
    RerankError,
    ScoreCache,
    missing_pairs,
    score_missing,
)
from rag.rerank import load_config as load_rerank_config
from rag.rerank import setting_name as reranker_name
from rag.retrieve import BM25, RRF, RRF_DEPTH, RRF_K, Expansion, Rerank, Retriever
from rag.runlog import DEPTH, dumps, make_run, same_results
from rag.vector import VectorIndexError, VectorSearch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
QUESTIONS = ROOT / "data" / "questions.v1.jsonl"
OUT = ROOT / "runs" / "retrieval"
EMBEDDING_CONFIG = ROOT / "config" / "embedding.yaml"
EMBEDDINGS = ROOT / "embeddings"
RERANK_CONFIG = ROOT / "config" / "rerank.yaml"
RERANKS = ROOT / "reranks"

# Checkpoint D: BM25 with its defaults over each chunking strategy. Checkpoint E: vector search
# with the configured embedding model, and hybrid search fusing the two by RRF; then each of the
# three again with cross-reference expansion. The model setting (name, revision, device, size) is
# in the vector and hybrid configs, so a new model makes new logs rather than silently new rows.
BM25_CONFIG = {"retriever": "bm25", "k1": 1.5, "b": 0.75}
# Checkpoint G: the reranker reorders hybrid search's top N over paragraph chunks, the setup the
# answers are compared from (docs/DECISIONS.md, #78), for two values of N. The reranker setting
# (name, revision, device) is in the config, as the embedding model's is.
RERANK_DEPTHS = (20, 50)


def configs(model: str) -> list[tuple[str, dict]]:
    """Each run's name prefix and retriever config, before the strategy and expansion."""
    vector = {"retriever": "vector", "model": model}
    hybrid = {**BM25_CONFIG, "retriever": "hybrid", "model": model}
    hybrid |= {"rrf_k": RRF_K, "rrf_depth": RRF_DEPTH}
    return [("bm25", BM25_CONFIG), ("vector", vector), ("hybrid", hybrid)]


def runs(model: str, reranker: str) -> list[tuple[str, str, dict]]:
    plain = [(f"{n}-{s}", s, c) for n, c in configs(model) for s in STRATEGIES]
    refs = [(f"{n}-{s}-refs", s, {**c, "expansion": "references"}) for n, c in configs(model)
            for s in STRATEGIES]  # fmt: skip
    hybrid = dict(configs(model))["hybrid"]
    reranked = [
        (f"hybrid-paragraph-rerank{n}", "paragraph", {**hybrid, "reranker": reranker,
                                                      "rerank_depth": n})
        for n in RERANK_DEPTHS
    ]  # fmt: skip
    return plain + refs + reranked


def retriever_for(
    config: dict,
    chunks: list[Chunk],
    tree: Tree,
    cache: dict,
    vectors: EmbeddingCache,
    scores: ScoreCache | None = None,
) -> Retriever:
    """The retriever a configuration describes; each index is built once per strategy.

    A reranked configuration reads its scores from `scores`, and never runs the model."""
    strategy = chunks[0].strategy

    def bm25() -> Retriever:
        key = ("bm25", strategy, config["k1"], config["b"])
        if key not in cache:
            cache[key] = BM25(chunks, k1=config["k1"], b=config["b"])
        return cache[key]

    def vector() -> Retriever:  # one embedding cache, so one vector index per strategy
        key = ("vector", strategy)
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
    if "rerank_depth" in config:
        if scores is None or config["reranker"] != reranker_name(scores.config):
            raise ValueError(f"no score cache for the reranker {config['reranker']}")
        retriever = Rerank(retriever, CachedScorer(scores), config["rerank_depth"])
    return retriever


def unreranked(config: dict) -> dict:
    """The configuration of the retriever a reranked one wraps."""
    return {k: v for k, v in config.items() if k not in ("reranker", "rerank_depth")}


class Recorder:
    """A stand-in scorer that notes each question and passage it is asked about."""

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def scores(self, query: str, texts) -> list[float]:
        self.pairs += [(query, text) for text in texts]
        return [0.0] * len(texts)


def pairs_to_score(made: list, questions: list) -> list[tuple[str, str]]:
    """Each pair the reranked runs will ask the reranker about.

    `made` holds (config, wrapped retriever) for the reranked runs. Each is searched as
    `make_run` searches it, with a `Recorder` in place of the reranker, so the pairs are exactly
    the ones the run will look up. The deepest run of a wrapped retriever covers the others.
    """
    deepest = max(config["rerank_depth"] for config, _ in made)
    recorder = Recorder()
    for config, retriever in made:
        if config["rerank_depth"] != deepest:
            continue
        reranked = Rerank(retriever, recorder, deepest)
        for question in (q for q in questions if q.split == "dev"):
            reranked.search(question.question, DEPTH)
    return recorder.pairs


def report_scores(pairs: list, cache: ScoreCache, run: bool) -> int:
    """`make rerank`: say how many pairs lack a score and, with `run`, score them locally."""
    config = cache.config
    missing = missing_pairs(pairs, cache)
    print(
        f"Reranker {config.model_id} at {config.revision[:12]} on {config.device}: "
        f"{len(dict.fromkeys(pairs)):,} question and passage pairs, {len(missing):,} not scored "
        f"yet; cache {cache.path.relative_to(ROOT)}."
    )
    if not missing:
        print("Nothing to score. `make retrieval-runs` writes the reranked runs.")
        return 0
    if not run:
        print(
            "Nothing was run. To score them on this machine, free: make rerank RUN=1 (the first "
            "run downloads the model's files, about 1.2 GB, to ~/.cache/huggingface)."
        )
        return 0
    try:
        model = LocalReranker.load(config)
    except (RerankError, OSError) as e:
        print(f"error: cannot load the reranker ({e}); nothing was scored", file=sys.stderr)
        return 1
    started = time.monotonic()

    def progress(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            print(f"  {done:,} of {total:,} scored, {time.monotonic() - started:.0f} s")

    error = score_missing(missing, cache, model, progress)
    if error:
        print(f"error: {error}; the scores so far are kept", file=sys.stderr)
        return 1
    print("Done. `make retrieval-runs` writes the reranked runs.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", action="store_true", help="report missing reranker scores")
    parser.add_argument("--run", action="store_true", help="with --scores, run the reranker")
    args = parser.parse_args(argv)
    if args.run and not args.scores:
        parser.error("--run goes with --scores")
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
        scores = ScoreCache.load(RERANKS, load_rerank_config(RERANK_CONFIG))
    except (ConfigError, EmbedError, RerankError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    # Every run is made before any log is written, so a run that fails (a missing vector, say)
    # leaves all the logs as they were.
    chunked: dict[str, list[Chunk]] = {}
    indexes: dict = {}
    made = []
    to_rerank = []
    for name, strategy, config in runs(setting_name(embedding), reranker_name(scores.config)):
        if args.scores and "rerank_depth" not in config:
            continue
        if strategy not in chunked:
            chunked[strategy] = STRATEGIES[strategy](tree)
        chunks = chunked[strategy]
        try:
            if args.scores:
                inner = retriever_for(unreranked(config), chunks, tree, indexes, vectors)
                to_rerank.append((config, inner))
                continue
            retriever = retriever_for(config, chunks, tree, indexes, vectors, scores)
            config = {**config, "strategy": strategy}
            made.append((name, make_run(name, config, retriever, chunks, questions, ROOT)))
        except (EmbedError, VectorIndexError, RerankError) as e:
            print(f"error: {name}: {e}; no run log was written", file=sys.stderr)
            return 1
    if args.scores:
        try:
            pairs = pairs_to_score(to_rerank, questions)
        except (EmbedError, VectorIndexError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return report_scores(pairs, scores, args.run)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, run in made:
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
