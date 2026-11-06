"""Run every retrieval configuration on the dev split and write a run log for each.

    retrieval_runs.py                 write runs/retrieval/<name>.json (`make retrieval-runs`)
    retrieval_runs.py --scores        report the reranker scores missing (`make rerank`)
    retrieval_runs.py --scores --run  score them with the local reranker (`make rerank RUN=1`),
                                      logging how long it took to runs/rerank-local/
    retrieval_runs.py --scores --with jev|haiku
                                      what asking a hosted reranker would send and cost
                                      (`make rerank WITH=…`); with --run it pays, within the
                                      approved cap, and logs the run to runs/rerank/

Configurations are listed by `runs`; later checkpoints add theirs. A log whose rankings haven't
changed is left as it is, so a rerun doesn't touch the committed logs and tables. Run logs are
committed, and `make tables` builds the results tables from them alone. Format: `rag.runlog`.

Writing the logs never runs a model: vectors come from the embedding cache and reranker scores
from the score cache (`rag.rerank`), and a missing one stops the run before any log is written.
"""

import argparse
import dataclasses
import datetime as dt
import json
import platform
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from rag.chunk import STRATEGIES, Chunk
from rag.corpus import CorpusError, load_manifest
from rag.embed import EmbeddingCache, EmbedError, setting_name
from rag.embed_estimate import ConfigError, load_config
from rag.generate import (
    AnthropicClient,
    GenerateError,
    RunResult,
    new_run,
    read_api_key,
    send_all,
    write_run_log,
)
from rag.generation_config import GenerationConfigError, load_generation_config
from rag.hosted_rerank import (
    JEV_FIELDS,
    LISTWISE_BLANK,
    LISTWISE_FIELDS,
    LISTWISE_SCHEMA,
    HostedCache,
    HostedConfig,
    HostedEstimate,
    JevClient,
    JevScorer,
    ListwiseScorer,
    Lot,
    cache_orders,
    estimate_jev,
    estimate_listwise,
    holds,
    hosted_run_log,
    jev_all,
    jev_key,
    listwise_calls,
    listwise_head,
    listwise_key,
    listwise_verdict,
    load_hosted_config,
    read_typesafe_key,
)
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
from rag.retrieve import BM25, RRF, RRF_DEPTH, RRF_K, Expansion, Rerank, Retriever, Scorer
from rag.runlog import DEPTH, dumps, git_state, make_run, same_results
from rag.spend import SpendLogError, spending, spending_report
from rag.vector import VectorIndexError, VectorSearch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
QUESTIONS = ROOT / "data" / "questions.v1.jsonl"
OUT = ROOT / "runs" / "retrieval"
EMBEDDING_CONFIG = ROOT / "config" / "embedding.yaml"
EMBEDDINGS = ROOT / "embeddings"
RERANK_CONFIG = ROOT / "config" / "rerank.yaml"
RERANKS = ROOT / "reranks"
LOCAL_RUNS = ROOT / "runs" / "rerank-local"  # free local scoring runs, for their timing
HOSTED_CONFIG = ROOT / "config" / "rerank-hosted.yaml"
GENERATION_CONFIG = ROOT / "config" / "generation.yaml"
HOSTED_RUNS = ROOT / "runs" / "rerank"  # paid runs, counted by rag.spend
ENV = ROOT / ".env"

# Checkpoint D: BM25 with its defaults over each chunking strategy. Checkpoint E: vector search
# with the configured embedding model, and hybrid search fusing the two by RRF; then each of the
# three again with cross-reference expansion. The model setting (name, revision, device, size) is
# in the vector and hybrid configs, so a new model makes new logs rather than silently new rows.
BM25_CONFIG = {"retriever": "bm25", "k1": 1.5, "b": 0.75}
# Checkpoint G: the reranker reorders hybrid search's top N over paragraph chunks, the setup the
# answers are compared from (docs/DECISIONS.md, #78), for two values of N. The reranker setting
# (name, revision, device) is in the config, as the embedding model's is.
RERANK_DEPTHS = (20, 50)
# Ways to improve on reranking the top 20 (#99), each changing one thing from it, fixed before
# any was measured: blending the reranker's order with search's by RRF, at hybrid search's own
# constant; the reranker asked with another instruction from config/rerank.yaml; and the chunks
# the reranked hits cite added after them, as cross-reference expansion adds them.
RERANK_VARIANTS = (
    ("blend", {"rerank_blend": RRF_K}),
    ("needed", {"rerank_instruction": "needed"}),
    ("refs", {"expansion_after_rerank": "references"}),
)
# Hosted rerankers (#99), paid, each in place of Qwen over the same top 20: Jev scoring each
# passage on its own, and Haiku reading them together (rag.hosted_rerank).
HOSTED = ("jev", "haiku")
# Settings of a reranked run that the retriever it wraps doesn't have.
RERANK_SETTINGS = ("reranker", "rerank_depth", "rerank_blend", "rerank_instruction",
                   "expansion_after_rerank")  # fmt: skip


def configs(model: str) -> list[tuple[str, dict]]:
    """Each run's name prefix and retriever config, before the strategy and expansion."""
    vector = {"retriever": "vector", "model": model}
    hybrid = {**BM25_CONFIG, "retriever": "hybrid", "model": model}
    hybrid |= {"rrf_k": RRF_K, "rrf_depth": RRF_DEPTH}
    return [("bm25", BM25_CONFIG), ("vector", vector), ("hybrid", hybrid)]


def runs(model: str, reranker: str, hosted: dict[str, str]) -> list[tuple[str, str, dict]]:
    """Every run: its name, strategy and config. `hosted` names each hosted reranker's setting."""
    plain = [(f"{n}-{s}", s, c) for n, c in configs(model) for s in STRATEGIES]
    refs = [(f"{n}-{s}-refs", s, {**c, "expansion": "references"}) for n, c in configs(model)
            for s in STRATEGIES]  # fmt: skip
    hybrid = dict(configs(model))["hybrid"]
    reranked = [
        (f"hybrid-paragraph-rerank{n}", "paragraph", {**hybrid, "reranker": reranker,
                                                      "rerank_depth": n})
        for n in RERANK_DEPTHS
    ]  # fmt: skip
    top20 = {**hybrid, "reranker": reranker, "rerank_depth": 20}
    variants = [(f"hybrid-paragraph-rerank20-{v}", "paragraph", {**top20, **change})
                for v, change in RERANK_VARIANTS]  # fmt: skip
    others = [(f"hybrid-paragraph-{h}20", "paragraph", {**top20, "reranker": hosted[h]})
              for h in HOSTED]  # fmt: skip
    return plain + refs + reranked + variants + others


def retriever_for(
    config: dict,
    chunks: list[Chunk],
    tree: Tree,
    cache: dict,
    vectors: EmbeddingCache,
    scorer_for: Callable[[dict], Scorer] | None = None,
) -> Retriever:
    """The retriever a configuration describes; each index is built once per strategy.

    A reranked configuration gets its scorer from `scorer_for`, which reads cached scores and
    never runs a model or calls an API."""
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
        if scorer_for is None:
            raise ValueError(f"no scores for the reranker {config['reranker']}")
        scorer = scorer_for(config)
        retriever = Rerank(retriever, scorer, config["rerank_depth"], config.get("rerank_blend"))
    if config.get("expansion_after_rerank") == "references":
        retriever = Expansion(retriever, chunks, tree)
    return retriever


def cached_scorers(
    scores: ScoreCache, hosted: HostedConfig, caches: dict[str, HostedCache]
) -> Callable[[dict], Scorer]:
    """The scorer a reranked configuration names, reading only cached scores."""

    def scorer_for(config: dict) -> Scorer:
        name = config["reranker"]
        if name == reranker_name(scores.config):
            instruction = scores.config.instruction_named(config.get("rerank_instruction"))
            return CachedScorer(scores, instruction)
        if name == hosted.jev.name:
            return JevScorer(caches["jev"], hosted.jev)
        if name == hosted.listwise.name:
            return ListwiseScorer(caches["haiku"], hosted.listwise)
        raise ValueError(f"no score cache for the reranker {name}")

    return scorer_for


def unreranked(config: dict) -> dict:
    """The configuration of the retriever a reranked one wraps."""
    return {k: v for k, v in config.items() if k not in RERANK_SETTINGS}


class Recorder:
    """A stand-in scorer that notes each question and passage it is asked about."""

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []
        self.lists: list[tuple[str, tuple[str, ...]]] = []

    def scores(self, query: str, texts) -> list[float]:
        self.pairs += [(query, text) for text in texts]
        self.lists.append((query, tuple(texts)))
        return [0.0] * len(texts)


def pairs_to_score(made: list, questions: list) -> dict[str | None, list[tuple[str, str]]]:
    """Each pair the reranked runs will ask the reranker about, by the instruction's name.

    `made` holds (config, wrapped retriever) for the reranked runs. Each is searched as
    `make_run` searches it, with a `Recorder` in place of the reranker, so the pairs are exactly
    the ones the run will look up. For each instruction, the deepest run covers the others: they
    all wrap the same retriever. The main instruction's name is None.
    """
    deepest: dict[str | None, int] = {}
    for config, _ in made:
        name = config.get("rerank_instruction")
        deepest[name] = max(deepest.get(name, 0), config["rerank_depth"])
    pairs: dict[str | None, list[tuple[str, str]]] = {}
    for config, retriever in made:
        name = config.get("rerank_instruction")
        if name in pairs or config["rerank_depth"] != deepest[name]:
            continue
        recorder = Recorder()
        reranked = Rerank(retriever, recorder, deepest[name])
        for question in (q for q in questions if q.split == "dev"):
            reranked.search(question.question, DEPTH)
        pairs[name] = recorder.pairs
    return pairs


def report_scores(pairs: dict[str | None, list], cache: ScoreCache, run: bool) -> int:
    """`make rerank`: say how many pairs lack a score and, with `run`, score them locally.

    `pairs` holds each instruction's pairs by its name (None for the main instruction). A run
    writes a log of how long scoring took to `LOCAL_RUNS`, for the time each question costs.
    """
    config = cache.config
    print(
        f"Reranker {config.model_id} at {config.revision[:12]} on {config.device}, "
        f"cache {cache.path.relative_to(ROOT)}:"
    )
    missing = {}
    for name, found in pairs.items():
        instruction = config.instruction_named(name)
        missing[name] = missing_pairs(found, cache, instruction)
        print(
            f"  {name or 'main'} instruction: {len(dict.fromkeys(found)):,} question and passage "
            f"pairs, {len(missing[name]):,} not scored yet"
        )
    if not any(missing.values()):
        print("Nothing to score. `make retrieval-runs` writes the reranked runs.")
        return 0
    if not run:
        print(
            "Nothing was run. To score them on this machine, free: make rerank RUN=1 (the first "
            "run downloads the model's files, about 1.2 GB, to ~/.cache/huggingface)."
        )
        return 0
    started = time.monotonic()
    try:
        model = LocalReranker.load(config)
    except (RerankError, OSError) as e:
        print(f"error: cannot load the reranker ({e}); nothing was scored", file=sys.stderr)
        return 1
    loaded = time.monotonic()
    for name, todo in missing.items():
        if not todo:
            continue
        print(f"Scoring {len(todo):,} pairs with the {name or 'main'} instruction:")
        begun = time.monotonic()

        def progress(done: int, total: int, begun: float = begun) -> None:
            if done % 100 == 0 or done == total:
                print(f"  {done:,} of {total:,} scored, {time.monotonic() - begun:.0f} s")

        instruction = config.instruction_named(name)
        error = score_missing(todo, cache, model, progress, instruction)
        log = write_local_log(config, name, todo, cache, loaded - started, time.monotonic() - begun,
                              error)  # fmt: skip
        print(f"  log: {log.relative_to(ROOT)}")
        if error:
            print(f"error: {error}; the scores so far are kept", file=sys.stderr)
            return 1
    print("Done. `make retrieval-runs` writes the reranked runs.")
    return 0


def write_local_log(
    config,
    name: str | None,
    pairs: list[tuple[str, str]],
    cache: ScoreCache,
    load_seconds: float,
    seconds: float,
    error: str | None,
) -> Path:
    """A free local run's log: what was scored, on what, and how long it took."""
    instruction = config.instruction_named(name)
    scored = [cache.get(q, t, instruction) for q, t in pairs]
    done = [e for e in scored if e is not None]
    commit, dirty = git_state(ROOT)
    created = datetime.now(UTC)
    log = {
        "run": {
            "created": created.isoformat(timespec="seconds"),
            "commit": commit,
            "dirty": dirty,
        },
        "config": {
            "reranker": reranker_name(config),
            "model_id": config.model_id,
            "revision": config.revision,
            "instruction_name": name or "main",
            "instruction": instruction,
        },
        "environment": {
            "device": config.device,
            "machine": platform.machine(),
            "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            **{lib: _version(lib) for lib in ("torch", "transformers", "tokenizers")},
        },
        "result": {
            "questions": len({q for q, _ in pairs}),
            "pairs": len(done),
            "tokens": sum(e.tokens for e in done),
            "load_seconds": round(load_seconds, 1),
            "seconds": round(seconds, 1),
        },
        "error": error,
    }
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    path = LOCAL_RUNS / f"{created.strftime('%Y-%m-%dT%H-%M-%SZ')}-{name or 'main'}.json"
    path.write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    return path


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def lots_for(made: list, questions: list) -> list[Lot]:
    """Each dev question with the passages a hosted reranker will be asked about, found as
    `pairs_to_score` finds them. `made` holds the (config, wrapped retriever) of its runs."""
    config, retriever = max(made, key=lambda m: m[0]["rerank_depth"])
    lots = []
    for question in (q for q in questions if q.split == "dev"):
        recorder = Recorder()
        Rerank(retriever, recorder, config["rerank_depth"]).search(question.question, DEPTH)
        [(_, texts)] = recorder.lists
        lots.append(Lot(question.id, question.question, texts))
    return lots


def report_hosted(which: str, lots: list[Lot], hosted: HostedConfig, cache: HostedCache,
                  run: bool) -> int:  # fmt: skip
    """`make rerank WITH=…`: what asking a hosted reranker would send and cost; with `run`,
    ask it, within the cap, logging every paid call to `HOSTED_RUNS`."""
    if which == "jev":
        config, name = hosted.jev, hosted.jev.name
        missing = [
            lot for lot in lots
            if any(jev_key(config, lot.question, t) not in cache.entries for t in lot.texts)
        ]  # fmt: skip
    else:
        config, name = hosted.listwise, hosted.listwise.name
        missing = [
            lot for lot in lots
            if listwise_key(config, lot.question, lot.texts) not in cache.entries
        ]  # fmt: skip
    print(
        f"Reranker {name}, cache {cache.path.relative_to(ROOT)}: {len(lots)} questions, "
        f"{sum(len(lot.texts) for lot in lots):,} passages; {len(missing)} questions not asked yet."
    )
    if not missing:
        print("Nothing to ask. `make retrieval-runs` writes the reranked runs.")
        return 0
    try:
        generation = load_generation_config(GENERATION_CONFIG)
        if which == "jev":
            e = estimate_jev(missing, hosted.jev)
            read_on = hosted.jev.prices_read_on
        else:
            e = estimate_listwise(missing, hosted.listwise, generation)
            read_on = generation.prices_read_on
        spent = spending(ROOT)
    except (GenerationConfigError, GenerateError, RerankError, SpendLogError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(estimate_lines(e))
    print(spending_report(spent))
    blocked = holds(e, read_on, spent.spent_usd, spent.cap_usd, dt.date.today())
    key = read_typesafe_key(ENV) if which == "jev" else read_api_key(ENV)
    if not key:
        variable = "TYPESAFE_API_KEY" if which == "jev" else "ANTHROPIC_API_KEY"
        blocked.append(f"{variable} is not set: add it to .env (see .env.example)")
    if not run:
        for problem in blocked:
            print(f"A paid run would not start: {problem}")
        print(f"Nothing was sent. To pay and ask: make rerank WITH={which} RUN=1")
        return 0
    for problem in blocked:
        print(f"error: {problem}", file=sys.stderr)
    if blocked:
        print("Nothing was sent.", file=sys.stderr)
        return 1
    settings = dataclasses.asdict(config) | {"price_read_on": read_on.isoformat()}
    settings = {k: str(v) if isinstance(v, dt.date) else v for k, v in settings.items()}
    try:  # before anything is sent, so a run that can't be logged never starts
        header, path = new_run(ROOT, HOSTED_RUNS, config.model)
    except OSError as err:
        print(f"error: cannot write the run log ({err}); nothing was sent", file=sys.stderr)
        return 1

    def save(result: RunResult) -> None:
        write_run_log(path, hosted_run_log(header, name, settings, e, result))

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    try:
        save(RunResult())
        if which == "jev":
            result = jev_all(missing, JevClient(hosted.jev, key), cache, save, progress)
        else:
            calls = listwise_calls(missing)
            settings_for_call = dataclasses.replace(
                generation, effort=config.effort, max_output_tokens=config.max_output_tokens
            )
            client = AnthropicClient(settings_for_call, config.model, api_key=key,
                                     schema=LISTWISE_SCHEMA)  # fmt: skip
            price = generation.prices[config.model]
            result = send_all(calls, client, price, listwise_head, listwise_verdict,
                              LISTWISE_BLANK, progress, save)  # fmt: skip
            cache_orders(result, calls, config, cache)
    except OSError as err:
        print(
            f"error: a log or the cache could not be written ({err}). Save the printed log to "
            "runs/rerank/ before anything else is run, so the spend is counted.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Asked about {len(result.answers)} of {len(missing)} questions: "
        f"{result.input_tokens:,} input and {result.output_tokens:,} output tokens as counted "
        f"by the API, US${result.cost_usd:.4f}."
    )
    print(
        f"Run log: {path.relative_to(ROOT)}. Add a row to the spend log in docs/PROGRESS.md "
        f"that names {path.name}."
    )
    if result.error:
        print(f"error: stopped early: {result.error}", file=sys.stderr)
        return 1
    return 0


def estimate_lines(e: HostedEstimate) -> str:
    each = e.most_usd / e.calls if e.calls else 0.0
    return (
        f"{e.calls} calls, one a question: about {e.input_tokens:,} input tokens before the "
        f"margin.\nExpected cost US${e.expected_usd:.4f}; the most it can cost US${e.most_usd:.4f}"
        f" (US${each:.5f} a question, US${each * 1000:.2f} per 1,000). The cap check uses the "
        "most."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", action="store_true", help="report missing reranker scores")
    parser.add_argument("--run", action="store_true", help="with --scores, run the reranker")
    parser.add_argument("--with", dest="hosted", choices=HOSTED, help="a hosted reranker")
    args = parser.parse_args(argv)
    if (args.run or args.hosted) and not args.scores:
        parser.error("--run and --with go with --scores")
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
        hosted = load_hosted_config(HOSTED_CONFIG)
        caches = {
            "jev": HostedCache.load(RERANKS, hosted.jev.name, JEV_FIELDS),
            "haiku": HostedCache.load(RERANKS, hosted.listwise.name, LISTWISE_FIELDS),
        }
    except (ConfigError, EmbedError, RerankError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    # Every run is made before any log is written, so a run that fails (a missing vector, say)
    # leaves all the logs as they were.
    chunked: dict[str, list[Chunk]] = {}
    indexes: dict = {}
    made = []
    to_rerank = []
    names = {"jev": hosted.jev.name, "haiku": hosted.listwise.name}
    asked = names.get(args.hosted) if args.hosted else reranker_name(scores.config)
    scorer_for = cached_scorers(scores, hosted, caches)
    for name, strategy, config in runs(
        setting_name(embedding), reranker_name(scores.config), names
    ):
        if args.scores and config.get("reranker") != asked:
            continue
        if strategy not in chunked:
            chunked[strategy] = STRATEGIES[strategy](tree)
        chunks = chunked[strategy]
        try:
            if args.scores:
                inner = retriever_for(unreranked(config), chunks, tree, indexes, vectors)
                to_rerank.append((config, inner))
                continue
            retriever = retriever_for(config, chunks, tree, indexes, vectors, scorer_for)
            config = {**config, "strategy": strategy}
            made.append((name, make_run(name, config, retriever, chunks, questions, ROOT)))
        except (EmbedError, VectorIndexError, RerankError) as e:
            print(f"error: {name}: {e}; no run log was written", file=sys.stderr)
            return 1
    if args.hosted:
        try:
            lots = lots_for(to_rerank, questions)
        except (EmbedError, VectorIndexError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return report_hosted(args.hosted, lots, hosted, caches[args.hosted], args.run)
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
