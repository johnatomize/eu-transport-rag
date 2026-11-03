"""Answer the dev questions with Claude, or show what that would send and cost (`make answers`).

Without `--run` it sends nothing and needs no API key: it retrieves the chunks for each
question, builds each prompt, and prints every call it would make (question, model, estimated
input and output tokens) and the estimated cost, with the margin stated. It also says whether a
paid run would be allowed to start.

With `--run` it pays, and only if:
- the prices in config/generation.yaml were read in the last 7 days;
- the most the run can cost (every answer at max_output_tokens) keeps what has been spent within
  the approved cap: the spend log in docs/PROGRESS.md plus every paid run whose log it doesn't
  name yet;
- no prompt may be over max_prompt_tokens;
- ANTHROPIC_API_KEY is set, in the environment or in .env.

A paid run writes its log to runs/generation/: per question the retrieved chunk IDs, the raw
reply, the check's verdict and reasons, the tokens the API counted, the cost and the time; for
the run the model, effort, prompt version and retrieval setup. Adding its row to the spend log,
naming the file, stays a manual step.

Retrieval is the setup E found best unless told otherwise: hybrid search (BM25 and vector search
fused by RRF) over paragraph chunks, top 5. Only the dev split can be answered; the test split
waits for G.
"""

import argparse
import datetime as dt
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from rag.chunk import STRATEGIES, Chunk
from rag.corpus import CorpusError, load_manifest
from rag.db import read_env_file
from rag.embed import EmbeddingCache, EmbedError
from rag.embed_estimate import PRICE_STALE_AFTER_DAYS, ConfigError, load_config
from rag.generate import (
    EXPECTED_OUTPUT_TOKENS,
    MARGIN,
    AnthropicClient,
    Call,
    Estimate,
    GenerateError,
    RunResult,
    answer_all,
    check_cap,
    estimate,
    input_problems,
    prompt_version,
)
from rag.generation_config import ROLES, GenerationConfig, GenerationConfigError
from rag.generation_config import load_generation_config as load_generation
from rag.parse import ParseError, parse_corpus
from rag.prompt import build_prompt
from rag.questions import validate
from rag.retrieve import BM25, RRF, RRF_DEPTH, RRF_K, Retriever
from rag.runlog import git_state
from rag.spend import SpendLogError, spending
from rag.vector import VectorIndexError, VectorSearch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
QUESTIONS = ROOT / "data" / "questions.v1.jsonl"
CONFIG = ROOT / "config" / "generation.yaml"
EMBEDDING_CONFIG = ROOT / "config" / "embedding.yaml"
EMBEDDINGS = ROOT / "embeddings"
RUNS = ROOT / "runs" / "generation"
ENV = ROOT / ".env"
RETRIEVERS = ("hybrid", "bm25", "vector")


def make_retriever(kind: str, chunks: list[Chunk]) -> Retriever:
    """The retriever by name, with E's settings; vector search reads the embedding cache."""
    if kind == "bm25":
        return BM25(chunks)
    vectors = EmbeddingCache.load(EMBEDDINGS, load_config(EMBEDDING_CONFIG))
    vector = VectorSearch(chunks, vectors)
    if kind == "vector":
        return vector
    return RRF(chunks, [BM25(chunks), vector], RRF_K, RRF_DEPTH)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="store_true", help="send the calls and pay for them")
    parser.add_argument(
        "--role", choices=ROLES, default="build", help="which model (default build)"
    )
    parser.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    parser.add_argument("--strategy", choices=STRATEGIES, default="paragraph")
    parser.add_argument("-k", type=int, default=5, help="chunks per prompt (default 5)")
    parser.add_argument("--only", help="answer only these dev questions: q001,q002")
    args = parser.parse_args(argv)

    try:
        config = load_generation(CONFIG)
        manifest = load_manifest(MANIFEST)
        tree = parse_corpus(manifest, ROOT)
        docs = {doc.id: doc for doc in manifest}
    except (GenerationConfigError, CorpusError, ParseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    questions, errors = validate(QUESTIONS.read_text(encoding="utf-8"), tree)
    if errors:
        print(f"error: {QUESTIONS.name} is not valid; run `make questions`", file=sys.stderr)
        return 1
    dev = [q for q in questions if q.split == "dev"]
    if args.only:
        wanted = [i.strip() for i in args.only.split(",") if i.strip()]
        known = {q.id for q in dev}
        if unknown := [i for i in wanted if i not in known]:
            print(f"error: not dev questions: {', '.join(unknown)}", file=sys.stderr)
            return 1
        dev = [q for q in dev if q.id in wanted]
        if not dev:
            print("error: --only names no question", file=sys.stderr)
            return 1

    model = config.model(args.role)
    chunks = STRATEGIES[args.strategy](tree)
    try:
        retriever = make_retriever(args.retriever, chunks)
        calls = []
        for q in dev:
            hits = retriever.search(q.question, args.k)
            found = tuple(hit.chunk for hit in hits)
            calls.append(Call(q.id, q.question, build_prompt(q.question, found, docs), found))
        e = estimate(calls, config, model)
    except (ConfigError, EmbedError, VectorIndexError, GenerateError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    retrieval = {"retriever": args.retriever, "strategy": args.strategy, "k": args.k}
    print(
        f"Model {model} ({args.role}), effort {config.effort}, at most "
        f"{config.max_output_tokens:,} output tokens an answer."
    )
    print(
        f"Retrieval: {args.retriever} over {args.strategy} chunks, top {args.k}. "
        f"Prompt version {prompt_version()}."
    )
    print(f"{'question':9} {'est. input':>10} {'est. output':>11} {'max output':>10}")
    for c in calls:
        print(
            f"{c.question_id:9} {c.input_tokens:>10,} {EXPECTED_OUTPUT_TOKENS:>11,} "
            f"{config.max_output_tokens:>10,}"
        )
    print(report(e))

    try:
        spent = spending(ROOT)
    except SpendLogError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(f"Spent so far US${spent.spent_usd:.4f} of the approved cap of US${spent.cap_usd:.2f}.")
    for name, cost in spent.pending:
        print(f"  including US${cost:.4f} from {name}, not in the spend log yet")
    holds = paid_holds(e, config, spent.spent_usd, spent.cap_usd)
    errors = input_problems(e, config)
    for problem in errors:
        print(f"error: {problem}", file=sys.stderr)

    if not args.run:
        for problem in holds:
            print(f"warning: a paid run would refuse to start: {problem}", file=sys.stderr)
        print("Nothing was sent. To send and pay: make answers RUN=1")
        return 1 if errors else 0
    key = os.environ.get("ANTHROPIC_API_KEY") or read_env_file(ENV).get("ANTHROPIC_API_KEY")
    if not key:
        holds.append("ANTHROPIC_API_KEY is not set: add it to .env (see .env.example)")
    for problem in holds:
        print(f"error: {problem}", file=sys.stderr)
    if errors or holds or not key:
        print("Nothing was sent.", file=sys.stderr)
        return 1

    commit, dirty = git_state(ROOT)
    created = datetime.now(UTC)
    run = {"created": created.isoformat(timespec="seconds"), "commit": commit, "dirty": dirty}
    path = RUNS / f"{created.strftime('%Y-%m-%dT%H-%M-%SZ')}-{model}.json"

    def save(result: RunResult) -> None:
        write_log(path, run, config, model, retrieval, e, result)

    try:  # before anything is sent, so a run that can't be logged never starts
        RUNS.mkdir(parents=True, exist_ok=True)
        save(RunResult())
    except OSError as err:
        print(f"error: cannot write the run log ({err}); nothing was sent", file=sys.stderr)
        return 1
    client = AnthropicClient(config, model, api_key=key)

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    try:
        result = answer_all(calls, client, config.prices[model], progress, save)
    except OSError as err:
        print(
            f"error: the run log could not be written ({err}); it is printed above. Save it "
            "to runs/generation/ before anything else is run, so the spend is counted.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Answered {len(result.answers)} of {len(calls)}: {result.passed} passed the check, "
        f"{result.refusals} of them saying the documents don't answer; "
        f"{len(result.answers) - result.passed} could not be answered reliably."
    )
    print(
        f"Tokens as counted by the API: {result.input_tokens:,} input, "
        f"{result.output_tokens:,} output; US${result.cost_usd:.4f} at list price."
    )
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(
        f"Run log: {shown}. Add a row to the spend log in docs/PROGRESS.md that names {path.name}."
    )
    if result.error:
        print(f"error: stopped early: {result.error}", file=sys.stderr)
        return 1
    return 0


def report(e: Estimate) -> str:
    return (
        f"{e.calls} calls: about {e.input_tokens:,} input tokens, {e.padded_input:,} with a "
        f"{MARGIN - 1:.0%} margin.\n"
        f"Expected cost, at {EXPECTED_OUTPUT_TOKENS:,} output tokens an answer plus the margin: "
        f"US${e.expected_usd:.4f}.\n"
        f"Most it can cost, every answer at {e.max_output_tokens:,} output tokens: "
        f"US${e.most_usd:.4f}. The cap check uses this figure."
    )


def paid_holds(e: Estimate, config: GenerationConfig, spent: float, cap: float) -> list[str]:
    """Why a paid run may not start: an old price, or a cost that could pass the cap."""
    holds = []
    age = (dt.date.today() - config.prices_read_on).days
    if age > PRICE_STALE_AFTER_DAYS:
        holds.append(
            f"the prices were read {age} days ago; read them again on Anthropic's pricing page "
            "and update config/generation.yaml"
        )
    try:
        check_cap(e, spent, cap)
    except GenerateError as err:
        holds.append(str(err))
    return holds


def write_log(
    path: Path,
    run: dict,
    config: GenerationConfig,
    model: str,
    retrieval: dict,
    e: Estimate,
    result: RunResult,
) -> None:
    """Write the run log whole, through a temporary file, so it is never left half written."""
    price = config.prices[model]
    log = {
        "run": run,
        "config": {
            "model": model,
            "effort": config.effort,
            "max_output_tokens": config.max_output_tokens,
            "prompt_version": prompt_version(),
            "price_usd_per_million": {"input": price.input, "output": price.output},
            "prices_read_on": config.prices_read_on.isoformat(),
            "retrieval": retrieval,
            "split": "dev",
        },
        "estimate": {
            "calls": e.calls,
            "input_tokens": e.input_tokens,
            "padded_input_tokens": e.padded_input,
            "expected_usd": round(e.expected_usd, 6),
            "most_usd": round(e.most_usd, 6),
        },
        "paid": {
            "calls": len(result.answers),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": round(result.cost_usd, 6),
        },
        "error": result.error,
        "failed": result.failed,
        "answers": result.answers,
    }
    text = json.dumps(log, indent=1, ensure_ascii=False) + "\n"
    try:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except OSError:
        print(text)  # the calls are paid for: never lose their record
        raise


if __name__ == "__main__":
    sys.exit(main())
