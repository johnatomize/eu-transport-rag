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

With `--batch` the calls go as one batch through Anthropic's Message Batches API, at half price
(`rag.batch`): the estimate and the cap check use the batch prices, and a paid run waits for the
batch, usually under an hour, at most 24. If it is stopped while it waits, the same command run
again picks up that batch instead of sending another.

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
import sys
from pathlib import Path

from rag.batch import (
    AnthropicBatchClient,
    BatchStatus,
    find_batch,
    found_note,
    new_batch,
    opening,
    pick_up,
    run_batch,
    status_line,
    stopped_note,
)
from rag.chunk import STRATEGIES
from rag.corpus import CorpusError, load_manifest
from rag.embed import EmbeddingCache, EmbedError
from rag.embed_estimate import ConfigError, load_config
from rag.generate import (
    ANSWER_BLANK,
    EXPECTED_OUTPUT_TOKENS,
    AnthropicClient,
    Call,
    GenerateError,
    RunResult,
    answer_all,
    answer_head,
    answer_verdict,
    estimate,
    estimate_report,
    input_problems,
    new_run,
    paid_holds,
    prompt_version,
    read_api_key,
    run_log,
    write_run_log,
)
from rag.generation_config import ROLES, GenerationConfigError
from rag.generation_config import load_generation_config as load_generation
from rag.parse import ParseError, parse_corpus
from rag.prompt import ANSWER_SCHEMA, build_prompt
from rag.questions import validate
from rag.spend import SpendLogError, spending, spending_report
from rag.vector import RETRIEVERS, VectorIndexError, make_retriever

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
QUESTIONS = ROOT / "data" / "questions.v1.jsonl"
CONFIG = ROOT / "config" / "generation.yaml"
EMBEDDING_CONFIG = ROOT / "config" / "embedding.yaml"
EMBEDDINGS = ROOT / "embeddings"
RUNS = ROOT / "runs" / "generation"
ENV = ROOT / ".env"


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
    parser.add_argument("--batch", action="store_true", help="send as one batch, at half price")
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
        cache = None
        if args.retriever != "bm25":
            cache = EmbeddingCache.load(EMBEDDINGS, load_config(EMBEDDING_CONFIG))
        retriever = make_retriever(args.retriever, chunks, cache)
        calls = []
        for q in dev:
            hits = retriever.search(q.question, args.k)
            shown = tuple(hit.chunk for hit in hits)
            calls.append(Call(q.id, q.question, build_prompt(q.question, shown, docs), shown))
        e = estimate(calls, config, model, batch=args.batch)
        record = new_batch(calls, config, model, ANSWER_SCHEMA)
        batch = record if args.batch else None
        pending = find_batch(RUNS, record)  # a batch of these calls that may still be paid for
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
    print(estimate_report(e))

    try:
        spent = spending(ROOT)
    except SpendLogError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(spending_report(spent))
    holds = paid_holds(e, config, spent.spent_usd, spent.cap_usd, dt.date.today())
    errors = input_problems(e, config)
    if pending:
        note, refused = found_note(*pending, batch_run=args.batch)
        print(note)
        if args.batch:  # nothing new is paid for: it passed the checks when it was sent
            holds, errors = [], refused
        else:
            errors += refused
    for problem in errors:
        print(f"error: {problem}", file=sys.stderr)

    if not args.run:
        for problem in holds:
            print(f"warning: a paid run would refuse to start: {problem}", file=sys.stderr)
        again = " BATCH=1" if args.batch else ""
        print(f"Nothing was sent. To send and pay: make answers{again} RUN=1")
        return 1 if errors else 0
    key = read_api_key(ENV)
    if not key:
        holds.append("ANTHROPIC_API_KEY is not set: add it to .env (see .env.example)")
    for problem in holds:
        print(f"error: {problem}", file=sys.stderr)
    if errors or holds or not key:
        print("Nothing was sent.", file=sys.stderr)
        return 1

    def save(result: RunResult) -> None:
        write_run_log(path, run_log(run, config, model, retrieval, "dev", e, result, batch))

    if pending:
        run, path, batch = pick_up(pending)
    else:
        try:  # before anything is sent, so a run that can't be logged never starts
            run, path = new_run(ROOT, RUNS, model)
            save(RunResult())
        except OSError as err:
            print(f"error: cannot write the run log ({err}); nothing was sent", file=sys.stderr)
            return 1

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    def waiting(status: BatchStatus) -> None:
        print(status_line(status, len(calls)), flush=True)

    try:
        if batch is None:
            client = AnthropicClient(config, model, api_key=key)
            result = answer_all(calls, client, e.price, progress, save)
        else:
            batches = AnthropicBatchClient(config, model, api_key=key)
            print(opening(batch, len(calls)))
            result = run_batch(
                calls,
                batches,
                batch,
                e.price,
                answer_head,
                answer_verdict,
                ANSWER_BLANK,
                save,
                waiting,
            )
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
        f"{result.output_tokens:,} output; US${result.cost_usd:.4f} at "
        f"{'the batch price' if batch else 'list price'}."
    )
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(
        f"Run log: {shown}. Add a row to the spend log in docs/PROGRESS.md that names {path.name}."
    )
    if result.error:
        print(f"error: stopped early: {result.error}", file=sys.stderr)
        if note := stopped_note(batch):
            print(note, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
