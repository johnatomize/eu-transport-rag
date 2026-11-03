"""Answer one typed question from the documents, or show what that would send and cost (`make ask`).

ask.py "Can I split my daily rest into two parts?"            shows the passages and the cost
ask.py --run "Can I split my daily rest into two parts?"      sends it to Claude and pays
ask.py --retriever bm25 -k 10 "daily rest on a ferry"

The whole chain for one question: retrieval, the prompt, the model and the check (design §F).
Without `--run` it sends nothing and needs no API key: it prints the passages found, the
prompt's size, the estimated cost, and whether a paid run would be allowed to start. With
`--run` it pays, under the same rules as `make answers` (prices read in the last 7 days, the
most it can cost within the cap, a prompt under max_prompt_tokens, ANTHROPIC_API_KEY set), and
prints one of three things:

- the answer, with each source's node, version and quote, as the corpus writes it;
- the fixed message for a question the documents don't answer;
- "could not answer reliably" with the places the search found, and the check's reasons.

Retrieval is the setup E found best unless told otherwise: hybrid search (BM25 and vector search
fused by RRF) over paragraph chunks, top 5. Vector search needs the question's vector: a
question already in the embedding cache (every question of the question set) uses the cached
one; any other is turned into a vector by the local embedding model on this Mac, which takes a
few seconds to load. Its files must already be in the Hugging Face cache; `make ask` runs it
offline, so nothing is downloaded. A typed question's vector is used and not saved: the cache
holds the question set's vectors only. `--retriever bm25` needs no model.

A paid question writes its log to runs/ask/, in the same form as `make answers`' run logs with
the question added, and `rag.spend` counts its cost until a row of the spend log names the file.
"""

import argparse
import datetime as dt
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from rag.chunk import STRATEGIES, Chunk
from rag.corpus import CorpusError, Document, load_manifest
from rag.embed import EmbeddingCache, EmbedError, LocalClient
from rag.embed_estimate import ConfigError, load_config, query_input
from rag.generate import (
    AnthropicClient,
    Call,
    GenerateError,
    RunResult,
    answer_all,
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
from rag.prompt import (
    NOT_COVERED,
    build_prompt,
    shown_answer,
    unreliable_message,
)
from rag.spend import SpendLogError, spending, spending_report
from rag.vector import RETRIEVERS, VectorIndexError, make_retriever

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
CONFIG = ROOT / "config" / "generation.yaml"
EMBEDDING_CONFIG = ROOT / "config" / "embedding.yaml"
EMBEDDINGS = ROOT / "embeddings"
RUNS = ROOT / "runs" / "ask"
ENV = ROOT / ".env"
QUESTION_ID = "typed"  # the question's ID in the run log; a typed question has none
# Run logs are named ask-<time>-<model>.json, so no file name can be the same as one in
# runs/generation/: the spend log names a run by its file name alone.
LOG_PREFIX = "ask-"


def embed_question(
    question: str, cache: EmbeddingCache, load_client: Callable | None = None
) -> bool:
    """Put the question's vector in `cache`, in memory only; True if the model had to run.

    The cache file is never saved here, so typed questions never enter the committed cache.
    Only the local model is run: a question for a paid embedding model is refused.
    """
    text = query_input(question, cache.config)
    if cache.get(text) is not None:
        return False
    if cache.config.provider != "local":
        raise EmbedError(
            f"the question has no cached vector, and embedding it with {cache.config.model_id} "
            "would be paid for; use --retriever bm25"
        )
    # Read once, when the Hugging Face libraries are first imported, which LocalClient.load
    # does: from then on they use only the files already on this machine.
    os.environ["HF_HUB_OFFLINE"] = "1"
    print(
        f"Turning the question into a vector with {cache.config.model_id} on this machine; "
        "nothing is downloaded or saved.",
        file=sys.stderr,
        flush=True,
    )
    try:
        client = (load_client or LocalClient.load)(cache.config)
        vector, tokens = client.embed(text)
    except EmbedError:
        raise
    except ImportError as e:
        raise EmbedError(
            f"the local model's libraries are missing ({e}); run this through `make ask`, "
            "which installs them, or use --retriever bm25"
        ) from e
    except Exception as e:  # files not on this machine, no GPU, a library error
        raise EmbedError(f"the local model could not embed the question: {e}") from e
    cache.put(text, vector, tokens)
    return True


def shown(entry: dict, chunks: Sequence[Chunk], docs: Mapping[str, Document]) -> str:
    """What the reader is shown for one checked reply, from its run log entry."""
    verdict = entry["verdict"]
    if verdict["ok"] and verdict["answerable"]:
        citations = [(c["node_id"], c["quote"]) for c in verdict["citations"]]
        return shown_answer(verdict["answer"], citations, docs)
    if verdict["ok"]:
        return NOT_COVERED
    reasons = "\n".join(f"  - {problem}" for problem in verdict["problems"])
    return (
        f"{unreliable_message(chunks, docs)}\n\n"
        f"Why, from the check (a reader isn't shown this):\n{reasons}"
    )


def print_hits(chunks: Sequence[Chunk], scores: Sequence[float]) -> None:
    for rank, (chunk, score) in enumerate(zip(chunks, scores, strict=True), 1):
        body = " ".join(chunk.body.split())
        print(f"{rank}. {score:7.4f}  {chunk.id}")
        print(f"   {(chunk.prefix.splitlines() or [''])[0]}")
        print(f"   {body[:200]}{'…' if len(body) > 200 else ''}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("question")
    parser.add_argument("--run", action="store_true", help="send the question and pay for it")
    parser.add_argument(
        "--role", choices=ROLES, default="build", help="which model (default build)"
    )
    parser.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    parser.add_argument("--strategy", choices=STRATEGIES, default="paragraph")
    parser.add_argument("-k", type=int, default=5, help="passages in the prompt (default 5)")
    args = parser.parse_args(argv)
    question = args.question.strip()
    if not question:
        print("error: the question is empty", file=sys.stderr)
        return 1

    try:
        config = load_generation(CONFIG)
        manifest = load_manifest(MANIFEST)
        tree = parse_corpus(manifest, ROOT)
    except (GenerationConfigError, CorpusError, ParseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    docs = {doc.id: doc for doc in manifest}
    model = config.model(args.role)
    chunks = STRATEGIES[args.strategy](tree)
    try:
        cache = None
        if args.retriever != "bm25":
            cache = EmbeddingCache.load(EMBEDDINGS, load_config(EMBEDDING_CONFIG))
            embed_question(question, cache)
        hits = make_retriever(args.retriever, chunks, cache).search(question, args.k)
        found = tuple(hit.chunk for hit in hits)
        call = Call(QUESTION_ID, question, build_prompt(question, found, docs), found)
        e = estimate([call], config, model)
    except (ConfigError, EmbedError, VectorIndexError, GenerateError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    retrieval = {"retriever": args.retriever, "strategy": args.strategy, "k": args.k}
    print(f"Question: {question}")
    print(f"Retrieval: {args.retriever} over {args.strategy} chunks, top {args.k}.")
    print_hits(found, [hit.score for hit in hits])
    size = len(call.prompt.system) + len(call.prompt.user)
    print(
        f"\nPrompt version {prompt_version()}: {size:,} characters, about "
        f"{call.input_tokens:,} input tokens. Model {model} ({args.role}), effort "
        f"{config.effort}, at most {config.max_output_tokens:,} output tokens."
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
    for problem in errors:
        print(f"error: {problem}", file=sys.stderr)

    if not args.run:
        for problem in holds:
            print(f"warning: a paid question would refuse to start: {problem}", file=sys.stderr)
        print('Nothing was sent. To send and pay: make ask Q="…" RUN=1')
        return 1 if errors else 0
    key = read_api_key(ENV)
    if not key:
        holds.append("ANTHROPIC_API_KEY is not set: add it to .env (see .env.example)")
    for problem in holds:
        print(f"error: {problem}", file=sys.stderr)
    if errors or holds:
        print("Nothing was sent.", file=sys.stderr)
        return 1

    def save(result: RunResult) -> None:
        log = run_log(run, config, model, retrieval, "typed", e, result)
        write_run_log(path, {**log, "question": question})

    try:  # before anything is sent, so a question whose cost can't be logged is never sent
        run, path = new_run(ROOT, RUNS, model, prefix=LOG_PREFIX)
        save(RunResult())
    except OSError as err:
        print(f"error: cannot write the run log ({err}); nothing was sent", file=sys.stderr)
        return 1
    try:
        result = answer_all(
            [call], AnthropicClient(config, model, api_key=key), config.prices[model], save=save
        )
    except OSError as err:
        print(
            f"error: the run log could not be written ({err}); it is printed above. Save it "
            "to runs/ask/ before anything else is run, so the spend is counted.",
            file=sys.stderr,
        )
        return 1

    shown_path = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    if result.answers:
        entry = result.answers[0]
        print(f"\n{shown(entry, found, docs)}\n")
        print(
            f"{entry['usage']['input_tokens']:,} input and {entry['usage']['output_tokens']:,} "
            f"output tokens, as counted by the API; US${result.cost_usd:.6f} at list price; "
            f"{entry['seconds']:.1f} s."
        )
    print(
        f"Run log: {shown_path}. Add a row to the spend log in docs/PROGRESS.md that names "
        f"{path.name}; until then the next paid run counts its cost."
    )
    if result.error:
        print(f"error: no answer: {result.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
