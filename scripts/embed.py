"""Embed every chunk and question that has no cached vector yet (`make embed`).

Without `--run` it runs nothing and loads no model: it reports how many texts are cached and how
many are missing, and warns about anything that would stop a run. With `--run` it embeds the
missing texts with the model in config/embedding.yaml, and only if every text can be embedded
(none empty, none possibly over the model's limit).

- **A local model** (provider `local`) costs nothing. The first run downloads the model's files
  into the Hugging Face cache. The run log, with the model's revision, the device, the library
  versions, the token count of each input and the time taken, goes to runs/embeddings-local/.
- **Bedrock** (provider `bedrock`) is paid, so a run also needs a price read in the last 7 days
  and an estimate, margin included, that keeps what has been spent within the approved cap.
  What has been spent is the spend log in docs/PROGRESS.md plus any paid run (embeddings or
  answers) whose log the spend log doesn't name yet. The call log holds the token count Bedrock
  returned for each call, so the spend log can be checked against the bill; adding its row to
  the spend log, naming that file, stays a manual step.
"""

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from rag.corpus import CorpusError
from rag.embed import (
    BedrockClient,
    EmbeddingCache,
    EmbeddingClient,
    EmbedError,
    LocalClient,
    Plan,
    RunResult,
    check_cap,
    check_inputs,
    corpus_texts,
    embed_missing,
    plan,
)
from rag.embed_estimate import (
    PRICE_STALE_AFTER_DAYS,
    ConfigError,
    EmbeddingConfig,
    cost_usd,
    load_config,
)
from rag.parse import ParseError
from rag.runlog import git_state
from rag.spend import SpendLogError
from rag.spend import spending as paid_runs_spending

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "embedding.yaml"
CACHE = ROOT / "embeddings"
RUNS = ROOT / "runs" / "embeddings"  # paid Bedrock runs; the spend log names each file
LOCAL_RUNS = ROOT / "runs" / "embeddings-local"  # free local runs, outside the spend log
PROFILE = "eu-transport-rag"
LIBRARIES = ("torch", "transformers", "sentence-transformers", "tokenizers")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="store_true", help="run the model (on Bedrock, pay for it)")
    args = parser.parse_args()
    try:
        config = load_config(CONFIG)
        texts = [t for batch in corpus_texts(ROOT, config).values() for t in batch]
        cache = EmbeddingCache.load(CACHE, config)
    except (ConfigError, CorpusError, ParseError, EmbedError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    p = plan(texts, cache)

    if config.paid:
        print(
            f"Model {config.model_id} on Bedrock in {config.region}, "
            f"{config.dimensions} dimensions."
        )
    else:
        print(
            f"Model {config.model_id} at revision {config.revision[:12]}, run on this machine, "
            f"{config.dimensions} dimensions."
        )
    print(f"{p.distinct} distinct texts: {p.cached} cached, {len(p.missing)} to embed.")

    holds = []  # these only stop a run
    if config.paid:
        try:
            spent, cap = spending()
        except (SpendLogError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        holds = paid_holds(p, config, spent, cap)
    if not p.missing:
        print("Nothing to embed.")
        return 0
    if config.paid:
        print(
            f"To send: about {p.tokens:,} tokens, {p.padded_tokens:,} with the margin; "
            f"estimated US${p.cost_usd:.5f}."
        )
    else:
        print(f"To embed: about {p.tokens:,} tokens (estimated); it costs nothing.")

    errors = check_inputs(p)  # these fail the report too, as they do in `make embed-dry-run`
    for problem in errors:
        print(f"error: {problem}", file=sys.stderr)
    if not args.run:
        for problem in holds:
            print(f"warning: a run would refuse to start: {problem}", file=sys.stderr)
        what = "To send and pay" if config.paid else "To run the model"
        print(f"Nothing was embedded. {what}: make embed RUN=1")
        return 1 if errors else 0
    for problem in holds:
        print(f"error: {problem}", file=sys.stderr)
    if errors or holds:
        print("Nothing was embedded.", file=sys.stderr)
        return 1

    commit, dirty = git_state(ROOT)
    created = datetime.now(UTC)
    try:
        client: EmbeddingClient
        if config.paid:
            client = BedrockClient(config, profile=os.environ.get("AWS_PROFILE", PROFILE))
        else:
            print(
                "Loading the model; the first run downloads its files into the Hugging Face cache.",
                flush=True,
            )
            client = LocalClient.load(config)
            print(f"Running on {client.device}.", flush=True)
    except (EmbedError, ImportError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        if isinstance(e, ImportError):
            print(
                "Install the local model's libraries: uv sync --group local-embed", file=sys.stderr
            )
        return 1

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    started = time.monotonic()
    result = embed_missing(p, cache, client, progress)
    seconds = time.monotonic() - started
    run = {"created": created.isoformat(timespec="seconds"), "commit": commit, "dirty": dirty}
    if config.paid:
        path = write_paid_log(run, config, p, result)
        print(
            f"Sent {len(result.calls)} of {len(p.missing)} texts: {result.tokens:,} tokens as "
            f"counted by Bedrock, US${cost_usd(result.tokens, config):.5f} before credits."
        )
        print(
            f"Call log: {path.relative_to(ROOT)}. Add a row to the spend log in docs/PROGRESS.md "
            f"that names {path.name}."
        )
    else:
        assert isinstance(client, LocalClient)
        path = write_local_log(run, config, client, p, result, seconds)
        print(
            f"Embedded {len(result.calls)} of {len(p.missing)} texts: {result.tokens:,} tokens "
            f"in {seconds:.0f} s on {client.device}. Run log: {path.relative_to(ROOT)}."
        )
    if result.error:
        print(f"error: stopped early: {result.error}", file=sys.stderr)
        return 1
    return 0


def spending() -> tuple[float, float]:
    """What has been spent, the spend log plus paid runs it doesn't name yet, and the cap."""
    spent = paid_runs_spending(ROOT)
    print(f"Spent so far US${spent.spent_usd:.5f} of the approved cap of US${spent.cap_usd:.2f}.")
    for name, cost in spent.pending:
        print(f"  including US${cost:.5f} from {name}, not in the spend log yet")
    return spent.spent_usd, spent.cap_usd


def paid_holds(p: Plan, config: EmbeddingConfig, spent: float, cap: float) -> list[str]:
    """Why a paid run may not start: an old price, or an estimate that would pass the cap."""
    holds = []
    assert config.price_read_on is not None  # load_config requires it for Bedrock
    age = (dt.date.today() - config.price_read_on).days
    if age > PRICE_STALE_AFTER_DAYS:
        holds.append(
            f"the price was read {age} days ago; read it again in the AWS Price List API and "
            "update config/embedding.yaml"
        )
    try:
        check_cap(p, spent, cap)
    except EmbedError as e:
        holds.append(str(e))
    return holds


def _write(directory: Path, run: dict, log: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    created = datetime.fromisoformat(run["created"])
    path = directory / f"{created.strftime('%Y-%m-%dT%H-%M-%SZ')}.json"
    path.write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    return path


def write_paid_log(run: dict, config: EmbeddingConfig, p: Plan, result: RunResult) -> Path:
    assert config.price_read_on is not None
    return _write(
        RUNS,
        run,
        {
            "run": run,
            "config": {
                "model_id": config.model_id,
                "region": config.region,
                "dimensions": config.dimensions,
                "normalize": config.normalize,
                "price_usd_per_million_input_tokens": config.price_usd_per_million_input_tokens,
                "price_read_on": config.price_read_on.isoformat(),
            },
            "estimate": {
                "texts": len(p.missing),
                "tokens": p.tokens,
                "padded_tokens": p.padded_tokens,
                "cost_usd": round(p.cost_usd, 6),
            },
            "paid": {
                "calls": len(result.calls),
                "tokens": result.tokens,
                "cost_usd": round(cost_usd(result.tokens, config), 6),
            },
            "error": result.error,
            "calls": result.calls,
        },
    )


def write_local_log(
    run: dict,
    config: EmbeddingConfig,
    client: LocalClient,
    p: Plan,
    result: RunResult,
    seconds: float,
) -> Path:
    return _write(
        LOCAL_RUNS,
        run,
        {
            "run": run,
            "config": {
                "model_id": config.model_id,
                "revision": config.revision,
                "dimensions": config.dimensions,
                "normalize": config.normalize,
                "max_input_tokens": config.max_input_tokens,
                "query_prefix": config.query_prefix,
            },
            "environment": {
                "device": client.device,
                "machine": platform.machine(),
                "system": f"{platform.system()} {platform.release()}",
                "python": platform.python_version(),
                **{name: _version(name) for name in LIBRARIES},
            },
            "estimate": {"texts": len(p.missing), "tokens": p.tokens},
            "result": {
                "texts": len(result.calls),
                "tokens": result.tokens,
                "seconds": round(seconds, 1),
            },
            "error": result.error,
            "calls": result.calls,
        },
    )


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    sys.exit(main())
