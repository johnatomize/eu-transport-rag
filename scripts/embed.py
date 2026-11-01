"""Embed every chunk and question that has no cached vector yet (`make embed`).

Without `--send` it calls nothing: it reports how many texts are cached, how many would be sent
and what that would cost, and warns about anything that would stop a paid run. With `--send` it
pays, and only if:

- every text can be sent (none empty, none possibly over the model's limit);
- the price in config/embedding.yaml was read in the last 7 days;
- the estimate, margin included, on top of what has been spent stays within the approved cap.
  What has been spent is the spend log in docs/PROGRESS.md plus any paid run whose call log in
  runs/embeddings/ the spend log doesn't name yet.

A paid run writes the token count Bedrock returned for each call to runs/embeddings/, so the
spend log can be checked against the bill. Adding its row to the spend log, naming that file,
stays a manual step.
"""

import argparse
import datetime as dt
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from rag.corpus import CorpusError
from rag.embed import (
    BedrockClient,
    EmbeddingCache,
    EmbedError,
    check_cap,
    check_inputs,
    corpus_texts,
    pay,
    plan,
)
from rag.embed_estimate import PRICE_STALE_AFTER_DAYS, ConfigError, cost_usd, load_config
from rag.parse import ParseError
from rag.runlog import git_state
from rag.spend import SpendLogError, read_spend_log, unlogged_runs

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "embedding.yaml"
CACHE = ROOT / "embeddings"
SPEND_LOG = ROOT / "docs" / "PROGRESS.md"
RUNS = ROOT / "runs" / "embeddings"
PROFILE = "eu-transport-rag"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--send", action="store_true", help="call Bedrock and pay for it")
    args = parser.parse_args()
    try:
        config = load_config(CONFIG)
        texts = [t for batch in corpus_texts(ROOT).values() for t in batch]
        cache = EmbeddingCache.load(CACHE, config)
        log_text = SPEND_LOG.read_text(encoding="utf-8")
        spend = read_spend_log(log_text)
        pending = unlogged_runs(log_text, RUNS)
    except (ConfigError, CorpusError, ParseError, EmbedError, SpendLogError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    p = plan(texts, cache)
    spent = spend.spent_usd + sum(cost for _, cost in pending)

    print(f"Model {config.model_id} in {config.region}, {config.dimensions} dimensions.")
    print(f"{p.distinct} distinct texts: {p.cached} cached, {len(p.missing)} to send.")
    print(f"Spent so far US${spent:.5f} of the approved cap of US${spend.cap_usd:.2f}.")
    for name, cost in pending:
        print(f"  including US${cost:.5f} from runs/embeddings/{name}, not in the spend log yet")
    if not p.missing:
        print("Nothing to send.")
        return 0
    print(
        f"To send: about {p.tokens:,} tokens, {p.padded_tokens:,} with the margin; "
        f"estimated US${p.cost_usd:.5f}."
    )

    errors = check_inputs(p)  # these fail the report too, as they do in `make embed-dry-run`
    holds = []  # these only stop a paid run
    age = (dt.date.today() - config.price_read_on).days
    if age > PRICE_STALE_AFTER_DAYS:
        holds.append(
            f"the price was read {age} days ago; read it again in the AWS Price List API and "
            "update config/embedding.yaml"
        )
    try:
        check_cap(p, spent, spend.cap_usd)
    except EmbedError as e:
        holds.append(str(e))
    for problem in errors:
        print(f"error: {problem}", file=sys.stderr)
    if not args.send:
        for problem in holds:
            print(f"warning: a paid run would refuse to start: {problem}", file=sys.stderr)
        print("Nothing was sent. To send and pay: make embed SEND=1")
        return 1 if errors else 0
    for problem in holds:
        print(f"error: {problem}", file=sys.stderr)
    if errors or holds:
        print("Nothing was sent.", file=sys.stderr)
        return 1

    commit, dirty = git_state(ROOT)
    created = datetime.now(UTC)
    try:
        client = BedrockClient(config, profile=os.environ.get("AWS_PROFILE", PROFILE))
    except EmbedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    paid = pay(p, cache, client, progress)
    log = {
        "run": {
            "created": created.isoformat(timespec="seconds"),
            "commit": commit,
            "dirty": dirty,
        },
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
            "calls": len(paid.calls),
            "tokens": paid.tokens,
            "cost_usd": round(cost_usd(paid.tokens, config), 6),
        },
        "error": paid.error,
        "calls": paid.calls,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{created.strftime('%Y-%m-%dT%H-%M-%SZ')}.json"
    path.write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")

    print(
        f"Sent {len(paid.calls)} of {len(p.missing)} texts: {paid.tokens:,} tokens as counted "
        f"by Bedrock, US${cost_usd(paid.tokens, config):.5f} before credits."
    )
    print(
        f"Call log: {path.relative_to(ROOT)}. Add a row to the spend log in docs/PROGRESS.md "
        f"that names {path.name}."
    )
    if paid.error:
        print(f"error: stopped early: {paid.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
