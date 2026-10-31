"""What embedding the corpus and the questions will cost, estimated without calling AWS (design §E).

Titan Text Embeddings V2's tokenizer is not published, so tokens are estimated from the text: the
higher of characters / 4 and words × 1.3, two common rules of thumb that disagree most on text
full of numbers and references, such as "Article 6(3) of Regulation (EC) No 561/2006". The total
then gets a safety margin of 50%, and so does each input when it is checked against the model's
limit. Bedrock returns the real count with every call, and the paid run logs it, so the estimate
is checked against the bill.

Vectors are cached by model and text, so a text that appears in more than one chunking, such as
an article without paragraphs, is paid for once; the total counts each distinct text once.
"""

import datetime as dt
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

MARGIN = 1.5  # estimates are multiplied by this before they are priced or checked
PRICE_STALE_AFTER_DAYS = 7
TITAN_V2_DIMENSIONS = (256, 512, 1024)
_KEYS = {
    "model_id": str,
    "region": str,
    "dimensions": int,
    "normalize": bool,
    "max_input_tokens": int,
    "price_usd_per_million_input_tokens": float,
    "price_read_on": dt.date,
}


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class EmbeddingConfig:
    model_id: str
    region: str
    dimensions: int
    normalize: bool
    max_input_tokens: int
    price_usd_per_million_input_tokens: float
    price_read_on: dt.date


def load_config(path: Path) -> EmbeddingConfig:
    """Read `config/embedding.yaml`, refusing missing, unknown, mistyped or impossible settings."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"{path.name}: cannot read it ({e.strerror})") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"{path.name}: not valid YAML ({e})") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name}: expected a mapping of settings")
    price = raw.get("price_usd_per_million_input_tokens")
    if type(price) is int:  # `type`, not isinstance: a YAML `true` must not become 1.0
        raw["price_usd_per_million_input_tokens"] = float(price)
    problems = [f"missing {k}" for k in _KEYS if k not in raw]
    problems += [f"unknown {k}" for k in raw if k not in _KEYS]
    problems += [
        f"{k} should be {'a date' if kind is dt.date else kind.__name__}"
        for k, kind in _KEYS.items()
        if k in raw and type(raw[k]) is not kind
    ]
    if not problems:
        if raw["price_usd_per_million_input_tokens"] <= 0:
            problems.append("price_usd_per_million_input_tokens should be above 0")
        if raw["max_input_tokens"] <= 0:
            problems.append("max_input_tokens should be above 0")
        if raw["dimensions"] not in TITAN_V2_DIMENSIONS:
            problems.append(f"dimensions should be one of {TITAN_V2_DIMENSIONS}")
    if problems:
        raise ConfigError(f"{path.name}: " + "; ".join(problems))
    return EmbeddingConfig(**raw)


def estimate_tokens(text: str) -> int:
    """Estimated tokens in one input: the higher of characters / 4 and words × 1.3."""
    return max(math.ceil(len(text) / 4), math.ceil(len(text.split()) * 1.3))


@dataclass(frozen=True)
class Batch:
    name: str  # "chunks: article", "questions"
    texts: int
    tokens: int  # estimated, before the margin
    largest: int  # estimated tokens of the longest input
    empty: int  # inputs with no text, which Bedrock refuses


def batch(name: str, texts: Sequence[str]) -> Batch:
    counts = [estimate_tokens(t) for t in texts]
    empty = sum(1 for t in texts if not t.strip())
    return Batch(name, len(counts), sum(counts), max(counts, default=0), empty)


def cost_usd(tokens: int, config: EmbeddingConfig) -> float:
    return tokens / 1_000_000 * config.price_usd_per_million_input_tokens


def _usd(amount: float) -> str:
    return f"{amount:.5f}"


def _price(config: EmbeddingConfig) -> str:
    return f"{config.price_usd_per_million_input_tokens:.6f}".rstrip("0").rstrip(".")


def report(
    named_texts: Mapping[str, Sequence[str]], config: EmbeddingConfig, today: dt.date
) -> tuple[list[str], bool]:
    """The dry-run's printout, and whether every input can be sent to the model."""
    batches = [batch(name, texts) for name, texts in named_texts.items()]
    distinct = {t for texts in named_texts.values() for t in texts}
    total = sum(estimate_tokens(t) for t in distinct)
    padded = math.ceil(total * MARGIN)
    lines = [f"{'':24} {'texts':>6} {'est. tokens':>12} {'largest':>8} {'cost (US$)':>11}"]
    for b in batches:
        lines.append(
            f"{b.name:24} {b.texts:>6} {b.tokens:>12,} {b.largest:>8,} "
            f"{_usd(cost_usd(b.tokens, config)):>11}"
        )
    lines += [
        f"{'total, distinct texts':24} {len(distinct):>6} {total:>12,} {'':>8} "
        f"{_usd(cost_usd(total, config)):>11}",
        "",
        f"With a {MARGIN - 1:.0%} margin: {padded:,} tokens, US${_usd(cost_usd(padded, config))}; "
        f"embedding everything twice: US${_usd(cost_usd(2 * padded, config))}.",
        f"Model {config.model_id} in {config.region}, {config.dimensions} dimensions; "
        f"US${_price(config)} per million input tokens, read {config.price_read_on}.",
        "Tokens are estimated (the higher of characters / 4 and words × 1.3); "
        "Bedrock reports the real count on each call. A text in several chunkings is paid once.",
    ]
    age = (today - config.price_read_on).days
    if age > PRICE_STALE_AFTER_DAYS:
        lines.append(
            f"warning: the price was read {age} days ago; check it in the AWS Price List API "
            "and update config/embedding.yaml before a paid run"
        )
    problems = [p for b in batches for p in input_problems(b, config)]
    lines += [f"error: {p}" for p in problems]
    return lines, not problems


def input_problems(b: Batch, config: EmbeddingConfig) -> list[str]:
    """Why batch `b` can't be sent, if it can't: Bedrock refuses empty input, and long input."""
    problems = []
    if math.ceil(b.largest * MARGIN) > config.max_input_tokens:
        problems.append(
            f"{b.name} has an input of about {b.largest:,} tokens; with the margin "
            f"that may be over the model's limit of {config.max_input_tokens:,}"
        )
    if b.empty:
        problems.append(f"{b.name} has {b.empty} empty input(s), which Bedrock refuses")
    return problems
