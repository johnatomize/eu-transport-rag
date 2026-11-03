"""The embedding model's settings, and an estimate of what embedding will cost (design §E).

`config/embedding.yaml` names the provider: `local`, a model run on this machine, which costs
nothing (Qwen3-Embedding 0.6B since 2026-11-03), or `bedrock`, a model paid per token on AWS
(Titan Text Embeddings V2, whose limit AWS left at zero). The checks on cost below apply to
Bedrock; for a local model the same report still checks every input against the model's limit.

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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

MARGIN = 1.5  # estimates are multiplied by this before they are priced or checked
PRICE_STALE_AFTER_DAYS = 7
TITAN_V2_DIMENSIONS = (256, 512, 1024)
PROVIDERS = ("local", "bedrock")
_REVISION = re.compile(r"[0-9a-f]{40}")  # a Hugging Face commit, so the model files never change
_COMMON = {
    "provider": str,
    "model_id": str,
    "dimensions": int,
    "normalize": bool,
    "max_input_tokens": int,
}
_BY_PROVIDER = {
    "local": {"revision": str, "device": str, "query_prefix": str},
    "bedrock": {
        "region": str,
        "price_usd_per_million_input_tokens": float,
        "price_read_on": dt.date,
    },
}
DEVICES = ("mps", "cpu")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str  # "local" or "bedrock"
    model_id: str
    dimensions: int
    normalize: bool
    max_input_tokens: int
    region: str | None = None  # Bedrock only
    price_usd_per_million_input_tokens: float = 0.0  # Bedrock only; a local model costs nothing
    price_read_on: dt.date | None = None  # Bedrock only
    revision: str | None = None  # local only: the model's commit on Hugging Face
    device: str | None = None  # local only: "mps" (the Mac's GPU) or "cpu"
    query_prefix: str = ""  # local only: put before a question, never a chunk (`query_input`)

    @property
    def paid(self) -> bool:
        return self.provider == "bedrock"


def query_input(question: str, config: EmbeddingConfig) -> str:
    """What the model is given for a question: the config's query prefix, then the question.

    Qwen3-Embedding is trained to see an instruction before a search query and nothing before
    the passages it searches, so a question and a chunk with the same words still get different
    vectors. Titan takes no prefix. Every caller that embeds or looks up a question goes
    through here, so the questions' cached vectors and the searches always agree.
    """
    return config.query_prefix + question


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
    provider = raw.get("provider")
    if provider not in PROVIDERS:
        raise ConfigError(f"{path.name}: provider should be one of {PROVIDERS}")
    allowed = _COMMON | _BY_PROVIDER[provider]
    price = raw.get("price_usd_per_million_input_tokens")
    if type(price) is int:  # `type`, not isinstance: a YAML `true` must not become 1.0
        raw["price_usd_per_million_input_tokens"] = float(price)
    known = _COMMON | _BY_PROVIDER["local"] | _BY_PROVIDER["bedrock"]
    problems = [f"missing {k}" for k in allowed if k not in raw]
    for k in raw:
        if k not in known:
            problems.append(f"unknown {k}")
        elif k not in allowed:
            problems.append(f"{k} is not a setting of a {provider} model")
    problems += [
        f"{k} should be {'a date' if kind is dt.date else kind.__name__}"
        for k, kind in allowed.items()
        if k in raw and type(raw[k]) is not kind
    ]
    if not problems:
        if raw["max_input_tokens"] <= 0:
            problems.append("max_input_tokens should be above 0")
        if provider == "bedrock":
            if raw["price_usd_per_million_input_tokens"] <= 0:
                problems.append("price_usd_per_million_input_tokens should be above 0")
            if raw["dimensions"] not in TITAN_V2_DIMENSIONS:
                problems.append(f"dimensions should be one of {TITAN_V2_DIMENSIONS}")
        else:
            if raw["dimensions"] <= 0:
                problems.append("dimensions should be above 0")
            if not _REVISION.fullmatch(raw["revision"]):
                problems.append("revision should be a full 40-character commit hash")
            if raw["device"] not in DEVICES:
                problems.append(f"device should be one of {DEVICES}")
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
    empty: int  # inputs with no text, which no model can embed


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

    def cost(tokens: int) -> str:  # a local model costs nothing, so it gets no cost column
        return f" {_usd(cost_usd(tokens, config)):>11}" if config.paid else ""

    header = f" {'cost (US$)':>11}" if config.paid else ""
    lines = [f"{'':24} {'texts':>6} {'est. tokens':>12} {'largest':>8}{header}"]
    for b in batches:
        lines.append(f"{b.name:24} {b.texts:>6} {b.tokens:>12,} {b.largest:>8,}{cost(b.tokens)}")
    lines += [
        f"{'total, distinct texts':24} {len(distinct):>6} {total:>12,} {'':>8}{cost(total)}",
        "",
    ]
    if config.paid:
        lines.append(
            f"With a {MARGIN - 1:.0%} margin: {padded:,} tokens, "
            f"US${_usd(cost_usd(padded, config))}; "
            f"embedding everything twice: US${_usd(cost_usd(2 * padded, config))}."
        )
    lines += _model_lines(config)
    age = (today - config.price_read_on).days if config.paid else 0
    if age > PRICE_STALE_AFTER_DAYS:
        lines.append(
            f"warning: the price was read {age} days ago; check it in the AWS Price List API "
            "and update config/embedding.yaml before a paid run"
        )
    problems = [p for b in batches for p in input_problems(b, config)]
    lines += [f"error: {p}" for p in problems]
    return lines, not problems


def _model_lines(config: EmbeddingConfig) -> list[str]:
    if not config.paid:
        return [
            f"Model {config.model_id} at revision {config.revision}, {config.dimensions} "
            "dimensions, run on this machine: it costs nothing.",
            "Tokens are estimated (the higher of characters / 4 and words × 1.3); the run counts "
            "them with the model's own tokenizer and refuses any input over the limit.",
        ]
    return [
        f"Model {config.model_id} in {config.region}, {config.dimensions} dimensions; "
        f"US${_price(config)} per million input tokens, read {config.price_read_on}.",
        "Tokens are estimated (the higher of characters / 4 and words × 1.3); "
        "Bedrock reports the real count on each call. A text in several chunkings is paid once.",
    ]


def input_problems(b: Batch, config: EmbeddingConfig) -> list[str]:
    """Why batch `b` can't be embedded, if it can't: empty input, or input that may be too long."""
    problems = []
    if math.ceil(b.largest * MARGIN) > config.max_input_tokens:
        problems.append(
            f"{b.name} has an input of about {b.largest:,} tokens; with the margin "
            f"that may be over the model's limit of {config.max_input_tokens:,}"
        )
    if b.empty:
        problems.append(f"{b.name} has {b.empty} empty input(s), which the model refuses")
    return problems
