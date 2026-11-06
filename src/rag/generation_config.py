"""The Claude models that write answers, and their prices, from `config/generation.yaml`.

`build_model` answers while checkpoint F is built and debugged; `scored_model` writes every
answer that G scores; `judge_model` judges faithfulness. Choices and reasons: "Haiku 5.5
while building, Opus 5.5 for scored answers" and "Opus 5.5 judges faithfulness" in
docs/DECISIONS.md.
"""

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

PROVIDERS = ("anthropic",)
ROLES = ("build", "scored")
PRICE_KINDS = ("input", "output", "batch_input", "batch_output")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
_SETTINGS = {
    "provider": str,
    "build_model": str,
    "scored_model": str,
    "judge_model": str,
    "effort": str,
    "max_output_tokens": int,
    "max_prompt_tokens": int,
    "prices_read_on": dt.date,
    "prices": dict,
}


class GenerationConfigError(Exception):
    pass


@dataclass(frozen=True)
class Price:
    """US$ per million tokens."""

    input: float
    output: float
    batch_input: float
    batch_output: float

    def for_batch(self) -> "Price":
        """The batch service's prices as the input and output prices, to cost a batch's calls."""
        return Price(self.batch_input, self.batch_output, self.batch_input, self.batch_output)


@dataclass(frozen=True)
class GenerationConfig:
    provider: str
    build_model: str
    scored_model: str
    judge_model: str  # judges faithfulness (#83)
    effort: str
    max_output_tokens: int
    max_prompt_tokens: int
    prices_read_on: dt.date
    prices: dict[str, Price]

    def model(self, role: str) -> str:
        """The model for a role: "build" while building F, "scored" for answers G scores."""
        if role not in ROLES:
            raise ValueError(f"role should be one of {ROLES}, not {role!r}")
        return self.build_model if role == "build" else self.scored_model


def _price(model: str, raw: object) -> tuple[Price | None, list[str]]:
    if not isinstance(raw, dict):
        return None, [f"prices of {model} should be a mapping"]
    problems = [f"prices of {model}: missing {k}" for k in PRICE_KINDS if k not in raw]
    problems += [f"prices of {model}: unknown {k}" for k in raw if k not in PRICE_KINDS]
    for k in PRICE_KINDS:
        # `type`, not isinstance: a YAML `true` must not become 1.0.
        number = k in raw and type(raw[k]) in (int, float) and math.isfinite(raw[k])
        if k in raw and (not number or raw[k] <= 0):
            problems.append(f"prices of {model}: {k} should be a number above 0")
    if problems:
        return None, problems
    return Price(**{k: float(raw[k]) for k in PRICE_KINDS}), []


def load_generation_config(path: Path) -> GenerationConfig:
    """Read `config/generation.yaml`, refusing missing, unknown, mistyped or impossible settings."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise GenerationConfigError(f"{path.name}: cannot read it ({e.strerror})") from e
    except yaml.YAMLError as e:
        raise GenerationConfigError(f"{path.name}: not valid YAML ({e})") from e
    if not isinstance(raw, dict):
        raise GenerationConfigError(f"{path.name}: expected a mapping of settings")
    problems = [f"missing {k}" for k in _SETTINGS if k not in raw]
    problems += [f"unknown {k}" for k in raw if k not in _SETTINGS]
    problems += [
        f"{k} should be {'a date' if kind is dt.date else kind.__name__}"
        for k, kind in _SETTINGS.items()
        if k in raw and type(raw[k]) is not kind
    ]
    prices: dict[str, Price] = {}
    if not problems:
        if raw["provider"] not in PROVIDERS:
            problems.append(f"provider should be one of {PROVIDERS}")
        if raw["effort"] not in EFFORTS:
            problems.append(f"effort should be one of {EFFORTS}")
        for k in ("max_output_tokens", "max_prompt_tokens"):
            if raw[k] <= 0:
                problems.append(f"{k} should be above 0")
        for model, entry in raw["prices"].items():
            price, found = _price(model, entry)
            problems += found
            if price is not None:
                prices[model] = price
        for role in (*ROLES, "judge"):
            model = raw[f"{role}_model"]
            if model not in raw["prices"]:
                problems.append(f"{role}_model {model} has no prices")
    if problems:
        raise GenerationConfigError(f"{path.name}: " + "; ".join(problems))
    return GenerationConfig(**{**raw, "prices": prices})
