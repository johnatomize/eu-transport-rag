"""Answers from Claude through Anthropic's API: what a run would cost, and the run itself.

Design §F. Every call is paid for, so nothing here sends anything unless it is handed a client,
and `scripts/answer.py` hands it one only with an explicit flag, after the checks below.

- **Estimate.** `estimate` prices a list of planned calls before anything is sent. Input tokens
  are estimated from the prompt's text, as for embeddings (`embed_estimate.estimate_tokens`),
  plus an allowance for the answer schema, then raised by `MARGIN`. Output is the uncertain
  part, since the model's thinking is billed as output, so two figures are given: the
  expected cost, at `EXPECTED_OUTPUT_TOKENS` an answer with the margin, and the most the run
  can cost, with every answer using all of `max_output_tokens`.
- **Cap.** A paid run may start only if the most it can cost, on top of what has been spent,
  stays within the approved cap (`check_cap`), and only if no prompt may be over
  `max_prompt_tokens`, above which Haiku 5.5's price rises.
- **Calls.** `AnthropicClient` sends one prompt and asks for the answer format as structured
  output. The SDK's own retries are off: `send_with_retries` tries again only after refusals
  that aren't billed (429, 5xx), at most `MAX_ATTEMPTS` tries with growing waits. A timeout or
  dropped connection may have been billed, so it is not tried again. A call that fails stops
  the run; it is logged as failed and never counted as an answer.
- **Record.** `answer_all` checks each reply (`validate.check_reply`) and keeps, for the run
  log, the retrieved chunk IDs, the raw reply, the verdict and its reasons, the tokens the API
  counted, the cost at the configured price and the latency, and hands the record to `save`
  after every call, so a run stopped by anything leaves every paid reply on disk.

Choices and reasons: "the answer client" in docs/DECISIONS.md.
"""

import copy
import datetime as dt
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Protocol

import anthropic

from rag.chunk import Chunk
from rag.db import read_env_file
from rag.embed_estimate import PRICE_STALE_AFTER_DAYS, estimate_tokens
from rag.generation_config import GenerationConfig, Price
from rag.prompt import ANSWER_SCHEMA, SYSTEM, Prompt
from rag.runlog import git_state
from rag.validate import check_reply

MARGIN = 1.5  # estimates are multiplied by this before they are priced
EXPECTED_OUTPUT_TOKENS = 1000  # an answer, thinking included: the expected figure only
# The API adds its own instructions for a JSON schema. Its size isn't published, so the
# schema's own length is counted twice over, plus a round allowance.
SCHEMA_TOKENS = 2 * estimate_tokens(json.dumps(ANSWER_SCHEMA)) + 200
MAX_ATTEMPTS = 4  # tries a call, the first included
BACKOFF_SECONDS = (5.0, 15.0, 45.0)  # waits before the second, third and fourth tries
TIMEOUT_SECONDS = 300.0
# Cache writes are billed at 1.25 times the input price. No prompt asks for caching, so both
# cache counts should be 0; if not, reads are counted at the full input price, which may
# overstate the cost but never understates it.
CACHE_WRITE_FACTOR = 1.25


class GenerateError(Exception):
    pass


class OverCapError(GenerateError):
    """A paid run whose most possible cost would take the spend log past the approved cap."""


class Throttled(GenerateError):
    """A call refused for now, and not billed: too many requests (429), or an overloaded or
    failing server (5xx, 529 included). Tried again, up to `MAX_ATTEMPTS`."""


class CallFailed(GenerateError):
    """A call not tried again: a bad key, an unknown model, a refused request, or a timeout or
    dropped connection, after which the call may have been billed."""


def prompt_version() -> str:
    """A short fingerprint of the instructions and the answer schema, for the run log."""
    text = SYSTEM + json.dumps(ANSWER_SCHEMA, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# --- Estimating --------------------------------------------------------------------------


class Sendable(Protocol):
    """One call as `estimate` and `send_all` need it: an answer's `Call`, or a judgement's
    (`rag.judge.JudgeCall`)."""

    question_id: str
    prompt: Prompt

    @property
    def input_tokens(self) -> int: ...


@dataclass(frozen=True)
class Call:
    """One question as it would be sent: its prompt and the chunks the prompt was built from."""

    question_id: str
    question: str
    prompt: Prompt
    chunks: tuple[Chunk, ...]

    @cached_property
    def input_tokens(self) -> int:
        """Estimated, before the margin."""
        return (
            estimate_tokens(self.prompt.system) + estimate_tokens(self.prompt.user) + SCHEMA_TOKENS
        )


def cost_usd(input_tokens: float, output_tokens: float, price: Price) -> float:
    return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


@dataclass(frozen=True)
class Estimate:
    calls: int
    input_tokens: int  # estimated, before the margin
    largest_input: int  # the longest prompt, estimated, before the margin
    max_output_tokens: int
    model: str
    price: Price
    expected_per_call: int = EXPECTED_OUTPUT_TOKENS  # output tokens a call, thinking included

    @property
    def padded_input(self) -> int:
        return math.ceil(self.input_tokens * MARGIN)

    @property
    def expected_output(self) -> int:
        return math.ceil(self.calls * self.expected_per_call * MARGIN)

    @property
    def expected_usd(self) -> float:
        return cost_usd(self.padded_input, self.expected_output, self.price)

    @property
    def most_usd(self) -> float:
        """The most the run can cost: the margin on input, every answer at its longest."""
        return cost_usd(self.padded_input, self.calls * self.max_output_tokens, self.price)


def estimate(
    calls: Sequence[Sendable],
    config: GenerationConfig,
    model: str,
    expected_per_call: int = EXPECTED_OUTPUT_TOKENS,
) -> Estimate:
    if model not in config.prices:
        raise GenerateError(f"{model} has no price in config/generation.yaml")
    counts = [c.input_tokens for c in calls]
    return Estimate(
        calls=len(calls),
        input_tokens=sum(counts),
        largest_input=max(counts, default=0),
        max_output_tokens=config.max_output_tokens,
        model=model,
        price=config.prices[model],
        expected_per_call=expected_per_call,
    )


def input_problems(e: Estimate, config: GenerationConfig) -> list[str]:
    """Why the calls can't be sent at the configured price: a prompt that may be too long."""
    if math.ceil(e.largest_input * MARGIN) > config.max_prompt_tokens:
        return [
            f"the longest prompt is about {e.largest_input:,} tokens, which with the margin "
            f"may pass max_prompt_tokens ({config.max_prompt_tokens:,})"
        ]
    return []


def check_cap(e: Estimate, spent_usd: float, cap_usd: float) -> None:
    """Refuse a run whose most possible cost, on top of what has been spent, passes the cap."""
    if spent_usd + e.most_usd > cap_usd:
        raise OverCapError(
            f"this run may cost up to US${e.most_usd:.4f}; with US${spent_usd:.4f} already "
            f"spent that could pass the approved cap of US${cap_usd:.2f}"
        )


def paid_holds(
    e: Estimate, config: GenerationConfig, spent_usd: float, cap_usd: float, today: dt.date
) -> list[str]:
    """Why a paid run may not start: prices read too long ago, or a cost that could pass the
    cap. Empty when it may."""
    holds = []
    age = (today - config.prices_read_on).days
    if age > PRICE_STALE_AFTER_DAYS:
        holds.append(
            f"the prices were read {age} days ago; read them again on Anthropic's pricing page "
            "and update config/generation.yaml"
        )
    try:
        check_cap(e, spent_usd, cap_usd)
    except GenerateError as err:
        holds.append(str(err))
    return holds


def estimate_report(e: Estimate, unit: str = "answer") -> str:
    """The estimate in words: input with the margin, the expected cost and the most it can cost."""
    calls = f"{e.calls} call{'s' if e.calls != 1 else ''}"
    return (
        f"{calls}: about {e.input_tokens:,} input tokens, {e.padded_input:,} with a "
        f"{MARGIN - 1:.0%} margin.\n"
        f"Expected cost, at {e.expected_per_call:,} output tokens {article(unit)} {unit} plus "
        f"the margin: US${e.expected_usd:.4f}.\n"
        f"Most it can cost, every {unit} at {e.max_output_tokens:,} output tokens: "
        f"US${e.most_usd:.4f}. The cap check uses this figure."
    )


def article(word: str) -> str:
    return "an" if word[:1] in "aeiou" else "a"


# --- Calling the model ---------------------------------------------------------------------


@dataclass(frozen=True)
class Reply:
    """What the API returned for one call, as it returned it."""

    text: str | None  # the reply's text, None if it has none
    stop_reason: str | None
    refusal_category: str | None
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    model: str
    request_id: str | None

    def cost_usd(self, price: Price) -> float:
        billed_input = (
            self.input_tokens
            + self.cache_write_tokens * CACHE_WRITE_FACTOR
            + self.cache_read_tokens
        )
        return cost_usd(billed_input, self.output_tokens, price)


class AnswerClient(Protocol):
    def send(self, prompt: Prompt) -> Reply:
        """One call. Raises `Throttled` for what may pass, `CallFailed` for what won't."""
        ...


class AnthropicClient:
    """Claude through Anthropic's Messages API. Each call is paid for.

    The answer format (or the judge's, `schema`) is passed as the reply's JSON schema, so the
    reply arrives as one text block of JSON. Thinking is on, as these models always think, at
    the configured effort.
    """

    def __init__(
        self,
        config: GenerationConfig,
        model: str,
        api_key: str,
        http_client=None,
        schema: dict = ANSWER_SCHEMA,
    ):
        self.model = model
        self.config = config
        self.schema = schema  # the reply's format: the answer's, or the judge's (`rag.judge`)
        self._client = anthropic.Anthropic(
            api_key=api_key, max_retries=0, timeout=TIMEOUT_SECONDS, http_client=http_client
        )

    def send(self, prompt: Prompt) -> Reply:
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.config.max_output_tokens,
                system=prompt.system,
                messages=[{"role": "user", "content": prompt.user}],
                output_config={
                    "effort": self.config.effort,
                    "format": {"type": "json_schema", "schema": self.schema},
                },
            )
        except anthropic.APIConnectionError as e:
            # A timeout or a dropped connection: the model may have answered, and been paid
            # for, without the reply arriving. Trying again could pay twice, so it isn't.
            raise CallFailed(f"{type(e).__name__} (may have been billed): {e}") from e
        except anthropic.APIStatusError as e:
            failure = f"HTTP {e.status_code} {type(e).__name__}: {e.message}"
            if e.status_code == 429 or e.status_code >= 500:  # refused or failed, not billed
                raise Throttled(failure) from e
            raise CallFailed(failure) from e
        except anthropic.AnthropicError as e:  # a reply the SDK couldn't read, say
            raise CallFailed(f"{type(e).__name__}: {e}") from e
        texts = [block.text for block in message.content if block.type == "text"]
        usage = message.usage
        details = message.stop_details
        return Reply(
            text="".join(texts) if texts else None,
            stop_reason=message.stop_reason,
            refusal_category=getattr(details, "category", None) if details else None,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            model=message.model,
            request_id=getattr(message, "_request_id", None),
        )


def send_with_retries(
    client: AnswerClient, prompt: Prompt, sleep: Callable[[float], None] = time.sleep
) -> tuple[Reply, int]:
    """Send `prompt`, trying again after throttling; the reply and the number of tries.

    Raises the last `Throttled` after `MAX_ATTEMPTS` tries, and `CallFailed` at once.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return client.send(prompt), attempt
        except Throttled as e:
            if attempt == MAX_ATTEMPTS:
                raise Throttled(f"gave up after {attempt} tries: {e}") from e
            sleep(BACKOFF_SECONDS[attempt - 1])
    raise AssertionError("unreachable")


# --- A run ----------------------------------------------------------------------------------


@dataclass
class RunResult:
    """Every call a run made, in order, for the run log and the spend log."""

    answers: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)  # calls that got no reply: head and error
    error: str | None = None

    @property
    def input_tokens(self) -> int:
        return sum(a["usage"]["input_tokens"] for a in self.answers)

    @property
    def output_tokens(self) -> int:
        return sum(a["usage"]["output_tokens"] for a in self.answers)

    @property
    def cost_usd(self) -> float:
        return sum(a["cost_usd"] for a in self.answers)

    @property
    def passed(self) -> int:
        return sum(1 for a in self.answers if a["verdict"]["ok"])

    @property
    def refusals(self) -> int:
        """Answers that passed the check by saying the documents don't answer."""
        return sum(1 for a in self.answers if a["verdict"]["ok"] and not a["verdict"]["answerable"])


def failed_stop(reply: Reply) -> str | None:
    """Why a reply that came back can't be an answer, before its text is read."""
    if reply.stop_reason == "refusal":
        return f"the model declined to answer (category {reply.refusal_category})"
    if reply.stop_reason == "max_tokens":
        return "the answer was cut off at max_output_tokens"
    if reply.stop_reason != "end_turn":
        return f"the reply stopped for an unexpected reason: {reply.stop_reason}"
    if reply.text is None:
        return "the reply had no text"
    return None


def _verdict(reply: Reply, call: Call) -> dict:
    """The check's verdict on a reply, for the run log."""
    stopped = failed_stop(reply)
    if stopped:
        return {
            "ok": False,
            "answerable": None,
            "answer": None,
            "problems": [stopped],
            "citations": [],
        }
    assert reply.text is not None
    checked = check_reply(reply.text, call.chunks)
    return {
        "ok": checked.ok,
        "answerable": checked.answer.answerable if checked.answer else None,
        "answer": checked.answer.answer if checked.answer else None,  # the model's text, as read
        "problems": list(checked.problems),
        "citations": [{"node_id": c.node_id, "quote": c.quote} for c in checked.citations],
    }


def answer_all(
    calls: Sequence[Call],
    client: AnswerClient,
    price: Price,
    progress: Callable[[int, int], None] | None = None,
    save: Callable[[RunResult], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult:
    """Send each call, check each reply, and keep everything a run log needs (`send_all`)."""

    def head(call: Call) -> dict:
        return {
            "id": call.question_id,
            "retrieved": [c.id for c in call.chunks],
            "estimated_input_tokens": call.input_tokens,
        }

    blank = {"ok": False, "answerable": None, "answer": None, "problems": [], "citations": []}
    return send_all(calls, client, price, head, _verdict, blank, progress, save, sleep, clock)


def send_all[C: Sendable](
    calls: Sequence[C],
    client: AnswerClient,
    price: Price,
    head: Callable[[C], dict],
    verdict: Callable[[Reply, C], dict],
    blank: dict,
    progress: Callable[[int, int], None] | None = None,
    save: Callable[[RunResult], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult:
    """Send each call and keep, for the run log, what `head` says about the call, the raw
    reply, `verdict`'s reading of it, the tokens the API counted, the cost and the time.

    The entry is kept before `verdict` reads the reply, with `blank` as its verdict, so a bug
    in the reading fails that one entry and never loses a paid reply. `save` is called after
    every call, answered or failed, so the run log on disk always holds every reply paid for,
    whatever stops the run. A call that fails after its retries, an interruption or any other
    error stops the run: the entries so far are kept, the failed call is listed in `failed`
    with what `head` says about it and the error, and the rest are not sent.
    """
    result = RunResult()
    try:
        for done, call in enumerate(calls, 1):
            started = clock()
            try:
                reply, attempts = send_with_retries(client, call.prompt, sleep)
            except (Exception, KeyboardInterrupt) as e:
                result.failed.append({**head(call), "error": _describe(e)})
                raise
            # Paid for from here on: recorded before the reply is read, which can't then lose it.
            entry = {
                **head(call),
                "raw": reply.text,
                "stop_reason": reply.stop_reason,
                "verdict": copy.deepcopy(blank),
                "usage": {
                    "input_tokens": reply.input_tokens,
                    "output_tokens": reply.output_tokens,
                    "cache_write_tokens": reply.cache_write_tokens,
                    "cache_read_tokens": reply.cache_read_tokens,
                },
                "cost_usd": round(reply.cost_usd(price), 8),
                "seconds": round(clock() - started, 2),
                "attempts": attempts,
                "model": reply.model,
                "request_id": reply.request_id,
            }
            result.answers.append(entry)
            try:
                entry["verdict"] = verdict(reply, call)
            except Exception as e:  # a bug in the reading: the entry fails, the run goes on
                entry["verdict"]["problems"] = [f"the check itself failed: {_describe(e)}"]
            if save:
                save(result)
            if progress:
                progress(done, len(calls))
    except (Exception, KeyboardInterrupt) as e:
        result.error = _describe(e)
    if save:
        save(result)
    return result


def _describe(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}" if str(e) else type(e).__name__


# --- The run log ----------------------------------------------------------------------------


def run_log(
    run: dict,
    config: GenerationConfig,
    model: str,
    retrieval: dict,
    split: str,
    e: Estimate,
    result: RunResult,
) -> dict:
    """What a paid run's log holds: the run, its settings, the estimate, and every call made.

    `split` is the question set: "dev", or "typed" for a question asked with `make ask`.
    `["paid"]["cost_usd"]` is what `rag.spend` counts until the spend log names the file.
    """
    price = config.prices[model]
    return {
        "run": run,
        "config": {
            "model": model,
            "effort": config.effort,
            "max_output_tokens": config.max_output_tokens,
            "prompt_version": prompt_version(),
            "price_usd_per_million": {"input": price.input, "output": price.output},
            "prices_read_on": config.prices_read_on.isoformat(),
            "retrieval": retrieval,
            "split": split,
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


def read_api_key(env: Path) -> str | None:
    """ANTHROPIC_API_KEY from the environment, or else from the `.env` file at `env`."""
    return os.environ.get("ANTHROPIC_API_KEY") or read_env_file(env).get("ANTHROPIC_API_KEY")


def new_run(root: Path, runs: Path, model: str, prefix: str = "") -> tuple[dict, Path]:
    """The run's header (time, commit, uncommitted changes) and a new, empty log file for it.

    The log is named `<prefix><time>-<model>.json` and created exclusively, so two runs started
    in the same second can't share one log and lose a paid reply's record: the second is
    refused (`FileExistsError`, an `OSError`) before it sends anything.
    """
    commit, dirty = git_state(root)
    created = datetime.now(UTC)
    run = {"created": created.isoformat(timespec="seconds"), "commit": commit, "dirty": dirty}
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{prefix}{created.strftime('%Y-%m-%dT%H-%M-%SZ')}-{model}.json"
    path.open("x").close()
    return run, path


def write_run_log(path: Path, log: dict) -> None:
    """Write the log whole, through a temporary file, so it is never left half written.

    If it can't be written, it is printed before the error is raised: the calls in it are paid
    for, and their record must not be lost.
    """
    text = json.dumps(log, indent=1, ensure_ascii=False) + "\n"
    try:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except OSError:
        print(text)
        raise
