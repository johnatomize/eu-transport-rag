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
  output. The SDK's own retries are off: `send` retries throttling and passing server errors
  itself, at most `MAX_ATTEMPTS` tries with growing waits, and a call that still fails stops
  the run. It is logged as failed and never counted as an answer.
- **Record.** `answer_all` checks each reply (`validate.check_reply`) and keeps, for the run
  log, the retrieved chunk IDs, the raw reply, the verdict and its reasons, the tokens the API
  counted, the cost at the configured price and the latency.

Choices and reasons: "the answer client" in docs/DECISIONS.md.
"""

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rag.chunk import Chunk
from rag.embed_estimate import estimate_tokens
from rag.generation_config import GenerationConfig, Price
from rag.prompt import ANSWER_SCHEMA, SYSTEM, Prompt
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
    """A call refused for now: too many requests, an overloaded or failing server, a timeout or
    a dropped connection. Tried again, up to `MAX_ATTEMPTS`."""


class CallFailed(GenerateError):
    """A call that trying again won't fix: a bad key, an unknown model, a refused request."""


def prompt_version() -> str:
    """A short fingerprint of the instructions and the answer schema, for the run log."""
    text = SYSTEM + json.dumps(ANSWER_SCHEMA, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# --- Estimating --------------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    """One question as it would be sent: its prompt and the chunks the prompt was built from."""

    question_id: str
    question: str
    prompt: Prompt
    chunks: tuple[Chunk, ...]

    @property
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

    @property
    def padded_input(self) -> int:
        return math.ceil(self.input_tokens * MARGIN)

    @property
    def expected_output(self) -> int:
        return math.ceil(self.calls * EXPECTED_OUTPUT_TOKENS * MARGIN)

    @property
    def expected_usd(self) -> float:
        return cost_usd(self.padded_input, self.expected_output, self.price)

    @property
    def most_usd(self) -> float:
        """The most the run can cost: the margin on input, every answer at its longest."""
        return cost_usd(self.padded_input, self.calls * self.max_output_tokens, self.price)


def estimate(calls: Sequence[Call], config: GenerationConfig, model: str) -> Estimate:
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

    The answer format is passed as the reply's JSON schema, so the reply arrives as one text
    block of JSON. Thinking is on, as these models always think, at the configured effort.
    """

    def __init__(self, config: GenerationConfig, model: str, api_key: str, http_client=None):
        import anthropic  # here, so that nothing without a client needs the SDK to start

        self._anthropic = anthropic
        self.model = model
        self.config = config
        self._client = anthropic.Anthropic(
            api_key=api_key, max_retries=0, timeout=TIMEOUT_SECONDS, http_client=http_client
        )

    def send(self, prompt: Prompt) -> Reply:
        anthropic = self._anthropic
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.config.max_output_tokens,
                system=prompt.system,
                messages=[{"role": "user", "content": prompt.user}],
                output_config={
                    "effort": self.config.effort,
                    "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
                },
            )
        except anthropic.APIConnectionError as e:  # timeouts included
            raise Throttled(f"{type(e).__name__}: {e}") from e
        except anthropic.APIStatusError as e:
            failure = f"HTTP {e.status_code} {type(e).__name__}: {e.message}"
            if e.status_code == 429 or e.status_code >= 500:
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


def _failed_stop(reply: Reply) -> str | None:
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


def answer_all(
    calls: Sequence[Call],
    client: AnswerClient,
    price: Price,
    progress: Callable[[int, int], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult:
    """Send each call, check each reply, and keep everything a run log needs.

    A call that fails after its retries, or an interruption, stops the run: the answers so far
    are kept, with the error, and the rest are not sent.
    """
    result = RunResult()
    try:
        for done, call in enumerate(calls, 1):
            started = clock()
            reply, attempts = send_with_retries(client, call.prompt, sleep)
            seconds = clock() - started
            # Paid for from here on: recorded before anything can refuse it.
            stopped = _failed_stop(reply)
            if stopped:
                verdict = {"ok": False, "answerable": None, "problems": [stopped], "citations": []}
            else:
                assert reply.text is not None
                checked = check_reply(reply.text, call.chunks)
                verdict = {
                    "ok": checked.ok,
                    "answerable": checked.answer.answerable if checked.answer else None,
                    "problems": list(checked.problems),
                    "citations": [
                        {"node_id": c.node_id, "quote": c.quote} for c in checked.citations
                    ],
                }
            result.answers.append(
                {
                    "id": call.question_id,
                    "retrieved": [c.id for c in call.chunks],
                    "estimated_input_tokens": call.input_tokens,
                    "raw": reply.text,
                    "stop_reason": reply.stop_reason,
                    "verdict": verdict,
                    "usage": {
                        "input_tokens": reply.input_tokens,
                        "output_tokens": reply.output_tokens,
                        "cache_write_tokens": reply.cache_write_tokens,
                        "cache_read_tokens": reply.cache_read_tokens,
                    },
                    "cost_usd": round(reply.cost_usd(price), 8),
                    "seconds": round(seconds, 2),
                    "attempts": attempts,
                    "model": reply.model,
                    "request_id": reply.request_id,
                }
            )
            if progress:
                progress(done, len(calls))
    except (GenerateError, KeyboardInterrupt) as e:
        result.error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
    return result
