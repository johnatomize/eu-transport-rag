"""Runs sent through Anthropic's Message Batches API, at half price (#79).

Design §G. A run's calls go as one batch, which the API answers within 24 hours, mostly within
one, at half the price of direct calls. `make answers BATCH=1` and `make judge … BATCH=1` build
the same calls as a direct run and pass the same checks before paying, at the batch prices
(`rag.generate.estimate` with `batch=True`): the cap check still assumes every call uses all
of `max_output_tokens`.

- **The batch's record** is the run log's `batch`: its ID, its status, and for each call its
  `custom_id`, its question and a fingerprint of the exact request. The log is written with
  status "sending" before the batch is created, and again with the ID as soon as the API
  returns it, before anything else happens.
- **Picking up.** A run stopped while it waits (Ctrl-C, a dropped connection, a laptop gone to
  sleep) leaves its log with status "submitted". Running the same command again finds that log,
  because its requests have the same fingerprints (`find_batch`), and waits for that batch
  instead of sending another, so nothing is paid for twice.
- **Spend.** Until a batch's results are in its log, `rag.spend` counts the most the batch can
  cost, so a run started meanwhile can't pass the cap.
- **Results** are matched to calls by `custom_id`, in whatever order they come. A reply is
  recorded as a direct run records it (`generate.paid_entry`), at the batch price, with no
  `seconds`: a reply from a batch has no time of its own, so speed is measured on direct runs
  only. A request that errored, expired or was canceled, or has no result, is listed in
  `failed` with the reason, is not billed, and is never counted as an answer.
- **When the batch can't be created:** a refusal for too many requests (429) is tried again,
  as for a direct call; any other refusal leaves status "not_sent", which costs nothing. A
  server error, a timeout, a dropped connection or Ctrl-C while the request is out leave
  "unknown", as the batch may have been stored without its ID reaching the log; the spend check
  keeps counting its most, and the same command refuses to send it again until someone has
  looked in the Console (see `find_batch`). Reading a batch's progress or results costs
  nothing, so a dropped connection there is tried again too.
- **A direct run of the same calls** is refused while a batch of them may still be paid for.

Choices and reasons: "runs through the batch service" in docs/DECISIONS.md.
"""

import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import anthropic

from rag.generate import (
    BACKOFF_SECONDS,
    MAX_ATTEMPTS,
    TIMEOUT_SECONDS,
    GenerateError,
    MaybeSent,
    Reply,
    RunResult,
    Sendable,
    Throttled,
    classify,
    describe,
    paid_entry,
    read_into,
    request_params,
    to_reply,
    with_retries,
)
from rag.generation_config import GenerationConfig, Price
from rag.prompt import ANSWER_SCHEMA, Prompt
from rag.spend import BATCH_SETTLED

SENDING, SUBMITTED, COLLECTED, NOT_SENT, UNKNOWN = (
    "sending", "submitted", "collected", "not_sent", "unknown"
)  # fmt: skip
assert set(BATCH_SETTLED) == {COLLECTED, NOT_SENT}
POLL_SECONDS = 60.0  # between two looks at the batch's progress
_CUSTOM_ID = re.compile(r"[^A-Za-z0-9_-]")


class BatchError(GenerateError):
    pass


# --- The batch's record --------------------------------------------------------------------


def fingerprint(config: GenerationConfig, model: str, schema: dict, prompt: Prompt) -> str:
    """A short fingerprint of one call's exact request: model, settings, format and prompt."""
    params = request_params(config, model, schema, prompt)
    text = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def new_batch(
    calls: Sequence[Sendable], config: GenerationConfig, model: str, schema: dict
) -> dict:
    """The run log's `batch` for calls not sent yet: no ID, and each call's request.

    A `custom_id` is the call's place in the run and its question, `007-q012`, so it is unique
    even when two answers to one question are judged together.
    """
    requests = [
        {
            "custom_id": f"{i:03d}-{_CUSTOM_ID.sub('_', call.question_id)}"[:64],
            "id": call.question_id,
            "fingerprint": fingerprint(config, model, schema, call.prompt),
        }
        for i, call in enumerate(calls)
    ]
    return {"id": None, "status": SENDING, "submitted": None, "ended": None, "requests": requests}


def find_batch(runs: Path, batch: dict) -> tuple[Path, dict] | None:
    """The log in `runs` of a batch of exactly these requests whose results aren't in yet.

    Its status is "submitted" when it can be picked up, or "sending" or "unknown" when the batch
    may exist without its ID in the log. The caller refuses to send the same requests again
    then: someone has to look for the batch in the Console first, and if there is none, set
    that log's status to "not_sent".
    """
    for path in sorted(runs.glob("*.json")):
        try:
            text = path.read_text(encoding="utf-8")
            if '"batch"' not in text:  # a direct run: no need to parse it
                continue
            log = json.loads(text)
            found = log.get("batch")
            if found and found["status"] not in BATCH_SETTLED:
                if found["requests"] == batch["requests"]:
                    return path, log
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            raise BatchError(f"{path.name}: cannot read its batch ({e!r})") from e
    return None


def found_note(path: Path, log: dict, batch_run: bool = True) -> tuple[str, list[str]]:
    """What a run of calls whose batch `find_batch` found will do, and why it can't, if it can't.

    A batch run picks a submitted batch up. A direct run of the same calls is refused while the
    batch may still be paid for, as it would pay for every call again.
    """
    batch = log["batch"]
    if not batch_run:
        return (
            f"A batch of these calls, logged in {path.name}, may still be paid for (status "
            f"{batch['status']!r}).",
            ["a direct run would pay for them again: run with BATCH=1 to pick the batch up"],
        )
    if batch["status"] == SUBMITTED:
        return (
            f"Batch {batch['id']} of these calls, logged in {path.name}, is still on its way: "
            "a paid run picks it up and sends nothing new.",
            [],
        )
    return (
        f"A batch of these calls may already have been sent: {path.name} has no batch ID "
        f"(status {batch['status']!r}).",
        [
            "look for the batch on the Batches page of the Anthropic Console; if there is none, "
            f'set the batch status in {path.name} to "not_sent" and run again'
        ],
    )


# --- The client ----------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchStatus:
    ended: bool
    processing: int  # requests not finished yet


class BatchClient(Protocol):
    def create(self, requests: Sequence[tuple[str, Prompt]]) -> str:
        """Send the requests, `(custom_id, prompt)`, as one batch; its ID. Raises `Throttled`
        when nothing was created and it may pass (429), `MaybeSent` when the batch may exist,
        and `CallFailed` otherwise."""
        ...

    def status(self, batch_id: str) -> BatchStatus: ...

    def results(self, batch_id: str) -> list[tuple[str, Reply | str]]:
        """Every request's result, by `custom_id`: its reply, or why it has none."""
        ...


class AnthropicBatchClient:
    """Claude through Anthropic's Message Batches API. Each request that succeeds is paid for,
    at half the direct price. Each request is exactly what `AnthropicClient` would send."""

    def __init__(
        self,
        config: GenerationConfig,
        model: str,
        api_key: str,
        http_client=None,
        schema: dict = ANSWER_SCHEMA,
    ):
        self.config = config
        self.model = model
        self.schema = schema
        self._client = anthropic.Anthropic(
            api_key=api_key, max_retries=0, timeout=TIMEOUT_SECONDS, http_client=http_client
        )

    def create(self, requests: Sequence[tuple[str, Prompt]]) -> str:
        body = [
            {
                "custom_id": custom_id,
                "params": request_params(self.config, self.model, self.schema, prompt),
            }
            for custom_id, prompt in requests
        ]
        try:
            return self._client.messages.batches.create(requests=body).id  # type: ignore[arg-type]
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:  # the batch may have been stored before the error
                raise MaybeSent(f"HTTP {e.status_code} {type(e).__name__}: {e.message}") from e
            raise classify(e) from e
        except anthropic.AnthropicError as e:
            raise classify(e) from e

    def status(self, batch_id: str) -> BatchStatus:
        try:
            batch = self._client.messages.batches.retrieve(batch_id)
        except anthropic.AnthropicError as e:
            raise _reading(e) from e
        return BatchStatus(batch.processing_status == "ended", batch.request_counts.processing)

    def results(self, batch_id: str) -> list[tuple[str, Reply | str]]:
        out: list[tuple[str, Reply | str]] = []
        try:
            for item in self._client.messages.batches.results(batch_id):
                result = item.result
                if result.type == "succeeded":
                    out.append((item.custom_id, to_reply(result.message, None)))
                elif result.type == "errored":
                    error = result.error.error
                    out.append((item.custom_id, f"errored ({error.type}): {error.message}"))
                elif result.type == "expired":
                    out.append((item.custom_id, "expired: not answered within 24 hours"))
                else:
                    out.append((item.custom_id, f"{result.type}: the batch was canceled"))
        except anthropic.AnthropicError as e:
            raise _reading(e) from e
        return out


def _reading(e: anthropic.AnthropicError) -> GenerateError:
    """An error while reading a batch's progress or results, which costs nothing, so a
    dropped connection may be tried again like throttling."""
    if isinstance(e, anthropic.APIConnectionError):
        return Throttled(f"{type(e).__name__}: {e}")
    return classify(e)


# --- A run ---------------------------------------------------------------------------------


def run_batch[C: Sendable](
    calls: Sequence[C],
    client: BatchClient,
    batch: dict,
    price: Price,
    head: Callable[[C], dict],
    verdict: Callable[[Reply, C], dict],
    blank: dict,
    save: Callable[[RunResult], None],
    progress: Callable[[BatchStatus], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = POLL_SECONDS,
) -> RunResult:
    """Send the calls as one batch, or pick up `batch` if it has an ID, wait for it to end,
    and record every result, as `generate.send_all` records a direct run's.

    `batch` is the run log's record (`new_batch`, or the one `find_batch` found), and is
    updated in place; `save` writes the log with it. It is saved before the batch is created,
    with its ID as soon as it is, and with every result once all are in. Whatever stops the run
    is in the result's `error`; a run stopped while waiting can be picked up again.
    """
    if len(batch["requests"]) != len(calls):
        raise BatchError("the batch's requests and the calls don't match")
    result = RunResult()
    try:
        if batch["id"] is None:
            _create(calls, client, batch, save, result, sleep)
        _wait(client, batch["id"], progress, sleep, poll_seconds)
        outcomes, _ = with_retries(lambda: client.results(batch["id"]), sleep)
        _record(calls, batch, outcomes, price, head, verdict, blank, result)
        batch["status"] = COLLECTED
        batch["ended"] = _now()
    except (Exception, KeyboardInterrupt) as e:
        result.error = describe(e)
    save(result)
    return result


def _create(calls, client, batch, save, result, sleep) -> None:
    """Create the batch and record its ID at once. Only a refusal for too many requests (429),
    which creates nothing, is tried again; anything that may have created it leaves "unknown".
    """
    batch["status"] = SENDING
    save(result)  # before anything is sent: a batch that can't be recorded is never created
    pairs = zip(batch["requests"], calls, strict=True)
    requests = [(r["custom_id"], call.prompt) for r, call in pairs]
    in_flight = False
    try:
        for tries in range(1, MAX_ATTEMPTS + 1):
            try:
                in_flight = True
                batch_id = client.create(requests)
                break
            except Throttled:
                in_flight = False
                if tries == MAX_ATTEMPTS:
                    raise
                sleep(BACKOFF_SECONDS[tries - 1])
    except (MaybeSent, KeyboardInterrupt):
        # Stopped while a request was out, the batch may exist and its ID is lost.
        batch["status"] = UNKNOWN if in_flight else NOT_SENT
        raise
    except GenerateError:
        batch["status"] = NOT_SENT  # refused, or still throttled after the retries
        raise
    batch.update(id=batch_id, status=SUBMITTED, submitted=_now())
    save(result)


def _wait(client, batch_id, progress, sleep, poll_seconds) -> None:
    last = None
    while True:
        status, _ = with_retries(lambda: client.status(batch_id), sleep)
        if status.ended:
            return
        if progress and status != last:
            progress(status)
        last = status
        sleep(poll_seconds)


def _record(calls, batch, outcomes, price, head, verdict, blank, result) -> None:
    """Every call's result into `result`, in the calls' order. All outcomes are checked before
    any is recorded: a batch whose results don't fit its requests records nothing, and stays
    "submitted", its results still to be fetched again."""
    place = {r["custom_id"]: i for i, r in enumerate(batch["requests"])}
    found: dict[int, Reply | str] = {}
    for custom_id, outcome in outcomes:
        if custom_id not in place:
            raise BatchError(f"a result for {custom_id}, which this batch didn't ask")
        if place[custom_id] in found:
            raise BatchError(f"two results for {custom_id}")
        found[place[custom_id]] = outcome
    for i, call in enumerate(calls):
        outcome = found.get(i, "no result in the batch")
        if isinstance(outcome, str):
            result.failed.append({**head(call), "error": outcome})
            continue
        entry = paid_entry(call, outcome, head, blank, price, None, 1)
        result.answers.append(entry)
        read_into(entry, outcome, call, verdict)


def pick_up(found: tuple[Path, dict]) -> tuple[dict, Path, dict]:
    """The run header, log path and batch record of a run to pick up (`find_batch`)."""
    path, log = found
    return log["run"], path, log["batch"]


def opening(batch: dict, calls: int) -> str:
    if batch["id"] is None:
        return f"Sending {calls} calls as one batch; waiting for its results."
    return f"Waiting for batch {batch['id']}; nothing new is sent."


def stopped_note(batch: dict | None) -> str | None:
    """What to do after a batch run stopped early, if the batch can still be picked up."""
    if batch is None or batch["status"] != SUBMITTED:
        return None
    return (
        f"Batch {batch['id']} is still on its way: run the same command again to pick it up. "
        "Nothing will be sent twice."
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def status_line(status: BatchStatus, total: int) -> str:
    return f"  {time.strftime('%H:%M')} {status.processing} of {total} still processing"
