"""The answer client, tested with stand-ins for Claude: nothing here reaches the network."""

import datetime as dt
import importlib.util
import json
from pathlib import Path

import anthropic
import httpx2
import pytest

from rag.chunk import Chunk
from rag.generate import (
    BACKOFF_SECONDS,
    MAX_ATTEMPTS,
    AnthropicClient,
    Call,
    CallFailed,
    OverCapError,
    Reply,
    Throttled,
    answer_all,
    check_cap,
    estimate,
    input_problems,
    send_with_retries,
)
from rag.generation_config import GenerationConfig, Price
from rag.prompt import ANSWER_SCHEMA, Prompt
from rag.spend import Spending

ROOT = Path(__file__).parents[1]
HAIKU = Price(input=0.10, output=0.50, batch_input=0.05, batch_output=0.25)
CONFIG = GenerationConfig(
    provider="anthropic",
    build_model="claude-haiku-5-5",
    scored_model="claude-opus-5-5",
    effort="medium",
    max_output_tokens=4000,
    max_prompt_tokens=100_000,
    prices_read_on=dt.date.today(),
    prices={"claude-haiku-5-5": HAIKU, "claude-opus-5-5": Price(4, 20, 2, 10)},
)
CHUNK = Chunk(
    "paragraph:561/2006:art6(2)",
    "paragraph",
    "561/2006",
    "Regulation (EC) No 561/2006 › Article 6",
    "2. The weekly driving time shall not exceed 56 hours.",
    ("561/2006:art6(2)",),
    line_owners=("561/2006:art6(2)",),
)
PROMPT = Prompt("system", "user")


def call(qid: str = "q001") -> Call:
    return Call(qid, "How long may I drive in a week?", PROMPT, (CHUNK,))


def answer(*citations: tuple[str, str], answerable: bool = True) -> str:
    return json.dumps(
        {
            "answerable": answerable,
            "answer": "56 hours.",
            "citations": [{"node_id": n, "quote": q} for n, q in citations],
        }
    )


def reply(text: str | None, stop: str = "end_turn", out: int = 300, category=None) -> Reply:
    return Reply(text, stop, category, 1200, out, 0, 0, "claude-haiku-5-5", "req_1")


class Scripted:
    """A stand-in client that gives each call the next scripted reply or raises it."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.sent: list[Prompt] = []

    def send(self, prompt: Prompt) -> Reply:
        self.sent.append(prompt)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


# --- Estimate and cap ---------------------------------------------------------------------


def test_estimate_gives_the_expected_cost_and_the_most_a_run_can_cost():
    e = estimate([call("q1"), call("q2")], CONFIG, "claude-haiku-5-5")
    per_call = call().input_tokens
    assert e.input_tokens == 2 * per_call
    assert e.padded_input == pytest.approx(2 * per_call * 1.5, abs=1)
    assert e.expected_usd == pytest.approx((e.padded_input * 0.10 + 3000 * 0.50) / 1e6)
    assert e.most_usd == pytest.approx((e.padded_input * 0.10 + 8000 * 0.50) / 1e6)
    assert e.most_usd > e.expected_usd


def test_a_run_that_could_pass_the_cap_refuses_to_start():
    e = estimate([call()] * 60, CONFIG, "claude-opus-5-5")
    check_cap(e, spent_usd=0.0, cap_usd=20.0)
    with pytest.raises(OverCapError, match="could pass the approved cap"):
        check_cap(e, spent_usd=20.0 - e.most_usd + 0.01, cap_usd=20.0)


def test_a_prompt_that_may_be_too_long_for_the_price_is_refused():
    huge = Call("q1", "?", Prompt("s", "x" * 300_000), (CHUNK,))
    assert input_problems(estimate([huge], CONFIG, "claude-haiku-5-5"), CONFIG)
    assert input_problems(estimate([call()], CONFIG, "claude-haiku-5-5"), CONFIG) == []


# --- Retries ------------------------------------------------------------------------------


def test_a_throttled_call_is_retried_with_growing_waits():
    client = Scripted(Throttled("429"), Throttled("529"), reply("{}"))
    waits: list[float] = []
    got, attempts = send_with_retries(client, PROMPT, waits.append)
    assert attempts == 3
    assert waits == list(BACKOFF_SECONDS[:2])
    assert got.text == "{}"


def test_a_call_still_throttled_after_the_last_try_is_given_up():
    client = Scripted(*[Throttled("429")] * MAX_ATTEMPTS)
    waits: list[float] = []
    with pytest.raises(Throttled, match=f"gave up after {MAX_ATTEMPTS} tries"):
        send_with_retries(client, PROMPT, waits.append)
    assert len(client.sent) == MAX_ATTEMPTS
    assert waits == list(BACKOFF_SECONDS)


def test_a_call_that_retrying_wont_fix_is_not_retried():
    client = Scripted(CallFailed("HTTP 401"))
    waits: list[float] = []
    with pytest.raises(CallFailed):
        send_with_retries(client, PROMPT, waits.append)
    assert len(client.sent) == 1 and waits == []


# --- A run --------------------------------------------------------------------------------


def test_each_reply_is_checked_and_recorded_with_its_tokens_cost_and_time():
    client = Scripted(
        reply(answer(("561/2006:art6(2)", "shall not exceed 56 hours"))),
        reply(answer(("561/2006:art8(1)", "shall take daily and weekly rest periods"))),
        reply(answer(answerable=False)),
    )
    ticks = iter([0.0, 1.5, 2.0, 2.25, 3.0, 4.0])
    result = answer_all(
        [call("q1"), call("q2"), call("q3")], client, HAIKU, clock=lambda: next(ticks)
    )
    assert result.error is None
    first, second, third = result.answers
    assert first["verdict"]["ok"] and first["verdict"]["answerable"] is True
    assert first["verdict"]["answer"] == "56 hours."
    assert first["verdict"]["citations"] == [
        {"node_id": "561/2006:art6(2)", "quote": "shall not exceed 56 hours"}
    ]
    assert first["retrieved"] == [CHUNK.id]
    assert first["usage"]["input_tokens"] == 1200 and first["usage"]["output_tokens"] == 300
    assert first["cost_usd"] == pytest.approx((1200 * 0.10 + 300 * 0.50) / 1e6)
    assert first["seconds"] == 1.5 and first["attempts"] == 1
    assert first["request_id"] == "req_1"
    # Cited a node that was not retrieved: kept, but never counted as an answer.
    assert not second["verdict"]["ok"]
    assert "not a node of the retrieved passages" in second["verdict"]["problems"][0]
    assert third["verdict"]["ok"] and third["verdict"]["answerable"] is False
    assert result.passed == 2
    assert result.input_tokens == 3600 and result.output_tokens == 900


@pytest.mark.parametrize(
    ("r", "reason"),
    [
        (reply('{"answerable": tru', stop="max_tokens"), "cut off at max_output_tokens"),
        (reply(None, stop="refusal", category="cyber"), "declined to answer (category cyber)"),
        (reply(None), "no text"),
    ],
)
def test_a_reply_that_stopped_short_fails_and_is_still_paid_for(r, reason):
    result = answer_all([call()], Scripted(r), HAIKU)
    (only,) = result.answers
    assert not only["verdict"]["ok"]
    assert reason in only["verdict"]["problems"][0]
    assert only["cost_usd"] > 0


def test_a_call_that_fails_stops_the_run_and_keeps_what_was_answered():
    client = Scripted(
        reply(answer(("561/2006:art6(2)", "shall not exceed 56 hours"))),
        CallFailed("HTTP 401 AuthenticationError: invalid x-api-key"),
    )
    result = answer_all([call("q1"), call("q2"), call("q3")], client, HAIKU)
    assert [a["id"] for a in result.answers] == ["q1"]
    assert result.error == "CallFailed: HTTP 401 AuthenticationError: invalid x-api-key"
    assert len(client.sent) == 2  # q3 was never sent


def test_any_error_stops_the_run_but_keeps_every_paid_reply_and_names_the_failed_call():
    client = Scripted(reply(answer(answerable=False)), RuntimeError("something unforeseen"))
    saved: list[int] = []
    result = answer_all(
        [call("q1"), call("q2"), call("q3")],
        client,
        HAIKU,
        save=lambda r: saved.append(len(r.answers)),
    )
    assert [a["id"] for a in result.answers] == ["q1"]
    assert [(f["id"], f["error"]) for f in result.failed] == [
        ("q2", "RuntimeError: something unforeseen")
    ]
    assert result.failed[0]["retrieved"] == [CHUNK.id]
    assert result.error == "RuntimeError: something unforeseen"
    assert saved == [1, 1]  # after q1, and once more when the run stopped


def test_the_log_is_saved_after_every_answer():
    canned = reply(answer(answerable=False))
    saved: list[int] = []
    answer_all(
        [call("q1"), call("q2")],
        Scripted(canned, canned),
        HAIKU,
        save=lambda r: saved.append(len(r.answers)),
    )
    assert saved == [1, 2, 2]


def test_a_check_that_crashes_fails_the_answer_and_keeps_its_cost(monkeypatch):
    import rag.generate as generate

    def broken(*args):
        raise TypeError("bug")

    monkeypatch.setattr(generate, "check_reply", broken)
    result = answer_all([call("q1"), call("q2")], Scripted(reply("{}"), reply("{}")), HAIKU)
    assert result.error is None and len(result.answers) == 2
    assert result.answers[0]["verdict"]["problems"] == ["the check itself failed: TypeError: bug"]
    assert result.answers[0]["cost_usd"] > 0


def test_refusals_count_only_answers_that_passed():
    client = Scripted(
        reply(answer(answerable=False)),
        reply(answer(("561/2006:art6(2)", "shall not exceed 56 hours"), answerable=False)),
        reply(answer(("561/2006:art6(2)", "shall not exceed 56 hours"))),
    )
    result = answer_all([call("q1"), call("q2"), call("q3")], client, HAIKU)
    assert (result.passed, result.refusals) == (2, 1)


def test_cache_tokens_are_never_counted_below_their_price():
    r = Reply("{}", "end_turn", None, 100, 10, 1000, 1000, "m", None)
    assert r.cost_usd(HAIKU) == pytest.approx(((100 + 1250 + 1000) * 0.10 + 10 * 0.50) / 1e6)


# --- The real client against a fake server ------------------------------------------------


def fake_api(*responses):
    """An Anthropic client whose HTTP calls go to a list of canned responses."""
    requests: list[dict] = []
    queue = list(responses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    return AnthropicClient(CONFIG, "claude-haiku-5-5", api_key="test", http_client=http), requests


def message(text: str, stop: str = "end_turn") -> httpx2.Response:
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "thinking", "thinking": "", "signature": "s"}]
        + [{"type": "text", "text": text}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 2100, "output_tokens": 640, "cache_read_input_tokens": 0},
    }
    return httpx2.Response(200, json=body, headers={"request-id": "req_abc"})


def error(status: int, kind: str) -> httpx2.Response:
    return httpx2.Response(status, json={"type": "error", "error": {"type": kind, "message": "no"}})


def test_the_client_asks_for_the_answer_format_at_the_configured_effort():
    client, requests = fake_api(message('{"answerable": false, "answer": "x", "citations": []}'))
    got = client.send(Prompt("the system prompt", "the passages and question"))
    (sent,) = requests
    assert sent["model"] == "claude-haiku-5-5"
    assert sent["max_tokens"] == 4000
    assert sent["system"] == "the system prompt"
    assert sent["messages"] == [{"role": "user", "content": "the passages and question"}]
    assert sent["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
    }
    assert got.text == '{"answerable": false, "answer": "x", "citations": []}'
    assert (got.input_tokens, got.output_tokens, got.cache_read_tokens) == (2100, 640, 0)
    assert got.stop_reason == "end_turn" and got.request_id == "req_abc"


@pytest.mark.parametrize(
    ("response", "raised"),
    [
        (error(429, "rate_limit_error"), Throttled),
        (error(529, "overloaded_error"), Throttled),
        (error(500, "api_error"), Throttled),
        (httpx2.ConnectError("no route"), CallFailed),  # may have been billed: not retried
        (httpx2.ReadTimeout("slow"), CallFailed),
        (error(401, "authentication_error"), CallFailed),
        (error(400, "invalid_request_error"), CallFailed),
        (error(404, "not_found_error"), CallFailed),
    ],
)
def test_the_client_sorts_failures_into_throttling_and_the_rest(response, raised):
    client, requests = fake_api(response)
    with pytest.raises(raised):
        client.send(PROMPT)
    assert len(requests) == 1  # the SDK's own retries are off; send_with_retries decides


# --- The command ----------------------------------------------------------------------------


@pytest.fixture
def script(monkeypatch, tmp_path):
    """scripts/answer.py, with its run logs in tmp_path and a client that must not be built."""
    spec = importlib.util.spec_from_file_location("answer_script", ROOT / "scripts" / "answer.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    built: list = []

    def no_client(*args, **kwargs):
        built.append(args)
        raise AssertionError("a client was built")

    monkeypatch.setattr(module, "AnthropicClient", no_client)
    monkeypatch.setattr(module, "RUNS", tmp_path / "generation")
    monkeypatch.setattr(module, "ENV", tmp_path / ".env")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    module.built = built
    return module


def test_without_the_flag_nothing_is_sent_and_every_call_is_shown(script, capsys):
    assert script.main(["--only", "q001,q002"]) == 0
    out = capsys.readouterr().out
    assert script.built == []
    assert "q001" in out and "q002" in out
    assert "2 calls: about" in out
    assert "Expected cost" in out and "Most it can cost" in out
    assert "Nothing was sent" in out


def test_a_paid_run_over_the_cap_refuses_to_start(script, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(script, "spending", lambda root: Spending(20.0, 19.9999, ()))
    assert script.main(["--only", "q001", "--run"]) == 1
    assert script.built == []
    assert "could pass the approved cap" in capsys.readouterr().err


def test_a_paid_run_without_a_key_refuses_to_start(script, capsys):
    assert script.main(["--only", "q001", "--run"]) == 1
    assert script.built == []
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_a_paid_run_with_old_prices_refuses_to_start(script, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    old = script.load_generation(script.CONFIG)
    stale = GenerationConfig(**{**old.__dict__, "prices_read_on": dt.date(2026, 1, 1)})
    monkeypatch.setattr(script, "load_generation", lambda path: stale)
    assert script.main(["--only", "q001", "--run"]) == 1
    assert script.built == []
    assert "prices were read" in capsys.readouterr().err


def test_a_test_split_question_is_refused(script, capsys):
    rows = (json.loads(line) for line in (ROOT / "data" / "questions.v1.jsonl").open())
    test_id = next(row["id"] for row in rows if row["split"] == "test")
    assert script.main(["--only", test_id]) == 1
    assert f"not dev questions: {test_id}" in capsys.readouterr().err
    assert script.built == []


def test_an_empty_selection_is_refused(script, capsys):
    assert script.main(["--only", ","]) == 1
    assert "names no question" in capsys.readouterr().err


def test_a_run_stopped_midway_leaves_a_log_the_spend_check_counts(script, monkeypatch, tmp_path):
    from rag.spend import unlogged_runs

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    first = reply(answer(answerable=False))
    monkeypatch.setattr(
        script, "AnthropicClient", lambda *a, **k: Scripted(first, KeyboardInterrupt())
    )
    assert script.main(["--only", "q001,q002", "--run"]) == 1
    (path,) = script.RUNS.glob("*.json")
    log = json.loads(path.read_text(encoding="utf-8"))
    assert [a["id"] for a in log["answers"]] == ["q001"]
    assert [(f["id"], f["error"]) for f in log["failed"]] == [("q002", "KeyboardInterrupt")]
    pending = unlogged_runs("# Progress\n\n## Spend log\n", script.RUNS)
    assert pending == [(path.name, log["paid"]["cost_usd"])] and pending[0][1] > 0
    assert not list(script.RUNS.glob("*.tmp"))


def test_a_paid_run_writes_a_log_with_every_answer(script, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    canned = reply(answer(answerable=False))
    monkeypatch.setattr(script, "AnthropicClient", lambda *a, **k: Scripted(canned, canned))
    assert script.main(["--only", "q001,q002", "--run"]) == 0
    (path,) = (script.RUNS).glob("*.json")
    assert path.name.endswith("-claude-haiku-5-5.json")
    log = json.loads(path.read_text(encoding="utf-8"))
    assert log["config"]["model"] == "claude-haiku-5-5"
    assert log["config"]["retrieval"] == {"retriever": "hybrid", "strategy": "paragraph", "k": 5}
    assert log["config"]["split"] == "dev" and len(log["config"]["prompt_version"]) == 12
    assert [a["id"] for a in log["answers"]] == ["q001", "q002"]
    assert all(len(a["retrieved"]) == 5 for a in log["answers"])
    assert log["paid"]["calls"] == 2 and log["paid"]["input_tokens"] == 2400
    assert log["paid"]["cost_usd"] == pytest.approx(2 * (1200 * 0.10 + 300 * 0.50) / 1e6, abs=1e-6)
    assert log["error"] is None
    assert f"names {path.name}" in capsys.readouterr().out


def test_two_runs_started_in_the_same_second_never_share_a_log(monkeypatch, tmp_path):
    import rag.generate

    class Frozen(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime(2026, 11, 3, 10, 0, 0, tzinfo=tz)

    monkeypatch.setattr(rag.generate, "datetime", Frozen)
    run, path = rag.generate.new_run(ROOT, tmp_path / "runs", "claude-haiku-5-5", prefix="ask-")
    assert path.name == "ask-2026-11-03T10-00-00Z-claude-haiku-5-5.json" and path.exists()
    assert run["created"] == "2026-11-03T10:00:00+00:00"
    with pytest.raises(FileExistsError):
        rag.generate.new_run(ROOT, tmp_path / "runs", "claude-haiku-5-5", prefix="ask-")
