"""Runs through the Message Batches API, tested with stand-ins: nothing here reaches the network."""

import datetime as dt
import importlib.util
import json
from pathlib import Path

import anthropic
import httpx2
import pytest

from rag.batch import (
    COLLECTED,
    NOT_SENT,
    SENDING,
    SUBMITTED,
    UNKNOWN,
    AnthropicBatchClient,
    BatchError,
    BatchStatus,
    find_batch,
    new_batch,
    run_batch,
)
from rag.chunk import Chunk
from rag.faithfulness import sentences
from rag.generate import (
    ANSWER_BLANK,
    Call,
    CallFailed,
    MaybeSent,
    Reply,
    RunResult,
    Throttled,
    answer_head,
    answer_verdict,
    estimate,
    request_params,
)
from rag.generation_config import GenerationConfig, Price
from rag.prompt import ANSWER_SCHEMA, Prompt
from rag.spend import Spending, unlogged_runs

ROOT = Path(__file__).parents[1]
HAIKU = Price(input=0.10, output=0.50, batch_input=0.05, batch_output=0.25)
CONFIG = GenerationConfig(
    provider="anthropic",
    build_model="claude-haiku-5-5",
    scored_model="claude-opus-5-5",
    judge_model="claude-opus-5-5",
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
REFUSAL = json.dumps({"answerable": False, "answer": "The documents don't say.", "citations": []})


def call(qid: str) -> Call:
    return Call(qid, f"Question {qid}?", Prompt("system", f"user {qid}"), (CHUNK,))


def reply(out: int = 300) -> Reply:
    return Reply(REFUSAL, "end_turn", None, 1200, out, 0, 0, "claude-haiku-5-5", None)


class Batches:
    """A stand-in batch service. `outcomes` gives each request's result by its question, a
    `Reply` or why it has none; results come back in reverse order. `stops` are raised, one a
    look, by the looks at the batch's progress before it ends."""

    def __init__(self, outcomes: dict, waits: int = 0, stops=(), create=None):
        self.outcomes = outcomes
        self.waits = waits
        self.stops = list(stops)
        self.create_error = create
        self.created: list[list[tuple[str, Prompt]]] = []
        self.looks = 0

    def create(self, requests):
        if self.create_error:
            raise self.create_error
        self.created.append(list(requests))
        return "msgbatch_01"

    def status(self, batch_id):
        assert batch_id == "msgbatch_01"
        self.looks += 1
        if self.stops:
            raise self.stops.pop(0)
        if self.waits:
            self.waits -= 1
            return BatchStatus(False, 2)
        return BatchStatus(True, 0)

    def results(self, batch_id):
        (requests,) = self.created
        out = [(cid, self.outcomes[cid.split("-", 1)[1]]) for cid, _ in requests]
        return [o for o in reversed(out) if o[1] is not None]


def run(calls, service, batch=None, saved=None, sleeps=None):
    batch = batch or new_batch(calls, CONFIG, "claude-haiku-5-5", ANSWER_SCHEMA)
    saved = saved if saved is not None else []
    sleeps = sleeps if sleeps is not None else []

    def save(result: RunResult) -> None:
        saved.append((json.loads(json.dumps(batch)), len(result.answers)))

    result = run_batch(
        calls,
        service,
        batch,
        HAIKU.for_batch(),
        answer_head,
        answer_verdict,
        ANSWER_BLANK,
        save,
        sleep=sleeps.append,
    )
    return result, batch, saved


# --- Estimate ---------------------------------------------------------------------------------


def test_a_batch_is_priced_at_half_and_the_cap_still_assumes_every_answer_at_its_longest():
    calls = [call("q001"), call("q002")]
    direct, batch = estimate(calls, CONFIG, "claude-haiku-5-5"), None
    batch = estimate(calls, CONFIG, "claude-haiku-5-5", batch=True)
    assert batch.price == Price(0.05, 0.25, 0.05, 0.25)
    assert batch.most_usd == pytest.approx(direct.most_usd / 2)
    assert batch.most_usd == pytest.approx((batch.padded_input * 0.05 + 2 * 4000 * 0.25) / 1e6)


# --- A run ------------------------------------------------------------------------------------


def test_the_batch_id_is_saved_before_anything_else_happens():
    calls = [call("q001"), call("q002")]
    service = Batches({"q001": reply(), "q002": reply()}, waits=2)
    result, batch, saved = run(calls, service)
    # Saved before sending (no ID), at once with the ID, and with every result at the end.
    assert [(s["status"], s["id"], n) for s, n in saved] == [
        (SENDING, None, 0),
        (SUBMITTED, "msgbatch_01", 0),
        (COLLECTED, "msgbatch_01", 2),
    ]
    assert service.looks == 3 and batch["submitted"] and batch["ended"]
    assert result.error is None


def test_each_request_is_the_call_a_direct_run_would_send_with_a_unique_id():
    calls = [call("q001"), call("q001"), call("q002")]  # one question's answer judged twice
    service = Batches({"q001": reply(), "q002": reply()})
    _, batch, _ = run(calls, service)
    (sent,) = service.created
    assert [cid for cid, _ in sent] == ["000-q001", "001-q001", "002-q002"]
    assert [prompt for _, prompt in sent] == [c.prompt for c in calls]
    assert [r["id"] for r in batch["requests"]] == ["q001", "q001", "q002"]


def test_results_out_of_order_are_matched_to_their_questions():
    calls = [call("q001"), call("q002"), call("q003")]
    outcomes = {"q001": reply(100), "q002": reply(200), "q003": reply(300)}
    result, _, _ = run(calls, Batches(outcomes))  # they come back q003, q002, q001
    assert [(a["id"], a["usage"]["output_tokens"]) for a in result.answers] == [
        ("q001", 100), ("q002", 200), ("q003", 300)
    ]  # fmt: skip


def test_a_reply_is_recorded_as_a_direct_run_records_it_at_the_batch_price_without_time():
    result, _, _ = run([call("q001")], Batches({"q001": reply(300)}))
    (entry,) = result.answers
    assert entry["verdict"]["ok"] and entry["verdict"]["answerable"] is False
    assert entry["retrieved"] == [CHUNK.id] and entry["raw"] == REFUSAL
    assert entry["cost_usd"] == pytest.approx((1200 * 0.05 + 300 * 0.25) / 1e6)
    assert entry["seconds"] is None and entry["attempts"] == 1
    assert set(entry) == {
        "id", "retrieved", "estimated_input_tokens", "raw", "stop_reason", "verdict", "usage",
        "cost_usd", "seconds", "attempts", "model", "request_id",
    }  # fmt: skip


def test_failed_expired_and_missing_requests_are_logged_as_failed_and_never_as_answers():
    calls = [call("q001"), call("q002"), call("q003"), call("q004")]
    outcomes = {
        "q001": reply(),
        "q002": "errored (overloaded_error): Overloaded",
        "q003": "expired: not answered within 24 hours",
        "q004": None,  # no result at all
    }
    result, batch, _ = run(calls, Batches(outcomes))
    assert [a["id"] for a in result.answers] == ["q001"]
    assert [(f["id"], f["error"]) for f in result.failed] == [
        ("q002", "errored (overloaded_error): Overloaded"),
        ("q003", "expired: not answered within 24 hours"),
        ("q004", "no result in the batch"),
    ]
    assert all(f["retrieved"] == [CHUNK.id] for f in result.failed)
    assert result.cost_usd == pytest.approx(result.answers[0]["cost_usd"])
    assert batch["status"] == COLLECTED and result.error is None


def test_results_that_dont_fit_the_batch_record_nothing_and_can_be_fetched_again():
    class Strange(Batches):
        def results(self, batch_id):
            return [("000-q001", reply()), ("009-q999", reply())]

    result, batch, _ = run([call("q001")], Strange({}))
    assert result.answers == [] and result.failed == []
    assert batch["status"] == SUBMITTED and "009-q999" in result.error


def test_a_run_stopped_while_waiting_is_picked_up_without_sending_again():
    calls = [call("q001"), call("q002")]
    service = Batches({"q001": reply(), "q002": reply()}, stops=[KeyboardInterrupt()])
    first, batch, _ = run(calls, service)
    assert first.error == "KeyboardInterrupt" and first.answers == []
    assert batch["status"] == SUBMITTED and batch["id"] == "msgbatch_01"
    second, batch, _ = run(calls, service, batch=batch)
    assert len(service.created) == 1  # the batch was sent once
    assert [a["id"] for a in second.answers] == ["q001", "q002"]
    assert batch["status"] == COLLECTED


def test_throttling_while_waiting_is_waited_out():
    service = Batches({"q001": reply()}, stops=[Throttled("HTTP 529")])
    sleeps: list[float] = []
    result, _, _ = run([call("q001")], service, sleeps=sleeps)
    assert result.error is None and sleeps == [5.0]


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (CallFailed("HTTP 400 BadRequestError: too big"), NOT_SENT),
        (MaybeSent("APITimeoutError (may have been billed)"), UNKNOWN),
        (KeyboardInterrupt(), UNKNOWN),
    ],
)
def test_a_batch_that_may_or_may_not_exist_is_told_apart(error, status):
    result, batch, _ = run([call("q001")], Batches({}, create=error))
    assert batch["status"] == status and batch["id"] is None and result.error


def test_only_too_many_requests_is_tried_again_when_creating():
    class Busy(Batches):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.refusals = [Throttled("HTTP 429")]

        def create(self, requests):
            if self.refusals:
                raise self.refusals.pop(0)
            return super().create(requests)

    sleeps: list[float] = []
    result, batch, _ = run([call("q001")], Busy({"q001": reply()}), sleeps=sleeps)
    assert result.error is None and batch["status"] == COLLECTED and sleeps == [5.0]


def test_stopping_while_waiting_to_try_again_creates_nothing_and_says_so():
    def stop(seconds):
        raise KeyboardInterrupt

    service = Batches({}, create=Throttled("HTTP 429"))
    batch = new_batch([call("q001")], CONFIG, "claude-haiku-5-5", ANSWER_SCHEMA)
    result = run_batch(
        [call("q001")], service, batch, HAIKU, answer_head, answer_verdict, ANSWER_BLANK,
        lambda r: None, sleep=stop,
    )  # fmt: skip
    assert batch["status"] == NOT_SENT and result.error == "KeyboardInterrupt"


# --- Finding a batch to pick up, and the spend check ----------------------------------------


def write_log(path: Path, batch: dict, paid: float = 0.0, most: float = 0.5) -> None:
    log = {"paid": {"cost_usd": paid}, "estimate": {"most_usd": most}, "batch": batch}
    path.write_text(json.dumps(log), encoding="utf-8")


def test_only_a_batch_of_the_same_requests_still_to_collect_is_found(tmp_path):
    calls = [call("q001")]
    batch = new_batch(calls, CONFIG, "claude-haiku-5-5", ANSWER_SCHEMA)
    other = new_batch([call("q002")], CONFIG, "claude-haiku-5-5", ANSWER_SCHEMA)
    opus = new_batch(calls, CONFIG, "claude-opus-5-5", ANSWER_SCHEMA)
    write_log(tmp_path / "a.json", {**other, "status": SUBMITTED})
    write_log(tmp_path / "b.json", {**opus, "status": SUBMITTED})
    write_log(tmp_path / "c.json", {**batch, "status": COLLECTED})
    assert find_batch(tmp_path, batch) is None
    write_log(tmp_path / "d.json", {**batch, "id": "msgbatch_01", "status": SUBMITTED})
    found = find_batch(tmp_path, batch)
    assert found is not None and found[0].name == "d.json"


def test_a_log_whose_batch_cant_be_read_is_named(tmp_path):
    (tmp_path / "broken.json").write_text('{"batch": {"id": null}}', encoding="utf-8")
    batch = new_batch([call("q001")], CONFIG, "claude-haiku-5-5", ANSWER_SCHEMA)
    with pytest.raises(BatchError, match="broken.json"):
        find_batch(tmp_path, batch)


def test_a_batch_without_results_is_counted_at_the_most_it_can_cost(tmp_path):
    batch = new_batch([call("q001")], CONFIG, "claude-haiku-5-5", ANSWER_SCHEMA)
    for status in (SENDING, SUBMITTED, UNKNOWN):
        write_log(tmp_path / f"{status}.json", {**batch, "status": status}, most=0.5)
    write_log(tmp_path / "collected.json", {**batch, "status": COLLECTED}, paid=0.1)
    write_log(tmp_path / "not_sent.json", {**batch, "status": NOT_SENT}, paid=0.0)
    pending = dict(unlogged_runs("# Progress\n\n## Spend log\n", tmp_path))
    assert pending == {
        "collected.json": 0.1, "not_sent.json": 0.0, "sending.json": 0.5,
        "submitted.json": 0.5, "unknown.json": 0.5,
    }  # fmt: skip


# --- The real client against a fake server ------------------------------------------------


def fake_api(*routes):
    """An Anthropic batch client whose HTTP calls are answered from `routes`, in order."""
    requests: list[httpx2.Request] = []
    queue = list(routes)

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return queue.pop(0)

    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    client = AnthropicBatchClient(CONFIG, "claude-haiku-5-5", api_key="test", http_client=http)
    return client, requests


def batch_body(status: str = "in_progress", processing: int = 2) -> httpx2.Response:
    counts = {"processing": processing, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
    return httpx2.Response(
        200,
        json={
            "id": "msgbatch_01",
            "type": "message_batch",
            "processing_status": status,
            "request_counts": counts,
            "created_at": "2026-11-06T10:00:00Z",
            "expires_at": "2026-11-07T10:00:00Z",
            "ended_at": None,
            "archived_at": None,
            "cancel_initiated_at": None,
            "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_01/results",
        },
    )


def test_the_client_sends_each_request_as_a_direct_call_would_be_sent():
    client, requests = fake_api(batch_body())
    prompt = Prompt("the system prompt", "the passages and question")
    assert client.create([("000-q001", prompt)]) == "msgbatch_01"
    (sent,) = requests
    assert sent.url.path == "/v1/messages/batches"
    body = json.loads(sent.content)
    assert body == {
        "requests": [
            {
                "custom_id": "000-q001",
                "params": request_params(CONFIG, "claude-haiku-5-5", ANSWER_SCHEMA, prompt),
            }
        ]
    }
    assert body["requests"][0]["params"]["output_config"]["effort"] == "medium"


def test_the_client_reads_progress_and_every_kind_of_result():
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "thinking", "thinking": "", "signature": "s"},
                    {"type": "text", "text": REFUSAL}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 2100, "output_tokens": 640},
    }  # fmt: skip
    error = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    lines = [
        {"custom_id": "001-q002", "result": {"type": "errored", "error": error}},
        {"custom_id": "000-q001", "result": {"type": "succeeded", "message": message}},
        {"custom_id": "002-q003", "result": {"type": "expired"}},
        {"custom_id": "003-q004", "result": {"type": "canceled"}},
    ]
    jsonl = "\n".join(json.dumps(line) for line in lines) + "\n"
    client, _ = fake_api(
        batch_body(),
        batch_body("ended", 0),
        batch_body("ended", 0),
        httpx2.Response(200, text=jsonl, headers={"content-type": "application/binary"}),
    )
    assert client.status("msgbatch_01") == BatchStatus(False, 2)
    assert client.status("msgbatch_01") == BatchStatus(True, 0)
    got = dict(client.results("msgbatch_01"))
    assert isinstance(got["000-q001"], Reply)
    assert (got["000-q001"].text, got["000-q001"].output_tokens) == (REFUSAL, 640)
    assert got["001-q002"] == "errored (overloaded_error): Overloaded"
    assert got["002-q003"].startswith("expired")
    assert got["003-q004"].startswith("canceled")


def api_error(status: int, kind: str) -> httpx2.Response:
    return httpx2.Response(status, json={"type": "error", "error": {"type": kind, "message": "no"}})


@pytest.mark.parametrize(
    ("response", "raised"),
    [
        (api_error(429, "rate_limit_error"), Throttled),  # nothing was created
        (api_error(529, "overloaded_error"), MaybeSent),  # it may have been stored
        (api_error(500, "api_error"), MaybeSent),
        (api_error(400, "invalid_request_error"), CallFailed),
    ],
)
def test_the_client_sorts_failures_to_create_a_batch(response, raised):
    client, requests = fake_api(response)
    with pytest.raises(raised):
        client.create([("000-q001", Prompt("s", "u"))])
    assert len(requests) == 1


def test_a_dropped_connection_while_reading_progress_may_be_tried_again():
    def handler(request):
        raise httpx2.ConnectError("no route")

    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    client = AnthropicBatchClient(CONFIG, "claude-haiku-5-5", api_key="test", http_client=http)
    with pytest.raises(Throttled):
        client.status("msgbatch_01")
    with pytest.raises(Throttled):
        client.results("msgbatch_01")


def test_a_dropped_connection_while_creating_may_have_sent_the_batch():
    def handler(request):
        raise httpx2.ReadTimeout("slow")

    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    client = AnthropicBatchClient(CONFIG, "claude-haiku-5-5", api_key="test", http_client=http)
    with pytest.raises(MaybeSent):
        client.create([("000-q001", Prompt("s", "u"))])


# --- The commands -------------------------------------------------------------------------


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script(monkeypatch, tmp_path):
    """scripts/answer.py with its run logs in tmp_path, and no client unless a test gives one."""
    module = load("answer")

    def no_client(*args, **kwargs):
        raise AssertionError("a client was built")

    monkeypatch.setattr(module, "AnthropicClient", no_client)
    monkeypatch.setattr(module, "AnthropicBatchClient", no_client)
    monkeypatch.setattr(module, "RUNS", tmp_path / "generation")
    monkeypatch.setattr(module, "ENV", tmp_path / ".env")
    monkeypatch.setattr(module, "spending", lambda root: Spending(20.0, 1.0, ()))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return module


def test_without_the_flag_nothing_is_sent_and_the_batch_price_is_shown(script, capsys):
    assert script.main(["--only", "q001,q002", "--batch"]) == 0
    out = capsys.readouterr().out
    assert "2 calls in one batch, at the batch service's half price" in out
    assert "make answers BATCH=1 RUN=1" in out
    assert not list(script.RUNS.glob("*.json"))


def test_a_batch_over_the_cap_does_not_start(script, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(script, "spending", lambda root: Spending(20.0, 19.9999, ()))
    assert script.main(["--only", "q001", "--batch", "--run"]) == 1
    assert "could pass the approved cap" in capsys.readouterr().err
    assert not list(script.RUNS.glob("*.json"))


def test_a_batch_run_writes_the_log_a_direct_run_writes_plus_the_batch(script, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    service = Batches({"q001": reply(), "q002": reply()})
    monkeypatch.setattr(script, "AnthropicBatchClient", lambda *a, **k: service)
    assert script.main(["--only", "q001,q002", "--batch", "--run"]) == 0
    (path,) = script.RUNS.glob("*.json")
    log = json.loads(path.read_text(encoding="utf-8"))
    assert [a["id"] for a in log["answers"]] == ["q001", "q002"]
    assert all(len(a["retrieved"]) == 5 and a["seconds"] is None for a in log["answers"])
    assert log["config"]["price_usd_per_million"] == {"input": 0.05, "output": 0.25}
    assert log["batch"]["id"] == "msgbatch_01" and log["batch"]["status"] == COLLECTED
    assert log["paid"]["cost_usd"] == pytest.approx(2 * (1200 * 0.05 + 300 * 0.25) / 1e6)
    assert set(log) == {"run", "config", "estimate", "paid", "error", "failed", "answers", "batch"}


def test_the_same_command_again_picks_up_a_stopped_batch(script, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    service = Batches({"q001": reply()}, stops=[KeyboardInterrupt()])
    monkeypatch.setattr(script, "AnthropicBatchClient", lambda *a, **k: service)
    assert script.main(["--only", "q001", "--batch", "--run"]) == 1
    assert "run the same command again" in capsys.readouterr().err
    (path,) = script.RUNS.glob("*.json")
    assert json.loads(path.read_text(encoding="utf-8"))["batch"]["status"] == SUBMITTED
    # Even with the cap all but spent by that batch, picking it up pays nothing new.
    monkeypatch.setattr(script, "spending", lambda root: Spending(20.0, 19.9999, ()))
    assert script.main(["--only", "q001", "--batch", "--run"]) == 0
    out = capsys.readouterr().out
    assert "is still on its way" in out and "Waiting for batch msgbatch_01" in out
    assert "Sending" not in out
    assert len(service.created) == 1 and list(script.RUNS.glob("*.json")) == [path]
    log = json.loads(path.read_text(encoding="utf-8"))
    assert log["batch"]["status"] == COLLECTED and [a["id"] for a in log["answers"]] == ["q001"]


def test_a_direct_run_doesnt_pay_again_for_calls_whose_batch_is_on_its_way(
    script, monkeypatch, capsys
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    service = Batches({"q001": reply()}, stops=[KeyboardInterrupt()])
    monkeypatch.setattr(script, "AnthropicBatchClient", lambda *a, **k: service)
    assert script.main(["--only", "q001", "--batch", "--run"]) == 1
    capsys.readouterr()
    assert script.main(["--only", "q001", "--run"]) == 1  # AnthropicClient would fail the test
    assert "run with BATCH=1 to pick the batch up" in capsys.readouterr().err


def test_a_batch_that_may_have_been_sent_is_not_sent_again(script, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    service = Batches({"q001": reply()}, create=MaybeSent("APITimeoutError"))
    monkeypatch.setattr(script, "AnthropicBatchClient", lambda *a, **k: service)
    assert script.main(["--only", "q001", "--batch", "--run"]) == 1
    capsys.readouterr()
    assert script.main(["--only", "q001", "--batch", "--run"]) == 1
    err = capsys.readouterr().err
    assert "Batches page of the Anthropic Console" in err and '"not_sent"' in err


def test_the_judge_sends_a_batch_too(monkeypatch, tmp_path):
    module = load("judge")
    monkeypatch.setattr(module, "RUNS", tmp_path / "judge")
    monkeypatch.setattr(module, "ENV", tmp_path / ".env")
    monkeypatch.setattr(module, "spending", lambda root: Spending(20.0, 1.0, ()))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    runs = sorted((ROOT / "runs" / "generation").glob("*claude-haiku-5-5.json"))
    log = json.loads(runs[0].read_text(encoding="utf-8"))
    labels = (ROOT / "data" / "faithfulness_labels.v1.jsonl").read_text(encoding="utf-8")
    labelled = {json.loads(line)["question_id"] for line in labels.splitlines() if line}
    answer = next(
        a for a in log["answers"]
        if a["verdict"]["ok"] and a["verdict"]["answerable"] and a["id"] not in labelled
    )  # fmt: skip
    qid, count = answer["id"], len(sentences(answer["verdict"]["answer"]))
    verdicts = [{"sentence": n, "reason": "r", "verdict": "supported"} for n in range(1, count + 1)]
    judged = json.dumps({"sentences": verdicts})

    class Judges(Batches):
        def results(self, batch_id):
            (requests,) = self.created
            return [(cid, Reply(judged, "end_turn", None, 900, 200, 0, 0, "m", None))
                    for cid, _ in requests]  # fmt: skip

    service = Judges({})
    monkeypatch.setattr(module, "AnthropicBatchClient", lambda *a, **k: service)
    assert module.main(["--log", runs[0].name, "--only", qid, "--batch", "--run"]) == 0
    (path,) = (tmp_path / "judge").glob("try-*.json")
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["batch"]["status"] == COLLECTED
    assert [j["id"] for j in written["judgements"]] == [qid]
    assert written["judgements"][0]["verdict"]["supported"] is True
