"""The hosted rerankers (#99) with stand-ins for Jev's API and Claude: nothing here sends anything
or needs a key."""

import dataclasses
import datetime as dt
import json
from pathlib import Path

import pytest

from rag.chunk import Chunk
from rag.generate import CallFailed, MaybeSent, Reply, RunResult, Throttled
from rag.generation_config import load_generation_config
from rag.hosted_rerank import (
    JEV_FIELDS,
    LISTWISE_FIELDS,
    HostedCache,
    HostedError,
    JevClient,
    JevConfig,
    JevScorer,
    ListwiseConfig,
    ListwiseScorer,
    Lot,
    cache_orders,
    estimate_jev,
    estimate_listwise,
    holds,
    jev_all,
    jev_key,
    jev_request,
    listwise_calls,
    listwise_key,
    listwise_verdict,
    load_hosted_config,
    order_scores,
    read_order,
)
from rag.rerank import MissingScoresError
from rag.retrieve import Hit, Rerank

ROOT = Path(__file__).parents[1]
JEV = JevConfig("jev-1.14.0", 0.042, dt.date(2026, 11, 6), 60.0)
HAIKU = ListwiseConfig("claude-haiku-5-5", "low", 2000)
Q = "How long may I drive?"
LOT = Lot("q001", Q, ("Nine hours a day.", "Ten hours twice a week.", "Tachographs record it."))


def test_the_committed_config_loads():
    config = load_hosted_config(ROOT / "config" / "rerank-hosted.yaml")
    assert config.jev.model == "jev-1.14.0"
    assert (
        config.listwise.model in load_generation_config(ROOT / "config" / "generation.yaml").prices
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r["jev"].update(model="jev-latest"), "pinned"),
        (lambda r: r["jev"].pop("timeout_seconds"), "missing timeout_seconds"),
        (lambda r: r["haiku"].update(effort="fast"), "effort"),
        (lambda r: r["haiku"].update(depth=20), "unknown depth"),
        (lambda r: r.pop("haiku"), "sections"),
    ],
)
def test_a_bad_config_is_refused(tmp_path, change, message):
    import yaml

    raw = yaml.safe_load((ROOT / "config" / "rerank-hosted.yaml").read_text(encoding="utf-8"))
    change(raw)
    path = tmp_path / "rerank-hosted.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(HostedError, match=message):
        load_hosted_config(path)


def test_a_change_of_model_or_wording_is_a_new_reranker():
    assert JEV.name.startswith("jev-1.14.0-")
    assert dataclasses.replace(JEV, model="jev-1.13.0").name != JEV.name
    assert HAIKU.name != dataclasses.replace(HAIKU, effort="medium").name
    assert jev_key(JEV, Q, "a") != jev_key(dataclasses.replace(JEV, model="jev-1.13.0"), Q, "a")
    assert listwise_key(HAIKU, Q, ["a", "b"]) != listwise_key(HAIKU, Q, ["b", "a"])


# --- Jev ----------------------------------------------------------------------------------


def test_the_request_holds_the_question_once_and_a_noul_per_passage():
    body = jev_request(JEV, LOT)
    assert body["model"] == "jev-1.14.0"
    assert body["state"] == {"question": Q}
    assert list(body["questions"]) == ["p01", "p02", "p03"]
    noul = body["questions"]["p02"]
    assert noul["type"] == "noul"
    assert noul["instructions"].endswith("Ten hours twice a week.")
    assert set(noul["criteria"]) == {"true", "false"}


class FakePost:
    """Stands in for the HTTP POST: answers with the given replies in turn."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent: list[tuple[str, dict, dict]] = []

    def __call__(self, url, body, headers, timeout):
        self.sent.append((url, json.loads(body), headers))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def ok(nouls: dict, tokens: int = 500) -> tuple[int, bytes]:
    answers = {name: {"type": "noul", "noul": value} for name, value in nouls.items()}
    body = {"model": "jev-1.14.0", "answers": answers, "usage": {"input_tokens": tokens,
                                                                 "output_tokens": 9}}  # fmt: skip
    return 200, json.dumps(body).encode()


def test_the_client_sends_the_key_and_reads_the_nouls():
    post = FakePost(ok({"p01": 0.9}))
    reply = JevClient(JEV, "test-key", post).send({"model": "m"})
    assert reply.nouls == {"p01": 0.9} and reply.input_tokens == 500
    url, _, headers = post.sent[0]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert headers["Authorization"] == "Bearer test-key"


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ((429, b"slow down"), Throttled),
        ((503, b"busy"), Throttled),
        ((422, b'{"detail": []}'), CallFailed),
        ((200, b"not json"), CallFailed),
        (TimeoutError("timed out"), MaybeSent),
        (OSError("connection reset"), MaybeSent),
    ],
)
def test_failures_are_sorted_into_retry_or_stop(reply, error):
    with pytest.raises(error):
        JevClient(JEV, "k", FakePost(reply)).send({})


def test_a_timeout_is_not_tried_again():
    assert not issubclass(MaybeSent, Throttled)


def test_a_run_caches_each_noul_and_logs_what_it_paid(tmp_path):
    cache = HostedCache.load(tmp_path, JEV.name, JEV_FIELDS)
    post = FakePost(ok({"p01": 0.9, "p02": 0.2, "p03": 0.5}, tokens=1_000_000))
    saved: list[RunResult] = []
    result = jev_all([LOT], JevClient(JEV, "k", post), cache, saved.append, sleep=lambda s: None)
    assert result.error is None
    [entry] = result.answers
    assert entry["cost_usd"] == pytest.approx(0.042) and entry["problem"] is None
    again = HostedCache.load(tmp_path, JEV.name, JEV_FIELDS)
    assert JevScorer(again, JEV).scores(Q, LOT.texts) == [0.9, 0.2, 0.5]
    assert saved


def test_a_throttled_request_is_tried_again():
    cache = HostedCache(Path("unused.jsonl"), JEV_FIELDS)
    cache.save = lambda: None
    post = FakePost((429, b""), ok({"p01": 0.9, "p02": 0.2, "p03": 0.5}))
    result = jev_all([LOT], JevClient(JEV, "k", post), cache, sleep=lambda s: None)
    assert result.answers[0]["attempts"] == 2


def test_a_reply_missing_a_passage_stops_the_run_and_keeps_the_paid_entry(tmp_path):
    cache = HostedCache.load(tmp_path, JEV.name, JEV_FIELDS)
    post = FakePost(ok({"p01": 0.9, "p02": 0.2}))
    second = Lot("q002", "Another?", ("x",))
    result = jev_all([LOT, second], JevClient(JEV, "k", post), cache, sleep=lambda s: None)
    assert "not ['p01', 'p02', 'p03']" in result.error
    assert len(result.answers) == 1 and result.answers[0]["problem"]
    assert result.cost_usd > 0  # paid for, so counted
    assert cache.entries == {}
    with pytest.raises(MissingScoresError):
        JevScorer(cache, JEV).scores(Q, LOT.texts)


def test_a_failed_request_is_listed_and_stops_the_run(tmp_path):
    cache = HostedCache.load(tmp_path, JEV.name, JEV_FIELDS)
    result = jev_all([LOT], JevClient(JEV, "k", FakePost((400, b"bad"))), cache)
    assert result.answers == [] and result.failed[0]["id"] == "q001"
    assert "HTTP 400" in result.error


def test_the_jev_estimate_is_priced_per_input_token():
    e = estimate_jev([LOT, LOT], JEV)
    assert e.calls == 2 and e.most_usd == e.expected_usd
    assert e.most_usd == pytest.approx(round(e.input_tokens * 1.5) * 0.042 / 1e6, rel=1e-3)


def test_a_run_over_the_cap_or_with_old_prices_is_held():
    e = estimate_jev([LOT], JEV)
    today = dt.date(2026, 11, 6)
    assert holds(e, JEV.prices_read_on, 0.0, 20.0, today) == []
    assert "cap" in holds(e, JEV.prices_read_on, 20.0, 20.0, today)[0]
    assert "days ago" in holds(e, JEV.prices_read_on, 0.0, 20.0, dt.date(2026, 11, 20))[0]


# --- Haiku --------------------------------------------------------------------------------


def test_the_prompt_numbers_the_passages_in_search_order():
    [call] = listwise_calls([LOT])
    assert "[1] Nine hours a day.\n\n[2] Ten hours twice a week." in call.prompt.user
    assert Q in call.prompt.user


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"order": [1, 1]}', "twice"),
        ('{"order": [4]}', "out of range"),
        ('{"order": [0]}', "out of range"),
        ('{"order": "1, 2"}', "no list"),
        ('{"order": [true]}', "no list"),
        ("1, 2", "not JSON"),
    ],
)
def test_an_order_that_cannot_be_used_is_refused(text, message):
    with pytest.raises(HostedError, match=message):
        read_order(text, 3)


def test_listed_passages_come_first_and_the_rest_keep_search_order():
    assert read_order('{"order": [3, 1]}', 3) == [3, 1]
    assert order_scores([3, 1], 4) == [3.0, 0.0, 4.0, 0.0]
    assert order_scores(None, 3) == [0.0, 0.0, 0.0]
    assert order_scores([], 2) == [0.0, 0.0]


def reply(text: str | None, stop: str = "end_turn") -> Reply:
    return Reply(text, stop, None, 3000, 200, 0, 0, "claude-haiku-5-5", "req")


def test_a_reply_cut_off_or_refused_has_no_order():
    [call] = listwise_calls([LOT])
    assert listwise_verdict(reply('{"order": [2]}'), call) == {
        "ok": True, "order": [2], "problems": []
    }  # fmt: skip
    cut = listwise_verdict(reply('{"order": [', "max_tokens"), call)
    assert cut["ok"] is False and "cut off" in cut["problems"][0]


def test_orders_are_cached_with_failures_and_rerank_from_the_cache(tmp_path):
    chunks = [
        Chunk(f"c{i}", "paragraph", "doc", "", t, (f"n{i}",)) for i, t in enumerate(LOT.texts)
    ]

    class Search:
        def search(self, query, k):
            return [Hit(c, 1.0) for c in chunks][:k]

    texts = tuple(c.text for c in chunks)
    calls = listwise_calls([Lot("q001", Q, texts), Lot("q002", "Other?", texts)])
    good = {"id": "q001", "verdict": {"ok": True, "order": [3], "problems": []}}
    bad = {"id": "q002", "verdict": {"ok": False, "order": None, "problems": ["cut off"]}}
    cache = HostedCache.load(tmp_path, HAIKU.name, LISTWISE_FIELDS)
    cache_orders(RunResult(answers=[good, bad]), calls, HAIKU, cache)
    again = HostedCache.load(tmp_path, HAIKU.name, LISTWISE_FIELDS)
    hits = Rerank(Search(), ListwiseScorer(again, HAIKU), depth=3).search(Q, 3)
    assert [h.chunk.id for h in hits] == ["c2", "c0", "c1"]
    hits = Rerank(Search(), ListwiseScorer(again, HAIKU), depth=3).search("Other?", 3)
    assert [h.chunk.id for h in hits] == ["c0", "c1", "c2"]  # search's order, as a service would
    assert again.entries[listwise_key(HAIKU, "Other?", texts)]["problem"] == "cut off"


def test_the_listwise_estimate_counts_every_reply_at_its_longest():
    generation = load_generation_config(ROOT / "config" / "generation.yaml")
    e = estimate_listwise([LOT], HAIKU, generation)
    price = generation.prices["claude-haiku-5-5"]
    assert e.most_usd == pytest.approx(
        (round(e.input_tokens * 1.5) * price.input + 2000 * price.output) / 1e6, rel=1e-3
    )
    assert e.expected_usd < e.most_usd


# --- The cache ----------------------------------------------------------------------------


def test_the_cache_file_is_sorted_and_refuses_damage(tmp_path):
    cache = HostedCache.load(tmp_path, "r", JEV_FIELDS)
    cache.put("b" * 64, {"noul": 0.1})
    cache.put("a" * 64, {"noul": 0.2})
    cache.save()
    lines = (tmp_path / "r.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["pair"][0] for line in lines] == ["a", "b"]
    with pytest.raises(HostedError):
        cache.put("c" * 64, {"score": 1})
    (tmp_path / "r.jsonl").write_text(lines[0] + "\n" + lines[0] + "\n", encoding="utf-8")
    with pytest.raises(HostedError, match="twice"):
        HostedCache.load(tmp_path, "r", JEV_FIELDS)
    (tmp_path / "r.jsonl").write_text('{"pair": "x", "noul": 1}\n', encoding="utf-8")
    with pytest.raises(HostedError, match="sha256"):
        HostedCache.load(tmp_path, "r", JEV_FIELDS)
