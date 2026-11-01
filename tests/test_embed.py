import dataclasses
import datetime as dt
import json
import math

import pytest

from rag.embed import (
    SAVE_EVERY,
    BedrockClient,
    EmbeddingCache,
    EmbedError,
    MissingVectorsError,
    OverCapError,
    cache_key,
    cache_path,
    check_cap,
    check_inputs,
    pay,
    plan,
    vectors,
)
from rag.embed_estimate import MARGIN, EmbeddingConfig

CONFIG = EmbeddingConfig(
    model_id="amazon.titan-embed-text-v2:0",
    region="eu-north-1",
    dimensions=4,
    normalize=True,
    max_input_tokens=100,
    price_usd_per_million_input_tokens=0.021,
    price_read_on=dt.date(2026, 10, 31),
)


class FakeClient:
    """Returns a vector made from the text; fails on the texts it is told to fail on."""

    def __init__(self, dimensions: int = 4, fail_on: frozenset[str] = frozenset()):
        self.dimensions = dimensions
        self.fail_on = fail_on
        self.sent: list[str] = []

    def embed(self, text: str) -> tuple[list[float], int]:
        self.sent.append(text)
        if text in self.fail_on:
            raise RuntimeError("ThrottlingException: too many requests")
        values = [float(len(text) + i) for i in range(self.dimensions)]
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values], len(text.split())


def test_the_key_covers_model_settings_and_text():
    base = cache_key("Article 6", CONFIG)
    assert base == cache_key("Article 6", CONFIG)
    assert base != cache_key("Article 7", CONFIG)
    for change in (
        {"model_id": "cohere.embed-multilingual-v3"},
        {"dimensions": 256},
        {"normalize": False},
    ):
        assert base != cache_key("Article 6", dataclasses.replace(CONFIG, **change))


def test_a_cache_hit_makes_no_call(tmp_path):
    cache = EmbeddingCache.load(tmp_path, CONFIG)
    client = FakeClient()
    pay(plan(["a b", "c"], cache), cache, client)
    assert client.sent == ["a b", "c"]

    again = EmbeddingCache.load(tmp_path, CONFIG)
    p = plan(["c", "a b"], again)
    assert (p.cached, p.missing) == (2, ())
    second = FakeClient()
    pay(p, again, second)
    assert second.sent == []


def test_a_text_seen_twice_is_sent_once(tmp_path):
    cache = EmbeddingCache.load(tmp_path, CONFIG)
    p = plan(["same", "other", "same"], cache)
    assert (p.distinct, p.missing) == (2, ("same", "other"))
    client = FakeClient()
    pay(p, cache, client)
    assert client.sent == ["same", "other"]


def test_a_changed_setting_misses_the_cache(tmp_path):
    cache = EmbeddingCache.load(tmp_path, CONFIG)
    pay(plan(["text"], cache), cache, FakeClient())
    smaller = dataclasses.replace(CONFIG, dimensions=2)
    p = plan(["text"], EmbeddingCache.load(tmp_path, smaller))
    assert p.missing == ("text",)
    assert cache_path(tmp_path, smaller) != cache_path(tmp_path, CONFIG)


def test_vectors_survive_a_reload_unchanged(tmp_path):
    cache = EmbeddingCache.load(tmp_path, CONFIG)
    pay(plan(["one", "two words"], cache), cache, FakeClient())
    before = vectors(["one", "two words"], cache)
    after = vectors(["one", "two words"], EmbeddingCache.load(tmp_path, CONFIG))
    assert before == after
    assert EmbeddingCache.load(tmp_path, CONFIG).get("two words").tokens == 2


def test_the_file_is_sorted_and_the_same_for_the_same_vectors(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    texts = ["zeta", "alpha", "mu"]
    for directory, order in ((a, texts), (b, list(reversed(texts)))):
        cache = EmbeddingCache.load(directory, CONFIG)
        pay(plan(order, cache), cache, FakeClient())
    first = cache_path(a, CONFIG).read_bytes()
    assert first == cache_path(b, CONFIG).read_bytes()
    keys = [json.loads(line)["key"] for line in first.decode().splitlines()]
    assert keys == sorted(keys)
    assert cache_path(a, CONFIG).name == "amazon.titan-embed-text-v2_0-4-normalized.jsonl"


def test_a_failed_call_caches_nothing_for_its_text_and_keeps_the_rest(tmp_path):
    cache = EmbeddingCache.load(tmp_path, CONFIG)
    client = FakeClient(fail_on=frozenset({"second"}))
    paid = pay(plan(["first", "second", "third"], cache), cache, client)
    assert "ThrottlingException" in paid.error
    assert client.sent == ["first", "second"]  # stops at the failure
    assert [c["tokens"] for c in paid.calls] == [1]
    reloaded = EmbeddingCache.load(tmp_path, CONFIG)
    assert reloaded.get("first") is not None
    assert reloaded.get("second") is None
    assert reloaded.get("third") is None


def test_a_wrong_sized_vector_is_not_cached(tmp_path):
    cache = EmbeddingCache.load(tmp_path, CONFIG)
    paid = pay(plan(["text"], cache), cache, FakeClient(dimensions=3))
    assert "3 dimensions" in paid.error
    assert EmbeddingCache.load(tmp_path, CONFIG).get("text") is None
    assert paid.calls == [{"key": cache_key("text", CONFIG), "tokens": 1}]  # billed all the same


def test_an_interrupted_run_keeps_and_reports_what_was_paid_for(tmp_path):
    class Interrupted(FakeClient):
        def embed(self, text):
            if text == "second":
                raise KeyboardInterrupt
            return super().embed(text)

    cache = EmbeddingCache.load(tmp_path, CONFIG)
    paid = pay(plan(["first", "second"], cache), cache, Interrupted())
    assert paid.error == "KeyboardInterrupt"
    assert len(paid.calls) == 1
    assert EmbeddingCache.load(tmp_path, CONFIG).get("first") is not None


def test_the_cache_is_saved_during_a_long_run(tmp_path):
    cache = EmbeddingCache.load(tmp_path, CONFIG)
    texts = [f"text {i}" for i in range(SAVE_EVERY + 1)]
    seen_on_disk = []

    def progress(done, total):
        if done == SAVE_EVERY:
            seen_on_disk.append(len(EmbeddingCache.load(tmp_path, CONFIG)))

    pay(plan(texts, cache), cache, FakeClient(), progress)
    assert seen_on_disk == [SAVE_EVERY]
    assert len(EmbeddingCache.load(tmp_path, CONFIG)) == SAVE_EVERY + 1


def test_each_call_logs_the_tokens_bedrock_returned(tmp_path):
    cache = EmbeddingCache.load(tmp_path, CONFIG)
    paid = pay(plan(["one two three", "four"], cache), cache, FakeClient())
    assert paid.calls == [
        {"key": cache_key("one two three", CONFIG), "tokens": 3},
        {"key": cache_key("four", CONFIG), "tokens": 1},
    ]
    assert paid.tokens == 4 and paid.error is None


def test_a_run_over_the_cap_refuses_to_start(tmp_path):
    cache = EmbeddingCache.load(tmp_path, CONFIG)
    p = plan(["word " * 40], cache)  # about 52 tokens
    assert p.padded_tokens == math.ceil(p.tokens * MARGIN)
    check_cap(p, spent_usd=0.0, cap_usd=20.0)
    with pytest.raises(OverCapError, match="over the approved cap"):
        check_cap(p, spent_usd=20.0 - p.cost_usd / 2, cap_usd=20.0)


def test_a_fully_cached_run_passes_the_cap_even_when_it_is_spent(tmp_path):
    cache = EmbeddingCache.load(tmp_path, CONFIG)
    pay(plan(["text"], cache), cache, FakeClient())
    check_cap(plan(["text"], cache), spent_usd=20.0, cap_usd=20.0)


def test_inputs_bedrock_would_refuse_are_reported(tmp_path):
    cache = EmbeddingCache.load(tmp_path, CONFIG)
    assert check_inputs(plan(["fine"], cache)) == []
    problems = check_inputs(plan(["  ", "word " * 70], cache))
    assert any("empty" in p for p in problems)
    assert any("over the model's limit" in p for p in problems)


def test_vectors_refuses_to_search_part_of_the_corpus(tmp_path):
    cache = EmbeddingCache.load(tmp_path, CONFIG)
    pay(plan(["cached"], cache), cache, FakeClient())
    with pytest.raises(MissingVectorsError, match="1 of 2 texts"):
        vectors(["cached", "not cached"], cache)


@pytest.mark.parametrize(
    "line, message",
    [
        ("not json", "not a cache entry"),
        ('{"key": "k", "tokens": 1}', "not a cache entry"),
        ('{"key": "' + "a" * 64 + '", "tokens": 1, "vector": "AAAAAA=="}', "1 dimensions"),
        ('{"key": ["x"], "tokens": 1, "vector": "AAAAAA=="}', "not a cache entry"),
        ('{"key": "k", "tokens": 1, "vector": "AAAAAA=="}', "the key is not a sha256"),
        ('{"key": "' + "a" * 64 + '", "tokens": "37", "vector": "AAAAAA=="}', "not a count"),
    ],
)
def test_a_damaged_cache_file_is_refused(tmp_path, line, message):
    cache_path(tmp_path, CONFIG).write_text(line + "\n", encoding="utf-8")
    with pytest.raises(EmbedError, match=message):
        EmbeddingCache.load(tmp_path, CONFIG)


def test_saving_an_unchanged_cache_writes_nothing(tmp_path):
    cache = EmbeddingCache.load(tmp_path, CONFIG)
    cache.save()
    assert not cache_path(tmp_path, CONFIG).exists()


class _Body:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class _Runtime:
    """Stands in for boto3's bedrock-runtime client, recording what it is asked."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.requests: list[dict] = []

    def invoke_model(self, **request):
        self.requests.append(request)
        return {"body": _Body(self.payload)}


def _bedrock(payload: dict) -> tuple[BedrockClient, _Runtime]:
    client = BedrockClient.__new__(BedrockClient)  # skips the AWS session and credentials
    client.config = CONFIG
    client._client = runtime = _Runtime(payload)
    return client, runtime


def test_bedrock_client_sends_titan_v2_settings_and_reads_tokens():
    client, runtime = _bedrock({"embedding": [0.5] * 4, "inputTextTokenCount": 7})
    assert client.embed("How long may I drive?") == ([0.5] * 4, 7)
    (request,) = runtime.requests
    assert request["modelId"] == "amazon.titan-embed-text-v2:0"
    assert json.loads(request["body"]) == {
        "inputText": "How long may I drive?",
        "dimensions": 4,
        "normalize": True,
    }


def test_bedrock_client_refuses_an_unexpected_response():
    client, _ = _bedrock({"message": "something else"})
    with pytest.raises(EmbedError, match="unexpected response"):
        client.embed("text")
