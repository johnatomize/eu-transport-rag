"""The reranker's settings, score cache and model wrapper, with a stand-in for the model: nothing
here downloads or runs one."""

import dataclasses
from pathlib import Path

import pytest

from rag.rerank import (
    PREFIX,
    SUFFIX,
    CachedScorer,
    LocalReranker,
    MissingScoresError,
    RerankConfig,
    RerankError,
    ScoreCache,
    cache_key,
    load_config,
    missing_pairs,
    model_input,
    score_missing,
    setting_name,
)

ROOT = Path(__file__).parents[1]
CONFIG = RerankConfig(
    model_id="Qwen/Qwen3-Reranker-0.6B",
    revision="e61197ed45024b0ed8a2d74b80b4d909f1255473",
    device="mps",
    max_input_tokens=1000,
    instruction="Find the rule",
)
Q = "How long may I drive?"


def test_the_committed_config_loads():
    config = load_config(ROOT / "config" / "rerank.yaml")
    assert config.model_id == "Qwen/Qwen3-Reranker-0.6B"
    embedding = (ROOT / "config" / "embedding.yaml").read_text(encoding="utf-8")
    assert config.instruction in embedding  # both models are asked the same thing


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("model_id: x\n", "missing revision"),
        ("model_id: x\nrevision: abc\ndevice: mps\nmax_input_tokens: 10\ninstruction: i\n", "40"),
        (
            "model_id: x\nrevision: " + "a" * 40 + "\ndevice: gpu\nmax_input_tokens: 10\n"
            "instruction: i\n",
            "device",
        ),
        (
            "model_id: x\nrevision: " + "a" * 40 + "\ndevice: mps\nmax_input_tokens: 10\n"
            "instruction: i\ndepth: 3\n",
            "unknown depth",
        ),
        ("- a list\n", "mapping"),
    ],
)
def test_a_bad_config_is_refused(tmp_path, text, message):
    path = tmp_path / "rerank.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RerankError, match=message):
        load_config(path)


def test_the_key_covers_every_setting_and_both_texts():
    base = cache_key(CONFIG, Q, "text")
    assert cache_key(CONFIG, Q, "text") == base
    assert cache_key(CONFIG, Q, "other text") != base
    assert cache_key(CONFIG, "Another question?", "text") != base
    for field, value in [("revision", "b" * 40), ("device", "cpu"), ("instruction", "Other")]:
        assert cache_key(dataclasses.replace(CONFIG, **{field: value}), Q, "text") != base


def test_the_cache_file_names_the_model_revision_and_device(tmp_path):
    cache = ScoreCache.load(tmp_path, CONFIG)
    assert setting_name(CONFIG) == "Qwen_Qwen3-Reranker-0.6B-e61197ed4502-mps"
    assert cache.path == tmp_path / "Qwen_Qwen3-Reranker-0.6B-e61197ed4502-mps.jsonl"


def test_scores_survive_a_reload_and_the_file_is_sorted(tmp_path):
    cache = ScoreCache.load(tmp_path, CONFIG)
    cache.put(Q, "b", -2.25, 30)
    cache.put(Q, "a", 3.8125, 31)
    cache.save()
    first = cache.path.read_text(encoding="utf-8")
    again = ScoreCache.load(tmp_path, CONFIG)
    assert again.get(Q, "a").score == 3.8125 and again.get(Q, "b").tokens == 30
    lines = first.splitlines()
    assert lines == sorted(lines)
    again.put(Q, "a", 3.8125, 31)
    again.save()
    assert cache.path.read_text(encoding="utf-8") == first


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"pair": "x", "tokens": 1, "score": 1.0}', "sha256"),
        ('{"pair": "' + "a" * 64 + '", "tokens": -1, "score": 1.0}', "count"),
        ('{"pair": "' + "a" * 64 + '", "tokens": 1, "score": NaN}', "finite"),
        ('{"pair": "' + "a" * 64 + '", "tokens": 1, "score": "1"}', "finite"),
        ('{"pair": "' + "a" * 64 + '", "tokens": 1}', "score"),
    ],
)
def test_a_damaged_cache_file_is_refused(tmp_path, line, message):
    (tmp_path / f"{setting_name(CONFIG)}.jsonl").write_text(line + "\n", encoding="utf-8")
    with pytest.raises(RerankError, match=message):
        ScoreCache.load(tmp_path, CONFIG)


def test_a_key_seen_twice_is_refused(tmp_path):
    line = '{"pair": "' + "a" * 64 + '", "tokens": 1, "score": 1.0}\n'
    (tmp_path / f"{setting_name(CONFIG)}.jsonl").write_text(line * 2, encoding="utf-8")
    with pytest.raises(RerankError, match="twice"):
        ScoreCache.load(tmp_path, CONFIG)


def test_the_cached_scorer_never_scores_part_of_a_list(tmp_path):
    cache = ScoreCache.load(tmp_path, CONFIG)
    cache.put(Q, "a", 1.5, 10)
    scorer = CachedScorer(cache)
    assert scorer.scores(Q, ["a"]) == [1.5]
    with pytest.raises(MissingScoresError, match="1 of 2"):
        scorer.scores(Q, ["a", "b"])


def test_missing_pairs_are_distinct_and_in_order(tmp_path):
    cache = ScoreCache.load(tmp_path, CONFIG)
    cache.put(Q, "a", 1.0, 10)
    pairs = [(Q, "c"), (Q, "a"), (Q, "b"), (Q, "c")]
    assert missing_pairs(pairs, cache) == [(Q, "c"), (Q, "b")]


# --- The model wrapper, with a stand-in for the model --------------------------------------


def encode(text: str) -> list[int]:
    """One token per character: enough to check what reaches the model."""
    return [ord(c) for c in text]


def reranker(run=None) -> tuple[LocalReranker, list[list[int]]]:
    fed: list[list[int]] = []

    def default(ids: list[int]) -> float:
        fed.append(ids)
        return float(len(ids)) / 100

    return LocalReranker(CONFIG, encode, run or default), fed


def test_the_model_reads_the_chat_its_card_gives():
    config = dataclasses.replace(CONFIG, max_input_tokens=10_000)
    fed: list[list[int]] = []
    model = LocalReranker(config, encode, lambda ids: fed.append(ids) or 0.5)
    score, tokens = model.score(Q, "Nine hours.")
    text = PREFIX + model_input(config, Q, "Nine hours.") + SUFFIX
    assert fed == [encode(text)]
    assert (score, tokens) == (0.5, len(text))
    assert model_input(config, Q, "Nine hours.") == (
        f"<Instruct>: Find the rule\n<Query>: {Q}\n<Document>: Nine hours."
    )


def test_an_input_over_the_limit_is_refused_not_cut():
    model, fed = reranker()
    with pytest.raises(RerankError, match="cut short"):
        model.score(Q, "x" * 1000)
    assert fed == []


def test_a_score_that_is_not_a_number_is_refused():
    model, _ = reranker(run=lambda ids: float("nan"))
    with pytest.raises(RerankError, match="finite"):
        model.score("q", "t")


def test_scoring_caches_each_pair_and_keeps_the_work_when_it_fails(tmp_path):
    cache = ScoreCache.load(tmp_path, CONFIG)
    calls = []

    def run(ids):
        calls.append(ids)
        if len(calls) == 3:
            raise RuntimeError("out of memory")
        return 1.0

    model = LocalReranker(dataclasses.replace(CONFIG, max_input_tokens=10_000), encode, run)
    error = score_missing([("q", "a"), ("q", "b"), ("q", "c")], cache, model)
    assert error == "RuntimeError: out of memory"
    again = ScoreCache.load(tmp_path, CONFIG)
    assert len(again) == 2 and again.get("q", "c") is None
