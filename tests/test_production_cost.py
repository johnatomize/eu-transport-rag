"""The production cost estimate (`rag.production_cost`), from hand-made run logs."""

import math

import pytest

from rag.generation_config import Price
from rag.production_cost import (
    MINUTES_A_MONTH,
    caching,
    fixed_tokens,
    hit_rate,
    production_cost,
    seconds,
    usage,
)

OPUS = Price(4, 20, 2, 10, cache_write_5m=5, cache_write_1h=8, cache_read=0.2)
HAIKU = Price(0.1, 0.5, 0.05, 0.25, cache_write_5m=0.125, cache_write_1h=0.2, cache_read=0.01)
PRICES = {"claude-opus-5-5": OPUS, "claude-haiku-5-5": HAIKU}
JUDGE = ("claude-opus-5-5", "j1")
RERANKER = "claude-haiku-5-5-low-abc"
BASE_RETRIEVAL = {"retriever": "hybrid", "strategy": "paragraph", "k": 5}


def call(qid, tokens_in, tokens_out, secs=None, estimated=None):
    return {
        "id": qid,
        "estimated_input_tokens": estimated if estimated is not None else tokens_in,
        "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
        "seconds": secs,
    }


def answer_log(split, calls, model="claude-opus-5-5", batch=True, **retrieval):
    log = {
        "config": {"model": model, "split": split, "retrieval": {**BASE_RETRIEVAL, **retrieval}},
        "answers": calls,
    }
    if batch:
        log["batch"] = {"status": "collected"}
    return log


def logs():
    """A base and a reranked finalist on the test split, and F's direct dev runs."""
    return [
        ("2026-11-03T20-00-00Z-claude-haiku-5-5.json",
         answer_log("dev", [call("q1", 1000, 300, 2.0), call("q2", 3000, 300, 4.0)],
                    model="claude-haiku-5-5", batch=False)),
        ("2026-11-03T21-00-00Z-claude-opus-5-5.json",
         answer_log("dev", [call("q1", 1000, 200, 5.0), call("q2", 3000, 200, 7.0)],
                    batch=False)),
        ("2026-11-09T07-00-02Z-claude-opus-5-5.json",
         answer_log("test", [call("q3", 2000, 500), call("q4", 2000, 500)],
                    reranker=RERANKER, rerank_depth=20)),
        ("2026-11-09T07-00-01Z-claude-opus-5-5.json",
         answer_log("test", [call("q3", 2000, 400, estimated=1000),
                             call("q4", 2000, 600, estimated=1000)])),
    ]  # fmt: skip


RERANKS = [
    (
        "2026-11-09T06-00-00Z-claude-haiku-5-5.json",
        {
            "config": {"reranker": RERANKER, "model": "claude-haiku-5-5", "split": "test"},
            "answers": [call("q3", 6000, 200, 1.0), call("q4", 6000, 200, 3.0)],
        },
    )
]
JUDGINGS = [
    (
        "run-2026-11-09T08-00-00Z-claude-opus-5-5.json",
        {
            "config": {"model": "claude-opus-5-5", "judge_prompt_version": "j1",
                       "answer_runs": ["2026-11-09T07-00-01Z-claude-opus-5-5.json"]},
            "judgements": [call("q3", 2000, 300)],
            "paid": {"cost_usd": 0.007},
        },
    )
]  # fmt: skip


def table(**overrides):
    args = dict(answer_logs=logs(), rerank_logs=RERANKS, judge_logs=JUDGINGS, prices=PRICES,
                prices_read_on="2026-11-09", estimated_fixed=600, judge=JUDGE)  # fmt: skip
    return production_cost(**{**args, **overrides})


def row(text, name):
    """The cells of the first table row in `text` naming `name`, after the name."""
    line = next(line for line in text.splitlines() if line.startswith(f"| {name} |"))
    return [cell.strip() for cell in line.strip("|").split("|")][1:]


def test_usage_and_seconds_skip_calls_without_a_reply_or_a_time():
    calls = [call("a", 100, 10, 1.0), {"id": "b", "usage": None}, call("c", 300, 30)]
    assert usage(calls) == usage([calls[0], calls[2]])
    assert usage(calls).input == 200 and usage(calls).output == 20
    assert usage([]).calls == 0
    timed = seconds([call(str(i), 1, 1, float(i)) for i in range(1, 11)])
    assert timed.mean == 5.5 and timed.p90 == 9.0  # nearest rank: the 9th of 10
    assert seconds([call("a", 1, 1)]) is None  # a batched run has no times


def test_hit_rate_is_the_chance_the_previous_question_came_within_the_lifetime():
    assert hit_rate(MINUTES_A_MONTH, 1) == pytest.approx(1 - math.exp(-1))
    assert hit_rate(0, 60) == 0


def test_caching_picks_the_cheaper_lifetime_or_none():
    # 1,000 a month: 5 minutes would hit 11% and cost more than not caching; 1 hour hits 75%.
    low = caching(1000, OPUS, 1000)
    assert low.lifetime == "1 hour" and low.hit_rate == pytest.approx(0.75, abs=0.01)
    p = hit_rate(1000, 60)
    assert low.saved_usd == pytest.approx(1000 * (4 - (p * 0.2 + (1 - p) * 8)) / 1e6)
    five_only = Price(4, 20, 2, 10, cache_write_5m=5, cache_read=0.2)
    assert caching(1000, five_only, 1000).lifetime is None
    assert caching(1000, five_only, 100_000).lifetime == "5 minutes"
    assert caching(500, OPUS, 100_000).lifetime is None  # too short to be cached
    assert caching(1000, Price(4, 20, 2, 10), 100_000).saved_usd == 0  # no cache prices


def test_fixed_tokens_are_raised_by_how_far_real_tokens_exceeded_the_estimates():
    log = logs()[3][1]  # estimated 1,000 a question, really 2,000
    assert fixed_tokens(log, 600) == 1200


def test_a_live_question_is_priced_at_real_time_prices_with_reranking_added():
    text = table()
    base, reranked = (
        "opus-5-5, hybrid paragraph top 5",
        ("opus-5-5, hybrid paragraph top 5 reranked by haiku-5-5-low-abc from top 20"),
    )
    assert text.index(f"| {base} |") < text.index(f"| {reranked} |")  # the base comes first
    # base: 2,000 in, 500 out at 4 and 20 a million
    assert row(text, base)[:6] == ["2,000", "500", "0.0180", "–", "0.0180", "18.00"]
    # reranked: the same answering plus 6,000 in and 200 out at Haiku's 0.10 and 0.50
    assert row(text, reranked)[2:5] == ["0.0180", "0.0007", "0.0187"]
    assert "read on 2026-11-09" in text


def test_a_month_without_and_with_caching():
    cells = row(table().split("## A month")[1], "opus-5-5, hybrid paragraph top 5")
    saved = caching(1200, OPUS, 100_000).saved_usd
    assert cells[0] == "18.00" and cells[4] == "1,800"
    assert cells[5] == f"{(0.018 - saved) * 100_000:,.0f}"


def test_time_adds_the_rerankers_round_trip_to_opus_answering_from_the_direct_run():
    text = table().split("## Time a question")[1]
    assert "`2026-11-03T21-00-00Z-claude-opus-5-5.json`" in text
    assert row(text, "opus-5-5, hybrid paragraph top 5") == ["–", "6.0 (90%: 7.0)", "6.0"]
    reranked = "opus-5-5, hybrid paragraph top 5 reranked by haiku-5-5-low-abc from top 20"
    assert row(text, reranked) == ["2.0 (90%: 3.0)", "6.0 (90%: 7.0)", "8.0"]


def test_haiku_answering_is_priced_with_its_measured_output_ratio():
    text = table().split("## Lever: Haiku 5.5 answering")[1].split("## Lever: caching")[0]
    assert "1.50 times Opus's output tokens" in text  # 300 against 200
    # base: 2,000 in at 0.10, 500 × 1.5 out at 0.50
    cost = float(row(text, "opus-5-5, hybrid paragraph top 5")[0])
    assert cost == pytest.approx(0.000575, abs=1e-5)


def test_judging_is_shown_apart_at_real_time_and_as_paid():
    text = table().split("## Evaluation only: judging")[1]
    cells = row(text, "opus-5-5, hybrid paragraph top 5")
    assert cells == ["`run-2026-11-09T08-00-00Z-claude-opus-5-5.json`", "1", "0.0140", "0.0070"]
    reranked = "opus-5-5, hybrid paragraph top 5 reranked by haiku-5-5-low-abc from top 20"
    assert row(text, reranked) == ["–", "0", "–", "–"]  # not judged
    other_judge = table(judge=("claude-opus-5-5", "other"))
    assert "`run-2026-11-09T08-00-00Z" not in other_judge


def test_a_reranked_finalist_without_its_rerankers_run_is_refused():
    with pytest.raises(ValueError, match="no scoring run of the reranker"):
        table(rerank_logs=[])


def test_no_test_runs_yet():
    text = table(answer_logs=logs()[:2])
    assert text.endswith("No runs on the test split yet.\n")


def test_a_rerankers_run_that_only_topped_up_its_cache_is_not_used():
    topped_up = {"config": RERANKS[0][1]["config"], "answers": [call("q3", 6000, 200, 9.0)]}
    later = ("2026-11-09T09-00-00Z-claude-haiku-5-5.json", topped_up)
    text = table(rerank_logs=[*RERANKS, later])
    reranked = "opus-5-5, hybrid paragraph top 5 reranked by haiku-5-5-low-abc from top 20"
    assert row(text.split("## Time a question")[1], reranked)[0] == "2.0 (90%: 3.0)"
    with pytest.raises(ValueError, match="that reranked every test question"):
        table(rerank_logs=[later])


def test_a_later_direct_run_of_a_few_questions_does_not_replace_the_whole_one():
    few = ("2026-11-05T00-00-00Z-claude-opus-5-5.json",
           answer_log("dev", [call("q1", 1000, 200, 30.0)], batch=False))  # fmt: skip
    text = table(answer_logs=[*logs(), few]).split("## Time a question")[1]
    assert "`2026-11-03T21-00-00Z-claude-opus-5-5.json`" in text
    assert row(text, "opus-5-5, hybrid paragraph top 5")[1] == "6.0 (90%: 7.0)"


def test_missing_cache_prices_are_refused_with_a_message():
    with pytest.raises(ValueError, match="no cache_write_5m, cache_write_1h, cache_read price"):
        table(prices={**PRICES, "claude-opus-5-5": Price(4, 20, 2, 10)})


def test_the_haiku_lever_needs_questions_both_models_answered():
    answers = logs()
    answers[0][1]["answers"] = [call("q9", 1000, 300, 2.0)]  # no question in common with Opus
    text = table(answer_logs=answers)
    assert "have no answered question in common" in text
