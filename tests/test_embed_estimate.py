import datetime as dt
from pathlib import Path

import pytest

from rag.embed_estimate import (
    MARGIN,
    ConfigError,
    EmbeddingConfig,
    batch,
    cost_usd,
    estimate_tokens,
    load_config,
    report,
)

ROOT = Path(__file__).resolve().parents[1]
READ_ON = dt.date(2026, 10, 31)

CONFIG = EmbeddingConfig(
    model_id="amazon.titan-embed-text-v2:0",
    region="eu-north-1",
    dimensions=1024,
    normalize=True,
    max_input_tokens=150,
    price_usd_per_million_input_tokens=0.02,
    price_read_on=READ_ON,
)
GOOD = (
    "model_id: m\nregion: r\ndimensions: 1024\nnormalize: true\nmax_input_tokens: 8192\n"
    "price_usd_per_million_input_tokens: 0.021\nprice_read_on: 2026-10-31\n"
)


def write(tmp_path, text):
    path = tmp_path / "embedding.yaml"
    path.write_text(text)
    return path


def test_the_committed_config_loads():
    config = load_config(ROOT / "config" / "embedding.yaml")
    assert config.model_id == "amazon.titan-embed-text-v2:0"
    assert config.region == "eu-north-1"
    assert config.price_read_on == READ_ON


def test_a_config_with_a_missing_unknown_or_mistyped_key_is_refused(tmp_path):
    text = GOOD.replace("dimensions: 1024", "dimensions: '1024'")
    text = text.replace("price_usd_per_million_input_tokens: 0.021\n", "colour: blue\n")
    with pytest.raises(ConfigError) as e:
        load_config(write(tmp_path, text))
    message = str(e.value)
    assert "missing price_usd_per_million_input_tokens" in message
    assert "unknown colour" in message
    assert "dimensions should be int" in message


def test_a_yes_for_the_price_is_refused_not_read_as_one_dollar(tmp_path):
    text = GOOD.replace("tokens: 0.021", "tokens: yes")
    with pytest.raises(ConfigError, match="price_usd_per_million_input_tokens should be float"):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize(
    ("old", "new", "problem"),
    [
        ("tokens: 0.021", "tokens: 0", "price_usd_per_million_input_tokens should be above 0"),
        ("max_input_tokens: 8192", "max_input_tokens: -1", "max_input_tokens should be above 0"),
        ("dimensions: 1024", "dimensions: 1536", "dimensions should be one of"),
    ],
)
def test_impossible_values_are_refused(tmp_path, old, new, problem):
    with pytest.raises(ConfigError, match=problem):
        load_config(write(tmp_path, GOOD.replace(old, new)))


def test_broken_yaml_or_a_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(write(tmp_path, "model_id: [unclosed\n"))
    with pytest.raises(ConfigError, match="cannot read it"):
        load_config(tmp_path / "absent.yaml")


def test_tokens_take_the_higher_of_the_two_rules():
    # Long words: characters / 4 wins (27 characters → 7; 2 words × 1.3 → 3).
    assert estimate_tokens("Interoperability Regulation") == 7
    # Short words: words × 1.3 wins (10 words → 13; 30 characters → 8).
    assert estimate_tokens("I may do it if he or she is ok") == 13
    assert estimate_tokens("") == 0


def test_a_batch_sums_tokens_keeps_the_largest_and_counts_empty_inputs():
    b = batch("chunks: test", ["one two three four", "x" * 40, " "])
    assert (b.texts, b.tokens, b.largest, b.empty) == (3, 6 + 10 + 1, 10, 1)


def test_cost_is_priced_per_million_tokens():
    assert cost_usd(1_000_000, CONFIG) == pytest.approx(0.02)
    assert cost_usd(250_000, CONFIG) == pytest.approx(0.005)


def test_the_report_adds_the_margin_and_states_the_price():
    texts = [f"{i:03d}" + "x" * 397 for i in range(10)]  # 100 tokens each, all different
    lines, fits = report({"questions": texts}, CONFIG, READ_ON)
    text = "\n".join(lines)
    assert fits
    assert f"{int(1000 * MARGIN):,} tokens" in text
    assert "US$0.02 per million input tokens, read 2026-10-31" in text
    assert "amazon.titan-embed-text-v2:0 in eu-north-1" in text
    assert "warning" not in text


def test_a_text_in_two_chunkings_is_counted_once_in_the_total():
    shared, other = "x" * 400, "y" * 400  # 100 tokens each
    lines, _ = report({"a": [shared], "b": [shared, other]}, CONFIG, READ_ON)
    total = next(line for line in lines if line.startswith("total, distinct texts"))
    assert total.split()[3:5] == ["2", "200"]


def test_the_report_fails_when_an_input_with_the_margin_may_be_over_the_limit():
    # 101 tokens × 1.5 = 152, over the fixture's limit of 150.
    lines, fits = report({"chunks: article": ["x" * 404]}, CONFIG, READ_ON)
    assert not fits
    assert "error: chunks: article has an input of about 101 tokens" in "\n".join(lines)


def test_the_report_fails_on_an_empty_input():
    lines, fits = report({"questions": ["What?", "  "]}, CONFIG, READ_ON)
    assert not fits
    assert "error: questions has 1 empty input(s)" in "\n".join(lines)


def test_an_old_price_is_warned_about():
    lines, fits = report({"questions": ["What?"]}, CONFIG, READ_ON + dt.timedelta(days=30))
    assert fits
    assert "warning: the price was read 30 days ago" in "\n".join(lines)
