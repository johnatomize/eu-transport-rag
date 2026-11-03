import datetime as dt
from pathlib import Path

import pytest

from rag.generation_config import (
    GenerationConfigError,
    Price,
    load_generation_config,
)

ROOT = Path(__file__).parents[1]
GOOD = """\
provider: anthropic
build_model: claude-haiku-5-5
scored_model: claude-opus-5-5
max_prompt_tokens: 100000
prices_read_on: 2026-11-03
prices:
  claude-haiku-5-5: {input: 0.10, output: 0.50, batch_input: 0.05, batch_output: 0.25}
  claude-opus-5-5: {input: 4, output: 20, batch_input: 2, batch_output: 10}
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "generation.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_committed_config_names_haiku_for_building_and_opus_for_scoring():
    config = load_generation_config(ROOT / "config" / "generation.yaml")
    assert config.provider == "anthropic"
    assert config.model("build") == "claude-haiku-5-5"
    assert config.model("scored") == "claude-opus-5-5"
    assert config.prices["claude-opus-5-5"] == Price(4.0, 20.0, 2.0, 10.0)
    assert config.prices["claude-haiku-5-5"] == Price(0.10, 0.50, 0.05, 0.25)
    assert config.prices_read_on == dt.date(2026, 11, 3)


def test_whole_numbers_are_read_as_prices(tmp_path):
    config = load_generation_config(write(tmp_path, GOOD))
    assert config.prices["claude-opus-5-5"].input == 4.0


def test_unknown_role_is_refused(tmp_path):
    config = load_generation_config(write(tmp_path, GOOD))
    with pytest.raises(ValueError, match="role"):
        config.model("cheap")


@pytest.mark.parametrize(
    ("old", "new", "problem"),
    [
        ("provider: anthropic", "provider: bedrock", "provider should be"),
        ("scored_model: claude-opus-5-5", "scored_model: claude-opus-5", "has no prices"),
        ("max_prompt_tokens: 100000\n", "", "missing max_prompt_tokens"),
        ("max_prompt_tokens: 100000", "max_prompt_tokens: 0", "above 0"),
        ("prices_read_on: 2026-11-03", "prices_read_on: soon", "should be a date"),
        ("provider: anthropic\n", "provider: anthropic\neffort: low\n", "unknown effort"),
        ("input: 4,", "input: true,", "input should be a number above 0"),
        ("input: 4,", "input: 0,", "input should be a number above 0"),
        ("input: 4,", "input: .nan,", "input should be a number above 0"),
        ("input: 4,", "input: .inf,", "input should be a number above 0"),
        ("batch_output: 10}", "batch_output: 10, cache: 1}", "unknown cache"),
        (", batch_output: 10}", "}", "missing batch_output"),
    ],
)
def test_impossible_settings_are_refused(tmp_path, old, new, problem):
    assert old in GOOD
    with pytest.raises(GenerationConfigError, match=problem):
        load_generation_config(write(tmp_path, GOOD.replace(old, new, 1)))


def test_unreadable_or_malformed_file_is_refused(tmp_path):
    with pytest.raises(GenerationConfigError, match="cannot read"):
        load_generation_config(tmp_path / "missing.yaml")
    with pytest.raises(GenerationConfigError, match="not valid YAML"):
        load_generation_config(write(tmp_path, "provider: [anthropic"))
