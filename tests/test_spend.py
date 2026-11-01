import json
from pathlib import Path

import pytest

from rag.spend import SpendLogError, load_spend_log, read_spend_log, unlogged_runs

ROOT = Path(__file__).resolve().parents[1]

LOG = """# Progress

## Spend log

Approved cap: **US$20 for the whole project** (approved 2026-10-31).

| Date | Item | Cost (US$) | Total (US$) |
| --- | --- | ---: | ---: |
| 2026-10-27 | Project start | 0.00 | 0.00 |
| 2026-11-01 | Embeddings | 0.01 | 0.01 |
| 2026-11-02 | Answer run | 1.25 | 1.26 |

## Later section

| 2026-11-03 | not part of the log | 9.00 | 9.00 |
"""


def test_reads_the_cap_and_the_sum_of_costs():
    log = read_spend_log(LOG)
    assert (log.cap_usd, log.spent_usd) == (20.0, 1.26)


def test_a_wrong_total_is_refused():
    with pytest.raises(SpendLogError, match="add up to US\\$1.26 but the last total says"):
        read_spend_log(LOG.replace("| 1.25 | 1.26 |", "| 1.25 | 0.26 |"))


@pytest.mark.parametrize(
    "change, message",
    [
        (("## Spend log", "## Spending"), "no '## Spend log' section"),
        (("Approved cap: **US$20", "Approved cap: none yet"), "expected one 'Approved cap"),
    ],
)
def test_a_log_it_cannot_read_is_refused(change, message):
    with pytest.raises(SpendLogError, match=message):
        read_spend_log(LOG.replace(*change))


def test_the_real_spend_log_can_be_read():
    log = load_spend_log(ROOT / "docs" / "PROGRESS.md")
    assert log.cap_usd == 20.0
    assert log.spent_usd >= 0


def test_whole_dollar_amounts_count():
    log = read_spend_log(LOG.replace("| 1.25 | 1.26 |", "| 1 | 1.01 |"))
    assert log.spent_usd == 1.01


@pytest.mark.parametrize("cells", ["| US$1.25 | 1.26 |", "| 1,250.00 | 1,250.01 |", "| | |"])
def test_a_row_it_cannot_read_is_refused_not_skipped(cells):
    with pytest.raises(SpendLogError, match="not plain amounts"):
        read_spend_log(LOG.replace("| 1.25 | 1.26 |", cells))


def _run_log(directory, name, cost):
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(json.dumps({"paid": {"cost_usd": cost}}), encoding="utf-8")


def test_paid_runs_the_spend_log_does_not_name_yet_are_counted(tmp_path):
    runs = tmp_path / "embeddings"
    _run_log(runs, "2026-11-01T10-00-00Z.json", 0.004)
    _run_log(runs, "2026-11-02T10-00-00Z.json", 0.002)
    named = LOG.replace("| Embeddings |", "| Embeddings (2026-11-01T10-00-00Z.json) |")
    assert unlogged_runs(named, runs) == [("2026-11-02T10-00-00Z.json", 0.002)]
    assert len(unlogged_runs(LOG, runs)) == 2
    assert unlogged_runs(LOG, tmp_path / "none yet") == []
