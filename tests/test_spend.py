import json
from pathlib import Path

import pytest

from rag.spend import (
    SpendLogError,
    load_spend_log,
    read_spend_log,
    spending,
    unlogged_runs,
)

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


def test_spending_counts_paid_answer_runs_and_embedding_runs_not_yet_logged(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "PROGRESS.md").write_text(LOG, encoding="utf-8")
    _run_log(tmp_path / "runs", "keep-dir.json", 0.0)  # makes runs/; not a paid directory
    _run_log(tmp_path / "runs" / "embeddings", "2026-11-02T10-00-00Z.json", 0.002)
    _run_log(tmp_path / "runs" / "generation", "2026-11-04T10-00-00Z-claude-haiku-5-5.json", 0.03)
    spent = spending(tmp_path)
    assert spent.cap_usd == read_spend_log(LOG).cap_usd
    assert spent.spent_usd == pytest.approx(read_spend_log(LOG).spent_usd + 0.032)
    assert [name for name, _ in spent.pending] == [
        "runs/embeddings/2026-11-02T10-00-00Z.json",
        "runs/generation/2026-11-04T10-00-00Z-claude-haiku-5-5.json",
    ]


def test_spending_counts_hosted_reranking_runs_not_yet_logged(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "PROGRESS.md").write_text(LOG, encoding="utf-8")
    (tmp_path / "runs").mkdir()
    _run_log(tmp_path / "runs" / "rerank", "2026-11-06T12-00-00Z-jev-1.14.0.json", 0.03)
    _run_log(tmp_path / "runs" / "rerank-local", "2026-11-06T10-46-34Z-needed.json", 0.0)
    spent = spending(tmp_path)
    assert [name for name, _ in spent.pending] == [
        "runs/rerank/2026-11-06T12-00-00Z-jev-1.14.0.json"
    ]
    assert spent.spent_usd == pytest.approx(read_spend_log(LOG).spent_usd + 0.03)
