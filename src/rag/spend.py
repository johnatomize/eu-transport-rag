"""The spend log in `docs/PROGRESS.md`, read so that a paid run can refuse to pass the cap.

The log is Markdown, written by hand. Two parts of it are read:

    Approved cap: **US$20 for the whole project** (…)

    | Date | Item | Cost (US$) | Total (US$) |
    | --- | --- | ---: | ---: |
    | 2026-10-27 | Project start | 0.00 | 0.00 |

What has been spent is the sum of the cost column, plus any paid run whose log in one of
`PAID_RUNS` (embeddings on Bedrock, answers from Claude) no row names yet (`unlogged_runs`).
The last total must match that sum, so a row added with a wrong total is caught before it can
hide spending.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

_CAP = re.compile(r"^Approved cap: \*\*US\$(\d+(?:\.\d+)?)\b", re.MULTILINE)
_ROW = re.compile(r"^\|\s*\d{4}-\d{2}-\d{2}\s*\|")  # a row of the table: it starts with a date
_AMOUNT = re.compile(r"-?\d+(?:\.\d+)?")


# Where paid runs write their logs; each log's ["paid"]["cost_usd"] is what the run cost.
PAID_RUNS = ("runs/embeddings", "runs/generation")


class SpendLogError(Exception):
    pass


@dataclass(frozen=True)
class SpendLog:
    cap_usd: float
    spent_usd: float


def read_spend_log(text: str) -> SpendLog:
    """The approved cap and what has been spent so far, from the text of `PROGRESS.md`."""
    section = _section(text)
    caps = _CAP.findall(section)
    if len(caps) != 1:
        raise SpendLogError(f"expected one 'Approved cap: **US$…' line, found {len(caps)}")
    rows = [_amounts(line) for line in section.splitlines() if _ROW.match(line)]
    if not rows:
        raise SpendLogError("no rows in the spend log table")
    spent = round(sum(cost for cost, _ in rows), 6)
    last_total = rows[-1][1]
    if abs(spent - last_total) > 0.005:
        raise SpendLogError(
            f"the costs add up to US${spent:.2f} but the last total says US${last_total:.2f}"
        )
    return SpendLog(cap_usd=float(caps[0]), spent_usd=spent)


def _section(text: str) -> str:
    _, found, section = text.partition("\n## Spend log\n")
    if not found:
        raise SpendLogError("no '## Spend log' section")
    return section.split("\n## ", 1)[0]


def unlogged_runs(text: str, runs: Path) -> list[tuple[str, float]]:
    """Paid runs whose call log in `runs` the spend log doesn't name yet, with their cost.

    A paid run writes its call log at once, but its row in the spend log is added by hand, and
    names the log's file. Until then the run's cost is counted here, so that two runs in a row
    can't both pass the cap check against a spend log that shows neither.
    """
    section = _section(text)
    pending = []
    for path in sorted(runs.glob("*.json")):
        if path.name in section:
            continue
        try:
            cost = json.loads(path.read_text(encoding="utf-8"))["paid"]["cost_usd"]
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise SpendLogError(f"{path.name}: cannot read its cost ({e})") from e
        pending.append((path.name, float(cost)))
    return pending


def _amounts(row: str) -> tuple[float, float]:
    """A row's cost and total, its last two cells: plain amounts in US$, such as 0.01 or 12."""
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    if len(cells) < 4 or not all(_AMOUNT.fullmatch(c) for c in cells[-2:]):
        raise SpendLogError(f"a row whose cost and total are not plain amounts: {row.strip()}")
    return float(cells[-2]), float(cells[-1])


def load_spend_log(path: Path) -> SpendLog:
    try:
        return read_spend_log(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise SpendLogError(f"{path.name}: cannot read it ({e.strerror})") from e
    except SpendLogError as e:
        raise SpendLogError(f"{path.name}: {e}") from e


@dataclass(frozen=True)
class Spending:
    cap_usd: float
    spent_usd: float  # the spend log plus every paid run it doesn't name yet
    pending: tuple[tuple[str, float], ...]  # those runs: their path from the root, and cost


def spending(root: Path) -> Spending:
    """What has been spent, counting every paid run's log the spend log doesn't name yet."""
    path = root / "docs" / "PROGRESS.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SpendLogError(f"{path.name}: cannot read it ({e.strerror})") from e
    log = read_spend_log(text)
    pending = tuple(
        (f"{runs}/{name}", cost)
        for runs in PAID_RUNS
        for name, cost in unlogged_runs(text, root / runs)
    )
    spent = log.spent_usd + sum(cost for _, cost in pending)
    return Spending(cap_usd=log.cap_usd, spent_usd=spent, pending=pending)
