"""What each finalist would cost to run as a live service (#102; design §G, production cost).

Built by `make tables` from the run logs and the prices in `config/generation.yaml`, without
calling anything. The evaluation runs went through the batch service at half price and were
judged; a live question is answered at once, at the real-time list price, and isn't judged. So
each finalist's cost is its test run's tokens a question at the real-time prices:

- **Answering**: the mean input and output tokens of its test-split answer run.
- **Reranking** (a finalist with a hosted reranker): the mean tokens of that reranker's scoring
  run on the test questions, at its model's real-time prices.
- **Judging** is shown on its own: it is part of the evaluation, not of the service.

The time a question takes comes from runs that were sent one call at a time, since a batched
reply has no time of its own: Opus answering from F's direct run on the dev questions, and the
reranker's round trip from its scoring run.

The levers:

- **Haiku 5.5 answering in place of Opus 5.5**, measured in F's direct dev runs, where both
  answered the same 60 questions with the same passages: the same input tokens, and Haiku's
  output tokens relative to Opus's. Their quality is compared in `results/answers.md`.
- **Prompt caching of the fixed instructions** (the system prompt and the answer schema), which
  is estimated, not measured: no run asked for caching. Their size is the prompt code's own
  estimate, raised by how much the run's real input tokens exceeded its estimates. Whether a
  question finds the cache still warm depends on how often questions come: if they come at
  random times at an even rate over a 30-day month, the chance that the previous question was
  within the cache's lifetime is 1 − e^(−rate × lifetime). A miss writes the cache, which costs
  more than not caching (1.25 times the input price for 5 minutes, 2 times for 1 hour); a hit
  reads it, at the cache-read price. The cheaper lifetime is chosen at each volume, or none.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag.generate import cost_usd
from rag.generation_config import Price
from rag.tables import BASE, _header, _row, is_base, run_label

VOLUMES = (1_000, 10_000, 100_000)  # questions a month (design §G; hosting in §I uses them too)
MINUTES_A_MONTH = 30 * 24 * 60
# Prompt caching's two lifetimes: (name, minutes, the write price's field in `Price`).
CACHE_LIFETIMES = (("5 minutes", 5, "cache_write_5m"), ("1 hour", 60, "cache_write_1h"))
CACHE_MIN_TOKENS = 512  # the shortest prefix Opus 5.5 and Haiku 5.5 cache; shorter isn't cached
OPUS, HAIKU = "claude-opus-5-5", "claude-haiku-5-5"  # the Haiku lever compares these two
CACHE_PRICES = ("cache_write_5m", "cache_write_1h", "cache_read")


@dataclass(frozen=True)
class Usage:
    """Mean tokens a call, over the calls with a reply."""

    calls: int
    input: float
    output: float

    def cost(self, price: Price) -> float:
        """US$ a call at the real-time prices."""
        return cost_usd(self.input, self.output, price)


@dataclass(frozen=True)
class Seconds:
    calls: int
    mean: float
    p90: float


@dataclass(frozen=True)
class Caching:
    """What caching the fixed instructions would save on one question at one volume."""

    lifetime: str | None  # the cheaper lifetime, or None when caching wouldn't pay
    hit_rate: float  # the chance a question finds the cache warm, with that lifetime
    saved_usd: float  # US$ a question, 0 when caching wouldn't pay


def usage(entries: Sequence[Mapping]) -> Usage:
    """Mean tokens over the entries (answers, judgements or rerankings) with a reply."""
    used = [e["usage"] for e in entries if e.get("usage")]
    if not used:
        return Usage(0, 0.0, 0.0)
    n = len(used)
    return Usage(
        n,
        sum(u["input_tokens"] for u in used) / n,
        sum(u["output_tokens"] for u in used) / n,
    )


def seconds(entries: Sequence[Mapping]) -> Seconds | None:
    """Mean and 90th percentile of the calls' times; None if no call was timed (a batch)."""
    timed = sorted(e["seconds"] for e in entries if e.get("seconds") is not None)
    if not timed:
        return None
    rank = max(math.ceil(0.9 * len(timed)) - 1, 0)  # nearest-rank percentile
    return Seconds(len(timed), sum(timed) / len(timed), timed[rank])


def hit_rate(questions_a_month: int, minutes: float) -> float:
    """The chance a question comes within `minutes` of the one before, at random times."""
    return 1 - math.exp(-questions_a_month / MINUTES_A_MONTH * minutes)


def caching(fixed_tokens: float, price: Price, questions_a_month: int) -> Caching:
    """The cheaper cache lifetime for `fixed_tokens` at this volume, and what it saves."""
    if fixed_tokens < CACHE_MIN_TOKENS or price.cache_read is None:
        return Caching(None, 0.0, 0.0)
    best = Caching(None, 0.0, 0.0)
    for name, minutes, write_field in CACHE_LIFETIMES:
        write = getattr(price, write_field)
        if write is None:
            continue
        p = hit_rate(questions_a_month, minutes)
        per_million = p * price.cache_read + (1 - p) * write
        saved = fixed_tokens * (price.input - per_million) / 1e6
        if saved > best.saved_usd:
            best = Caching(name, p, saved)
    return best


def fixed_tokens(log: Mapping, estimated_fixed: int) -> float:
    """The fixed instructions' tokens in this run's prompts: the prompt code's estimate, raised by
    how far the run's real input tokens exceeded its estimated ones."""
    answered = [e for e in log["answers"] if e.get("usage")]
    estimated = sum(e["estimated_input_tokens"] for e in answered)
    real = sum(e["usage"]["input_tokens"] for e in answered)
    return estimated_fixed * (real / estimated if estimated else 1.0)


def _direct(logs: Sequence[tuple[str, Mapping]], model: str) -> tuple[str, Mapping] | None:
    """The direct (not batched) run of the base setup on the dev questions by `model` with the
    most timed answers, the latest of those, so a later run of a few questions can't stand in for
    a whole one."""
    found = [
        (file, log)
        for file, log in logs
        if log["config"]["split"] == "dev"
        and log["config"]["model"] == model
        and "batch" not in log
        and dict(log["config"]["retrieval"]) == BASE["retrieval"]
        and seconds(log["answers"]) is not None
    ]
    return max(found, key=lambda item: (seconds(item[1]["answers"]).calls, item[0]), default=None)


def _reranking(log: Mapping, rerank_logs: Sequence[tuple[str, Mapping]]) -> tuple[str, Mapping]:
    """The latest scoring run on the same split by the hosted reranker this answer run used that
    reranked every question the answer run answered. A run that only topped up the reranker's
    cache scored fewer, and its cost and time would stand for a question wrongly."""
    reranker = log["config"]["retrieval"]["reranker"]
    split = log["config"]["split"]
    wanted = {e["id"] for e in log["answers"]}
    found = [
        (file, r)
        for file, r in rerank_logs
        if r["config"]["reranker"] == reranker
        and r["config"].get("split", "dev") == split
        and wanted <= {e["id"] for e in r["answers"] if e.get("usage")}
    ]
    if not found:
        raise ValueError(
            f"no scoring run of the reranker {reranker} that reranked every {split} question"
        )
    return max(found, key=lambda item: item[0])


def _judging(
    file: str, judge_logs: Sequence[tuple[str, Mapping]], judge: tuple[str, str]
) -> tuple[str, Mapping] | None:
    """The chosen judge's latest whole run on this answer run, if there is one."""
    model, version = judge
    found = [
        (name, j)
        for name, j in judge_logs
        if j["config"]["model"] == model
        and j["config"]["judge_prompt_version"] == version
        and j["config"]["answer_runs"] == [file]
    ]
    return max(found, key=lambda item: item[0]) if found else None


def _usd(value: float) -> str:
    return f"{value:.4f}"


def _money(value: float) -> str:
    return f"{value:,.0f}" if value >= 100 else f"{value:,.2f}"


def _secs(s: Seconds | None) -> str:
    return f"{s.mean:.1f} (90%: {s.p90:.1f})" if s else "–"


@dataclass(frozen=True)
class Finalist:
    name: str
    file: str
    model: str  # the model that answers
    answering: Usage
    answer_usd: float
    rerank_file: str | None
    reranking: Usage | None
    rerank_usd: float
    rerank_seconds: Seconds | None
    fixed: float  # tokens of fixed instructions in each prompt

    @property
    def total_usd(self) -> float:
        return self.answer_usd + self.rerank_usd


def production_cost(
    answer_logs: Sequence[tuple[str, Mapping]],
    rerank_logs: Sequence[tuple[str, Mapping]],
    judge_logs: Sequence[tuple[str, Mapping]],
    prices: Mapping[str, Price],
    prices_read_on: str,
    estimated_fixed: int,
    judge: tuple[str, str],
) -> str:
    """The production cost estimate, as one Markdown document.

    `answer_logs` are every answer run `(file name, log)`; the finalists are the runs on the test
    split, the base setup first. `rerank_logs` are the hosted rerankers' scoring runs,
    `judge_logs` the faithfulness judge's whole runs and `judge` the chosen judge, `(model,
    instructions version)`. `estimated_fixed` is the prompt code's estimate of the fixed
    instructions' tokens.
    """
    lines = [
        "# Production cost",
        "",
        "Generated by `make tables` from the run logs and `config/generation.yaml`; don't edit "
        "by hand.",
        "",
        "What each finalist would cost as a live service, from its run on the test questions at "
        f"Anthropic's real-time list prices, read on {prices_read_on}. The evaluation runs went "
        "through the batch service at half price; a live question can't wait for a batch, which "
        "answers within hours rather than seconds, so the batch price doesn't apply. Hosting, "
        "and the search and question embedding that run on it, are estimated in checkpoint I "
        "at the same volumes. Rules in `src/rag/production_cost.py`.",
    ]
    finals = sorted(
        ((f, log) for f, log in answer_logs if log["config"]["split"] == "test"),
        key=lambda item: (not is_base(item[1]["config"]), item[0]),
    )
    if not finals:
        return "\n".join([*lines, "", "No runs on the test split yet."]) + "\n"

    finalists = []
    for file, log in finals:
        model = log["config"]["model"]
        missing = [k for k in CACHE_PRICES if getattr(prices[model], k) is None]
        if missing:
            raise ValueError(f"no {', '.join(missing)} price for {model} in the prices")
        answering = usage(log["answers"])
        rerank_file, reranking, rerank_usd, rerank_secs = None, None, 0.0, None
        if "reranker" in log["config"]["retrieval"]:
            rerank_file, rerank_log = _reranking(log, rerank_logs)
            reranking = usage(rerank_log["answers"])
            rerank_usd = reranking.cost(prices[rerank_log["config"]["model"]])
            rerank_secs = seconds(rerank_log["answers"])
        finalists.append(
            Finalist(
                run_label(log["config"]),
                file,
                model,
                answering,
                answering.cost(prices[model]),
                rerank_file,
                reranking,
                rerank_usd,
                rerank_secs,
                fixed_tokens(log, estimated_fixed),
            )  # fmt: skip
        )

    lines += ["", "## One live question", ""]
    lines += [
        "Mean tokens a question in each finalist's test run, at the real-time prices. A "
        "reranker reads the question and the passages search ranked highest and returns their "
        "order; the answering model then reads the first few of that order, as in the setup's "
        "name.",
        "",
    ]
    titles = ["Finalist", "Input tokens", "Output tokens", "Answering (US$)",
              "Reranking (US$)", "A question (US$)", "Per 1,000 questions (US$)"]  # fmt: skip
    lines += _header(titles)
    for f in finalists:
        lines.append(
            _row(
                [
                    f.name,
                    f"{f.answering.input:,.0f}",
                    f"{f.answering.output:,.0f}",
                    _usd(f.answer_usd),
                    _usd(f.rerank_usd) if f.reranking else "–",
                    _usd(f.total_usd),
                    _money(f.total_usd * 1000),
                ]
            )  # fmt: skip
        )

    lines += ["", "## A month", ""]
    lines += [
        "US$ a month at three volumes, without caching and with the cheaper cache lifetime at "
        "each volume (see the levers below).",
        "",
    ]
    titles = ["Finalist"]
    for v in VOLUMES:
        titles += [f"{v:,} a month", f"{v:,}, cached"]
    lines += _header(titles)
    for f in finalists:
        cells: list[object] = [f.name]
        for v in VOLUMES:
            saved = caching(f.fixed, prices[f.model], v).saved_usd
            cells += [_money(f.total_usd * v), _money((f.total_usd - saved) * v)]
        lines.append(_row(cells))

    direct = {m: _direct(answer_logs, m) for m in {OPUS, HAIKU, *(f.model for f in finalists)}}
    answer_secs = {
        m: seconds(found[1]["answers"]) if found else None for m, found in direct.items()
    }
    sources = sorted(f"`{direct[m][0]}`" for m in {f.model for f in finalists} if direct[m])
    lines += ["", "## Time a question", ""]
    lines += [
        "Seconds from sending the prompt to the whole reply, mean and 90th percentile. The "
        "answer time is the answering model's in F's direct run on the dev questions "
        f"({', '.join(sources) or 'none yet'}), with the base setup's prompts; the test runs were "
        "batched and have no times. A longer prompt, as with article chunks, adds a little; "
        "most of the time is the model writing. The reranker's time is its round trip in its "
        "scoring run. Search and the question's embedding aren't counted: they run on the "
        "service's own machine (checkpoint I).",
        "",
    ]
    lines += _header(["Finalist", "Reranking (s)", "Answering (s)", "Together, mean (s)"])
    for f in finalists:
        answer = answer_secs[f.model]
        rerank = f.rerank_seconds.mean if f.rerank_seconds else 0.0
        lines.append(
            _row(
                [
                    f.name,
                    _secs(f.rerank_seconds) if f.reranking else "–",
                    _secs(answer),
                    f"{answer.mean + rerank:.1f}" if answer else "–",
                ]
            )  # fmt: skip
        )

    lines += _haiku_lever(finalists, direct, prices)
    lines += _caching_lever(finalists, prices)
    lines += _judging_section(finals, finalists, judge_logs, judge, prices)
    return "\n".join(lines) + "\n"


def _haiku_lever(
    finalists: Sequence[Finalist],
    direct: Mapping[str, tuple[str, Mapping] | None],
    prices: Mapping[str, Price],
) -> list[str]:
    lines = ["", "## Lever: Haiku 5.5 answering", ""]
    opus, haiku = direct[OPUS], direct[HAIKU]
    if not (opus and haiku):
        return [*lines, "Needs F's direct runs of both models on the dev questions."]
    by_id = {e["id"]: e for e in opus[1]["answers"] if e.get("usage")}
    answered = [e for e in haiku[1]["answers"] if e.get("usage") and e["id"] in by_id]
    pairs = [(by_id[e["id"]], e) for e in answered]
    o = usage([p[0] for p in pairs])
    h = usage([p[1] for p in pairs])
    o_usd, h_usd = o.cost(prices[OPUS]), h.cost(prices[HAIKU])
    if not (h_usd and o.output):
        return [*lines, "F's direct runs of the two models have no answered question in common."]
    output_ratio = h.output / o.output
    o_secs, h_secs = seconds(opus[1]["answers"]), seconds(haiku[1]["answers"])
    lines += [
        f"Measured in F's direct runs (`{opus[0]}` and `{haiku[0]}`), where both models "
        f"answered the same {len(pairs)} dev questions from the same passages: the same input "
        f"tokens ({o.input:,.0f} and {h.input:,.0f} a question), Haiku wrote "
        f"{output_ratio:.2f} times Opus's output tokens ({h.output:,.0f} against "
        f"{o.output:,.0f}), and a question cost US${h_usd:.5f} against US${o_usd:.4f}, "
        f"{o_usd / h_usd:.0f} times less. Haiku took {_secs(h_secs)} seconds against "
        f"{_secs(o_secs)}. How the two answered is compared in `results/answers.md` (dev "
        "split). Below, each finalist's answering is priced for Haiku: its own input tokens, "
        "and its output tokens times that ratio; the reranking is unchanged.",
        "",
    ]
    titles = ["Finalist", "Haiku answering (US$)", "A question (US$)"]
    titles += [f"{v:,} a month (US$)" for v in VOLUMES]
    lines += _header(titles)
    hp = prices[HAIKU]
    for f in finalists:
        answer = Usage(f.answering.calls, f.answering.input, f.answering.output * output_ratio)
        total = answer.cost(hp) + f.rerank_usd
        lines.append(
            _row(
                [
                    f.name,
                    f"{answer.cost(hp):.5f}",
                    f"{total:.5f}",
                    *[_money(total * v) for v in VOLUMES],
                ]
            )  # fmt: skip
        )
    return lines


def _caching_lever(finalists: Sequence[Finalist], prices: Mapping[str, Price]) -> list[str]:
    lines = ["", "## Lever: caching the fixed instructions", ""]
    models = sorted({f.model for f in finalists})
    rates = "; ".join(
        f"{m}: a hit costs US${prices[m].cache_read:.2f} per million tokens instead of "
        f"US${prices[m].input:.2f}, and a miss writes the cache at "
        f"US${prices[m].cache_write_5m:.2f} (5 minutes) or US${prices[m].cache_write_1h:.2f} "
        "(1 hour)"
        for m in models
    )
    lines += [
        "Estimated, not measured. Every answer prompt starts with the same instructions and "
        f"answer schema, about {min(f.fixed for f in finalists):,.0f}–"
        f"{max(f.fixed for f in finalists):,.0f} tokens, which can be cached; the passages and "
        f"the question change every time. Prices, {rates}. Anthropic's documentation doesn't "
        "say whether the instructions the API adds for the answer schema are part of the cached "
        "prefix; if they aren't, a little over half as many tokens are cached and the savings "
        "shrink in proportion. The share of hits assumes questions come at random times at an "
        "even rate; real traffic bunches in working hours, which raises it. The reranker's "
        "prompts are left out: they cost too little for caching to matter.",
        "",
    ]
    titles = ["Finalist", "Fixed tokens"]
    for v in VOLUMES:
        titles += [f"{v:,} a month: lifetime, hits", "Saved a question (US$)"]
    lines += _header(titles, numeric_from=1)
    for f in finalists:
        cells: list[object] = [f.name, f"{f.fixed:,.0f}"]
        for v in VOLUMES:
            c = caching(f.fixed, prices[f.model], v)
            when = f"{c.lifetime}, {c.hit_rate:.0%}" if c.lifetime else "none"
            cells += [when, _usd(c.saved_usd)]
        lines.append(_row(cells))
    return lines


def _judging_section(
    finals: Sequence[tuple[str, Mapping]],
    finalists: Sequence[Finalist],
    judge_logs: Sequence[tuple[str, Mapping]],
    judge: tuple[str, str],
    prices: Mapping[str, Price],
) -> list[str]:
    lines = ["", "## Evaluation only: judging", ""]
    lines += [
        f"The faithfulness judge ({judge[0]}, instructions `{judge[1]}`) checked each answer "
        "that wasn't a refusal. A live service doesn't judge its answers, so this isn't in the "
        "figures above; it is what checking a sample of live answers would cost. Paid is what "
        "the batch run cost; real-time is the same tokens at the real-time prices.",
        "",
    ]
    titles = ["Finalist", "Judge log", "Answers judged", "Real-time a judged answer (US$)",
              "Paid a judged answer, batch (US$)"]  # fmt: skip
    lines += _header(titles, numeric_from=2)
    for (file, _), f in zip(finals, finalists, strict=True):
        found = _judging(file, judge_logs, judge)
        if found is None:
            lines.append(_row([f.name, "–", 0, "–", "–"]))
            continue
        name, log = found
        u = usage(log["judgements"])
        paid = log["paid"]["cost_usd"] / u.calls if u.calls else 0.0
        lines.append(
            _row([f.name, f"`{name}`", u.calls, _usd(u.cost(prices[judge[0]])), _usd(paid)])
        )
    return lines
