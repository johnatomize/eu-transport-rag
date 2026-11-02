"""Score retrieval rankings against the gold node IDs of the question set.

For each question, a ranking is the list of retrieved chunks, best first, each given by the node
IDs it holds (`Chunk.node_ids`). The rules, decided in #32 and recorded in docs/DECISIONS.md
("how retrieval evaluation scores a ranking"):

- A chunk finds a gold node when its node IDs hold that node or a descendant of it.
- recall@k: 1 when every gold node is found in the top k, else 0. Fractional recall@k: the
  share of gold nodes found in the top k.
- MRR@10: 1 / the rank of the first chunk that finds any gold node, or 0 if none is in the top 10.
- nDCG@10: binary relevance, each gold node credited once, at the first chunk that finds it. The
  ideal ranking holds one gold node at each of the first ranks. A chunk that finds several gold
  nodes at once can beat that, so nDCG is capped at 1.
- Unanswerable questions have no gold nodes; they are left out and counted.
- Intervals are 95%: Wilson for recall@k, a seeded percentile bootstrap over questions for the
  means (fractional recall, MRR and nDCG).

Nothing here depends on how the ranking was made, so BM25, vector and hybrid search share it.
The mean chunk length that goes beside the scores belongs to the run, so the run log carries it
(#36). The test split is scored only when asked for explicitly, which is for checkpoint G alone.
"""

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag.parse import Tree
from rag.questions import PHRASINGS, SPLITS, TYPES, Question

KS = (1, 3, 5, 10)
CUTOFF = 10  # for MRR and nDCG
Z95 = 1.959963984540054  # the 97.5th percentile of the standard normal distribution
RESAMPLES = 10_000
SEED = 35

Ranking = Sequence[Sequence[str]]  # node IDs of each retrieved chunk, best first


class UnseenSplitError(Exception):
    """The test split was asked for without the flag that only checkpoint G sets."""


# --- Intervals -----------------------------------------------------------------------------


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval for a proportion (Wilson 1927; Newcombe 1998, method 3)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap(
    values: Sequence[float], resamples: int = RESAMPLES, seed: int | str = SEED
) -> tuple[float, float]:
    """A 95% percentile bootstrap interval for the mean: resample with replacement, seeded."""
    return _bootstrap({"": values}, resamples, seed)[""]


def _bootstrap(
    columns: Mapping[str, Sequence[float]], resamples: int, seed: int | str
) -> dict[str, tuple[float, float]]:
    """Bootstrap intervals for several means over the same questions, with shared resamples.

    Every column is a value per question, in the same order, so one set of resampled questions
    serves them all.
    """
    n = len(next(iter(columns.values())))
    if n == 0:
        return dict.fromkeys(columns, (0.0, 1.0))
    rng = random.Random(seed)
    draws = [[rng.randrange(n) for _ in range(n)] for _ in range(resamples)]
    low, high = int(0.025 * resamples), math.ceil(0.975 * resamples) - 1
    intervals = {}
    for name, values in columns.items():
        means = sorted(sum(values[i] for i in draw) / n for draw in draws)
        intervals[name] = (means[low], means[high])
    return intervals


# --- One question --------------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionScore:
    question_id: str
    # The rank (from 1) of the first chunk that finds each gold node, or None if none does.
    found_at: Mapping[str, int | None]
    recall: Mapping[int, int]  # recall@k, 0 or 1, for each k in KS
    fractional_recall: Mapping[int, float]
    mrr: float
    ndcg: float


def score(question: Question, ranking: Ranking, tree: Tree) -> QuestionScore:
    """Score one ranking for an answerable question, by the rules in the module docstring.

    Only the first CUTOFF chunks count; a gold node found later has `found_at` None.
    """
    gold = question.gold_node_ids
    if not gold:
        raise ValueError(f"{question.id}: no gold nodes to score against")
    # A chunk finds gold node G when it holds G or a descendant: a node of G's subtree.
    subtrees = {g: {node.id for node in tree.subtree(g)} for g in gold}
    found_at: dict[str, int | None] = dict.fromkeys(gold)
    for rank, node_ids in enumerate(ranking[:CUTOFF], 1):
        unknown = [n for n in node_ids if n not in tree]
        if unknown:
            raise ValueError(
                f"{question.id}: the chunk at rank {rank} holds {', '.join(unknown)}, which the "
                "parsed corpus doesn't have; was the index built from another parse?"
            )
        for g in gold:
            if found_at[g] is None and not subtrees[g].isdisjoint(node_ids):
                found_at[g] = rank
    ranks = [r for r in found_at.values() if r is not None]
    recall = {k: int(len(ranks) == len(gold) and max(ranks) <= k) for k in KS}
    fractional = {k: sum(r <= k for r in ranks) / len(gold) for k in KS}
    mrr = 1 / min(ranks) if ranks else 0.0
    dcg = sum(1 / math.log2(r + 1) for r in ranks)
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(len(gold), CUTOFF) + 1))
    return QuestionScore(question.id, found_at, recall, fractional, mrr, min(1.0, dcg / ideal))


# --- A question set ------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    value: float
    low: float
    high: float


@dataclass(frozen=True)
class GroupResult:
    group: str  # "all", a phrasing such as "paraphrased", or a type such as "single_fact"
    n: int
    metrics: Mapping[str, Metric]  # "recall@5", "fractional_recall@5", "mrr@10", "ndcg@10"
    question_ids: tuple[str, ...] = ()  # the questions scored in this group, in question order


@dataclass(frozen=True)
class Evaluation:
    splits: tuple[str, ...]
    scores: tuple[QuestionScore, ...]  # one per scored question, in question order
    groups: tuple[GroupResult, ...]
    unanswerable: int  # questions in the splits left out because they have no gold nodes


def _group(name: str, scores: Sequence[QuestionScore], seed: int | str) -> GroupResult:
    n = len(scores)
    metrics = {}
    for k in KS:
        hits = sum(s.recall[k] for s in scores)
        metrics[f"recall@{k}"] = Metric(hits / n, *wilson(hits, n))
    means = {f"fractional_recall@{k}": [s.fractional_recall[k] for s in scores] for k in KS}
    means[f"mrr@{CUTOFF}"] = [s.mrr for s in scores]
    means[f"ndcg@{CUTOFF}"] = [s.ndcg for s in scores]
    intervals = _bootstrap(means, RESAMPLES, f"{seed}:{name}")
    for metric, values in means.items():
        metrics[metric] = Metric(sum(values) / n, *intervals[metric])
    return GroupResult(name, n, metrics, tuple(s.question_id for s in scores))


def evaluate(
    questions: Sequence[Question],
    rankings: Mapping[str, Ranking],
    tree: Tree,
    splits: Sequence[str] = ("dev",),
    allow_test: bool = False,
    seed: int | str = SEED,
) -> Evaluation:
    """Score every answerable question in `splits`, overall, by phrasing and by type.

    `rankings` maps a question ID to its ranking. The test split is used once, for checkpoint
    G's final table (design §C), so asking for it needs `allow_test`.
    """
    splits = (splits,) if isinstance(splits, str) else tuple(splits)
    unknown = [split for split in splits if split not in SPLITS]
    if unknown:
        raise ValueError(f"unknown split {', '.join(unknown)}; the splits are {', '.join(SPLITS)}")
    if "test" in splits and not allow_test:
        raise UnseenSplitError(
            "the test split is scored only once, for checkpoint G's final table (design §C); "
            "pass allow_test=True there and nowhere else"
        )
    chosen = [q for q in questions if q.split in splits]
    answerable = [q for q in chosen if q.gold_node_ids]
    if not answerable:
        raise ValueError(f"no answerable questions in the {' and '.join(splits)} split")
    missing = [q.id for q in answerable if q.id not in rankings]
    if missing:
        raise ValueError(f"no ranking for {len(missing)} questions: {', '.join(missing)}")
    by_id = {q.id: q for q in answerable}
    scores = [score(q, rankings[q.id], tree) for q in answerable]
    groups = [("all", scores)]
    for attribute, values in (("phrasing", PHRASINGS), ("type", TYPES)):
        for value in values:
            members = [s for s in scores if getattr(by_id[s.question_id], attribute) == value]
            if members:
                groups.append((value, members))
    return Evaluation(
        splits=splits,
        scores=tuple(scores),
        groups=tuple(_group(name, members, seed) for name, members in groups),
        unanswerable=len(chosen) - len(answerable),
    )
