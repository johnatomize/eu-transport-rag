"""Score an answer run log against the question set, as `retrieval_eval` scores a ranking.

Nothing here calls a model: a run log (`rag.generate.run_log`) holds every reply, the check's
verdict on it, the chunks the model was shown, the tokens, the cost and the time, and that is
all scoring needs. The rules, recorded in docs/DECISIONS.md ("how an answer run is scored"):

- **What the reader saw.** Each question's reply is one of: an answer that passed the check;
  a refusal that passed it ("the documents don't answer"); "could not answer reliably", when the
  reply failed the check; or no reply at all, when the call failed.
- **Correctness** is scored on answerable questions with key facts: correct when the answer
  passed the check and states every key fact (`rag.keyfacts`). A refusal, an unreliable answer
  or a missing reply is not correct. Questions without key facts get no correctness score here;
  on the test split the owner grades them by hand (#75, #88).
- **Citations.** A citation finds gold node G when it cites G or a node inside G. The check has
  already made sure the quote is word for word in the cited node's own text, so the quoted words
  lie inside G. A citation of a node above G, such as the paragraph whose point is gold, quotes
  that node's own words, the lead-in, which the gold leaves out (the decision of 2026-10-29).
  Precision is the share of an answer's distinct cited nodes that find a gold node, over the
  answerable questions answered. Recall is the share of gold nodes found, over every answerable
  question; a question not answered finds none.
- **Refusals.** Refusal accuracy is the share of unanswerable questions refused; false refusals
  the share of answerable questions refused. "Could not answer reliably" and missing replies are
  counted on their own, as neither.
- **Tracing.** A question whose answer is not correct, refused, unreliable or missing is a
  retrieval miss when none of its gold nodes was among the chunks the model was shown (a chunk
  holds G when it holds G or a node inside it, as in retrieval evaluation), and a generation
  error when at least one was. An unanswerable question answered is a generation error. A
  question whose call got no reply is not traced: the log doesn't say what search found.
- **No reply** happens when a batch request fails or expires (#79); a direct run stops at its
  first failed call, and a run with questions missing is refused as unfinished.
- **Cost and speed:** the cost, the input and output tokens and the seconds of each reply, as
  means over the questions that got one.
- **Intervals** are 95%: Wilson for shares of questions, a seeded percentile bootstrap over
  questions for means. Each measure is given overall, by phrasing and by type.

The test split is scored only when asked for explicitly, as in `retrieval_eval`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag.chunk import STRATEGIES
from rag.keyfacts import missing_facts
from rag.parse import Tree
from rag.questions import PHRASINGS, SPLITS, TYPES, Question
from rag.retrieval_eval import RESAMPLES, SEED, Metric, UnseenSplitError, _bootstrap, wilson

ANSWERED, REFUSED, UNRELIABLE, NO_REPLY = "answered", "refused", "unreliable", "no_reply"
OUTCOMES = (ANSWERED, REFUSED, UNRELIABLE, NO_REPLY)
RETRIEVAL, GENERATION = "retrieval", "generation"

# The measures, by name. Shares of questions get Wilson intervals; the rest are means.
SHARES = ("correct", "refusal_accuracy", "false_refusals", "unreliable", "no_reply")
# Cost and speed are taken over the same questions, those with a reply, so they share resamples.
USAGE = ("cost_usd", "input_tokens", "output_tokens", "seconds")
MEANS = ("citation_precision", "citation_recall", *USAGE)


@dataclass(frozen=True)
class AnswerScore:
    question_id: str
    answerable: bool  # by the question set: it has gold nodes
    outcome: str  # one of OUTCOMES
    # None when not scored: an unanswerable question, or one without key facts.
    correct: bool | None
    missing_facts: tuple[str, ...]  # the key facts the answer doesn't state, or all if no answer
    cited: tuple[str, ...]  # distinct cited node IDs, in the order first cited
    citation_precision: float | None  # None unless an answerable question was answered
    citation_recall: float | None  # None for an unanswerable question
    gold_retrieved: tuple[str, ...]  # gold nodes held by a chunk the model was shown
    trace: str | None  # RETRIEVAL or GENERATION for a question that went wrong, else None
    cost_usd: float | None  # None when the call got no reply
    input_tokens: int | None
    output_tokens: int | None
    seconds: float | None


@dataclass(frozen=True)
class AnswerGroup:
    group: str  # "all", a phrasing such as "paraphrased", or a type such as "single_fact"
    n: int  # questions in the group
    counts: Mapping[str, int]  # each outcome, and "retrieval" and "generation" for the traces
    # Each measure in SHARES and MEANS, with `n` the questions it was taken over; a measure
    # with no questions to take it over is left out.
    metrics: Mapping[str, Metric]
    over: Mapping[str, int]
    question_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerEvaluation:
    split: str
    scores: tuple[AnswerScore, ...]  # one per question of the split, in question order
    groups: tuple[AnswerGroup, ...]


def chunk_nodes(tree: Tree, strategy: str) -> dict[str, tuple[str, ...]]:
    """The node IDs each chunk of `strategy` holds, by chunk ID, cut from `tree` now."""
    if strategy not in STRATEGIES:
        raise ValueError(f"no chunking strategy {strategy!r}")
    return {chunk.id: chunk.node_ids for chunk in STRATEGIES[strategy](tree)}


# --- One question --------------------------------------------------------------------------


def _outcome(entry: Mapping | None) -> str:
    if entry is None:
        return NO_REPLY
    verdict = entry["verdict"]
    if not verdict["ok"]:
        return UNRELIABLE
    return ANSWERED if verdict["answerable"] else REFUSED


def score(
    question: Question,
    entry: Mapping | None,
    tree: Tree,
    nodes_of: Mapping[str, Sequence[str]],
) -> AnswerScore:
    """Score one question's entry in a run log; `entry` is None when its call got no reply.

    `nodes_of` maps a chunk ID to the node IDs it holds (`chunk_nodes`).
    """
    gold = question.gold_node_ids
    outcome = _outcome(entry)
    subtrees = {g: {node.id for node in tree.subtree(g)} for g in gold}

    retrieved: list[str] = []
    if entry is not None:
        unknown = [c for c in entry["retrieved"] if c not in nodes_of]
        if unknown:
            raise ValueError(
                f"{question.id}: the run log shows chunks {', '.join(unknown)}, which chunking "
                "the corpus now doesn't give; was it made from another parse?"
            )
        retrieved = [n for c in entry["retrieved"] for n in nodes_of[c]]
        unknown = sorted({n for n in retrieved if n not in tree})
        if unknown:
            raise ValueError(
                f"{question.id}: the chunks shown hold {', '.join(unknown)}, which the parsed "
                "corpus doesn't have; was the chunk map made from another parse?"
            )
    gold_retrieved = tuple(g for g in gold if not subtrees[g].isdisjoint(retrieved))

    cited: tuple[str, ...] = ()
    if outcome == ANSWERED:
        assert entry is not None
        cited = tuple(dict.fromkeys(c["node_id"] for c in entry["verdict"]["citations"]))
        unknown = [n for n in cited if n not in tree]
        if unknown:
            raise ValueError(f"{question.id}: cites {', '.join(unknown)}, not in the corpus")

    precision = recall = None
    if gold:
        recall = sum(not subtrees[g].isdisjoint(cited) for g in gold) / len(gold)
        if outcome == ANSWERED:  # the check fails an answer that cites nothing
            precision = sum(any(n in s for s in subtrees.values()) for n in cited) / len(cited)

    correct = None
    missing: tuple[str, ...] = ()
    if gold and question.key_facts:
        if outcome == ANSWERED:
            assert entry is not None
            missing = tuple(missing_facts(entry["verdict"]["answer"], question.key_facts))
        else:
            missing = question.key_facts
        correct = outcome == ANSWERED and not missing

    if outcome == NO_REPLY:
        trace = None
    elif gold:
        wrong = outcome != ANSWERED or correct is False
        trace = (GENERATION if gold_retrieved else RETRIEVAL) if wrong else None
    else:
        trace = GENERATION if outcome == ANSWERED else None

    usage = entry["usage"] if entry is not None else None
    return AnswerScore(
        question_id=question.id,
        answerable=bool(gold),
        outcome=outcome,
        correct=correct,
        missing_facts=missing,
        cited=cited,
        citation_precision=precision,
        citation_recall=recall,
        gold_retrieved=gold_retrieved,
        trace=trace,
        cost_usd=entry["cost_usd"] if entry is not None else None,
        input_tokens=usage["input_tokens"] if usage else None,
        output_tokens=usage["output_tokens"] if usage else None,
        seconds=entry["seconds"] if entry is not None else None,
    )


# --- A run ---------------------------------------------------------------------------------


def _shares(scores: Sequence[AnswerScore]) -> dict[str, list[bool]]:
    """For each share in SHARES, whether each question it is taken over counts."""
    answerable = [s for s in scores if s.answerable]
    unanswerable = [s for s in scores if not s.answerable]
    return {
        "correct": [s.correct for s in scores if s.correct is not None],
        "refusal_accuracy": [s.outcome == REFUSED for s in unanswerable],
        "false_refusals": [s.outcome == REFUSED for s in answerable],
        "unreliable": [s.outcome == UNRELIABLE for s in scores],
        "no_reply": [s.outcome == NO_REPLY for s in scores],
    }


def _group(name: str, scores: Sequence[AnswerScore], seed: int | str) -> AnswerGroup:
    metrics: dict[str, Metric] = {}
    over: dict[str, int] = {}
    for metric, hits in _shares(scores).items():
        if hits:
            n, k = len(hits), sum(hits)
            metrics[metric], over[metric] = Metric(k / n, *wilson(k, n)), n
    columns = [(metric,) for metric in MEANS if metric not in USAGE] + [USAGE]
    for names in columns:
        # Every measure in `names` is None for the same questions, so they share resamples.
        taken = [s for s in scores if getattr(s, names[0]) is not None]
        if not taken:
            continue
        values = {metric: [getattr(s, metric) for s in taken] for metric in names}
        intervals = _bootstrap(values, RESAMPLES, f"{seed}:{name}:{names[0]}")
        for metric in names:
            mean = sum(values[metric]) / len(taken)
            metrics[metric], over[metric] = Metric(mean, *intervals[metric]), len(taken)
    counts = {outcome: sum(s.outcome == outcome for s in scores) for outcome in OUTCOMES}
    for trace in (RETRIEVAL, GENERATION):
        counts[trace] = sum(s.trace == trace for s in scores)
    return AnswerGroup(
        name, len(scores), counts, metrics, over, tuple(s.question_id for s in scores)
    )


def evaluate(
    log: Mapping,
    questions: Sequence[Question],
    tree: Tree,
    nodes_of: Mapping[str, Sequence[str]] | None = None,
    allow_test: bool = False,
    seed: int | str = SEED,
) -> AnswerEvaluation:
    """Score every question of the run's split, overall, by phrasing and by type.

    `nodes_of` maps chunk IDs to node IDs; by default the run's chunking strategy is cut from
    `tree`. A run must hold a reply, or a failed call, for every question of its split and for
    nothing else. The test split is used once, for checkpoint G's final table, so scoring a
    test run needs `allow_test`.
    """
    split = log["config"]["split"]
    if split not in SPLITS:
        raise ValueError(f"a run on the {split!r} questions is not scored; the splits are {SPLITS}")
    if split == "test" and not allow_test:
        raise UnseenSplitError(
            "the test split is scored only once, for checkpoint G's final table (design §C); "
            "pass allow_test=True there and nowhere else"
        )
    if nodes_of is None:
        nodes_of = chunk_nodes(tree, log["config"]["retrieval"]["strategy"])

    chosen = [q for q in questions if q.split == split]
    entries: dict[str, Mapping | None] = {}
    for entry in log["answers"]:
        if entry["id"] in entries:
            raise ValueError(f"the run log answers {entry['id']} twice")
        entries[entry["id"]] = entry
    for failed in log["failed"]:
        if failed["id"] in entries:
            raise ValueError(f"the run log has {failed['id']} both answered and failed")
        entries[failed["id"]] = None
    wanted = {q.id for q in chosen}
    missing = sorted(wanted - entries.keys())
    extra = sorted(entries.keys() - wanted)
    if missing:
        raise ValueError(
            f"the run log has nothing for {len(missing)} {split} questions ({', '.join(missing)}); "
            "an unfinished run can't be compared with the others"
        )
    if extra:
        raise ValueError(f"the run log has questions not in the {split} split: {', '.join(extra)}")

    scores = [score(q, entries[q.id], tree, nodes_of) for q in chosen]
    by_id = {q.id: q for q in chosen}
    groups = [("all", scores)]
    for attribute, values in (("phrasing", PHRASINGS), ("type", TYPES)):
        for value in values:
            members = [s for s in scores if getattr(by_id[s.question_id], attribute) == value]
            if members:
                groups.append((value, members))
    return AnswerEvaluation(
        split=split,
        scores=tuple(scores),
        groups=tuple(_group(name, members, seed) for name, members in groups),
    )
