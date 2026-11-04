"""The answer results table (`rag.tables.answers`), built from hand-made run logs."""

import copy

import pytest
from test_answer_eval import NODES_OF, QUESTIONS, RUN, TREE, entry, log

from rag.chunk import Chunk
from rag.tables import answers, is_base, run_label

# Chunks matching NODES_OF, with bodies of known length: p:1(1) holds 3 words with its
# one-word prefix, p:1(2) 5, p:2(1) 2.
CHUNKS = {
    "paragraph": [
        Chunk(cid, "paragraph", "a", "Act", " ".join(["w"] * (n - 1)), NODES_OF[cid])
        for cid, n in (("p:1(1)", 3), ("p:1(2)", 5), ("p:2(1)", 2))
    ]
}


def settle(run: dict, model="claude-opus-5-5", k=5, **retrieval) -> dict:
    """`run` with the settings and run details a real log carries."""
    run = copy.deepcopy(run)
    run["config"] = {
        "model": model,
        "effort": "medium",
        "max_output_tokens": 4000,
        "prompt_version": "abc123",
        "retrieval": {"retriever": "hybrid", "strategy": "paragraph", "k": k, **retrieval},
        "split": run["config"]["split"],
    }
    run["run"] = {"commit": "0123456789abcdef", "dirty": False}
    run["paid"] = {"cost_usd": 0.12}
    return run


BASE = settle(RUN)
# Gets q004, q005 and q008 right where the base doesn't, and q007 wrong where it is right.
BETTER = settle(
    log(
        [
            entry("q001", answer="9 hours"),
            entry("q002", answer="10 hours"),
            entry("q003"),
            entry("q004"),
            entry("q005"),
            entry("q006", answer="Ten hours."),
            entry("q007"),
            entry("q008", answerable=False, retrieved=("p:1(1)", "p:1(2)")),
        ]
    ),
    k=3,
)


def table(*logs) -> str:
    return answers(list(logs), QUESTIONS, TREE, CHUNKS)


def rows(text: str, starting: str) -> list[list[str]]:
    """The cells of every table row whose first cell starts with `starting`."""
    found = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if line.startswith("|") and cells[0].startswith(starting):
            found.append(cells)
    return found


def section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n#", start + len(heading))
    return text[start : end if end != -1 else None]


def test_a_run_is_named_by_its_settings():
    assert run_label(BASE["config"]) == "opus-5-5, hybrid paragraph top 5"
    reranked = settle(RUN, model="claude-haiku-5-5", rerank=20)["config"]
    assert run_label(reranked) == "haiku-5-5, hybrid paragraph top 5 rerank 20"
    assert is_base(BASE["config"]) and not is_base(BETTER["config"])
    assert not is_base(reranked)


def test_each_run_gets_a_row_of_quality_cost_and_counts():
    text = table(("b.json", BETTER), ("a.json", BASE))
    quality = section(text, "### Quality")
    # The base setup comes first, whatever its file name.
    first, second = rows(quality, "opus-5-5")
    assert first[0] == "opus-5-5, hybrid paragraph top 5"
    assert second[0] == "opus-5-5, hybrid paragraph top 3"
    # Correct over the five with key facts (q001 only); refusal accuracy 1 of 2.
    assert first[1].startswith("0.20 (") and first[4].startswith("0.50 (")
    assert first[6].startswith("0.12 (")  # q005 of 8 not answered reliably

    cost = rows(section(text, "### Cost and speed"), "opus-5-5")
    assert cost[0][1].startswith(f"{0.12 / 7:.4f} (")  # over the seven replies
    # Words shown: q004 saw p:2(1) (2 words), the six others with a reply p:1(1) (3 words).
    assert cost[0][-1] == f"{(2 + 6 * 3) / 7:,.0f}"
    assert cost[1][-1] == f"{(7 * 3 + 3 + 5) / 8:,.0f}"  # q008 saw p:1(1) and p:1(2)

    counts = rows(section(text, "### What happened"), "opus-5-5")
    assert counts[0][1:] == ["4", "2", "1", "1", "1", "3"]


def test_breakdowns_by_phrasing_and_type_skip_groups_without_the_measure():
    text = table(("a.json", BASE))
    by_type = section(text, "### Correct by question type")
    assert rows(by_type, "single_fact")[0][1] == "5"
    assert not rows(by_type, "unanswerable")
    assert rows(section(text, "### Correct by phrasing"), "verbatim")[0][1:3] == [
        "1",
        "0.00 (0.00–0.79)",
    ]
    assert "### Citation recall by question type" in text


def test_each_run_is_compared_with_the_base_setup():
    text = table(("a.json", BASE), ("b.json", BETTER))
    paired = section(text, "### Paired comparisons")
    assert "against the base setup (opus-5-5, hybrid paragraph top 5)" in paired
    (row,) = rows(paired, "opus-5-5, hybrid paragraph top 3")
    assert row[2:] == ["3", "1", "1", "1", "0.625"]  # q006 got no reply in the base run
    assert "1 comparison on this split" in paired and "Bonferroni" not in paired


def test_without_the_base_setup_every_pair_is_compared_and_the_count_noted():
    haiku = settle(RUN, model="claude-haiku-5-5")
    top10 = settle(RUN, k=10)
    paired = section(table(("b.json", BETTER), ("h.json", haiku), ("t.json", top10)), "### Paired")
    assert "against each other" in paired
    assert len(rows(paired, "opus-5-5")) + len(rows(paired, "haiku-5-5")) == 3
    assert "3 comparisons on this split" in paired and "0.0167 (0.05 / 3, Bonferroni)" in paired


def test_one_run_alone_has_no_comparison():
    assert "### Paired comparisons" not in table(("a.json", BASE))


def test_an_unfinished_run_is_listed_not_scored():
    partial = settle(log(RUN["answers"][:3]), model="claude-haiku-5-5")
    text = table(("a.json", BASE), ("p.json", partial))
    assert not rows(section(text, "### Quality"), "haiku")
    assert "- `p.json`: haiku-5-5, hybrid paragraph top 5, 5 questions missing" in text


def test_two_runs_of_the_same_setup_on_one_split_are_refused():
    with pytest.raises(ValueError, match="a.json and c.json are both opus-5-5"):
        table(("a.json", BASE), ("c.json", BASE))


def test_runs_differing_only_in_other_settings_are_named_by_them():
    newer = copy.deepcopy(BASE)
    newer["config"]["prompt_version"] = "def456"
    text = table(("a.json", BASE), ("b.json", newer))
    names = [r[0] for r in rows(section(text, "### Quality"), "opus-5-5")]
    # The latest base run is the base, and comes first.
    assert names == [
        "opus-5-5, hybrid paragraph top 5, effort medium, max output 4000, prompt def456",
        "opus-5-5, hybrid paragraph top 5, effort medium, max output 4000, prompt abc123",
    ]
    assert "against the base setup (opus-5-5, hybrid paragraph top 5, effort medium" in text


def test_test_runs_get_their_own_section_after_dev():
    test_run = settle(log([entry("q100")], split="test"))
    text = table(("t.json", test_run), ("a.json", BASE))
    assert text.index("## Dev split") < text.index("## Test split") < text.index("## Runs")
    runs = rows(section(text, "## Runs"), "opus-5-5")
    assert [r[1] for r in runs] == ["dev", "test"]
    assert runs[0][4:] == ["0123456", "`a.json`", "0.12"]


def test_a_typed_question_log_is_refused():
    typed = settle(log([], split="typed"))
    with pytest.raises(ValueError, match="'typed' questions"):
        table(("x.json", typed))


def test_the_table_is_reproducible():
    logs = (("a.json", BASE), ("b.json", BETTER))
    assert table(*logs) == table(*reversed(logs))


def test_without_runs_the_table_says_so():
    assert table().endswith("No answer runs yet.\n")


def test_a_log_that_is_not_an_answer_run_log_is_refused():
    broken = copy.deepcopy(BASE)
    del broken["config"]["retrieval"]
    with pytest.raises(ValueError, match="x.json is not an answer run log: it has no 'retrieval'"):
        table(("x.json", broken))


def test_an_unknown_strategy_is_refused():
    other = settle(RUN)
    other["config"]["retrieval"]["strategy"] = "window-300-50"
    with pytest.raises(ValueError, match="no chunking strategy 'window-300-50'"):
        table(("x.json", other))


def test_an_unfinished_run_from_another_parse_is_refused():
    partial = settle(log([entry("q001", retrieved=("p:9(9)",))]))
    with pytest.raises(ValueError, match="p.json shows chunks p:9\\(9\\)"):
        table(("p.json", partial))
