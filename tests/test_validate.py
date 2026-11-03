import json

import pytest

from rag.chunk import Chunk
from rag.prompt import Answer, Citation
from rag.validate import check_answer, check_reply, find_quote, normalise, quotable

# A hand-built chunk shaped like 561/2006 Art. 6, with paragraph 3's text split over two lines.
ARTICLE = Chunk(
    "article:561/2006:art6",
    "article",
    "561/2006",
    "Regulation (EC) No 561/2006 › Article 6",
    "1. The daily driving time shall not exceed 9 hours.\n"
    "However, the daily driving time may be extended to at most 10 hours not more than twice "
    "during the week.\n"
    "2. The weekly driving time shall not exceed 56 hours.\n"
    "3. The total accumulated driving time during any two consecutive weeks\n"
    "shall not exceed 90 hours.",
    ("561/2006:art6(1)", "561/2006:art6(2)", "561/2006:art6(3)"),
    line_owners=(
        "561/2006:art6(1)",
        "561/2006:art6(1)",
        "561/2006:art6(2)",
        "561/2006:art6(3)",
        "561/2006:art6(3)",
    ),
)
# A definition with the act's own typography: an em dash, curly quotes, a decimal comma.
DEFINITION = Chunk(
    "paragraph:561/2006:art4(a)",
    "paragraph",
    "561/2006",
    "Regulation (EC) No 561/2006 › Article 4",
    "— ‘carriage by road’ means any journey made on roads open to the public, of a vehicle of "
    "more than 3,5 tonnes;",
    ("561/2006:art4(a)",),
    line_owners=("561/2006:art4(a)",),
)
CHUNKS = [ARTICLE, DEFINITION]


def reply(answerable: bool, *citations: tuple[str, str], answer: str = "An answer.") -> str:
    return json.dumps(
        {
            "answerable": answerable,
            "answer": answer,
            "citations": [{"node_id": n, "quote": q} for n, q in citations],
        }
    )


# --- Each rule, passing and failing ---------------------------------------------------------


def test_a_quote_from_the_cited_node_passes_and_carries_the_corpus_text():
    verdict = check_reply(
        reply(True, ("561/2006:art6(2)", "The weekly driving time shall not exceed 56 hours.")),
        CHUNKS,
    )
    assert verdict.ok, verdict.problems
    assert [(c.node_id, c.quote) for c in verdict.citations] == [
        ("561/2006:art6(2)", "The weekly driving time shall not exceed 56 hours.")
    ]


def test_a_refusal_without_citations_passes():
    verdict = check_reply(reply(False, answer="The passages don't say."), CHUNKS)
    assert verdict.ok
    assert verdict.answer is not None and not verdict.answer.answerable
    assert verdict.citations == ()


def test_a_refusal_with_citations_fails():
    verdict = check_reply(reply(False, ("561/2006:art6(2)", "shall not exceed 56 hours")), CHUNKS)
    assert not verdict.ok
    assert verdict.problems == ("says the passages don't answer it, but gives 1 citations",)


def test_an_answer_without_citations_fails():
    verdict = check_reply(reply(True), CHUNKS)
    assert not verdict.ok
    assert verdict.problems == ("gives an answer without citations",)


def test_a_node_in_the_corpus_that_was_not_retrieved_fails(tree):
    """The quote is real, word for word, but its node was not among the retrieved chunks."""
    assert "shall take daily and weekly rest periods" in tree.nodes["561/2006:art8(1)"].text
    verdict = check_reply(
        reply(True, ("561/2006:art8(1)", "shall take daily and weekly rest periods")), CHUNKS
    )
    assert not verdict.ok
    assert verdict.problems == (
        "citation 1 (561/2006:art8(1)): not a node of the retrieved passages",
    )


def test_the_right_quote_under_another_retrieved_node_fails():
    verdict = check_reply(reply(True, ("561/2006:art6(1)", "shall not exceed 90 hours")), CHUNKS)
    assert not verdict.ok
    assert verdict.problems[0].startswith("citation 1 (561/2006:art6(1)): the quote is not in")


def test_the_article_when_only_its_paragraphs_were_marked_fails():
    """Citations name the most specific node: the prompt marked art6(3), not art6."""
    verdict = check_reply(reply(True, ("561/2006:art6", "shall not exceed 90 hours")), CHUNKS)
    assert verdict.problems == ("citation 1 (561/2006:art6): not a node of the retrieved passages",)


@pytest.mark.parametrize(
    "quote",
    [
        "The weekly driving time shall not exceed 57 hours.",  # one digit
        "The weekly driving time shall not exceed 56 hour.",  # one letter missing
        "the weekly driving time shall not exceed 56 hours.",  # case
        "The weekly driving time shall not exceed 56 hours;",  # punctuation
        "The weekly driving time ... 56 hours.",  # shortened
    ],
)
def test_a_quote_that_differs_by_one_character_fails(quote):
    verdict = check_reply(reply(True, ("561/2006:art6(2)", quote)), CHUNKS)
    assert not verdict.ok
    assert "the quote is not in that node's text" in verdict.problems[0]


def test_every_citation_is_checked_and_every_failure_reported():
    verdict = check_reply(
        reply(
            True,
            ("561/2006:art6(2)", "shall not exceed 56 hours"),
            ("561/2006:art9(1)", "a ferry"),
            ("561/2006:art6(3)", "shall not exceed 99 hours"),
        ),
        CHUNKS,
    )
    assert not verdict.ok
    assert verdict.citations == ()
    assert [p.split(":")[0] for p in verdict.problems] == [
        "citation 2 (561/2006",
        "citation 3 (561/2006",
    ]


# --- What is normalised, and what isn't ---------------------------------------------------


def test_a_quote_across_a_line_break_may_use_a_space():
    verdict = check_reply(
        reply(True, ("561/2006:art6(3)", "two consecutive weeks shall not exceed 90 hours")),
        CHUNKS,
    )
    assert verdict.ok, verdict.problems
    # The reader is shown the corpus text, line break and all.
    assert verdict.citations[0].quote == "two consecutive weeks\nshall not exceed 90 hours"


def test_straight_quote_marks_and_a_hyphen_match_the_acts_typography():
    verdict = check_reply(
        reply(True, ("561/2006:art4(a)", "- 'carriage by road' means any journey")), CHUNKS
    )
    assert verdict.ok, verdict.problems
    assert verdict.citations[0].quote == "— ‘carriage by road’ means any journey"


def test_extra_spaces_inside_and_around_a_quote_match():
    verdict = check_reply(
        reply(True, ("561/2006:art6(2)", "  weekly   driving time\tshall ")), CHUNKS
    )
    assert verdict.ok, verdict.problems
    assert verdict.citations[0].quote == "weekly driving time shall"


def test_a_decimal_point_for_the_acts_decimal_comma_fails():
    verdict = check_reply(reply(True, ("561/2006:art4(a)", "more than 3.5 tonnes")), CHUNKS)
    assert not verdict.ok


def test_normalise_maps_every_character_back_to_the_text():
    text = "a  ‘b’\n—\tc"
    flat, starts = normalise(text)
    assert flat == "a 'b' - c"
    assert len(starts) == len(flat) + 1 and starts[-1] == len(text)
    assert [text[s] for s in starts[:-1]] == ["a", " ", "‘", "b", "’", "\n", "—", "\t", "c"]


def test_a_quote_of_nothing_but_space_matches_nothing():
    assert find_quote("  \n ", ["any text"]) is None


# --- What a node offers to quote ----------------------------------------------------------


def test_a_window_offers_only_the_part_of_a_node_it_holds():
    """A window cut inside paragraph 3 holds its second line only: the first can't be quoted."""
    window = Chunk(
        "window-200-50:561/2006:0001",
        "window-200-50",
        "561/2006",
        "Regulation (EC) No 561/2006 › Article 6",
        "shall not exceed 90 hours.",
        ("561/2006:art6(3)",),
        line_owners=("561/2006:art6(3)",),
    )
    assert check_reply(reply(True, ("561/2006:art6(3)", "90 hours")), [window]).ok
    verdict = check_reply(
        reply(True, ("561/2006:art6(3)", "during any two consecutive weeks")), [window]
    )
    assert not verdict.ok


def test_a_quote_may_not_run_across_a_point_into_its_paragraphs_later_text():
    """A paragraph with text before and after its points offers two pieces, not one."""
    split = Chunk(
        "paragraph:561/2006:art2(3)",
        "paragraph",
        "561/2006",
        "Regulation (EC) No 561/2006 › Article 2",
        "3. The AETR shall apply instead.\n(a) vehicles registered here;\nThe provisions align.",
        ("561/2006:art2(3)", "561/2006:art2(3)(a)"),
        line_owners=("561/2006:art2(3)", "561/2006:art2(3)(a)", "561/2006:art2(3)"),
    )
    assert quotable([split])["561/2006:art2(3)"] == [
        "3. The AETR shall apply instead.",
        "The provisions align.",
    ]
    assert check_reply(reply(True, ("561/2006:art2(3)", "The provisions align.")), [split]).ok
    across = check_reply(
        reply(True, ("561/2006:art2(3)", "apply instead. The provisions")), [split]
    )
    assert not across.ok


def test_a_heading_line_is_not_quotable():
    chunk = Chunk(
        "window-200-50:561/2006:0002",
        "window-200-50",
        "561/2006",
        "Regulation (EC) No 561/2006",
        "Article 7\nAfter a driving period of four and a half hours a driver shall take a break.",
        ("561/2006:art7",),
        line_owners=(None, "561/2006:art7"),
    )
    assert quotable([chunk]) == {
        "561/2006:art7": [
            "After a driving period of four and a half hours a driver shall take a break."
        ]
    }


def test_a_chunk_without_line_owners_is_refused():
    old = Chunk("paragraph:x:art1", "paragraph", "x", "X", "text", ("x:art1",))
    with pytest.raises(ValueError, match="no line_owners"):
        quotable([old])


# --- Replies that don't fit the format ------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "The daily driving time is 9 hours.",
        '{"answerable": true, "answer": "9 hours"}',
        '{"answerable": "true", "answer": "9 hours", "citations": []}',
        '{"answerable": true, "answer": "9 hours", "citations": [], "confidence": 1}',
        '{"answerable": true, "answer": "9 hours", "citations": [{"node_id": "x"}]}',
        "[]",
    ],
)
def test_a_reply_that_does_not_fit_the_format_fails_instead_of_crashing(raw):
    verdict = check_reply(raw, CHUNKS)
    assert not verdict.ok
    assert verdict.answer is None
    assert len(verdict.problems) == 1
    assert verdict.problems[0].startswith("not in the answer format: ")


def test_check_answer_takes_a_parsed_answer():
    answer = Answer(True, "9 hours", (Citation("561/2006:art6(1)", "shall not exceed 9 hours"),))
    assert check_answer(answer, CHUNKS).ok


# --- On the real corpus -------------------------------------------------------------------


def test_a_quote_from_the_real_article_6_passes_against_each_chunking(chunks):
    """Paragraph 3 of the real Art. 6, cited from the chunk of each strategy that holds it."""
    quote = "shall not exceed 90 hours"
    for strategy, cut in chunks.items():
        holding = [c for c in cut if "561/2006:art6(3)" in c.line_owners]
        assert holding, strategy
        verdict = check_reply(reply(True, ("561/2006:art6(3)", quote)), holding)
        assert verdict.ok, (strategy, verdict.problems)
        assert verdict.citations[0].quote == quote


def test_every_node_of_the_real_corpus_can_quote_its_own_text(chunks):
    """Each node's text, as its chunks hold it, is found again: nothing in the corpus trips
    the normalisation, and the corpus text read back is the text itself."""
    for strategy, cut in chunks.items():
        for node_id, pieces in quotable(cut).items():
            for piece in pieces:
                assert find_quote(piece, [piece]) == piece.strip(), (strategy, node_id)
