import re
from pathlib import Path

import pytest

from rag.corpus import Document, load_manifest
from rag.parse import (
    Amendment,
    Node,
    ParseError,
    Ref,
    Tree,
    _legal_basis,
    parse_corpus,
    parse_document,
    resolve_refs,
)

ROOT = Path(__file__).parents[1]
SNAPSHOT = ROOT / "tests" / "snapshots" / "node_ids.txt"
ACTS = {"561/2006", "165/2014", "2002/15"}
NOTES = {f"guidance-{i}" for i in range(1, 11)} | {f"clarification-{i}" for i in range(1, 8)}

# Everything the parser reports as not understood. A change here must be reviewed by hand.
UNPARSED = [
    "561/2006:art26(4): image, left as [image] in the text",
    "561/2006:art26(4): image, left as [image] in the text",
    "165/2014:art34(5)(b)(i): image, left as [image] in the text",
    "165/2014:art34(5)(b)(ii): image, left as [image] in the text",
    "165/2014:art34(5)(b)(iii): image, left as [image] in the text",
    "165/2014:art34(5)(b)(iv): image, left as [image] in the text",
    "165/2014:art34(5)(b)(v): image, left as [image] in the text",
    "165/2014:anxI.III.e.2: image, left as [image] in the text",
    "165/2014:anxII.I.3: image, left as [image] in the text",
    "guidance-4:p1: 1 image, not extracted",
    "guidance-7:p2: 15 images, not extracted",
    "guidance-7:p3: 15 images, not extracted",
    "guidance-7:p4: 15 images, not extracted",
    "guidance-7:p5: 29 images, not extracted",
    "guidance-7:p5: drawn as a diagram, table or form (250 lines and boxes)",
    "guidance-7:p6: 31 images, not extracted",
    "guidance-7:p6: drawn as a diagram, table or form (271 lines and boxes)",
    "guidance-9:p3: drawn as a diagram, table or form (126 lines and boxes)",
    "guidance-10:p3: 2 images, not extracted",
    "guidance-10:p4: 3 images, not extracted",
    "guidance-10:p5: drawn as a diagram, table or form (686 lines and boxes)",
]

# References in the guidance notes that stay unresolved. Each was checked by hand: annexes of
# acts outside the corpus, the note's own "Annex C", the AETR, 3821/85, and bare references in
# notes that interpret more than one act. A change here must be reviewed by hand.
NOTE_UNRESOLVED = [
    ("guidance-4:p1", "Annex IB"),
    ("guidance-8:p1", "Article 13"),
    ("guidance-8:p1", "paragraph 3"),
    ("guidance-8:p1", "Article 14 (4)"),
    ("guidance-9:p4", "Annex IC"),
    ("guidance-9:p4", "Annex IB"),
    ("guidance-10:p2", "Annex C"),
    ("guidance-10:p4", "Annex C"),
    ("clarification-4:p1", "Article 4"),
    ("clarification-7:p1", "Article 34(3)"),
    ("clarification-7:p1", "Second paragraph of Article 34(3)"),
]

M3 = Amendment("M3", "32020R1054", "REPLACED")
M4_INSERTED = Amendment("M4", "32024R1258", "INSERTED")


@pytest.fixture(scope="module")
def tree() -> Tree:
    return parse_corpus(load_manifest(ROOT / "corpus" / "manifest.yaml"), ROOT)


def children(tree: Tree, node_id: str) -> list[str]:
    return [c.removeprefix(node_id) for c in tree[node_id].children]


# --- The real corpus ------------------------------------------------------------------------


def test_every_document_parses_with_no_duplicates_and_only_known_gaps(tree):
    assert tree.errors == []
    assert tree.unparsed == UNPARSED
    assert {n.doc_id for n in tree} == ACTS | NOTES


def test_ids_match_the_snapshot(tree):
    # A failure means node IDs changed. Gold answers cite them: check that the change is
    # deliberate, then rewrite the snapshot with scripts/parse_corpus.py --snapshot.
    assert [n.id for n in tree] == SNAPSHOT.read_text(encoding="utf-8").splitlines()


def test_parsing_twice_gives_identical_output(tree):
    again = parse_corpus(load_manifest(ROOT / "corpus" / "manifest.yaml"), ROOT)
    assert [n.to_dict() for n in again] == [n.to_dict() for n in tree]


def test_a_known_article_parses_into_the_expected_structure(tree):
    article = tree["561/2006:art6"]
    assert article.kind == "article"
    assert article.heading_path == (
        "Regulation (EC) No 561/2006",
        "Chapter II — Crews, driving times, breaks and rest periods",
        "Article 6",
    )
    assert children(tree, "561/2006:art6") == ["(1)", "(2)", "(3)", "(4)", "(5)"]
    assert tree["561/2006:art6(3)"].kind == "paragraph"
    assert tree["561/2006:art6(3)"].text == (
        "The total accumulated driving time during any two consecutive weeks shall not exceed "
        "90 hours."
    )
    assert tree["561/2006:art6(1)"].text.splitlines() == [
        "The daily driving time shall not exceed nine hours.",
        "However, the daily driving time may be extended to at most 10 hours not more than "
        "twice during the week.",
    ]


def test_subparagraphs_outside_their_paragraph_element_stay_with_it(tree):
    # In Art. 8(6) the 2nd-4th subparagraphs follow the points, outside the paragraph's div.
    assert children(tree, "561/2006:art8(6)") == ["(a)", "(b)"]
    lines = tree.full_text("561/2006:art8(6)").splitlines()
    assert lines[0] == "In any two consecutive weeks a driver shall take at least:"
    assert lines[1] == "(a) two regular weekly rest periods; or"
    assert lines[3].startswith("A weekly rest period shall start no later than")
    assert lines[5].startswith("For the purpose of this paragraph, a driver shall be considered")
    assert len(lines) == 6
    # Points after a paragraph's element belong to it as well.
    assert children(tree, "561/2006:art9(1)") == ["(a)", "(b)"]


def test_points_under_an_article_and_indents_in_their_text(tree):
    assert children(tree, "561/2006:art4")[:3] == ["(a)", "(b)", "(c)"]
    vehicle = tree["561/2006:art4(b)"]
    assert vehicle.children == []
    assert "— ‘motor vehicle’: any self-propelled vehicle" in vehicle.text
    assert "561/2006:art34(5)(b)(iii)" not in tree
    assert "165/2014:art34(5)(b)(iii)" in tree


def test_amendment_markers_are_metadata_not_text(tree):
    assert tree["561/2006:art8(2a)"].amendments == [M4_INSERTED]
    assert M3 in tree["561/2006:art6(5)"].amendments
    assert M3 not in tree["561/2006:art6(1)"].amendments
    # An article lists the amendments of everything in it.
    assert M3 in tree["561/2006:art6"].amendments
    # An inline ►M4 … ◄ span marks the node it is in.
    assert Amendment("M4", "32024R1258", "REPLACED") in tree["561/2006:art8(6a)"].amendments
    for node in tree:
        assert not re.search(r"[▼►◄]", node.text), node.id


def test_footnote_calls_are_removed(tree):
    text = tree["561/2006:art6(5)"].text
    assert "of the European Parliament and of the Council. This record shall" in text
    assert "relationship(6)" not in tree.full_text("2002/15:art9")


def test_the_html_directive_is_rebuilt_from_its_labels(tree):
    assert [n.label for n in tree if n.doc_id == "2002/15" and n.kind == "article"] == [
        str(i) for i in range(1, 17)
    ]
    assert tree["2002/15:art3"].heading == "Article 3 — Definitions"
    assert children(tree, "2002/15:art3(a)") == ["(1)", "(2)"]
    assert children(tree, "2002/15:art3(a)(1)") == ["(i)", "(ii)", "(iii)", "(iv)", "(v)"]
    # (i) after (h) is the next letter, not a roman numeral.
    assert tree["2002/15:art3(i)"].parent == "2002/15:art3"
    assert tree["2002/15:art3(i)"].text.startswith('"night work" shall mean')
    # The second indent of 3(a)(1) comes after its roman list and belongs to 3(a)(1).
    assert tree["2002/15:art3(a)(1)"].text.splitlines()[-1].startswith("— the times during")
    assert (
        tree["2002/15:art3(a)"]
        .text.splitlines()[-1]
        .startswith("The break times referred to in Article 5")
    )
    # Recitals are left out (#12): the act's own text is its title.
    assert tree["2002/15"].text.startswith("Directive 2002/15/EC of the European Parliament")
    assert "Whereas" not in tree.full_text("2002/15")


def test_annex_ids_follow_headings_and_decimal_labels(tree):
    assert tree["165/2014:anxI.III.a.3"].heading == "Materials"
    assert tree["165/2014:anxI.III.a.3.1"].parent == "165/2014:anxI.III.a.3"
    assert tree["165/2014:anxI.III.a.1.1"].parent == "165/2014:anxI.III.a.1"
    assert tree["165/2014:anxI.I(a)"].text.startswith("‘recording equipment’")
    assert tree["165/2014:anxI.V.4(f)"].parent == "165/2014:anxI.V.4"  # a layout table
    assert tree["165/2014:anxI.VI.3(b)"].parent == "165/2014:anxI.VI.3"
    lines = tree.full_text("165/2014:anxI.III.a").splitlines()
    assert lines[0] == "1. Recording equipment shall include the following:"
    assert "3. Materials" in lines


def test_amending_articles_are_flagged(tree):
    assert {n.id for n in tree if n.amending} == {
        "561/2006:art26",
        "561/2006:art27",
        "165/2014:art45",
    }
    # Quoted amending text stays text: no nodes for the points it quotes.
    assert children(tree, "165/2014:art45") == ["(1)", "(2)"]
    assert children(tree, "165/2014:art45(2)") == []


def test_cross_references_resolve(tree):
    refs = tree["561/2006:art6(5)"].refs
    assert Ref("point (e) of Article 4", "561/2006", "561/2006:art4(e)") in refs
    assert (
        Ref("point (b) of Article 3 of Directive 2002/15/EC", "2002/15", "2002/15:art3(b)") in refs
    )
    assert (
        Ref(
            "point (b)(iii) of Article 34(5) of Regulation (EU) No 165/2014",
            "165/2014",
            "165/2014:art34(5)(b)(iii)",
        )
        in refs
    )
    # An amending article's references point into the act it amends.
    assert "561/2006:art3" in {r.target for r in tree["165/2014:art45(1)"].refs}
    # Inside an annex, points are cited relative to where the citing text is.
    assert "165/2014:anxI.V.4(f)" in {r.target for r in tree["165/2014:anxI.VI.1"].refs}


def test_every_reference_target_exists(tree):
    for node in tree:
        for ref in node.refs:
            assert ref.target is None or ref.target in tree, (node.id, ref)


# --- The guidance notes -------------------------------------------------------------------


def test_a_note_has_one_node_per_page(tree):
    note = tree["guidance-7"]
    assert note.kind == "note"
    assert note.heading == 'Guidance note 7: The meaning of "each period of 24 hours"'
    assert note.children == [f"guidance-7:p{i}" for i in range(1, 7)]
    page = tree["guidance-7:p3"]
    assert (page.kind, page.label, page.parent) == ("page", "p3", "guidance-7")
    assert page.heading_path == (note.heading, "Page 3")
    assert sum(1 for n in tree if n.kind == "page") == 30


def test_a_known_passage_is_on_its_page(tree):
    assert "This article does not authorize a driver to derogate from the Regulation" in (
        tree["guidance-1:p1"].text
    )
    passage = "The time of the border crossing should be written in 24-hour format"
    assert passage in tree["guidance-10:p2"].text
    assert passage not in tree["guidance-10:p1"].text


def test_footnote_markers_are_dropped_and_footnote_text_kept(tree):
    text = tree["guidance-10:p1"].text
    assert "Legal basis: Article 34(6), paragraph (f) of Regulation (EU) No 165/2014." in text
    assert "165/20141" not in text
    assert "as amended by Regulation (EU) No 2020/1054 sets out" in text
    assert "within eight working days after the receipt" in tree["guidance-8:p1"].text


def test_running_headers_and_page_numbers_are_removed_after_page_one(tree):
    header = "SOCIAL LEGISLATION IN ROAD TRANSPORT"
    assert tree["guidance-7:p1"].text.startswith(header)
    assert all(header not in tree[f"guidance-7:p{i}"].text for i in range(2, 7))
    assert tree["guidance-7:p2"].text.startswith("Example 1:")
    assert not tree["guidance-10:p2"].text.endswith("2")


def test_text_inside_figures_is_extracted(tree):
    # The labels of guidance-7's last diagram are drawn inside a figure.
    assert "driving time (multi-manning)" in tree["guidance-7:p6"].text


def test_words_split_over_lines_are_joined(tree):
    assert "The form may not be pre-signed nor may it be altered" in tree["guidance-5:p1"].text


def test_a_notes_legal_basis_is_read_whatever_its_form(tree):
    basis = {n.id: n.legal_basis for n in tree if n.kind == "note"}
    assert basis["guidance-7"] == "Article 8 (2) and (5) of Regulation (EC) No 561/2006"
    # The act is named only in the header.
    assert basis["clarification-2"] == "Article 1, 2, 4(a), 4(c) of Regulation (EC) No 561/2006"
    assert basis["clarification-7"].startswith(
        "Article 34(1) and (3) of Regulation (EU) No 165/2014, Article 15(7) of Regulation (EEC)"
    )
    assert basis["guidance-4"].startswith("Article 1 of Regulation (EEC) No 3821/85")
    interprets = {n.id: n.interprets for n in tree if n.kind == "note"}
    assert interprets["guidance-7"] == "561/2006"
    assert interprets["guidance-10"] == "165/2014"
    assert interprets["guidance-5"] is None  # Directive 2006/22/EC, outside the corpus
    assert interprets["clarification-7"] is None  # two acts in the corpus


def test_note_references_resolve_into_the_acts(tree):
    def targets(node_id: str) -> set[str | None]:
        return {r.target for r in tree[node_id].refs}

    assert {"561/2006:art8(2)", "561/2006:art8(5)"} <= targets("guidance-7:p1")
    # "Articles 6 to 9" without an act points into the act the note interprets.
    assert {"561/2006:art6", "561/2006:art8a", "561/2006:art9"} <= targets("guidance-1:p1")
    # Spaces before brackets, as the notes write them: "Article 13 (1) (d)".
    assert "561/2006:art13(1)(d)" in targets("clarification-3:p1")
    # "of the repealed Regulation 3820/85" names an act outside the corpus.
    assert Ref("Article 13(f) of the repealed Regulation 3820/85", "Regulation 3820/85", None) in (
        tree["clarification-3:p1"].refs
    )
    # A reference followed by another act is not guessed into the note's act.
    assert "165/2014:art14" not in targets("guidance-8:p1")


def test_a_reference_is_written_as_in_its_text(tree):
    # References are found in the page text as it is, so each can be located in it; those
    # from a note's legal basis carry the rewritten wording of the basis.
    for node in tree:
        basis = tree[node.parent].legal_basis if node.kind == "page" else None
        for ref in node.refs:
            assert ref.text in node.text or ref.text in (basis or ""), (node.id, ref.text)


def test_only_known_note_references_stay_unresolved(tree):
    unresolved = [
        (n.id, r.text)
        for n in tree
        if n.kind == "page"
        for r in n.refs
        if r.target is None and (r.act in ACTS or r.act == n.doc_id)
    ]
    assert unresolved == NOTE_UNRESOLVED


# --- Hand-written fixtures for edge cases ---------------------------------------------------


def document(root: Path, fmt: str, body: str, doc_id: str = "1/2000") -> Document:
    name = f"fixture.en.{fmt}"
    (root / "corpus" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "corpus" / "raw" / name).write_text(body, encoding="utf-8")
    return Document(
        id=doc_id,
        title=f"Regulation (EU) No {doc_id} (a fixture)",
        kind="act",
        phase=1,
        format=fmt,
        url="https://cellar.test/fixture",
        file=f"corpus/raw/{name}",
        sha256=None,
        retrieved=None,
        celex="32000R0001",
    )


def xhtml(content: str) -> str:
    return (
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><div class="eli-container">'
        f'<div class="eli-subdivision" id="enc_1">{content}</div></div></body></html>'
    )


def modref(marker: str, celex: str, title: str) -> str:
    return f'<p class="modref"><a href="http://x/{celex}" title="{title}">▼{marker}</a></p>'


def paragraph(label: str, text: str) -> str:
    return (
        f'<div class="norm"><span class="no-parag">{label}.  </span>'
        f'<div class="norm inline-element">{text}</div></div>'
    )


def item(label: str, text: str) -> str:
    return (
        f'<div class="grid-container grid-list"><div class="list grid-list-column-1">'
        f'<span>{label} </span></div><div class="grid-list-column-2">'
        f'<p class="norm">{text}</p></div></div>'
    )


def article(label: str, content: str) -> str:
    return (
        f'<div class="eli-subdivision" id="art_{label}">'
        f'<p class="title-article-norm">Article {label}</p>{content}</div>'
    )


def parse(tmp_path: Path, fmt: str, body: str) -> Tree:
    tree = parse_document(document(tmp_path, fmt, body), tmp_path)
    resolve_refs(tree, {"1/2000"})
    return tree


def test_a_marker_applies_until_the_next_one(tmp_path):
    tree = parse(
        tmp_path,
        "xhtml",
        xhtml(
            article(
                "1",
                paragraph("1", "Old.")
                + modref("M1", "32010R0001", "32010R0001: INSERTED")
                + paragraph("1a", "Inserted.")
                + modref("B", "32000R0001", "32000R0001")
                + paragraph("2", "Old again."),
            )
        ),
    )
    assert tree["1/2000:art1(1)"].amendments == []
    assert tree["1/2000:art1(1a)"].amendments == [Amendment("M1", "32010R0001", "INSERTED")]
    assert tree["1/2000:art1(2)"].amendments == []


def test_inline_markers_and_footnotes_leave_clean_text(tmp_path):
    text = (
        '<span><a href="http://x/32010R0001" title="32010R0001: REPLACED">'
        '<span class="boldface">►M1</span></a></span> New words<span class="boldface"> ◄ </span>'
        ' of the Council (<a href="#E0001" id="src.E0001"><span class="superscript">1</span>'
        "</a>) apply."
    )
    tree = parse(tmp_path, "xhtml", xhtml(article("1", paragraph("1", text))))
    node = tree["1/2000:art1(1)"]
    assert node.text == "New words of the Council apply."
    assert node.amendments == [Amendment("M1", "32010R0001", "REPLACED")]


def test_a_duplicate_id_is_an_error(tmp_path):
    tree = parse(
        tmp_path, "xhtml", xhtml(article("1", paragraph("1", "A.") + paragraph("1", "B.")))
    )
    assert tree.errors == ["1/2000: duplicate node ID 1/2000:art1(1)"]


def test_unknown_structure_is_listed_not_dropped(tmp_path):
    tree = parse(tmp_path, "xhtml", xhtml(article("1", "<ul><li>Something new</li></ul>")))
    assert tree.unparsed == ["1/2000:art1: ul.None: Something new"]


def test_html_labels_close_and_open_the_right_levels(tmp_path):
    paragraphs = [
        "Title of the act",
        "Article 1",
        "Heading",
        "For the purposes:",
        "(a) first:",
        "1. numbered under (a);",
        "(i) roman under 1;",
        "(ii) second roman;",
        "2. second number;",
        "(b) back to letters;",
        "- an indent of (b),",
        "text after the indent;",
        "(c) third letter.",
        "Done at Brussels.",
    ]
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    tree = parse(tmp_path, "html", f"<html><body>{body}</body></html>")
    assert [n.id.removeprefix("1/2000:") for n in tree if n.kind == "point"] == [
        "art1(a)",
        "art1(a)(1)",
        "art1(a)(1)(i)",
        "art1(a)(1)(ii)",
        "art1(a)(2)",
        "art1(b)",
        "art1(c)",
    ]
    assert tree["1/2000:art1(b)"].text.splitlines() == [
        "back to letters;",
        "— an indent of (b),",
        "text after the indent;",
    ]
    assert tree["1/2000:art1"].heading == "Article 1 — Heading"


def test_reference_forms(tmp_path):
    content = (
        article("4", item("(a)", "a definition"))
        + article("6", paragraph("1", "One.") + paragraph("2", "Two."))
        + article("8", "")
        + article("8a", "")
        + article("9", "")
        + article(
            "10",
            paragraph(
                "1",
                "As in Articles 6 to 9, Article 6(1) and (2), point (a) of Article 4, "
                "paragraph 2 of this Article, Annex I and Annex IB of Regulation (EEC) No 3821/85, "
                "and Article 5 of Directive 2003/59/EC.",
            )
            + paragraph("2", "See paragraph 1."),
        ),
    )
    tree = parse(tmp_path, "xhtml", xhtml(content))
    refs = tree["1/2000:art10(1)"].refs
    targets = [r.target for r in refs if r.target]
    assert targets == [
        "1/2000:art6",  # Articles 6 to 9, in document order, including 8a
        "1/2000:art8",
        "1/2000:art8a",
        "1/2000:art9",
        "1/2000:art6(1)",
        "1/2000:art6(2)",
        "1/2000:art4(a)",
        "1/2000:art10(2)",
    ]
    outside = [(r.text, r.act) for r in refs if r.target is None]
    assert outside == [
        ("Annex I", "Regulation (EEC) No 3821/85"),
        ("Annex IB of Regulation (EEC) No 3821/85", "Regulation (EEC) No 3821/85"),
        ("Article 5 of Directive 2003/59/EC", "Directive 2003/59/EC"),
    ]
    assert tree["1/2000:art10(2)"].refs == [Ref("paragraph 1", "1/2000", "1/2000:art10(1)")]


def test_references_into_annexes_and_ranges_in_the_corpus(tree):
    def targets(node_id: str) -> set[str]:
        return {r.target for r in tree[node_id].refs}

    assert "165/2014:anxI.III.c.4.1" in targets("165/2014:anxI.III.c.4.3")  # "point 4.1"
    assert "165/2014:anxI.III.f.3" in targets("165/2014:anxI.VI.3(b)")  # "(f)(3) of Part III"
    assert {"561/2006:art7", "561/2006:art8a"} <= targets("561/2006:art11")  # "Articles 6 to 9"


def test_only_known_references_fall_back_to_a_parent(tree):
    # 165/2014 Art. 10 has no numbered paragraphs but cites "paragraph 1" (its first one). Two
    # clarification notes cite points of 561/2006 Art. 13 without its paragraph ("13(d)" for
    # Art. 13(1)(d)).
    partial = [(n.id, r.text, r.target) for n in tree for r in n.refs if r.partial]
    assert partial == [
        ("165/2014:art10", "paragraph 1", "165/2014:art10"),
        ("clarification-3:p1", "Article 13 (d)", "561/2006:art13"),
        (
            "clarification-5:p1",
            "Article 3(h), 13(i) of Regulation (EC) No 561/2006",
            "561/2006:art13",
        ),
    ]


def annex(content: str) -> str:
    return (
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><div class="eli-container">'
        '<div class="eli-subdivision" id="enc_1"></div>'
        f'<div id="anx_I"><p class="title-annex-1">ANNEX I</p>{content}</div>'
        "</div></body></html>"
    )


def test_ranges_and_annex_references(tmp_path):
    points = "".join(item(f"({x})", x) for x in "abcd")
    content = xhtml(
        article("4", points)
        + article("5", paragraph("1", "A.") + paragraph("2", "B.") + paragraph("3", "C."))
        + article(
            "6",
            paragraph(
                "1",
                "See points (a) to (c) of Article 4, paragraphs 1 to 3 of Article 5, "
                "point 2 of Part II of Annex I and Annexes I and II.",
            ),
        )
    ).replace(
        "</div></body>",
        '<div id="anx_I"><p class="title-gr-seq-level-2">II. PART</p>'
        '<div><p class="norm">2. A point.</p></div></div>'
        '<div id="anx_II"><p class="title-gr-seq-level-2">I. OTHER</p></div></div></body>',
    )
    tree = parse(tmp_path, "xhtml", content)
    assert [r.target for r in tree["1/2000:art6(1)"].refs] == [
        "1/2000:art4(a)",
        "1/2000:art4(b)",
        "1/2000:art4(c)",
        "1/2000:art5(1)",
        "1/2000:art5(2)",
        "1/2000:art5(3)",
        "1/2000:anxI.II.2",
        "1/2000:anxI",
        "1/2000:anxII",
    ]


def test_the_amended_act_comes_from_the_body_not_the_heading(tmp_path):
    body = (
        '<div class="eli-subdivision" id="art_9"><p class="title-article-norm">Article 9</p>'
        '<div class="eli-title"><p class="stitle-article-norm">Amendment of Directive 2006/22/EC'
        '</p></div><p class="norm">Regulation (EEC) No 3821/85 is hereby amended as follows:</p>'
        + item("(1)", "in Article 2, point (a) is deleted;")
        + "</div>"
    )
    tree = parse(tmp_path, "xhtml", xhtml(body))
    node = tree["1/2000:art9"]
    assert node.amending
    assert node.amends == "Regulation (EEC) No 3821/85"
    assert tree["1/2000:art9(1)"].refs[0].act == "Regulation (EEC) No 3821/85"


def test_table_cells_keep_their_inline_amendments(tmp_path):
    cell = (
        '<td><p><a href="http://x/32010R0001" title="32010R0001: REPLACED">►M1</a> 45 minutes'
        " ◄</p></td>"
    )
    tree = parse(
        tmp_path, "xhtml", xhtml(article("1", f"<table><tr><td>Break</td>{cell}</tr></table>"))
    )
    node = tree["1/2000:art1"]
    assert node.text == "Break | 45 minutes"
    assert node.amendments == [Amendment("M1", "32010R0001", "REPLACED")]


def test_an_image_in_a_labelled_annex_point_is_listed_once_under_that_point(tmp_path):
    content = annex(
        '<p class="title-gr-seq-level-2">I. MARKS</p>'
        '<div><p class="norm">1. The mark <img src="x.png" alt="image"/> is used.</p></div>'
    )
    tree = parse(tmp_path, "xhtml", content)
    assert tree["1/2000:anxI.I.1"].text == "The mark [image] is used."
    assert tree.unparsed == ["1/2000:anxI.I.1: image, left as [image] in the text"]


def test_html_without_its_closing_formula_is_a_parse_error(tmp_path):
    body = "".join(f"<p>{p}</p>" for p in ["Title", "Article 1", "Heading", "Text."])
    with pytest.raises(ParseError, match="Done at"):
        parse(tmp_path, "html", f"<html><body>{body}</body></html>")


def test_capitalised_references_and_a_range_whose_start_has_no_node(tmp_path):
    points = "".join(item(f"({x})", x) for x in "abfg")  # no (c), (d) or (e)
    content = xhtml(
        article("4", points)
        + article("5", paragraph("1", "Points (a) and (b) of Article 4 apply."))
        + article("6", paragraph("1", "See points (a), (e) to (g) of Article 4."))
    )
    tree = parse(tmp_path, "xhtml", content)
    assert [r.target for r in tree["1/2000:art5(1)"].refs] == ["1/2000:art4(a)", "1/2000:art4(b)"]
    # (e) has no node: it falls back to its article, flagged as partial, and the range keeps
    # only its end instead of starting from (a).
    targets = [(r.target, r.partial) for r in tree["1/2000:art6(1)"].refs]
    assert targets == [("1/2000:art4(a)", False), ("1/2000:art4", True), ("1/2000:art4(g)", False)]


def test_this_regulation_in_amending_text_means_the_amended_act(tmp_path):
    body = article(
        "9",
        '<p class="norm">Regulation (EU) No 2/2000 is hereby amended as follows:</p>'
        + item("(1)", "Article 4 of this Regulation is replaced."),
    )
    tree = parse_document(document(tmp_path, "xhtml", xhtml(body)), tmp_path)
    resolve_refs(tree, {"1/2000", "2/2000"})
    assert [(r.act, r.target) for r in tree["1/2000:art9(1)"].refs] == [("2/2000", None)]


def note(tree: Tree, legal_basis: str, text: str) -> Node:
    """A guidance note with one page, added to `tree` by hand rather than from a PDF."""
    tree.add(Node("n-1", "n-1", "note", None, "Note 1", ("Note 1",), None, legal_basis=legal_basis))
    page = Node("n-1:p1", "n-1", "page", "p1", "Page 1", ("Note 1", "Page 1"), "n-1")
    page.segments = [(1, text)]
    return tree.add(page)


def test_a_note_reference_without_an_act_is_not_guessed_when_another_act_follows(tmp_path):
    tree = parse_document(document(tmp_path, "xhtml", xhtml(article("4", ""))), tmp_path)
    page = note(
        tree,
        "Article 4 of Regulation (EU) No 1/2000",
        "Article 4 applies. Article 4 (1) a of Regulation (EEC) No 9/99 applies. Article 4 of "
        "Regulation (EU) No 1/2000 as amended by Regulation (EU) No 2/2000 applies.",
    )
    resolve_refs(tree, {"1/2000"})
    assert tree["n-1"].interprets == "1/2000"
    assert [(r.text, r.act, r.target) for r in page.refs] == [
        ("Article 4", "1/2000", "1/2000:art4"),
        ("Article 4 (1)", "n-1", None),
        ("Article 4 of Regulation (EU) No 1/2000", "1/2000", "1/2000:art4"),
    ]


def test_a_damaged_pdf_is_a_parse_error(tmp_path):
    with pytest.raises(ParseError, match="not a readable PDF"):
        parse_document(document(tmp_path, "pdf", "not a PDF"), tmp_path)


def test_a_legal_basis_takes_its_act_from_the_header_when_it_names_none():
    # The header names the act twice, in two forms, and shares a text box with the field.
    segments = [
        "Regulation (EC) No 561/2006",
        "Rules of Regulation 561/2006 apply. Articles: 1, 2",
        "Issue raised: something.",
    ]
    assert _legal_basis(segments) == "Article 1, 2 of Regulation (EC) No 561/2006"
    # A number after an act is another article, unless it is another act's number.
    segments = ["Article: 3 of Regulation (EC) No 561/2006 and 2002/15/EC, 7 of Directive 2/2000"]
    assert _legal_basis(segments) == (
        "Article 3 of Regulation (EC) No 561/2006 and 2002/15/EC, Article 7 of Directive 2/2000"
    )
