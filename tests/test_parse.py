import re
from pathlib import Path

import pytest

from rag.corpus import Document, load_manifest
from rag.parse import (
    Amendment,
    ParseError,
    Ref,
    Tree,
    parse_corpus,
    parse_document,
    resolve_refs,
)

ROOT = Path(__file__).parents[1]
SNAPSHOT = ROOT / "tests" / "snapshots" / "node_ids.txt"
ACTS = {"561/2006", "165/2014", "2002/15"}

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
]

M3 = Amendment("M3", "32020R1054", "REPLACED")
M4_INSERTED = Amendment("M4", "32024R1258", "INSERTED")


@pytest.fixture(scope="module")
def tree() -> Tree:
    return parse_corpus(load_manifest(ROOT / "corpus" / "manifest.yaml"), ROOT)


def children(tree: Tree, node_id: str) -> list[str]:
    return [c.removeprefix(node_id) for c in tree[node_id].children]


# --- The real corpus ------------------------------------------------------------------------


def test_every_act_parses_with_no_duplicates_and_only_known_gaps(tree):
    assert tree.errors == []
    assert tree.unparsed == UNPARSED
    assert {n.doc_id for n in tree} == ACTS


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
    # 165/2014 Art. 10 has no numbered paragraphs but cites "paragraph 1" (its first one).
    partial = [(n.id, r.text, r.target) for n in tree for r in n.refs if r.partial]
    assert partial == [("165/2014:art10", "paragraph 1", "165/2014:art10")]


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
