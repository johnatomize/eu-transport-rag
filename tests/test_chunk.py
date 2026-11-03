import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from rag.chunk import STRATEGIES, Chunk, window_chunks
from rag.corpus import load_manifest
from rag.parse import Node, Tree, parse_corpus, printed_label

ROOT = Path(__file__).parents[1]
AMENDING = ("561/2006:art26", "561/2006:art27", "165/2014:art45")
ART6 = "Regulation (EC) No 561/2006 › Chapter II — Crews, driving times, breaks and rest periods"
ART6_PREFIX = f"{ART6} › Article 6"


def by_id(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {chunk.id: chunk for chunk in chunks}


def ancestors(tree: Tree, node_id: str) -> Iterator[str]:
    while node_id is not None:
        yield node_id
        node_id = tree[node_id].parent


def left_out(tree: Tree, node: Node) -> bool:
    """An act's title text, or anything inside an amending article."""
    return node.kind == "act" or any(a in AMENDING for a in ancestors(tree, node.id))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_node_with_text_is_in_a_chunk(tree, chunks, strategy):
    """Each piece of text is in a body, or as a lead-in in a prefix, unless it is left out."""
    texts = [chunk.text if strategy == "paragraph" else chunk.body for chunk in chunks[strategy]]
    covered = {node_id for chunk in chunks[strategy] for node_id in chunk.node_ids}
    for node in tree:
        if not node.segments or left_out(tree, node):
            continue
        assert node.id in covered or strategy == "paragraph", node.id
        if not strategy.startswith("window"):
            for _, segment in node.segments:
                assert any(segment in text for text in texts), (node.id, segment)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_left_out_parts_are_in_no_chunk(tree, chunks, strategy):
    for chunk in chunks[strategy]:
        assert not any(left_out(tree, tree[node_id]) for node_id in chunk.node_ids), chunk.id
        assert "OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL" not in chunk.text, chunk.id


@pytest.fixture(scope="module")
def fresh_chunks() -> dict[str, list[Chunk]]:
    """The chunks again, from a second parse of the corpus."""
    fresh = parse_corpus(load_manifest(ROOT / "corpus" / "manifest.yaml"), ROOT)
    return {name: strategy(fresh) for name, strategy in STRATEGIES.items()}


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_chunk_ids_are_unique_and_the_same_across_runs(chunks, fresh_chunks, strategy):
    ids = [chunk.id for chunk in chunks[strategy]]
    assert len(set(ids)) == len(ids)
    assert all(chunk_id.startswith(f"{strategy}:") for chunk_id in ids)
    assert fresh_chunks[strategy] == chunks[strategy]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_gold_node_can_be_found(tree, chunks, strategy):
    """A chunk finds a gold node when its node_ids hold the node or a descendant of it."""
    found = {a for chunk in chunks[strategy] for n in chunk.node_ids for a in ancestors(tree, n)}
    lines = (ROOT / "data" / "questions.v1.jsonl").read_text(encoding="utf-8").splitlines()
    gold = {node_id for line in lines for node_id in json.loads(line)["gold_node_ids"]}
    assert gold
    assert gold <= found, sorted(gold - found)


def test_article_6_as_one_article_chunk(chunks):
    chunk = by_id(chunks["article"])["article:561/2006:art6"]
    assert chunk.prefix == ART6_PREFIX
    assert chunk.body.startswith("1. The daily driving time shall not exceed nine hours.\n")
    assert "\n3. The total accumulated driving time during any two consecutive weeks" in chunk.body
    assert chunk.node_ids == tuple(f"561/2006:art6({i})" for i in range(1, 6))
    assert chunk.text == f"{ART6_PREFIX}\n{chunk.body}"


def test_article_6_as_paragraph_chunks(chunks):
    art6 = [c for c in chunks["paragraph"] if c.id.startswith("paragraph:561/2006:art6(")]
    assert [c.id for c in art6] == [f"paragraph:561/2006:art6({i})" for i in range(1, 6)]
    assert all(c.prefix == ART6_PREFIX for c in art6)
    assert art6[0].body.startswith("1. The daily driving time shall not exceed nine hours.")
    assert art6[2].body.startswith("3. The total accumulated driving time")
    assert art6[2].node_ids == ("561/2006:art6(3)",)


def test_article_6_in_windows(chunks):
    in_art6 = "561/2006:art6("
    art6 = [c for c in chunks["window-200-50"] if any(in_art6 in n for n in c.node_ids)]
    assert {n for c in art6 for n in c.node_ids if in_art6 in n} == {
        f"561/2006:art6({i})" for i in range(1, 6)
    }
    assert any("\nArticle 6\n1. The daily driving time shall not exceed" in c.body for c in art6)


def test_points_carry_their_lead_in(chunks):
    chunk = by_id(chunks["paragraph"])["paragraph:561/2006:art4(e)"]
    assert chunk.prefix == (
        "Regulation (EC) No 561/2006 › Chapter I — Introductory provisions › Article 4\n"
        "For the purposes of this Regulation the following definitions shall apply:"
    )
    assert chunk.body.startswith("(e) ")


def test_closing_text_gets_its_own_paragraph_chunk(chunks):
    chunks = by_id(chunks["paragraph"])
    closing = chunks["paragraph:165/2014:art10"]
    assert closing.body.startswith("The tachographs of vehicles registered for the first time in")
    assert closing.node_ids == ("165/2014:art10",)
    first = chunks["paragraph:165/2014:art10(a)"]
    lead = first.prefix.split("\n")[1:]
    assert len(lead) == 1
    assert lead[0].startswith("The tachographs of vehicles registered for the first time 36 months")
    assert lead[0].endswith("provided that the following conditions are met:")


def test_guidance_pages_carry_their_date(tree, chunks):
    for strategy, chunk_list in chunks.items():
        page = next(c for c in chunk_list if c.doc_id == "guidance-7")
        assert page.published == tree["guidance-7"].published is not None, strategy
        assert page.prefix.startswith('Guidance note 7: The meaning of "each period of 24 hours"')
        act = next(c for c in chunk_list if c.doc_id == "561/2006")
        assert act.published is None


def test_annex_i_is_split_into_its_parts(chunks):
    ids = by_id(chunks["article"])
    assert "article:165/2014:anxI" not in ids
    assert [i for i in ids if i.startswith("article:165/2014:anxI.")] == [
        f"article:165/2014:anxI.{part}" for part in ("I", "II", "III", "IV", "V", "VI")
    ]


def test_windows_overlap_and_stay_in_their_document(tree, chunks):
    windows = chunks["window-200-50"]
    articles = chunks["article"]
    for doc_id in dict.fromkeys(c.doc_id for c in windows):
        doc = [c for c in windows if c.doc_id == doc_id]
        numbers = range(1, len(doc) + 1)
        assert [c.id for c in doc] == [f"window-200-50:{doc_id}:w{i}" for i in numbers]
        assert all(n.startswith(f"{doc_id}:") for c in doc for n in c.node_ids)
        words = [c.body.split() for c in doc]
        assert all(len(w) == 200 for w in words[:-1])
        for before, after in zip(words, words[1:], strict=False):
            assert before[-50:] == after[:50]
        # Without the overlaps, the windows are the document's text in reading order: each
        # unit's heading, then its text, exactly as the article strategy holds it.
        stream = words[0] + [w for later in words[1:] for w in later[50:]]
        units = [c for c in articles if c.doc_id == doc_id]
        expected = [w for c in units for w in c.prefix.split(" › ")[-1].split() + c.body.split()]
        assert stream == expected, doc_id


def test_window_inside_article_13_1_carries_its_opening_sentence(chunks):
    inside = [c for c in chunks["window-200-50"] if c.node_ids[0].startswith("561/2006:art13(1)(")]
    assert inside
    opening = "\n1. Provided the objectives set out in Article 1 are not prejudiced"
    for chunk in inside:
        assert opening in chunk.prefix


def test_a_window_starting_inside_a_heading_carries_that_heading(chunks):
    window = by_id(chunks["window-200-50"])["window-200-50:165/2014:w94"]
    assert window.body.startswith("recording instruments)\n1. On the test bench")
    assert window.prefix.endswith(
        "› III. Construction requirements for recording equipment › "
        "(f) Maximum tolerances (visual and recording instruments)"
    )


def small_tree() -> Tree:
    """An act with one article of two five-word paragraphs, for counting words by hand."""
    tree = Tree()
    path = ("Act",)
    tree.add(Node("a", "a", "act", None, "Act", path, None))
    article = Node("a:art1", "a", "article", "1", "Article 1", (*path, "Article 1"), "a")
    article.position = 1
    tree.add(article)
    for i, (label, words) in enumerate((("(1)", "a b c d e"), ("(2)", "f g h i j")), 1):
        node_id = f"a:art1{label}"
        node = Node(node_id, "a", "paragraph", label, None, (*path, "Article 1"), "a:art1")
        node.position = 2 * i
        node.segments = [(2 * i + 1, words)]
        tree.add(node)
    return tree


def test_window_node_ids_are_exactly_the_nodes_it_touches():
    # Words: "Article 1" (2), "1. a b c d e" (6), "2. f g h i j" (6): 14 words, step 4.
    windows = window_chunks(small_tree(), size=6, overlap=2)
    assert [c.body for c in windows] == [
        "Article 1\n1. a b c",
        "b c d e\n2. f",
        "2. f g h i j",
    ]
    assert [c.node_ids for c in windows] == [
        ("a:art1(1)",),
        ("a:art1(1)", "a:art1(2)"),
        ("a:art1(2)",),
    ]
    assert [c.id for c in windows] == ["window-6-2:a:w1", "window-6-2:a:w2", "window-6-2:a:w3"]


def test_window_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        window_chunks(small_tree(), size=50, overlap=50)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_body_line_names_the_node_it_belongs_to(tree, chunks, strategy):
    """`line_owners` has one entry per body line, and its nodes are the chunk's, in order."""
    for chunk in chunks[strategy]:
        lines = chunk.body.split("\n")
        assert len(chunk.line_owners) == len(lines), chunk.id
        owners = [owner for owner in chunk.line_owners if owner is not None]
        assert tuple(dict.fromkeys(owners)) == chunk.node_ids, chunk.id
        for owner, line in zip(chunk.line_owners, lines, strict=True):
            if owner is not None and strategy != "window-200-50":
                # Windows can start or end inside a line, so only whole lines are compared. A
                # node's first line carries its printed label, which full_text leaves out.
                node = tree[owner]
                label = printed_label(node) if node.kind in {"paragraph", "point"} else ""
                text = line.removeprefix(label).strip()
                assert text in tree.full_text(owner), (chunk.id, line)


def test_line_owners_must_match_the_body():
    with pytest.raises(ValueError, match="line_owners"):
        Chunk("t:a:art1", "t", "a", "Act", "one\ntwo", ("a:art1",), line_owners=("a:art1",))
