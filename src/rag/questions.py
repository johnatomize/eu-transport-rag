"""The question set in `data/questions.v1.jsonl`, and the validator that `make check` runs.

One JSON object per line, one question each (design §C):

    {"id": "q001", "question": "How many hours can I drive in two weeks in a row?",
     "answer": "At most 90 hours in any two consecutive weeks.", "key_facts": ["90 hours"],
     "gold_node_ids": ["561/2006:art6(3)"], "type": "single_fact", "phrasing": "paraphrased",
     "split": "dev", "checked": false, "notes": ""}

`key_facts` are the values that answer evaluation checks by exact match (design problem 4).
`gold_node_ids` are the most specific nodes that hold the answer; how a retrieved chunk matches
them is for retrieval evaluation to decide. `checked` stays false until the owner has checked
the gold answer against the text. A version trap's old answer goes in `notes`, the only optional
field. IDs are never reused or renumbered, so that run logs stay comparable. See
docs/DECISIONS.md.
"""

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass

from rag.parse import Node, Tree

TYPES = (
    "single_fact",
    "definition",
    "multi_article",
    "exception",
    "cross_act",
    "unanswerable",
    "version_trap",
)
PHRASINGS = ("paraphrased", "verbatim")
SPLITS = ("dev", "test")

_FIELDS = {
    "id": str,
    "question": str,
    "answer": str,
    "key_facts": list,
    "gold_node_ids": list,
    "type": str,
    "phrasing": str,
    "split": str,
    "checked": bool,
    "notes": str,
}
_OPTIONAL = {"notes": ""}
_EXPECTED = {str: "a string", list: "a list of strings", bool: "true or false"}
_ID = re.compile(r"q\d{3}")
_TOO_BROAD = {"act", "chapter", "note"}


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    key_facts: tuple[str, ...]
    gold_node_ids: tuple[str, ...]
    type: str
    phrasing: str
    split: str
    checked: bool
    notes: str = ""


def validate(text: str, tree: Tree) -> tuple[list[Question], list[str]]:
    """Check every line of a questions file against the schema and the parsed corpus.

    Returns the valid questions, and one message per problem naming the line, the question ID
    and the field, such as `line 7 (q007): gold_node_ids: '561/2006:art6(9)' is not a node in
    the parsed corpus`. Blank lines are skipped.
    """
    questions, errors = [], []
    seen: dict[str, int] = {}
    # Not splitlines(): JSON strings may hold U+2028 and other characters it splits on.
    for number, line in enumerate(text.removeprefix("\ufeff").split("\n"), 1):
        if not line.strip():
            continue
        row, problems = _check(line, tree)
        qid = row.get("id") if row is not None else None
        if isinstance(qid, str) and qid:
            where = f"line {number} ({qid})"
            if qid in seen:
                problems.append(f"id: already used on line {seen[qid]}")
            else:
                seen[qid] = number
        else:
            where = f"line {number}"
        errors.extend(f"{where}: {p}" for p in problems)
        if not problems:
            row = {**_OPTIONAL, **row}
            for name in ("key_facts", "gold_node_ids"):
                row[name] = tuple(row[name])
            questions.append(Question(**row))
    return questions, errors


def coverage(questions: list[Question]) -> list[str]:
    """Problems with the set as a whole: every type must appear in both splits.

    Results are reported per type and split, so a type missing from one split would leave a
    hole in every table.
    """
    present = {(q.type, q.split) for q in questions}
    return [
        f"type {kind!r} has no {split} questions"
        for kind in TYPES
        for split in SPLITS
        if (kind, split) not in present
    ]


def _check(line: str, tree: Tree) -> tuple[dict | None, list[str]]:
    """The row as read, and its problems, each starting with the field it is about."""
    try:
        row = json.loads(line, object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as e:
        return None, [f"not valid JSON ({e.msg})"]
    except _DuplicateKey as e:
        return None, [f"{e}: given twice"]
    if not isinstance(row, dict):
        return None, ["not a JSON object"]
    problems = [f"{name}: missing" for name in _FIELDS if name not in row and name not in _OPTIONAL]
    problems += [f"{name}: unknown field" for name in row if name not in _FIELDS]
    problems += [
        f"{name}: must be {_EXPECTED[kind]}"
        for name, kind in _FIELDS.items()
        if name in row and not _is(row[name], kind)
    ]
    if problems:
        return row, problems

    if not _ID.fullmatch(row["id"]):
        problems.append(f"id: {row['id']!r} is not q followed by three digits")
    if not row["question"].strip():
        problems.append("question: empty")
    for name, allowed in (("type", TYPES), ("phrasing", PHRASINGS), ("split", SPLITS)):
        if row[name] not in allowed:
            problems.append(f"{name}: {row[name]!r} is not one of {', '.join(allowed)}")
    for fact in row["key_facts"]:
        if not fact.strip():
            problems.append("key_facts: an empty fact")
        elif fact not in row["answer"]:
            problems.append(f"key_facts: {fact!r} is not in the answer")
    problems += _gold_problems(row["gold_node_ids"], tree)
    if row["type"] in TYPES:
        problems += _type_problems(row, tree)
    return row, problems


class _DuplicateKey(Exception):
    pass


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    # json.loads would keep the last of two values silently, and a hand edit can leave both.
    row = {}
    for key, value in pairs:
        if key in row:
            raise _DuplicateKey(key)
        row[key] = value
    return row


def _is(value: object, kind: type) -> bool:
    if kind is list:
        return isinstance(value, list) and all(isinstance(v, str) for v in value)
    return isinstance(value, kind)


def _gold_problems(node_ids: list[str], tree: Tree) -> list[str]:
    problems = []
    for node_id in sorted(set(node_ids), key=node_ids.index):
        if node_ids.count(node_id) > 1:
            problems.append(f"gold_node_ids: {node_id!r} is listed twice")
        if node_id not in tree:
            problems.append(f"gold_node_ids: {node_id!r} is not a node in the parsed corpus")
            continue
        node = tree[node_id]
        # Gold nodes are the most specific ones that hold the answer, so that a chunk holding
        # the whole act doesn't count as finding it.
        if node.kind in _TOO_BROAD:
            problems.append(f"gold_node_ids: {node_id!r} is a whole {node.kind}; name the nodes")
        inside = next(
            (a.id for a in _ancestors(tree, node) if a is not node and a.id in node_ids), None
        )
        if inside is not None:
            problems.append(f"gold_node_ids: {node_id!r} is inside {inside!r}, also listed")
        # Amending articles hold wording that has since been replaced (design problem 1), so
        # they can't be the evidence for a current answer.
        amending = next((n for n in _ancestors(tree, node) if n.amending), None)
        if amending is not None:
            problems.append(
                f"gold_node_ids: {node_id!r} is in an amending article ({amending.id}), "
                "whose wording has since been replaced"
            )
    return problems


def _type_problems(row: dict, tree: Tree) -> list[str]:
    """The rules that depend on the question type."""
    kind, gold = row["type"], row["gold_node_ids"]
    if kind == "unanswerable":
        return [
            f"{name}: must be empty for an unanswerable question"
            for name in ("answer", "key_facts", "gold_node_ids")
            if row[name]
        ]
    problems = []
    if not row["answer"].strip():
        problems.append("answer: empty; only an unanswerable question has no answer")
    if not gold:
        problems.append("gold_node_ids: empty; only an unanswerable question has none")
    nodes = [tree[node_id] for node_id in gold if node_id in tree]
    if len(nodes) < len(gold):
        return problems  # already reported; the rules below would only add noise
    if kind == "multi_article":
        provisions = {_provision(tree, n) for n in nodes} - {None}
        if len(provisions) < 2:
            problems.append(
                "gold_node_ids: a multi_article question needs two articles or annexes or more"
            )
    if kind == "cross_act":
        acts = {n.doc_id for n in nodes if tree[n.doc_id].kind == "act"}
        if len(acts) < 2:
            problems.append("gold_node_ids: a cross_act question needs two acts or more")
    if kind == "version_trap" and not row.get("notes", "").strip():
        problems.append("notes: a version_trap question needs what changed and the old answer")
    return problems


def _ancestors(tree: Tree, node: Node) -> Iterator[Node]:
    """The node itself, then its parent, up to the document's root."""
    while True:
        yield node
        if node.parent is None:
            return
        node = tree[node.parent]


def _provision(tree: Tree, node: Node) -> str | None:
    """The ID of the article or annex the node is in, or None if neither (a guidance page)."""
    return next((n.id for n in _ancestors(tree, node) if n.kind in {"article", "annex"}), None)
