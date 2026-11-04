"""The owner's faithfulness labels (#82): which answers get labelled, and the labels file.

The faithfulness judge (#83) is measured against about 30 answers the owner labels by hand.
The rules, recorded in docs/DECISIONS.md ("which answers the owner labels for faithfulness"):

- **Which answers.** 15 dev questions that both of F's runs (Opus 5.5 and Haiku 5.5) answered
  with a reply that passed the check, and both answers to each: 30 answers. The questions are
  spread over the answerable types by fixed quotas, drawn with a fixed seed from the questions
  that qualify, and q001 is always among them, for its "at most 34 hours" worked out from
  quoted numbers. The rule was fixed before anyone read the answers for it.
- **Blind.** The page calls a question's two answers A and B, in an order drawn with the same
  seed, and never names the model; `assign` maps the letters back to run logs.
- **Sentences.** An answer is split into sentences by `sentences`, the same way for the page
  and for the judge, so "the first unsupported sentence" means the same thing to both.
- **The labels file**, `data/faithfulness_labels.v1.jsonl`, one answer per line:

      {"question_id": "q001", "run": "2026-11-03T21-03-51Z-claude-opus-5-5.json",
       "supported": true, "first_unsupported": null, "sentence": null, "note": ""}

  `supported` is the owner's verdict: every claim follows from the full text of the nodes the
  answer cites, simple arithmetic on cited numbers included. When it is false,
  `first_unsupported` is the index of the first sentence that doesn't, and `sentence` its text,
  so that a reader of the file doesn't need the run log. `note` is the owner's own words.

The labels stay out of the judge's prompt and its development, so that the agreement measured
in #83 is honest.
"""

import hashlib
import json
import random
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag.questions import Question
from rag.validate import normalise

# 15 questions, spread over the answerable types roughly as the dev split is.
QUOTAS = {
    "single_fact": 4,
    "exception": 3,
    "version_trap": 3,
    "multi_article": 2,
    "definition": 2,
    "cross_act": 1,
}
ALWAYS = ("q001",)
SEED = "faithfulness-labels-v1"
LETTERS = ("A", "B")

_FIELDS = {
    "question_id": str,
    "run": str,
    "supported": bool,
    "first_unsupported": (int, type(None)),
    "sentence": (str, type(None)),
    "note": str,
}

# A sentence ends after . ! or ? when the next starts with a capital, a digit or an opening
# mark. A full stop doesn't end one after an abbreviation that is always one, after an
# abbreviation that is one before a number ("Art. 6", "No. 165/2014"; but "the answer is no."),
# or after a list item's own number or letter, alone at the start ("2. Between the dials").
_ABBREVIATIONS = {"e.g", "i.e", "cf", "reg", "dir"}
_BEFORE_NUMBER = {"art", "arts", "no", "nos", "para", "paras", "p", "pp"}
_BOUNDARY = re.compile(r"(?<=[.!?])([\"'”’)\]]*)\s+(?=[\"“‘(\[]?[A-Z0-9])")


@dataclass(frozen=True)
class Label:
    question_id: str
    run: str
    supported: bool
    first_unsupported: int | None
    sentence: str | None
    note: str


def sentences(text: str) -> list[str]:
    """The sentences of an answer, in order: each line apart, then split after . ! or ?"""
    out = []
    for line in text.split("\n"):
        start = 0
        for match in _BOUNDARY.finditer(line):
            end = match.start() + len(match.group(1))
            if line[end - 1 : end] != "." or _ends(line[start:end], line[match.end()]):
                out.append(line[start:end].strip())
                start = match.end()
        if line[start:].strip():
            out.append(line[start:].strip())
    return out


def _ends(sentence: str, following: str) -> bool:
    """Whether a full stop at the end of `sentence`, before the character `following`, ends it."""
    words = sentence.split()
    last = words[-1].strip("\"'“‘([.!?").lower() if words else ""
    if last in _ABBREVIATIONS or (last in _BEFORE_NUMBER and following.isdigit()):
        return False
    is_marker = last.isdigit() or (len(last) == 1 and last.isalpha())
    return not (len(words) == 1 and is_marker)


def by_id(log: Mapping) -> dict:
    """A run log's entries by question ID."""
    return {e["id"]: e for e in log["answers"]}


def build_id(question_ids: Sequence[str], letters: Mapping, logs: Mapping[str, Mapping]) -> str:
    """A short fingerprint of what the page shows: the questions, the letters and the answers.

    The page keeps labels under it and puts it in its export, so labels made on a page built
    from other answers are never applied to these.
    """
    record = [
        [qid, letters[qid], [by_id(logs[r])[qid]["verdict"]["answer"] for r in sorted(logs)]]
        for qid in question_ids
    ]
    digest = hashlib.sha256(json.dumps(record, ensure_ascii=False).encode()).hexdigest()
    return digest[:12]


def quote_spans(text: str, quotes: Sequence[str]) -> tuple[list[list[int]], list[str]]:
    """Where each quote lies in `text`, as [start, end] offsets, and the quotes not found.

    Matched the way the answer check matched them (`rag.validate.normalise`): spacing, invisible
    characters and typography such as curly quotes and dashes don't matter.
    """
    flat, starts = normalise(text)
    spans, missing = [], []
    for quote in quotes:
        wanted, _ = normalise(unicodedata.normalize("NFC", quote))
        wanted = wanted.strip()
        at = flat.find(wanted) if wanted else -1
        if at < 0:
            missing.append(quote)
        else:
            spans.append([starts[at], starts[at + len(wanted) - 1] + 1])
    return sorted(spans), missing


def _answered(entry: Mapping | None) -> bool:
    """Whether a run log entry is an answer that passed the check (not a refusal)."""
    return entry is not None and entry["verdict"]["ok"] and entry["verdict"]["answerable"]


def select(
    questions: Sequence[Question],
    logs: Mapping[str, Mapping],
    quotas: Mapping[str, int] = QUOTAS,
    always: Sequence[str] = ALWAYS,
    seed: str = SEED,
) -> list[str]:
    """The IDs of the questions whose answers in every one of `logs` get labelled, sorted.

    A question qualifies when it is an answerable dev question and every log answered it with a
    reply that passed the check. Each type's quota is filled with the questions in `always`
    first, then drawn from the rest with a seed per type, so changing one quota leaves the
    other types' draws alone.
    """
    entries = [by_id(log) for log in logs.values()]
    qualify = [
        q
        for q in questions
        if q.split == "dev" and q.gold_node_ids and all(_answered(e.get(q.id)) for e in entries)
    ]
    unknown = [qid for qid in always if qid not in {q.id for q in qualify}]
    if unknown:
        raise ValueError(f"{', '.join(unknown)}: not answered by every run, so can't be labelled")
    chosen = []
    for kind, quota in quotas.items():
        pool = sorted(q.id for q in qualify if q.type == kind)
        forced = [qid for qid in pool if qid in always]
        rest = [qid for qid in pool if qid not in always]
        if len(forced) > quota or len(pool) < quota:
            raise ValueError(f"{kind}: {len(pool)} questions qualify for a quota of {quota}")
        chosen += forced + random.Random(f"{seed}:{kind}").sample(rest, quota - len(forced))
    stray = [qid for qid in always if qid not in chosen]
    if stray:
        raise ValueError(f"{', '.join(stray)}: its type has no quota")
    return sorted(chosen)


def assign(question_ids: Sequence[str], runs: Sequence[str], seed: str = SEED) -> dict:
    """Each question's answers as letters, {question_id: {"A": run, "B": run}}, seeded."""
    if len(runs) > len(LETTERS):
        raise ValueError(f"at most {len(LETTERS)} runs, not {len(runs)}")
    out = {}
    for qid in question_ids:
        order = sorted(runs)
        random.Random(f"{seed}:{qid}:order").shuffle(order)
        out[qid] = dict(zip(LETTERS, order, strict=False))
    return out


def answers(question_ids: Sequence[str], logs: Mapping[str, Mapping]) -> dict:
    """The answer text of each selected answer, {(question_id, run): text}."""
    out = {}
    for run, log in logs.items():
        entries = by_id(log)
        for qid in question_ids:
            out[qid, run] = entries[qid]["verdict"]["answer"]
    return out


def validate_labels(text: str, expected: Mapping[tuple[str, str], str]) -> tuple[list, list]:
    """Check a labels file against the answers that should be labelled.

    `expected` maps (question_id, run) to the answer's text, as `answers` gives it. Returns the
    valid labels and one message per problem, naming the line. Every expected answer must be
    labelled once, and nothing else.
    """
    labels, errors = [], []
    seen: dict[tuple[str, str], int] = {}
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        problems = []
        try:
            row = json.loads(line)
        except ValueError as e:
            errors.append(f"line {number}: not JSON: {e}")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {number}: not a JSON object")
            continue
        for name, kind in _FIELDS.items():
            value = row.get(name)
            if name not in row:
                problems.append(f"{name}: missing")
            elif _wrong_type(value, kind):
                problems.append(f"{name}: {value!r} is the wrong type")
        problems += [f"{name}: not a field" for name in row if name not in _FIELDS]
        key = (row.get("question_id"), row.get("run"))
        where = f"line {number} ({key[0]}, {key[1]})"
        if not problems:
            if key not in expected:
                problems.append("not one of the answers chosen for labelling")
            elif key in seen:
                problems.append(f"already labelled on line {seen[key]}")
            else:
                seen[key] = number
                problems += _verdict_problems(row, sentences(expected[key]))
        errors += [f"{where}: {p}" for p in problems]
        if not problems:
            labels.append(Label(**row))
    missing = [k for k in expected if k not in seen]
    if missing and not errors:
        errors.append(f"{len(missing)} of {len(expected)} chosen answers not labelled")
    return labels, errors


def _wrong_type(value: object, kind: type | tuple) -> bool:
    kinds = kind if isinstance(kind, tuple) else (kind,)
    if isinstance(value, bool):  # a bool is an int to isinstance, but not here
        return bool not in kinds
    return not isinstance(value, kinds)


def _verdict_problems(row: Mapping, parts: Sequence[str]) -> list[str]:
    index, sentence = row["first_unsupported"], row["sentence"]
    if row["supported"]:
        if index is not None or sentence is not None:
            return ["a supported answer has no first unsupported sentence"]
        return []
    if index is None or not 0 <= index < len(parts):
        return [f"first_unsupported: needs a sentence number from 0 to {len(parts) - 1}"]
    if sentence != parts[index]:
        return [f"sentence: is not sentence {index} of the answer, {parts[index]!r}"]
    return []
