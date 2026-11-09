"""Run logs: what one retrieval run returned, kept so that tables never need to run it again.

One JSON file per run, under `runs/`, committed (plan approved on #36):

    {"run":       {"name": "bm25-paragraph", "created": "2026-10-31T09:00:00+00:00",
                   "commit": "6fa9f32…", "dirty": false},
     "config":    {"retriever": "bm25", "k1": 1.5, "b": 0.75, "strategy": "paragraph",
                   "chunks": 412, "mean_chunk_words": 112.3, "chunks_sha256": "<sha256>",
                   "split": "dev", "depth": 50},
     "inputs":    {"data/questions.v1.jsonl": "<sha256>", "corpus/manifest.yaml": "<sha256>"},
     "questions": [{"id": "q001", "ranking": [["paragraph:561/2006:art6(3)", 12.345678], …]}, …],
     "chunks":    {"paragraph:561/2006:art6(3)": ["561/2006:art6(3)"], …},
     "words":     {"paragraph:561/2006:art6(3)": 87, …}}

- `questions` holds every question of the split, unanswerable ones too, each with its top
  `depth` chunks and their scores, best first. Scores are rounded to 6 decimals, so two runs of
  the same code write the same file apart from `run`. A chunk that cross-reference expansion
  brought in has a third entry, the chunk that cites it (`Hit.via`).
- `chunks` holds the node IDs of every chunk that appears in a ranking, so a log can be scored
  again, under any rule, without chunking the corpus again.
- `words` holds the length of every such chunk as indexed, prefix included, so a table can say
  how much text the top k hold.
- `inputs` fingerprints the files the run depended on, and `config.chunks_sha256` the chunks it
  searched, so a table can refuse a log made from other questions or by other parse or chunking
  code.
"""

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from rag.chunk import STRATEGIES, Chunk
from rag.parse import Tree
from rag.questions import Question
from rag.retrieve import Retriever

DEPTH = 50
INPUTS = ("data/questions.v1.jsonl", "corpus/manifest.yaml")


class StaleRunError(Exception):
    """A run log made from other inputs than the ones in the repo now."""


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(root: Path) -> tuple[str | None, bool]:
    """The commit checked out in `root`, and whether anything differs from it."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            # New files count as changes; the logs, tables and cached vectors runs write don't.
            ["git", "status", "--porcelain", "--", ".", ":!runs", ":!results", ":!embeddings"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, False
    return commit, bool(status.strip())


def chunks_fingerprint(chunks: Sequence[Chunk]) -> str:
    """A sha256 of every chunk's ID, node IDs and text, in order.

    It changes whenever the parser or a chunking strategy changes what a run searched, so a
    table can refuse a log made by older code.
    """
    digest = hashlib.sha256()
    for chunk in chunks:
        record = [chunk.id, list(chunk.node_ids), chunk.text]
        digest.update(json.dumps(record, ensure_ascii=False).encode() + b"\n")
    return digest.hexdigest()


def same_results(a: Mapping, b: Mapping) -> bool:
    """Whether two logs hold the same run, apart from when and from which commit it was made."""
    return {k: v for k, v in a.items() if k != "run"} == {k: v for k, v in b.items() if k != "run"}


def make_run(
    name: str,
    config: Mapping[str, object],
    retriever: Retriever,
    chunks: Sequence[Chunk],
    questions: Sequence[Question],
    root: Path,
    split: str = "dev",
    depth: int = DEPTH,
    allow_test: bool = False,
) -> dict:
    """Run `retriever` on every question of `split` and return the run log.

    `config` describes the retriever (name and parameters); the chunk count, mean chunk length,
    split and depth are added here. The test split is run once, for checkpoint G's finalists
    (#86), so running it needs `allow_test`.
    """
    if split not in ("dev", "test"):
        raise ValueError(f"no split {split!r}; the splits are dev and test")
    if split == "test" and not allow_test:
        raise ValueError("the test split is run only for checkpoint G's finalists; pass allow_test")
    if not chunks:
        raise ValueError(f"{name}: no chunks")
    commit, dirty = git_state(root)
    rankings = []
    used: dict[str, list[str]] = {}
    for question in (q for q in questions if q.split == split):
        hits = retriever.search(question.question, depth)
        ranking = [[h.chunk.id, round(h.score, 6), *([h.via] if h.via else [])] for h in hits]
        rankings.append({"id": question.id, "ranking": ranking})
        for hit in hits:
            used.setdefault(hit.chunk.id, list(hit.chunk.node_ids))
    words = {chunk.id: len(chunk.text.split()) for chunk in chunks}
    lengths = list(words.values())
    return {
        "run": {
            "name": name,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "commit": commit,
            "dirty": dirty,
        },
        "config": {
            **config,
            "chunks": len(chunks),
            "mean_chunk_words": round(sum(lengths) / len(lengths), 1),
            "chunks_sha256": chunks_fingerprint(chunks),
            "split": split,
            "depth": depth,
        },
        "inputs": {path: fingerprint(root / path) for path in INPUTS},
        "questions": rankings,
        "chunks": {chunk_id: used[chunk_id] for chunk_id in sorted(used)},
        "words": {chunk_id: words[chunk_id] for chunk_id in sorted(used)},
    }


def dumps(run: Mapping) -> str:
    """The log as JSON, one line per question and per chunk, so that diffs stay readable."""

    def block(items: list[str], opening: str, closing: str) -> str:
        return opening + "\n" + ",\n".join(f"  {item}" for item in items) + "\n " + closing

    def line(value: object) -> str:
        return json.dumps(value, ensure_ascii=False)

    parts = [f' "{key}": {line(run[key])}' for key in ("run", "config", "inputs")]
    questions = [line(q) for q in run["questions"]]
    parts.append(' "questions": ' + block(questions, "[", "]"))
    chunks = [f"{line(cid)}: {line(ids)}" for cid, ids in run["chunks"].items()]
    parts.append(' "chunks": ' + block(chunks, "{", "}"))
    words = [f"{line(cid)}: {n}" for cid, n in run["words"].items()]
    parts.append(' "words": ' + block(words, "{", "}"))
    return "{\n" + ",\n".join(parts) + "\n}\n"


def load(path: Path, root: Path) -> dict:
    """Read a run log, and refuse it if the files it was made from have changed since."""
    run = json.loads(path.read_text(encoding="utf-8"))
    changed = [p for p, digest in run["inputs"].items() if fingerprint(root / p) != digest]
    if changed:
        raise StaleRunError(
            f"{path.name}: made from another version of {', '.join(changed)}; "
            "run it again with `make retrieval-runs`"
        )
    return run


def check_chunks(runs: Sequence[Mapping], tree: Tree) -> None:
    """Refuse a log whose chunks the current parser and chunking code no longer produce."""
    fingerprints: dict[str, str] = {}
    for run in runs:
        strategy = run["config"]["strategy"]
        if strategy not in STRATEGIES:
            raise StaleRunError(f"{run['run']['name']}: no chunking strategy {strategy!r} now")
        if strategy not in fingerprints:
            fingerprints[strategy] = chunks_fingerprint(STRATEGIES[strategy](tree))
        if run["config"]["chunks_sha256"] != fingerprints[strategy]:
            raise StaleRunError(
                f"{run['run']['name']}: its {strategy} chunks differ from what the parser and "
                "chunking code give now; run `make retrieval-runs`"
            )


def rankings(run: Mapping) -> dict[str, list[tuple[str, ...]]]:
    """Each question's ranking as the node IDs of its chunks, as `retrieval_eval` takes it."""
    chunks = run["chunks"]
    return {q["id"]: [tuple(chunks[entry[0]]) for entry in q["ranking"]] for q in run["questions"]}
