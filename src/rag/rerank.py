"""The reranking model, its settings and the scores it gave, cached on disk (design §G).

Since 2026-11-04 the reranker is Qwen3-Reranker 0.6B, run on this machine by `LocalReranker`
(the owner's choice in #77). `rag.retrieve.Rerank` uses its scores to reorder a retriever's best
hits; this module only says how well one passage answers one question.

**How the model scores.** Qwen3-Reranker is a small language model asked a yes-or-no question:
does this document meet the query and the instruction? It reads the instruction, the question
and the passage in the form of its model card (`PREFIX`, `model_input`, `SUFFIX`), and its next
token's logits for "yes" and "no" give the score: their difference, the log-odds of "yes".
Ranking by log-odds is the same as ranking by the probability the model card prints, but the
log-odds don't round to 1.0 for two passages that are both very likely, so they never tie
where the probability would. Pairs go in one at a time, with no padding, so a score never
depends on what it was batched with, and an input over `max_input_tokens` is refused rather
than cut short.

**The cache.** Every score the model gives is kept, so a pair is never scored twice, and the
cache is committed, like the embedding cache, so the retrieval runs and their tables can be
rebuilt without the model. One file per model setting, under `reranks/`:

    reranks/Qwen_Qwen3-Reranker-0.6B-e61197ed4502-mps.jsonl
    {"pair": "<sha256>", "tokens": 212, "score": 3.8125}

- `pair` is the key: the sha256 of the model ID, its revision, the device, the instruction,
  the question and the passage's text, so a change to any of them misses the cache instead of
  returning a stale score. The file name carries the revision and the device too.
- The field is called `pair`, not `key` as in the embedding cache, so that the secret scan's
  generic-API-key rule, which looks for a long random value after a word such as "key", doesn't
  take the hashes for secrets; no exception to the scan is needed.
- Lines are sorted by key and the file is rewritten whole, through a temporary file, so the same
  scores always give the same file.
- `tokens` is the length of the input the model read, prefix and suffix included.
"""

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

SAVE_EVERY = 50  # new scores between cache writes during a run
DEVICES = ("mps", "cpu")
_REVISION = re.compile(r"[0-9a-f]{40}")
_KEY = re.compile(r"[0-9a-f]{64}")
_SETTINGS = {
    "model_id": str,
    "revision": str,
    "device": str,
    "max_input_tokens": int,
    "instruction": str,
}

# The chat the model was trained to answer, from its model card at the pinned revision.
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class RerankError(Exception):
    pass


class MissingScoresError(RerankError):
    """Pairs with no cached score, where the caller may not run the model for new ones."""


@dataclass(frozen=True)
class RerankConfig:
    model_id: str
    revision: str  # the model's commit on Hugging Face
    device: str  # "mps" (the Mac's GPU) or "cpu"
    max_input_tokens: int
    instruction: str


def load_config(path: Path) -> RerankConfig:
    """Read `config/rerank.yaml`, refusing missing, unknown, mistyped or impossible settings."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise RerankError(f"{path.name}: cannot read it ({e.strerror})") from e
    except yaml.YAMLError as e:
        raise RerankError(f"{path.name}: not valid YAML ({e})") from e
    if not isinstance(raw, dict):
        raise RerankError(f"{path.name}: expected a mapping of settings")
    problems = [f"missing {k}" for k in _SETTINGS if k not in raw]
    problems += [f"unknown {k}" for k in raw if k not in _SETTINGS]
    problems += [
        f"{k} should be {kind.__name__}"
        for k, kind in _SETTINGS.items()
        if k in raw and type(raw[k]) is not kind
    ]
    if not problems:
        if not _REVISION.fullmatch(raw["revision"]):
            problems.append("revision should be a full 40-character commit hash")
        if raw["device"] not in DEVICES:
            problems.append(f"device should be one of {DEVICES}")
        if raw["max_input_tokens"] <= 0:
            problems.append("max_input_tokens should be above 0")
        if not raw["instruction"].strip():
            problems.append("instruction should not be empty")
    if problems:
        raise RerankError(f"{path.name}: " + "; ".join(problems))
    return RerankConfig(**raw)


def model_input(config: RerankConfig, query: str, text: str) -> str:
    """The user's turn of the chat: the instruction, the question and the passage."""
    return f"<Instruct>: {config.instruction}\n<Query>: {query}\n<Document>: {text}"


def setting_name(config: RerankConfig) -> str:
    """The model setting as a name: the cache file's stem, and the run logs' `reranker`."""
    model = config.model_id.replace(":", "_").replace("/", "_")
    return f"{model}-{config.revision[:12]}-{config.device}"


def cache_key(config: RerankConfig, query: str, text: str) -> str:
    record = {
        "model": config.model_id,
        "revision": config.revision,
        "device": config.device,
        "instruction": config.instruction,
        "query": query,
        "text": text,
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Entry:
    score: float
    tokens: int


class ScoreCache:
    """The scores of one model setting, read from and written to one file."""

    def __init__(self, path: Path, config: RerankConfig):
        self.path = path
        self.config = config
        self._entries: dict[str, Entry] = {}
        self._dirty = False

    @classmethod
    def load(cls, directory: Path, config: RerankConfig) -> "ScoreCache":
        cache = cls(directory / f"{setting_name(config)}.jsonl", config)
        if not cache.path.exists():
            return cache
        for number, line in enumerate(cache.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
                key, tokens, score = row["pair"], row["tokens"], row["score"]
                if set(row) != {"pair", "tokens", "score"}:
                    raise ValueError("unexpected fields")
                if not (isinstance(key, str) and _KEY.fullmatch(key)):
                    raise ValueError("the key is not a sha256")
                if type(tokens) is not int or tokens < 0:
                    raise ValueError("tokens is not a count")
                if type(score) not in (int, float) or not math.isfinite(score):
                    raise ValueError("the score is not a finite number")
            except (ValueError, KeyError, TypeError) as e:
                raise RerankError(f"{cache.path.name}:{number}: not a cache entry ({e})") from e
            if key in cache._entries:
                raise RerankError(f"{cache.path.name}:{number}: key {key[:12]}… appears twice")
            cache._entries[key] = Entry(float(score), tokens)
        return cache

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, query: str, text: str) -> Entry | None:
        return self._entries.get(cache_key(self.config, query, text))

    def put(self, query: str, text: str, score: float, tokens: int) -> None:
        if not math.isfinite(score):
            raise RerankError("got a score that is not a finite number; not cached")
        self._entries[cache_key(self.config, query, text)] = Entry(float(score), tokens)
        self._dirty = True

    def save(self) -> None:
        """Write the file if anything was added, sorted by key, replacing it in one step."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps({"pair": key, "tokens": e.tokens, "score": e.score})
            for key, e in sorted(self._entries.items())
        ]
        partial = self.path.with_name(self.path.name + ".part")
        partial.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        os.replace(partial, self.path)
        self._dirty = False


class CachedScorer:
    """A `rag.retrieve.Scorer` that reads every score from the cache and never runs the model.

    A pair with no cached score raises `MissingScoresError`, so a reranked run never mixes
    scored and unscored passages; `make rerank` reports what is missing and scores it.
    """

    def __init__(self, cache: ScoreCache):
        self.cache = cache

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        entries = [self.cache.get(query, t) for t in texts]
        missing = sum(1 for e in entries if e is None)
        if missing:
            raise MissingScoresError(
                f"{missing} of {len(texts)} passages have no cached reranker score for this "
                "question; run `make rerank` to see what scoring them would take"
            )
        return [e.score for e in entries if e is not None]


def missing_pairs(pairs: Iterable[tuple[str, str]], cache: ScoreCache) -> list[tuple[str, str]]:
    """The distinct (question, passage) pairs with no cached score, in the order first seen."""
    return [p for p in dict.fromkeys(pairs) if cache.get(*p) is None]


# --- The local model -----------------------------------------------------------------------


class LocalReranker:
    """Qwen3-Reranker run on this machine with transformers. Costs nothing.

    `load` reads the model's files at the configured revision from the Hugging Face cache
    (`~/.cache/huggingface`), downloading them on first use, and runs the model in float32 on
    the configured device; a machine without MPS is refused rather than quietly falling back to
    the CPU. `encode` turns text into token IDs without adding anything, and `run` returns the
    "yes" logit minus the "no" logit for the last position of one input.
    """

    def __init__(
        self,
        config: RerankConfig,
        encode: Callable[[str], list[int]],
        run: Callable[[list[int]], float],
    ):
        self.config = config
        self._encode = encode
        self._run = run
        self._prefix = encode(PREFIX)
        self._suffix = encode(SUFFIX)

    @classmethod
    def load(cls, config: RerankConfig) -> "LocalReranker":
        import torch  # here, so that nothing without a local model needs torch to start
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if config.device == "mps" and not torch.backends.mps.is_available():
            raise RerankError("config/rerank.yaml says device: mps, but this machine has no MPS")
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id, revision=config.revision, dtype=torch.float32
        )
        model.to(config.device)
        model.eval()
        yes, no = (tokenizer.convert_tokens_to_ids(t) for t in ("yes", "no"))
        single = isinstance(yes, int) and isinstance(no, int)
        if not single or tokenizer.unk_token_id in (yes, no):
            raise RerankError(f"{config.model_id}'s tokenizer has no single 'yes' and 'no' tokens")

        # The model card tokenizes with the defaults, which for this tokenizer add nothing; if a
        # library version starts adding tokens, the input would differ from the card's.
        sample = model_input(config, "q", "d")
        if tokenizer(sample)["input_ids"] != tokenizer.encode(sample, add_special_tokens=False):
            raise RerankError("the tokenizer adds tokens by default; check the library version")

        def encode(text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False)

        def run(ids: list[int]) -> float:
            with torch.inference_mode():  # only the last position's logits are needed
                out = model(input_ids=torch.tensor([ids], device=config.device), logits_to_keep=1)
            logits = out.logits[0, -1]
            return float(logits[yes].item()) - float(logits[no].item())

        return cls(config, encode, run)

    def score(self, query: str, text: str) -> tuple[float, int]:
        """The log-odds that `text` answers `query`, and the input's length in tokens."""
        ids = self._prefix + self._encode(model_input(self.config, query, text)) + self._suffix
        if len(ids) > self.config.max_input_tokens:
            raise RerankError(
                f"an input of {len(ids):,} tokens is over the limit of "
                f"{self.config.max_input_tokens:,}; it would be cut short, so it is refused"
            )
        score = self._run(ids)
        if not math.isfinite(score):
            raise RerankError("the model gave a score that is not a finite number")
        return score, len(ids)


def score_missing(
    pairs: Sequence[tuple[str, str]],
    cache: ScoreCache,
    model: LocalReranker,
    progress: Callable[[int, int], None] | None = None,
) -> str | None:
    """Score each pair with `model` and cache it; returns the error that stopped the run, if any.

    The cache is written every `SAVE_EVERY` scores and once more at the end, also when the model
    fails or the run is interrupted, so the work done is kept.
    """
    error = None
    try:
        for done, (query, text) in enumerate(pairs, 1):
            score, tokens = model.score(query, text)
            cache.put(query, text, score, tokens)
            if done % SAVE_EVERY == 0:
                cache.save()
            if progress:
                progress(done, len(pairs))
    except (Exception, KeyboardInterrupt) as e:  # keep what was scored, report the rest
        error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
    try:
        cache.save()
    except OSError as e:
        failed = f"the cache could not be written ({e})"
        error = f"{error}; {failed}" if error else failed
    return error
