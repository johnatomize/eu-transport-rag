"""Vectors for chunks and questions, from Titan Text Embeddings V2, cached on disk (design §E).

Every vector Bedrock returns is kept in the cache, so the same text is never paid for twice, and
the cache is committed (decided by the owner on 2026-10-31), so the tables and CI's quality gate
can use the vectors without AWS. One file per model setting, under `embeddings/`:

    embeddings/amazon.titan-embed-text-v2_0-1024-normalized.jsonl
    {"key": "<sha256>", "tokens": 37, "vector": "<base64 of 1024 little-endian float32>"}

- The key is the sha256 of the model ID, the vector size, the normalisation and the text, so a
  change to any of them misses the cache instead of returning a stale vector.
- Lines are sorted by key and the file is rewritten whole, through a temporary file, so the same
  vectors always give the same file, and an interrupted write never leaves half a line.
- `tokens` is the count Bedrock returned when the vector was paid for.
- Vectors are stored as float32, the precision Titan computes in, as base64: about 5.5 KB a
  vector, against about 20 KB as JSON numbers.

Nothing here calls AWS unless it is handed a client; `BedrockClient` is the only one that does.
"""

import array
import base64
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rag.embed_estimate import (
    MARGIN,
    EmbeddingConfig,
    batch,
    cost_usd,
    estimate_tokens,
    input_problems,
)

MAX_ATTEMPTS = 6  # per call, the first try included; botocore backs off between them
SAVE_EVERY = 50  # new vectors between cache writes during a paid run
_KEY = re.compile(r"[0-9a-f]{64}")


class EmbedError(Exception):
    pass


class OverCapError(EmbedError):
    """A paid run whose estimate would take the spend log past the approved cap."""


class MissingVectorsError(EmbedError):
    """Texts with no cached vector, where the caller may not pay for new ones."""


def cache_key(text: str, config: EmbeddingConfig) -> str:
    record = {
        "model": config.model_id,
        "dimensions": config.dimensions,
        "normalize": config.normalize,
        "text": text,
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def setting_name(config: EmbeddingConfig) -> str:
    """The model setting as a name: the cache file's stem, and the vector index's `model`."""
    model = config.model_id.replace(":", "_").replace("/", "_")
    norm = "-normalized" if config.normalize else ""
    return f"{model}-{config.dimensions}{norm}"


def cache_path(directory: Path, config: EmbeddingConfig) -> Path:
    return directory / f"{setting_name(config)}.jsonl"


@dataclass(frozen=True)
class Entry:
    vector: tuple[float, ...]
    tokens: int


def _encode(vector: Sequence[float]) -> str:
    values = array.array("f", vector)
    if sys.byteorder != "little":
        values.byteswap()
    return base64.b64encode(values.tobytes()).decode("ascii")


def _decode(text: str) -> tuple[float, ...]:
    values = array.array("f")
    values.frombytes(base64.b64decode(text, validate=True))
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(values)


class EmbeddingCache:
    """The vectors of one model setting, read from and written to one file."""

    def __init__(self, path: Path, config: EmbeddingConfig):
        self.path = path
        self.config = config
        self._entries: dict[str, Entry] = {}
        self._dirty = False

    @classmethod
    def load(cls, directory: Path, config: EmbeddingConfig) -> "EmbeddingCache":
        cache = cls(cache_path(directory, config), config)
        if not cache.path.exists():
            return cache
        for number, line in enumerate(cache.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
                key, tokens, vector = row["key"], row["tokens"], _decode(row["vector"])
                if not (isinstance(key, str) and _KEY.fullmatch(key)):
                    raise ValueError("the key is not a sha256")
                if type(tokens) is not int or tokens < 0:
                    raise ValueError("tokens is not a count")
                if not all(math.isfinite(v) for v in vector):
                    raise ValueError("the vector holds a value that is not a finite number")
            except (ValueError, KeyError, TypeError) as e:
                raise EmbedError(f"{cache.path.name}:{number}: not a cache entry ({e})") from e
            if len(vector) != config.dimensions:
                raise EmbedError(
                    f"{cache.path.name}:{number}: {len(vector)} dimensions, "
                    f"expected {config.dimensions}"
                )
            if key in cache._entries:
                raise EmbedError(f"{cache.path.name}:{number}: key {key[:12]}… appears twice")
            cache._entries[key] = Entry(vector, tokens)
        return cache

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, text: str) -> Entry | None:
        return self._entries.get(cache_key(text, self.config))

    def put(self, text: str, vector: Sequence[float], tokens: int) -> None:
        if len(vector) != self.config.dimensions:
            raise EmbedError(
                f"got {len(vector)} dimensions, expected {self.config.dimensions}; not cached"
            )
        if not all(math.isfinite(v) for v in vector):
            raise EmbedError("got a vector with a non-finite value; not cached")
        stored = _decode(_encode(vector))  # what a reload will return
        self._entries[cache_key(text, self.config)] = Entry(stored, tokens)
        self._dirty = True

    def save(self) -> None:
        """Write the file if anything was added, sorted by key, replacing it in one step."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps({"key": key, "tokens": e.tokens, "vector": _encode(e.vector)})
            for key, e in sorted(self._entries.items())
        ]
        partial = self.path.with_name(self.path.name + ".part")
        partial.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        os.replace(partial, self.path)
        self._dirty = False


# --- Clients -------------------------------------------------------------------------------


class EmbeddingClient(Protocol):
    def embed(self, text: str) -> tuple[list[float], int]:
        """The vector for `text` and the input tokens it was billed for. Raises on failure."""
        ...


class BedrockClient:
    """Titan Text Embeddings V2 through Bedrock's InvokeModel. Each call is paid for.

    Credentials come from the AWS CLI profile (`aws login`). Throttling and other passing
    errors are retried by botocore with exponential backoff, at most `MAX_ATTEMPTS` tries a call;
    anything still failing raises, and the caller caches nothing for that text.
    """

    def __init__(self, config: EmbeddingConfig, profile: str | None = None):
        import boto3  # here, so that nothing without a client needs boto3 to start
        from botocore.config import Config

        self.config = config
        session = boto3.Session(profile_name=profile, region_name=config.region)
        if session.get_credentials() is None:
            raise EmbedError(
                f"no AWS credentials for profile {profile or '(default)'}; "
                f"sign in with: aws login --profile {profile or 'default'}"
            )
        self._client = session.client(
            "bedrock-runtime",
            config=Config(
                retries={"total_max_attempts": MAX_ATTEMPTS, "mode": "standard"},
                connect_timeout=10,
                read_timeout=60,
            ),
        )

    def embed(self, text: str) -> tuple[list[float], int]:
        body = {
            "inputText": text,
            "dimensions": self.config.dimensions,
            "normalize": self.config.normalize,
        }
        response = self._client.invoke_model(
            modelId=self.config.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        data = json.loads(response["body"].read())
        vector, tokens = data.get("embedding"), data.get("inputTextTokenCount")
        if not isinstance(vector, list) or not isinstance(tokens, int):
            raise EmbedError(f"unexpected response from {self.config.model_id}: {sorted(data)}")
        return vector, tokens


# --- Planning and paying -------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """What embedding some texts would send to Bedrock, given what is already cached."""

    distinct: int
    cached: int
    missing: tuple[str, ...]  # distinct texts with no vector yet, in first-seen order
    tokens: int  # estimated tokens of `missing`, before the margin
    largest: int  # estimated tokens of the longest missing text
    config: EmbeddingConfig

    @property
    def padded_tokens(self) -> int:
        return math.ceil(self.tokens * MARGIN)

    @property
    def cost_usd(self) -> float:
        """The estimate a paid run is held to: the margin included."""
        return cost_usd(self.padded_tokens, self.config)


def distinct(texts: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(texts))


def plan(texts: Iterable[str], cache: EmbeddingCache) -> Plan:
    unique = distinct(texts)
    missing = tuple(t for t in unique if cache.get(t) is None)
    counts = [estimate_tokens(t) for t in missing]
    return Plan(
        distinct=len(unique),
        cached=len(unique) - len(missing),
        missing=missing,
        tokens=sum(counts),
        largest=max(counts, default=0),
        config=cache.config,
    )


def check_inputs(p: Plan) -> list[str]:
    """Why the missing texts can't be sent, if they can't: Bedrock refuses empty or long input."""
    return input_problems(batch("texts to send", p.missing), p.config)


def check_cap(p: Plan, spent_usd: float, cap_usd: float) -> None:
    """Refuse a run whose estimate, on top of what the spend log has, would pass the cap."""
    if spent_usd + p.cost_usd > cap_usd:
        raise OverCapError(
            f"this run is estimated at US${p.cost_usd:.5f} (margin included); with "
            f"US${spent_usd:.2f} already spent that is over the approved cap of US${cap_usd:.2f}"
        )


@dataclass
class Paid:
    """What a paid run sent, call by call, for the run log and the spend log."""

    calls: list[dict] = field(default_factory=list)  # {"key": …, "tokens": …}, in call order
    error: str | None = None

    @property
    def tokens(self) -> int:
        return sum(c["tokens"] for c in self.calls)


def pay(
    p: Plan,
    cache: EmbeddingCache,
    client: EmbeddingClient,
    progress: Callable[[int, int], None] | None = None,
) -> Paid:
    """Send every missing text of plan `p` to `client` and cache what comes back.

    The cache is written every `SAVE_EVERY` vectors and once more at the end, also when a call
    fails or the run is interrupted, so what was paid for is kept. A failed call caches nothing
    for its text and stops the run. Every call Bedrock answered is in the result, with its tokens,
    even when its vector was refused; so is the error, and the caller decides what to do about it.
    """
    paid = Paid()
    try:
        for done, text in enumerate(p.missing, 1):
            vector, tokens = client.embed(text)
            # Billed from here on, so logged before anything can refuse the vector.
            paid.calls.append({"key": cache_key(text, cache.config), "tokens": tokens})
            cache.put(text, vector, tokens)
            if done % SAVE_EVERY == 0:
                cache.save()
            if progress:
                progress(done, len(p.missing))
    except (Exception, KeyboardInterrupt) as e:  # keep what was paid for, report the rest
        paid.error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
    try:
        cache.save()
    except OSError as e:
        failed = f"the cache could not be written ({e})"
        paid.error = f"{paid.error}; {failed}" if paid.error else failed
    return paid


def vectors(texts: Sequence[str], cache: EmbeddingCache) -> list[tuple[float, ...]]:
    """The cached vector of every text, in order, without calling anything.

    Raises `MissingVectorsError` when any text has none, so that a search never silently runs on
    part of the corpus; `make embed` reports what is missing and what sending it would cost.
    """
    entries = [cache.get(t) for t in texts]
    missing = sum(1 for e in entries if e is None)
    if missing:
        raise MissingVectorsError(
            f"{missing} of {len(texts)} texts have no cached vector for {cache.config.model_id}; "
            "run `make embed` to see what sending them would cost"
        )
    return [e.vector for e in entries if e is not None]


def corpus_texts(root: Path) -> dict[str, list[str]]:
    """Every text E embeds, by batch name: each chunking's chunks, and the questions of both splits.

    G embeds the test questions too, and embedding a question is not scoring it.
    """
    from rag.chunk import STRATEGIES
    from rag.corpus import load_manifest
    from rag.parse import parse_corpus
    from rag.questions import validate

    tree = parse_corpus(load_manifest(root / "corpus" / "manifest.yaml"), root)
    questions_file = root / "data" / "questions.v1.jsonl"
    questions, errors = validate(questions_file.read_text(encoding="utf-8"), tree)
    if errors:
        raise EmbedError(f"{questions_file.name} is not valid; run `make questions`")
    texts = {f"chunks: {name}": [c.text for c in chunk(tree)] for name, chunk in STRATEGIES.items()}
    texts["questions"] = [q.question for q in questions]
    return texts
