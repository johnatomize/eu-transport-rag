"""The corpus manifest, and fetching and checking the raw documents it pins.

`corpus/manifest.yaml` lists every document with the exact version to use and the sha256 of its
file under `corpus/raw/`. The raw files are committed (see `docs/DECISIONS.md`), so checking the
corpus normally needs no network. Documents are downloaded only when a file is missing, when a
document is pinned, or when checking that the corpus can still be rebuilt from its sources.

All HTTP goes through `http_get`; tests give `Fetcher` a fake instead.
"""

import hashlib
import http.client
import re
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import yaml

RAW_DIR = Path("corpus/raw")
KINDS = {"act", "guidance"}
FORMATS = {"xhtml", "html", "pdf"}
REQUIRED = {"id", "title", "kind", "phase", "format", "url", "file", "sha256", "retrieved"}
OPTIONAL = {"celex"}

# Document IDs prefix every node ID ("561/2006:art6(3)"), so they may not contain the colon.
_ID = re.compile(r"[a-z0-9]+(?:[/-][a-z0-9]+)*")
_SHA256 = re.compile(r"[0-9a-f]{64}")

USER_AGENT = "eu-transport-rag/0.1 (corpus fetch)"
# A longer Retry-After than this stops the run instead of silently waiting.
MAX_RETRY_AFTER = 300.0
# CELLAR picks the language from Accept-Language; the Commission's PDF URLs name it themselves.
_HEADERS = {
    "xhtml": {"Accept": "application/xhtml+xml", "Accept-Language": "eng"},
    "html": {"Accept": "text/html", "Accept-Language": "eng"},
    "pdf": {"Accept": "application/pdf"},
}
_CONTENT_TYPES = {"xhtml": "application/xhtml+xml", "html": "text/html", "pdf": "application/pdf"}


class CorpusError(Exception):
    """A manifest, download or hash problem. The message names the document."""


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    kind: str
    phase: int
    format: str
    url: str
    file: str
    sha256: str | None
    retrieved: date | None
    celex: str | None = None

    @property
    def source(self) -> str:
        """What a message names the document by: its CELEX, or its URL if it has none."""
        return self.celex or self.url


def load_manifest(path: Path) -> list[Document]:
    return parse_manifest(path.read_text(encoding="utf-8"))


def parse_manifest(text: str) -> list[Document]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise CorpusError(f"manifest: not valid YAML: {e}") from e
    if not isinstance(data, dict) or data.get("version") != 1:
        raise CorpusError("manifest: expected 'version: 1' at the top")
    entries = data.get("documents")
    if not isinstance(entries, list) or not entries:
        raise CorpusError("manifest: 'documents' must be a non-empty list")

    docs = [_document(i, entry) for i, entry in enumerate(entries)]
    for field in ("id", "file"):
        seen = set()
        for doc in docs:
            value = getattr(doc, field)
            if value in seen:
                raise CorpusError(f"manifest: duplicate {field} {value!r}")
            seen.add(value)
    return docs


def _document(index: int, entry: object) -> Document:
    if not isinstance(entry, dict):
        raise CorpusError(f"manifest: document {index} is not a mapping")
    name = entry.get("id", f"document {index}")

    def fail(message: str) -> CorpusError:
        return CorpusError(f"manifest: {name}: {message}")

    missing = REQUIRED - entry.keys()
    if missing:
        raise fail(f"missing {', '.join(sorted(missing))}")
    unknown = entry.keys() - REQUIRED - OPTIONAL
    if unknown:
        raise fail(f"unknown field {', '.join(sorted(unknown))}")

    doc = Document(**entry)
    if not isinstance(doc.id, str) or not _ID.fullmatch(doc.id):
        raise fail("id must be lowercase letters and digits joined by '/' or '-'")
    if doc.kind not in KINDS:
        raise fail(f"kind must be one of {', '.join(sorted(KINDS))}")
    if doc.format not in FORMATS:
        raise fail(f"format must be one of {', '.join(sorted(FORMATS))}")
    if not isinstance(doc.title, str) or not doc.title:
        raise fail("title must be text")
    if doc.celex is not None and not isinstance(doc.celex, str):
        raise fail("celex must be text")
    if doc.kind == "act" and not doc.celex:
        raise fail("an act needs its celex")
    if type(doc.phase) is not int or doc.phase not in (1, 2):
        raise fail("phase must be 1 or 2")
    # A document is pinned by whatever arrives, so it must arrive over an authenticated channel.
    if not isinstance(doc.url, str) or not doc.url.startswith("https://"):
        raise fail("url must be https")
    if not isinstance(doc.file, str) or Path(doc.file).parent != RAW_DIR:
        raise fail(f"file must be directly inside {RAW_DIR}/")
    if (doc.sha256 is None) != (doc.retrieved is None):
        raise fail("sha256 and retrieved are set together, by --pin")
    if doc.sha256 is not None and not (
        isinstance(doc.sha256, str) and _SHA256.fullmatch(doc.sha256)
    ):
        raise fail("sha256 must be 64 lowercase hex digits, in quotes")
    if doc.retrieved is not None and not isinstance(doc.retrieved, date):
        raise fail("retrieved must be a date (YYYY-MM-DD)")
    return doc


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_hash(doc: Document, data: bytes) -> None:
    actual = sha256_hex(data)
    if actual != doc.sha256:
        raise CorpusError(
            f"{doc.id} ({doc.source}): sha256 mismatch\n"
            f"  expected {doc.sha256}\n"
            f"  actual   {actual}"
        )


@dataclass(frozen=True)
class Response:
    status: int
    content_type: str
    retry_after: str | None
    body: bytes


class _HttpsRedirects(urllib.request.HTTPRedirectHandler):
    """Follow redirects over HTTPS only.

    CELLAR answers an https:// request with a redirect to a plain http:// URL, which it also
    serves over HTTPS with the same content (checked 2026-10-27).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if newurl.startswith("http://"):
            newurl = "https://" + newurl.removeprefix("http://")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_HttpsRedirects)


def http_get(url: str, headers: dict[str, str]) -> Response:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
    try:
        with _opener.open(request, timeout=60) as r:
            if not r.url.startswith("https://"):
                raise CorpusError(f"{url}: ended up at {r.url}, which isn't https")
            return Response(r.status, r.headers.get("Content-Type", ""), None, r.read())
    except urllib.error.HTTPError as e:
        headers = e.headers
        return Response(e.code, headers.get("Content-Type", ""), headers.get("Retry-After"), b"")


class Fetcher:
    """Downloads documents politely: a pause between requests, and retries after HTTP 429."""

    def __init__(
        self,
        get: Callable[[str, dict[str, str]], Response] = http_get,
        sleep: Callable[[float], None] = time.sleep,
        pause: float = 2.0,
        retries: int = 3,
        default_retry_after: float = 30.0,
    ):
        self._get = get
        self._sleep = sleep
        self._pause = pause
        self._retries = retries
        self._default_retry_after = default_retry_after
        self._requests = 0

    def fetch(self, doc: Document) -> bytes:
        for attempt in range(self._retries + 1):
            if self._requests:
                self._sleep(self._pause)
            self._requests += 1
            try:
                response = self._get(doc.url, _HEADERS[doc.format])
            except (OSError, http.client.HTTPException) as e:
                raise CorpusError(f"{doc.id} ({doc.source}): download failed: {e}") from e
            if response.status == 429 and attempt < self._retries:
                delay = self._retry_delay(response.retry_after)
                if delay > MAX_RETRY_AFTER:
                    raise CorpusError(
                        f"{doc.id} ({doc.source}): HTTP 429, asked to wait {delay:.0f} s; try later"
                    )
                self._sleep(delay)
                continue
            if response.status != 200:
                raise CorpusError(f"{doc.id} ({doc.source}): HTTP {response.status} from {doc.url}")
            _check_content(doc, response)
            return response.body
        raise AssertionError("unreachable")

    def _retry_delay(self, retry_after: str | None) -> float:
        # Retry-After may also be an HTTP date; the default covers that rare case.
        if retry_after and retry_after.strip().isdigit():
            return float(retry_after)
        return self._default_retry_after


def _check_content(doc: Document, response: Response) -> None:
    """Refuse anything that isn't the expected format, such as a rate-limit page."""
    expected = _CONTENT_TYPES[doc.format]
    if doc.format == "pdf":
        ok = response.body.startswith(b"%PDF-")
    else:
        ok = response.content_type.startswith(expected) and b"<html" in response.body[:4096].lower()
    if not ok:
        got = response.content_type or "no content type"
        raise CorpusError(f"{doc.id} ({doc.source}): expected {expected}, got {got}")


def ensure(doc: Document, root: Path, fetcher: Fetcher) -> str:
    """Check the raw file against its pinned hash, downloading it first if it is missing.

    Returns "ok" or "fetched". A file that fails the check is never written.
    """
    if doc.sha256 is None:
        raise CorpusError(f"{doc.id}: not pinned yet; run fetch_corpus.py --pin {doc.id}")
    path = root / doc.file
    if path.exists():
        check_hash(doc, path.read_bytes())
        return "ok"
    data = fetcher.fetch(doc)
    check_hash(doc, data)
    _write(path, data)
    return "fetched"


def refetch(
    docs: list[Document], fetcher: Fetcher
) -> tuple[list[tuple[Document, Path]], list[CorpusError]]:
    """Download every document again and compare it with its pinned hash.

    Nothing in the corpus is written. Each download that differs is saved to a new temporary
    directory so it can be compared by hand. Returns the mismatches with those paths, and the
    documents that couldn't be checked; one failure doesn't stop the others being checked.
    """
    mismatches = []
    failures = []
    tmp = None
    for doc in docs:
        if doc.sha256 is None:
            failures.append(CorpusError(f"{doc.id}: not pinned yet; nothing to compare with"))
            continue
        try:
            data = fetcher.fetch(doc)
        except CorpusError as e:
            failures.append(e)
            continue
        if sha256_hex(data) != doc.sha256:
            tmp = tmp or Path(tempfile.mkdtemp(prefix="corpus-refetch-"))
            saved = tmp / Path(doc.file).name
            saved.write_bytes(data)
            mismatches.append((doc, saved))
    return mismatches, failures


def pin(doc_id: str, manifest: Path, root: Path, fetcher: Fetcher, today: date) -> Document:
    """Download one document, write its raw file, and record its sha256 and date in the manifest.

    The manifest is edited as text, so its comments and layout are kept.
    """
    text = manifest.read_text(encoding="utf-8")
    docs = parse_manifest(text)
    doc = next((d for d in docs if d.id == doc_id), None)
    if doc is None:
        raise CorpusError(f"{doc_id}: not in {manifest}")
    data = fetcher.fetch(doc)
    pinned = replace(doc, sha256=sha256_hex(data), retrieved=today)
    new_text = _set_pin(text, pinned)
    # The text edit must change exactly this entry's two fields and nothing else.
    if parse_manifest(new_text) != [pinned if d.id == doc_id else d for d in docs]:
        raise CorpusError(f"{doc_id}: couldn't update the manifest safely; nothing was written")
    _write(root / doc.file, data)
    _write(manifest, new_text.encode("utf-8"))
    return pinned


def _set_pin(text: str, doc: Document) -> str:
    """Rewrite the sha256 and retrieved values in one entry, keeping any trailing comment."""
    lines = text.splitlines(keepends=True)
    entry_start = re.compile(r"\s*- ")
    this_entry = re.compile(rf'\s*- id: (["\']?){re.escape(doc.id)}\1\s*(#.*)?')
    starts = [i for i, line in enumerate(lines) if this_entry.fullmatch(line.rstrip("\n"))]
    if len(starts) != 1:
        raise CorpusError(f"{doc.id}: can't find '- id: {doc.id}'; id must be an entry's first key")
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if entry_start.match(lines[i])), len(lines))
    for key, value in (("sha256", f'"{doc.sha256}"'), ("retrieved", doc.retrieved.isoformat())):
        field = re.compile(rf"(\s*){key}:\s*[^\s#]*(\s+#.*)?")
        hits = [(i, m) for i in range(start, end) if (m := field.fullmatch(lines[i].rstrip("\n")))]
        if len(hits) != 1:
            raise CorpusError(f"{doc.id}: expected one '{key}:' line in its manifest entry")
        i, m = hits[0]
        lines[i] = f"{m.group(1)}{key}: {value}{m.group(2) or ''}\n"
    return "".join(lines)


def _write(path: Path, data: bytes) -> None:
    """Write via a temporary file, so an interrupted run never leaves half a document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
