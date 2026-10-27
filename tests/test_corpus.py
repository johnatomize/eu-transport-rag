import urllib.error
import urllib.request
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from rag.corpus import (
    CorpusError,
    Fetcher,
    Response,
    _HttpsRedirects,
    ensure,
    load_manifest,
    parse_manifest,
    pin,
    refetch,
    sha256_hex,
)

ROOT = Path(__file__).parents[1]
PDF = b"%PDF-1.7 a guidance note"
XHTML = b'<?xml version="1.0"?>\n<html xmlns="http://www.w3.org/1999/xhtml">an act</html>'

MANIFEST = f"""\
# A comment that --pin must keep
version: 1
documents:
  - id: "561/2006"
    title: "Regulation (EC) No 561/2006"
    kind: act
    phase: 1
    celex: 02006R0561-20241231
    format: xhtml
    url: https://cellar.test/02006R0561-20241231
    file: corpus/raw/02006R0561-20241231.en.xhtml
    sha256: "{sha256_hex(XHTML)}"
    retrieved: 2026-10-27
  - id: guidance-1
    title: "Guidance note 1"
    kind: guidance
    phase: 1
    format: pdf
    url: https://commission.test/guidance_1_en.pdf
    file: corpus/raw/guidance-1.en.pdf
    sha256: null
    retrieved: null
"""


class FakeGet:
    """Stands in for HTTP: replies from a queue per URL and records every request."""

    def __init__(self, replies: dict[str, list[Response]] | None = None):
        self.replies = replies or {}
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str]) -> Response:
        self.calls.append((url, headers))
        return self.replies[url].pop(0)


def ok(body: bytes, content_type: str = "application/pdf") -> Response:
    return Response(200, content_type, None, body)


def fetcher(get: FakeGet, sleeps: list[float] | None = None) -> Fetcher:
    return Fetcher(get=get, sleep=(sleeps if sleeps is not None else []).append)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "manifest.yaml").write_text(MANIFEST)
    return tmp_path


def docs(root: Path):
    return {d.id: d for d in load_manifest(root / "corpus" / "manifest.yaml")}


def write_raw(root: Path, name: str, data: bytes) -> Path:
    path = root / "corpus" / "raw" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# The committed corpus


def test_committed_manifest_is_valid_and_fully_pinned():
    manifest = load_manifest(ROOT / "corpus" / "manifest.yaml")
    assert all(d.sha256 and d.retrieved for d in manifest)


def test_committed_raw_files_match_their_hashes_without_network():
    offline = Fetcher(get=lambda url, headers: pytest.fail(f"network call to {url}"))
    for doc in load_manifest(ROOT / "corpus" / "manifest.yaml"):
        assert ensure(doc, ROOT, offline) == "ok"


# Checking and fetching


def test_matching_file_is_ok_without_network(root):
    write_raw(root, "02006R0561-20241231.en.xhtml", XHTML)
    get = FakeGet()
    assert ensure(docs(root)["561/2006"], root, fetcher(get)) == "ok"
    assert get.calls == []


def test_mismatch_names_document_celex_and_both_hashes(root):
    path = write_raw(root, "02006R0561-20241231.en.xhtml", b"<html>changed</html>")
    with pytest.raises(CorpusError) as e:
        ensure(docs(root)["561/2006"], root, fetcher(FakeGet()))
    message = str(e.value)
    assert "561/2006" in message and "02006R0561-20241231" in message
    assert sha256_hex(XHTML) in message and sha256_hex(b"<html>changed</html>") in message
    assert path.read_bytes() == b"<html>changed</html>"


def test_missing_file_is_fetched_checked_and_written(root):
    doc = docs(root)["561/2006"]
    get = FakeGet({doc.url: [ok(XHTML, "application/xhtml+xml;charset=UTF-8")]})
    assert ensure(doc, root, fetcher(get)) == "fetched"
    assert (root / doc.file).read_bytes() == XHTML
    assert get.calls[0][1] == {"Accept": "application/xhtml+xml", "Accept-Language": "eng"}


def test_missing_file_with_wrong_hash_is_not_written(root):
    doc = docs(root)["561/2006"]
    get = FakeGet({doc.url: [ok(XHTML + b"x", "application/xhtml+xml")]})
    with pytest.raises(CorpusError, match="sha256 mismatch"):
        ensure(doc, root, fetcher(get))
    assert not (root / doc.file).exists()


def test_unpinned_document_asks_for_pin(root):
    with pytest.raises(CorpusError, match="--pin guidance-1"):
        ensure(docs(root)["guidance-1"], root, fetcher(FakeGet()))


def test_retries_after_429_using_retry_after(root):
    doc = docs(root)["guidance-1"]
    get = FakeGet({doc.url: [Response(429, "text/html", "7", b""), ok(PDF)]})
    sleeps: list[float] = []
    assert fetcher(get, sleeps).fetch(doc) == PDF
    assert sleeps == [7.0, 2.0]  # the Retry-After wait, then the usual pause before the retry


def test_gives_up_after_three_retries(root):
    doc = docs(root)["guidance-1"]
    get = FakeGet({doc.url: [Response(429, "text/html", None, b"")] * 4})
    with pytest.raises(CorpusError, match="HTTP 429"):
        fetcher(get).fetch(doc)
    assert len(get.calls) == 4


def test_too_long_retry_after_stops_instead_of_waiting(root):
    doc = docs(root)["guidance-1"]
    get = FakeGet({doc.url: [Response(429, "text/html", "86400", b"")]})
    sleeps: list[float] = []
    with pytest.raises(CorpusError, match="asked to wait 86400 s"):
        fetcher(get, sleeps).fetch(doc)
    assert sleeps == []


def test_network_error_names_the_document(root):
    doc = docs(root)["guidance-1"]

    def unreachable(url, headers):
        raise urllib.error.URLError("nodename nor servname provided")

    with pytest.raises(CorpusError, match="guidance-1 .*download failed: .*nodename"):
        Fetcher(get=unreachable).fetch(doc)


def test_redirects_to_plain_http_are_followed_over_https():
    request = urllib.request.Request("https://publications.europa.eu/resource/celex/32002L0015")
    new = _HttpsRedirects().redirect_request(
        request, None, 303, "See Other", {}, "http://publications.europa.eu/resource/cellar/x/DOC_1"
    )
    assert new.full_url == "https://publications.europa.eu/resource/cellar/x/DOC_1"


def test_html_page_where_pdf_expected_is_rejected(root):
    doc = docs(root)["guidance-1"]
    get = FakeGet({doc.url: [ok(b"<html>Too many requests</html>", "text/html")]})
    with pytest.raises(CorpusError, match="expected application/pdf"):
        fetcher(get).fetch(doc)


def test_pdf_where_xhtml_expected_is_rejected(root):
    doc = docs(root)["561/2006"]
    get = FakeGet({doc.url: [ok(PDF)]})
    with pytest.raises(CorpusError, match="expected application/xhtml"):
        fetcher(get).fetch(doc)


def test_pauses_between_downloads(root):
    d = docs(root)
    get = FakeGet(
        {d["561/2006"].url: [ok(XHTML, "application/xhtml+xml")], d["guidance-1"].url: [ok(PDF)]}
    )
    sleeps: list[float] = []
    f = fetcher(get, sleeps)
    f.fetch(d["561/2006"])
    f.fetch(d["guidance-1"])
    assert sleeps == [2.0]


# Pinning and refetching


def test_pin_writes_hash_date_and_file_and_keeps_the_rest(root):
    manifest = root / "corpus" / "manifest.yaml"
    doc = docs(root)["guidance-1"]
    get = FakeGet({doc.url: [ok(PDF)]})
    pinned = pin("guidance-1", manifest, root, fetcher(get), date(2026, 11, 2))
    assert pinned.sha256 == sha256_hex(PDF) and pinned.retrieved == date(2026, 11, 2)
    assert (root / doc.file).read_bytes() == PDF
    expected = MANIFEST.replace(
        "    sha256: null\n    retrieved: null\n",
        f'    sha256: "{sha256_hex(PDF)}"\n    retrieved: 2026-11-02\n',
    )
    assert manifest.read_text() == expected


def test_pin_handles_quotes_and_keeps_trailing_comments(root):
    manifest = root / "corpus" / "manifest.yaml"
    text = MANIFEST.replace("id: guidance-1", "id: 'guidance-1'  # added later").replace(
        "    retrieved: null\n", "    retrieved: null  # pinned by hand\n"
    )
    manifest.write_text(text)
    doc = docs(root)["guidance-1"]
    pin("guidance-1", manifest, root, fetcher(FakeGet({doc.url: [ok(PDF)]})), date(2026, 11, 2))
    assert "    retrieved: 2026-11-02  # pinned by hand\n" in manifest.read_text()
    assert docs(root)["guidance-1"].sha256 == sha256_hex(PDF)


def test_pin_refuses_entry_whose_id_is_not_first(root):
    manifest = root / "corpus" / "manifest.yaml"
    manifest.write_text(
        MANIFEST.replace(
            '  - id: guidance-1\n    title: "Guidance note 1"\n',
            '  - title: "Guidance note 1"\n    id: guidance-1\n',
        )
    )
    doc = docs(root)["guidance-1"]
    with pytest.raises(CorpusError, match="first key"):
        pin("guidance-1", manifest, root, fetcher(FakeGet({doc.url: [ok(PDF)]})), date.today())
    assert not (root / doc.file).exists()


def test_pin_unknown_id_fails(root):
    with pytest.raises(CorpusError, match="not in"):
        pin("guidance-9", root / "corpus" / "manifest.yaml", root, fetcher(FakeGet()), date.today())


def test_refetch_reports_and_saves_changed_downloads_only(root):
    doc = docs(root)["561/2006"]
    raw = write_raw(root, "02006R0561-20241231.en.xhtml", XHTML)
    new = XHTML.replace(b"an act", b"an amended act")
    get = FakeGet({doc.url: [ok(new, "application/xhtml+xml")]})
    mismatches, failures = refetch([doc], fetcher(get))
    assert [(d.id, saved.read_bytes()) for d, saved in mismatches] == [("561/2006", new)]
    assert failures == []
    assert raw.read_bytes() == XHTML


def test_refetch_matching_reports_nothing(root):
    doc = docs(root)["561/2006"]
    get = FakeGet({doc.url: [ok(XHTML, "application/xhtml+xml")]})
    assert refetch([doc], fetcher(get)) == ([], [])


def test_refetch_keeps_going_after_a_failure(root):
    d = docs(root)
    act = d["561/2006"]
    pinned_note = pin_in_memory(d["guidance-1"])
    get = FakeGet(
        {
            pinned_note.url: [Response(500, "text/html", None, b"")],
            act.url: [ok(XHTML + b"changed", "application/xhtml+xml")],
        }
    )
    mismatches, failures = refetch([pinned_note, act], fetcher(get))
    assert [doc.id for doc, _ in mismatches] == ["561/2006"]
    assert [str(e) for e in failures] == [
        f"guidance-1 ({pinned_note.url}): HTTP 500 from {pinned_note.url}"
    ]


def test_refetch_reports_unpinned_document_as_a_failure(root):
    get = FakeGet()
    mismatches, failures = refetch([docs(root)["guidance-1"]], fetcher(get))
    assert mismatches == [] and "not pinned yet" in str(failures[0])
    assert get.calls == []


def pin_in_memory(doc):
    return replace(doc, sha256=sha256_hex(PDF), retrieved=date(2026, 10, 27))


# Manifest validation


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        ("id: guidance-1", 'id: "561/2006"', "duplicate id"),
        ("kind: guidance", "kind: note", "kind must be"),
        ("format: pdf", "format: docx", "format must be"),
        ("file: corpus/raw/guidance-1.en.pdf", "file: corpus/guidance-1.en.pdf", "directly inside"),
        ("file: corpus/raw/guidance-1.en.pdf", "file: corpus/raw/../x.pdf", "directly inside"),
        ("id: guidance-1", 'id: "guidance:1"', "id must be"),
        ("    celex: 02006R0561-20241231\n", "", "needs its celex"),
        ('sha256: "' + sha256_hex(XHTML), 'sha256: "ABC', "64 lowercase hex"),
        ("sha256: null", 'sha256: "' + "0" * 64 + '"', "set together"),
        ("kind: guidance\n", "kind: guidance\n    language: en\n", "unknown field"),
        ('    title: "Guidance note 1"\n', "", "missing title"),
        ("version: 1", "version: 2", "version: 1"),
        ("url: https://commission.test/", "url: http://commission.test/", "url must be https"),
        ("kind: guidance\n    phase: 1", "kind: guidance\n    phase: true", "phase must be"),
        ('sha256: "' + sha256_hex(XHTML) + '"', "sha256: " + "1" * 64, "in quotes"),
        ('title: "Guidance note 1"', "title: 7", "title must be text"),
        ("kind: act", "kind: act: x", "not valid YAML"),
    ],
)
def test_manifest_validation(old, new, error):
    assert old in MANIFEST
    with pytest.raises(CorpusError, match=error):
        parse_manifest(MANIFEST.replace(old, new, 1))
