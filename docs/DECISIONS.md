# Decisions

Choices made along the way, with the reason and the alternatives considered. Newest first. Open questions are `decision-needed` issues; the answer is recorded here or in the design once the owner decides.

## 2026-10-27: node IDs, and what the parser leaves out

**Decision:** `src/rag/parse.py` gives every node an ID `<document ID>:<path>` (#12, plan approved by the owner on the issue).

- **Articles:** numbered and lettered enumerations go in brackets, the way the acts cite themselves: `561/2006:art6(3)`, `561/2006:art8(6)(a)`, `561/2006:art4(e)`, `165/2014:art34(5)(b)(iii)`, `2002/15:art3(a)(1)`. Chapters are nodes (`561/2006:chII`) but aren't part of article IDs.
- **Annexes:** headings and decimal numbers are joined with dots and lettered points go in brackets: `165/2014:anxI.III.c.4`, `165/2014:anxI.III.a.3.1`, `165/2014:anxI.I(a)`.
- **Not nodes:** subparagraphs and indents (`—`) stay in their parent's text, so "paragraph 2, first subparagraph" resolves to `art8(2)`.
- **Left out:** the 16 recitals of 2002/15. The consolidated acts have none, and recitals aren't binding. They can be added later as `2002/15:recN` without changing any other ID.
- **Flagged:** articles that amend another act (561/2006 Art. 26–27 and 165/2014 Art. 45) are parsed normally, with `amending: true`. Their text holds wording that was later replaced: Art. 45 has the 2014 wording of 561/2006 Art. 3(aa), which the Mobility Package has since replaced. Chunking (checkpoint D) decides whether to index them. The default proposed there is to leave them out.

**Why:** gold answers and citations cite these IDs, so they follow the structure a reader sees in the act and survive changes around them. Leaving chapters out of article IDs means an ID survives a chapter being restructured. Brackets for article enumerations make the IDs read like citations. Annexes are numbered by headings and decimals ("point 4 of Part VI"), so dots fit them better.

**How it is held stable:** `tests/snapshots/node_ids.txt` lists every ID, and a test compares the parse with it, so any change to an ID shows up in a pull request diff. A duplicate ID is an error, never given a silent suffix. Anything the parser doesn't understand is listed by `make parse` and checked against a reviewed list in the tests.

**Also decided:** the parser uses only the standard library (`xml.etree` for the XHTML, `html.parser` for the HTML 4 of 2002/15), so it adds no dependency. 2002/15's flat HTML gets its structure from the labels. One sentence can't be placed that way, so the parser has an explicit override for it, which is safe because the file is pinned by hash.

**Alternatives:** dots throughout (`art8.6.a`), which doesn't read like a citation; subparagraphs as nodes (`art8(6)-2`), which the acts cite rarely and which shift when a subparagraph is inserted; dropping amending articles in the parser, which would hide a choice that retrieval can measure instead.

## 2026-10-27: document IDs, and pinned versions

**Decision:** each document in `corpus/manifest.yaml` has a short ID that prefixes all its node IDs: `561/2006`, `165/2014` and `2002/15` for the acts, and `guidance-1`…`guidance-10` and `clarification-1`…`clarification-7` for the Commission's notes (#11). The notes keep the Commission's own numbering, which has two separate series. IDs carry no version and no language; the manifest pins the version, and the file name carries the language.

**Why:** node IDs end up in the gold answers, which the owner checks by hand, so they must survive changes around them. The Swedish texts (checkpoint L) need the same IDs as the English ones. EU drafting inserts new paragraphs as `8a` rather than renumbering, so most IDs would also survive a newer consolidated version.

**Trade-off:** moving to a newer consolidated version (re-pinning, with `scripts/fetch_corpus.py --pin`) is a deliberate step: compare the parsed versions, then re-check the gold answers whose nodes changed. A re-pin isn't expected during this project, so there is no automatic check for newer versions.

**Alternatives:** the version in the ID (`561/2006@20241231:art6(3)`), which breaks every ID on a re-pin; a type letter (`R561/2006`), which isn't needed because the acts in the design don't collide.

## 2026-10-27: Docker runs in Colima

**Decision:** the local Docker engine is Colima, installed with Homebrew (`brew install colima docker docker-compose`), and isn't started at login (`colima start` when needed).

**Why:** it is free, has no GUI, and installs the same way as the rest of the toolchain. Compose (#7) is all this project needs from Docker until checkpoint H.

**Alternatives:** Docker Desktop, the standard GUI app (free for personal use, heavier).

## 2026-10-27: raw corpus files are committed to git

**Decision:** the documents that `scripts/fetch_corpus.py` downloads are committed under `corpus/raw/` (#9).

**Why:** tests and the CI retrieval quality gate then run offline and are reproducible, and the corpus survives CELLAR changing or dropping a document. Both phases together are about 10–13 MB (measured 2026-10-27: phase 1 about 1–3 MB including the guidance PDFs, phase 2 about 9.5 MB), well below the 50 MB limit set in #9 for committing.

**Trade-off:** every re-pinned consolidated version stays in git history for good. The largest is Reg. 2018/858 at 5.9 MB, so a few updates over the project's life are fine, but frequent re-pinning is not.

**Alternatives:** a GitHub release asset (keeps the repo small, but needs a new release whenever the manifest changes), or not storing the files at all (CI depends on CELLAR being up, and a removed document breaks the rebuild).

## 2026-10-27: gitleaks runs as a local pre-commit hook

**Decision:** `.pre-commit-config.yaml` defines a `repo: local` hook that runs the installed `gitleaks` on staged changes, rather than the hook published in the gitleaks repository.

**Why:** the published hook builds gitleaks from source with Go, which means downloading a Go toolchain on first use. The local hook needs only the `gitleaks` binary and no network when committing.

**Trade-off:** every clone needs gitleaks installed (`brew install gitleaks`), and the version isn't pinned in the repo. CI (checkpoint J) will install a pinned version.
