# Decisions

Choices made along the way, with the reason and the alternatives considered. Newest first. Open questions are `decision-needed` issues; the answer is recorded here or in the design once the owner decides.

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
