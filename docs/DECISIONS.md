# Decisions

Choices made along the way, with the reason and the alternatives considered. Newest first. Open questions are `decision-needed` issues; the answer is recorded here or in the design once the owner decides.

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
