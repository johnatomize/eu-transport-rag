# Decisions

Choices made along the way, with the reason and the alternatives considered. Newest first. Open questions are `decision-needed` issues; the answer is recorded here or in the design once the owner decides.

## 2026-10-27: gitleaks runs as a local pre-commit hook

**Decision:** `.pre-commit-config.yaml` defines a `repo: local` hook that runs the installed `gitleaks` on staged changes, rather than the hook published in the gitleaks repository.

**Why:** the published hook builds gitleaks from source with Go, which means downloading a Go toolchain on first use. The local hook needs only the `gitleaks` binary and no network when committing.

**Trade-off:** every clone needs gitleaks installed (`brew install gitleaks`), and the version isn't pinned in the repo. CI (checkpoint J) will install a pinned version.
