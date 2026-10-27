# Progress

What is built and verified, as of 27 October 2026. The board holds the status of open work; past evidence is in git history and on the closed issues.

Checkpoint A is in progress.

## Built and verified

- **Repo and tracking:** private GitHub repo, with milestones, tracking issues and sub-issues for checkpoints A–C on the [board](https://github.com/users/johnatomize/projects/2).
- **Secret scanning (#4):** a gitleaks pre-commit hook guards every commit since the first. Before the first commit, a staged fake token was refused.
- **Python tooling (#5):** a `uv` project on Python 3.12 with ruff and pytest, locked in `uv.lock`. `make check` runs lint, tests and a gitleaks scan of the full history, and passes on a fresh clone with only `uv` and gitleaks installed.
- **Local Postgres + pgvector (#7):** `docker-compose.yml` runs `pgvector/pgvector` 0.8.7 on Postgres 18, pinned by tag and digest, with credentials from `.env` and the port bound to localhost only. `make db-up` waits until it is healthy, `make db-check` runs `CREATE EXTENSION vector` and prints `pgvector 0.8.7`, and data survives `make db-down` / `make db-up`. Docker runs in Colima.

## Spend log

Every cost goes here: date, what it paid for, amount and running total. A paid step needs the owner's OK and a dry-run estimate first.

Approved cap: none yet (checkpoint E proposes US$20 for the whole project).

| Date | Item | Cost (US$) | Total (US$) |
| --- | --- | ---: | ---: |
| 2026-10-27 | Project start | 0.00 | 0.00 |
