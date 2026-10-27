# eu-transport-rag

Read [`docs/DESIGN.md`](docs/DESIGN.md) first: what the project is, its checkpoints and its hard rules. Then read [`docs/PROGRESS.md`](docs/PROGRESS.md): what is built and verified, and the spend log.

## Hard rules

- No spending without the owner's OK: a dry-run cost estimate first, then a cap the owner approves. Every cost goes in the spend log in `docs/PROGRESS.md`. Issues labelled `paid` wait for that OK.
- No local model download or run without telling the owner what will happen and getting an OK.
- Secrets live only in `.env` or AWS: never committed, never pasted into chat. The gitleaks pre-commit hook stays enabled; never commit with `--no-verify`.
- Public data only: EU legislation and Commission guidance. The repo stays private until the owner has checked and approved a public release.
- `make destroy` after every AWS session, unless the owner decides to keep it running.
- `docs/plan/` holds the owner's private notes and learning notes, and is git-ignored. Nothing from it goes into committed files, commit messages, issues or pull requests; `docs/DESIGN.md` is the design.

## Work tracking

Work items live as GitHub Issues in `johnatomize/eu-transport-rag`, on the [board](https://github.com/users/johnatomize/projects/2). Each checkpoint in the design has a GitHub milestone and one tracking issue labelled `milestone` (#1 for A, #2 for B, #3 for C), and its work items are that issue's sub-issues. "Blocked by #n" at the top of an issue body names its dependencies. Labels: `owner-only`, `decision-needed`, `paid`, and the areas `corpus`, `eval`, `retrieval`, `generation`, `service`, `infra`, `documentation`. Only checkpoints A–C have issues so far; create the next checkpoint's issues from its section in the design when the previous checkpoint is nearly done.

**Who owns what**

- The board owns status: what is open, in progress or done, and what blocks what. Don't track status in Markdown.
- The Markdown owns design: `docs/DESIGN.md` for the design, `docs/DECISIONS.md` for choices and their reasons, and `docs/PROGRESS.md` for what is built and verified and for the spend log. The owner's learning notes live in `docs/plan/LEARNING.md`, which is private and git-ignored like the rest of `docs/plan/`. When the owner decides something, the decision is written into the design or `DECISIONS.md`; the `decision-needed` issue keeps the question and the answer.
- The owner decides scope, spending, model choices and accounts, and reviews every checkpoint. The owner also does everything labelled `owner-only`: accounts, secrets, approvals and checking gold answers. Claude prepares the commands and checklists for these and never asks for secrets in chat.
- Claude implements issues, keeps the issue and its docs in step, and proposes changes to scope or design instead of making them silently. A needed decision becomes an issue labelled `decision-needed`, not a guess.

**Working an issue**

1. Take an issue that is Ready (design clear, dependencies done). If it is vague or blocked, say so on the issue rather than starting.
2. Create a branch for it, make the change, and run `make check`.
3. Open a pull request containing `Fixes #<number>`, so merging closes the issue. Pure documentation changes still follow the Git rule below.
4. Link each issue to the design section it implements, and put acceptance criteria in the issue. Don't copy design text into issues.
5. If work turns up something new (a bug, a follow-up, a design that is wrong), open a new issue and link it, and don't widen the current one.
6. Passing tests alone don't close an `owner-only` issue; the owner confirms it.
7. Each checkpoint ends with its close issue: write the checkpoint's section in the private `docs/plan/LEARNING.md`, check the exit gate on the tracking issue and update `docs/PROGRESS.md`. The owner reviews the section and closes the issue.

**Review and autonomy**

- An agent reviews the code; the owner reviews direction. Before opening a pull request, run `/code-review` on the diff and fix what it finds.
- Every pull request has a short **Direction** section for the owner: which design section it implements, what the change means for the project, and each assumption the design doesn't cover. An assumption that would change direction becomes a `decision-needed` issue instead of shipping.
- For issues that shape later work (node IDs, the question format, schemas, infrastructure), post the plan as a comment on the issue and wait for the owner's OK before writing code.
- Board status: set the issue to In Progress when starting and In review when its pull request opens; merging moves it to Done.
- Merging: the owner merges every pull request until they say otherwise.

Issues and comments may be created, edited and closed by Claude when working an issue. Ask the owner before closing an `owner-only` or `decision-needed` issue, or before deleting any issue.

## Environment

- `gitleaks`, `uv`, `colima` and `docker` are Homebrew installs in `/opt/homebrew/bin`, which may not be on PATH in Claude's shell; prefix commands with `export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"`.
- After cloning, run `make install`: it creates `.venv` from `uv.lock` and installs the pre-commit hook. `make help` lists the other targets; run `make check` before every pull request.
- Docker runs in a Colima VM that is not started at login: run `colima start` before `make db-up`, and `colima stop` when done. Copy `.env.example` to `.env` and set a local `POSTGRES_PASSWORD` first. Homebrew's Compose plugin needs `"cliPluginsExtraDirs": ["/opt/homebrew/lib/docker/cli-plugins"]` in `~/.docker/config.json`.

## Git

Pure documentation changes (Markdown files only, with no code or configuration changes) may be committed straight to `main` and pushed without asking. Everything else goes on a branch unless the owner says otherwise.
