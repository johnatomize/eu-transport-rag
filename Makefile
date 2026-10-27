# Developer commands. Needs uv and gitleaks on PATH (brew install uv gitleaks).
# Later checkpoints add targets (corpus, tables, deploy, destroy) and extend check.

.PHONY: help install lint format test secrets check gitleaks-present

help: ## List the targets
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install: gitleaks-present ## Create .venv from uv.lock and install the pre-commit hook
	uv sync --locked
	uv run --locked pre-commit install

lint: ## Ruff lint and format check
	uv run --locked ruff check .
	uv run --locked ruff format --check .

format: ## Apply ruff fixes and formatting
	uv run --locked ruff check --fix .
	uv run --locked ruff format .

test: ## Run the test suite
	uv run --locked pytest

secrets: gitleaks-present ## Scan the full git history for secrets
	gitleaks git --redact --no-banner .

check: lint test secrets ## Everything that must pass before a pull request

gitleaks-present:
	@command -v gitleaks >/dev/null || { echo "gitleaks not found on PATH: brew install gitleaks" >&2; exit 1; }
