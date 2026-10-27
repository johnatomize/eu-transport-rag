# Developer commands. Needs uv and gitleaks on PATH (brew install uv gitleaks).
# Later checkpoints add targets (corpus, tables, deploy, destroy) and extend check.

.PHONY: help install lint format test secrets check

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install: ## Create .venv from uv.lock and install the pre-commit hook
	uv sync --locked
	uv tool run pre-commit install

lint: ## Ruff lint and format check
	uv run --locked ruff check .
	uv run --locked ruff format --check .

format: ## Apply ruff fixes and formatting
	uv run --locked ruff check --fix .
	uv run --locked ruff format .

test: ## Run the test suite
	uv run --locked pytest

secrets: ## Scan the full git history for secrets
	gitleaks git --redact --no-banner .

check: lint test secrets ## Everything that must pass before a pull request
