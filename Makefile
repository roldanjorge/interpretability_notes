.PHONY: help install lint format fix typecheck test test-all test-cov pre-commit commit bump-patch bump-minor bump-major changelog clean

help:
	@echo "Available targets:"
	@echo "  install      Install all dependencies (uv sync)"
	@echo "  lint         Run ruff lint check (no auto-fix)"
	@echo "  format       Run ruff format check (no auto-fix)"
	@echo "  fix          Run ruff format + lint with auto-fix"
	@echo "  typecheck    Run mypy on src/arena3/"
	@echo "  test         Run fast tests (excludes slow GPT-2 tests)"
	@echo "  test-all     Run all tests including slow GPT-2 tests"
	@echo "  test-cov     Run fast tests with HTML coverage report"
	@echo "  pre-commit   Run all pre-commit hooks on all files"
	@echo "  commit       Interactive commitizen commit (enforces conventional commits)"
	@echo "  bump-patch   Bump patch version (fix: commits)"
	@echo "  bump-minor   Bump minor version (feat: commits)"
	@echo "  bump-major   Bump major version (breaking changes)"
	@echo "  changelog    Generate CHANGELOG.md dry-run"
	@echo "  clean        Remove caches and coverage artifacts"

install:
	uv sync

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format --check src/ tests/

fix:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

typecheck:
	uv run mypy src/arena3/

test:
	uv run pytest tests/ -m "not slow" --tb=short -q

test-all:
	uv run pytest tests/ --tb=short -q

test-cov:
	uv run pytest tests/ -m "not slow" --tb=short --cov=src --cov-report=term-missing --cov-report=html -q

pre-commit:
	uv run pre-commit run --all-files

commit:
	uv run cz commit

bump-patch:
	uv run cz bump --increment PATCH --changelog

bump-minor:
	uv run cz bump --increment MINOR --changelog

bump-major:
	uv run cz bump --increment MAJOR --changelog

changelog:
	uv run cz changelog --dry-run

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml
