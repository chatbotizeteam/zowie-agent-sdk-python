.PHONY: install test lint format typecheck check-lint check-format fix check all setup clean

# Setup development environment
setup:
	poetry install
	poetry run pre-commit install
	@echo "Development environment ready!"

# Install dependencies only
install:
	poetry install

# Run tests (real API tests will be skipped if no API keys)
test:
	poetry run pytest tests/ -v

# Run linting
lint:
	poetry run ruff check --fix .

# Format code
format:
	poetry run ruff format .

# Type check
typecheck:
	poetry run mypy src/zowie_agent_sdk tests/

# Fix all auto-fixable issues (format + lint with fixes)
fix: format
	poetry run ruff check --fix .
	@echo "All auto-fixes applied!"

# Run all quality checks (no fixes, just checks)
check-lint:
	poetry run ruff check .

check-format:
	poetry run ruff format --check .

# Run all quality checks (no fixes, just checks)
check: check-lint check-format typecheck test
	@echo "All checks passed!"

# Fix everything then run all checks
all: fix check
	@echo "Code is clean and all tests pass!"

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
