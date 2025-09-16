.PHONY: install test lint format check setup

# Setup development environment
setup:
	poetry install
	poetry run pre-commit install
	@echo "✅ Development environment ready!"

# Install dependencies only
install:
	poetry install

# Run tests
test:
	poetry run pytest tests/ -k "not real" -v

# Run linting
lint:
	poetry run ruff check --fix .

# Format code
format:
	poetry run ruff format .

# Type check
typecheck:
	poetry run mypy src/zowie_agent_sdk

# Run all quality checks
check: lint typecheck test
	@echo "✅ All checks passed!"

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
