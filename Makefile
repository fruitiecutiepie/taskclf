.PHONY: lint test typecheck docs-serve docs-build

lint:
	uv run ruff check .

test:
	uv run pytest

typecheck:
	uv run mypy src

docs-serve:
	uv run --group docs zensical serve

docs-build:
	uv run --group docs zensical build
