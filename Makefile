.PHONY: lint test typecheck docs-serve docs-build ci ui-build ui-dev

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

ui-build:
	cd src/taskclf/ui/frontend && npm ci && npm run build

ui-dev:
	cd src/taskclf/ui/frontend && npm run dev

ci: lint test
