.PHONY: install \
       py-lint py-test py-typecheck \
       ui-lint ui-test ui-typecheck ui-build ui-dev \
       lint test typecheck ci check \
       nuitka-build \
       docs-serve docs-build \
       version bump-patch bump-minor bump-major \
       retag

PNPM := pnpm --dir src/taskclf/ui/frontend

# --- setup ---

install:
	uv sync & $(PNPM) install --frozen-lockfile & wait

# --- python ---

py-lint:
	uv run ruff check .

py-test:
	uv run pytest

py-typecheck:
	uv run mypy src

py-format:
	uv run ruff format

py-build: ui-build
	uv build

# --- ui ---

ui-lint:
	$(PNPM) run lint

ui-lint-fix:
	$(PNPM) run lint:fix

ui-test:
	$(PNPM) run test

ui-typecheck:
	$(PNPM) run typecheck

ui-format:
	$(PNPM) run format

ui-build:
	$(PNPM) install --frozen-lockfile && $(PNPM) run build

ui-dev:
	$(PNPM) run dev

# --- nuitka ---

nuitka-build: ui-build
	uv run --group bundle python -m nuitka \
		--standalone \
		--output-dir=build/nuitka \
		--include-package=taskclf \
		--include-package=lightgbm \
		--include-package=pandas \
		--include-package=duckdb \
		--include-package=sklearn \
		--include-package=scipy \
		--include-package=pydantic \
		--include-package=uvicorn \
		--include-package=fastapi \
		--include-package=typer \
		--include-package=rich \
		--include-package=yaml \
		--include-package=pyarrow \
		--include-package=PIL \
		--include-package=pystray \
		--include-data-dir=src/taskclf/ui/static=taskclf/ui/static \
		--python-flag=no_site \
		--macos-create-app-bundle \
		src/taskclf/cli/entry.py

# --- aggregates ---

lint: py-lint ui-lint

test: py-test ui-test

typecheck: py-typecheck ui-typecheck

format: py-format ui-format

build: py-build ui-build

check: lint test typecheck

ci: check build

# --- docs ---

docs-serve:
	uv run --group docs zensical serve

docs-build:
	uv run --group docs zensical build

# --- versioning ---

CURRENT_VERSION := $(shell python3 -c "import re, pathlib; \
	m = re.search(r'^version\s*=\s*\"([^\"]+)\"', pathlib.Path('pyproject.toml').read_text(), re.M); \
	print(m.group(1) if m else '')")

version:
	@echo $(CURRENT_VERSION)

define bump_version
	$(eval NEW_VERSION := $(shell python3 -c "\
import re, pathlib; \
parts = '$(CURRENT_VERSION)'.split('.'); \
idx = {'major': 0, 'minor': 1, 'patch': 2}['$(1)']; \
parts[idx] = str(int(parts[idx]) + 1); \
parts[idx+1:] = ['0'] * (2 - idx); \
print('.'.join(parts))"))
	python3 -c "\
import re, pathlib; \
p = pathlib.Path('pyproject.toml'); \
p.write_text(re.sub(r'^(version\s*=\s*)\"[^\"]+\"', r'\1\"$(NEW_VERSION)\"', p.read_text(), count=1, flags=re.M)); \
print('$(CURRENT_VERSION) -> $(NEW_VERSION)')"
	uv lock
	git add pyproject.toml uv.lock
	git commit -m "bump v$(NEW_VERSION)"
	git tag -a v$(NEW_VERSION) -m "v$(NEW_VERSION)"
endef

bump-patch: check
	$(call bump_version,patch)

bump-minor: check
	$(call bump_version,minor)

bump-major: check
	$(call bump_version,major)

retag: check
	@if [ -n "$$(git log origin/HEAD..HEAD --oneline)" ]; then \
		echo "WARNING: Local commits not pushed to remote."; \
		printf "Push now and continue? [y/N] "; \
		read ans; \
		case "$$ans" in [yY]*) git push ;; *) echo "Aborted."; exit 1 ;; esac; \
	fi
	git tag -d v$(CURRENT_VERSION)
	git push origin :refs/tags/v$(CURRENT_VERSION)
	git tag -a v$(CURRENT_VERSION) -m "v$(CURRENT_VERSION)"
	git push origin v$(CURRENT_VERSION)
