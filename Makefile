.PHONY: install lint test typecheck docs-serve docs-build ci ui-build ui-dev ui-test \
       version bump-patch bump-minor bump-major

PNPM := pnpm --dir src/taskclf/ui/frontend

install:
	uv sync & $(PNPM) install --frozen-lockfile & wait

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
	$(PNPM) install --frozen-lockfile && $(PNPM) run build

ui-dev:
	$(PNPM) run dev

ui-test:
	$(PNPM) run test

ci: lint test ui-test

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

bump-patch:
	$(call bump_version,patch)

bump-minor:
	$(call bump_version,minor)

bump-major:
	$(call bump_version,major)
