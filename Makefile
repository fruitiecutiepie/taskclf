.PHONY: lint test typecheck docs-serve docs-build ci ui-build ui-dev \
       version bump-patch bump-minor bump-major

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

CURRENT_VERSION := $(shell python3 -c "import re, pathlib; \
	m = re.search(r'^version\s*=\s*\"([^\"]+)\"', pathlib.Path('pyproject.toml').read_text(), re.M); \
	print(m.group(1) if m else '')")

version:
	@echo $(CURRENT_VERSION)

define bump_version
	python3 -c "\
import re, pathlib, sys; \
parts = '$(CURRENT_VERSION)'.split('.'); \
idx = {'major': 0, 'minor': 1, 'patch': 2}['$(1)']; \
parts[idx] = str(int(parts[idx]) + 1); \
parts[idx+1:] = ['0'] * (2 - idx); \
nv = '.'.join(parts); \
p = pathlib.Path('pyproject.toml'); \
p.write_text(re.sub(r'^(version\s*=\s*)\"[^\"]+\"', rf'\1\"{nv}\"', p.read_text(), count=1, flags=re.M)); \
print(f'$(CURRENT_VERSION) -> {nv}')"
endef

bump-patch:
	$(call bump_version,patch)

bump-minor:
	$(call bump_version,minor)

bump-major:
	$(call bump_version,major)
