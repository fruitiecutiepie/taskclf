.PHONY: install \
       py-lint py-test py-typecheck \
       ui-lint ui-test ui-typecheck ui-build ui-dev electron-typecheck electron-dist \
       lint test typecheck ci check \
       pyinstaller-build \
       docs-serve docs-build \
       version \
			 bump-patch bump-minor bump-major \
			 bump-launcher-patch bump-launcher-minor bump-launcher-major \
       retag

PNPM := pnpm --dir src/taskclf/ui/frontend

# --- setup ---

install:
	uv sync & $(PNPM) install --frozen-lockfile & pnpm --dir electron install --frozen-lockfile & wait

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

# --- electron ---

electron-typecheck:
	pnpm --dir electron run typecheck

# Same packaging as .github/workflows/electron-release.yml (unsigned; set CSC_IDENTITY_AUTO_DISCOVERY=true to sign locally).
electron-dist:
	CSC_IDENTITY_AUTO_DISCOVERY=false pnpm --dir electron run dist

# --- PyInstaller one-folder sidecar (Electron backend payload) ---

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
# LLVM-style host target triple (e.g. x86_64-unknown-linux-gnu); see scripts/host_target_triple.py
PLATFORM_NAME ?= $(shell python3 $(REPO_ROOT)/scripts/host_target_triple.py 2>/dev/null)
ifeq ($(PLATFORM_NAME),)
	PLATFORM_NAME := $(shell uname -s | tr A-Z a-z)
endif

pyinstaller-build: ui-build
	uv run --group bundle python $(REPO_ROOT)/scripts/payload_build.py

build-payload: pyinstaller-build
	@echo "Payload built at build/payload-$(PLATFORM_NAME).zip"

# --- aggregates ---

lint: py-lint ui-lint

test: py-test ui-test

typecheck: py-typecheck ui-typecheck electron-typecheck

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

define bump_launcher_version
	$(eval NEW_VERSION := $(shell python3 -c "\
import re, pathlib; \
p = pathlib.Path('electron/package.json'); \
data = p.read_text(); \
m = re.search(r'\"version\":\s*\"([^\"]+)\"', data); \
parts = m.group(1).split('.'); \
idx = {'major': 0, 'minor': 1, 'patch': 2}['$(1)']; \
parts[idx] = str(int(parts[idx]) + 1); \
parts[idx+1:] = ['0'] * (2 - idx); \
print('.'.join(parts))"))
	python3 -c "\
import re, pathlib; \
p = pathlib.Path('electron/package.json'); \
p.write_text(re.sub(r'(\"version\":\s*)\"[^\"]+\"', r'\1\"$(NEW_VERSION)\"', p.read_text(), count=1)); \
print('Launcher version -> $(NEW_VERSION)')"
	git add electron/package.json
	git commit -m "bump launcher v$(NEW_VERSION)"
	git tag -a launcher-v$(NEW_VERSION) -m "launcher-v$(NEW_VERSION)"
endef

bump-launcher-patch:
	$(call bump_launcher_version,patch)

bump-launcher-minor:
	$(call bump_launcher_version,minor)

bump-launcher-major:
	$(call bump_launcher_version,major)

define bump_payload_version
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
	$(call bump_payload_version,patch)

bump-minor: check
	$(call bump_payload_version,minor)

bump-major: check
	$(call bump_payload_version,major)

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
