# scripts/payload_index.py

Build helper for the GitHub Pages `payload-index.json` consumed by the packaged
Electron launcher.

## Role

- Provides one small, testable interface for creating `site/payload-index.json`.
- Treats the **payload release workflow** as the source of truth for payload
  availability on GitHub Pages.
- Lets the **docs deploy workflow** carry forward the currently published
  `payload-index.json` instead of regenerating it from releases and clobbering a
  just-published payload release.
- Falls back to GitHub Releases only when no published payload index exists yet.
- Shares a repository-wide `pages` concurrency group with other Pages publishers
  so docs deploys and payload metadata updates serialize instead of canceling
  each other.

That split avoids the race where a `master` docs deploy and a `v*` payload
release both publish `site/` in close succession.

## Modes

### `generate-from-releases`

Used by `.github/workflows/payload-release.yml`.

- Reads the GitHub Releases API for the repo.
- Includes only published `v*` releases with a `manifest.json` asset.
- Sorts payload versions newest-first.
- Writes a fresh `generated_at` timestamp.
- Also powers the workflow's manual **index refresh only** path for repairing a
  stale GitHub Pages index without rebuilding or republishing payload zips.

### `preserve-current`

Used by `.github/workflows/deploy-docs.yml`.

- Fetches the currently published
  `https://fruitiecutiepie.github.io/taskclf/payload-index.json`.
- Reuses that document when it is valid.
- Falls back to `generate-from-releases` only when the published copy is missing
  or invalid.

## Usage

Generate a fresh index from releases:

```bash
python3 scripts/payload_index.py generate-from-releases \
  --repo fruitiecutiepie/taskclf \
  --output site/payload-index.json
```

Preserve the published copy during docs deployment:

```bash
python3 scripts/payload_index.py preserve-current \
  --repo fruitiecutiepie/taskclf \
  --output site/payload-index.json \
  --current-url https://fruitiecutiepie.github.io/taskclf/payload-index.json
```

## See also

- [`ui/electron_shell`](../ui/electron_shell.md) — launcher manifest and payload
  discovery
- [`scripts/payload_build.py`](payload_build.md) — payload zip build helper
