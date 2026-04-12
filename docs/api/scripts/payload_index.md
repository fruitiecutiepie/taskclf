# scripts/payload_index.py

Build helper for the GitHub Pages `payload-index.json` consumed by the packaged
Electron launcher.

## Role

- Provides one small, testable interface for creating `site/payload-index.json`.
- Treats **GitHub Releases** as the source of truth for payload availability.
- Lets both the **payload release** and **docs deploy** workflows regenerate
  `payload-index.json` from the same published release metadata instead of
  copying whatever GitHub Pages currently serves.
- Makes the **payload release** workflow the only Pages publisher for commits
  tagged `v*`; the ordinary docs deploy skips those release-tagged `master`
  pushes to avoid two GitHub Pages deployments racing on the same commit.
- Shares a repository-wide `pages` concurrency group with other Pages publishers
  so docs deploys and payload metadata updates serialize instead of canceling
  each other.
- After each `zensical build`, both workflows assert `site/index.html` exists
  before uploading the Pages artifact so a broken docs build cannot publish a
  site that 404s at the project URL.

That setup keeps later ordinary `master` docs deploys refreshing metadata from
published releases, while avoiding duplicate Pages deployments on the release
commit itself.

## Modes

### `generate-from-releases`

Used by `.github/workflows/payload-release.yml` and
`.github/workflows/deploy-docs.yml`.

- Reads the GitHub Releases API for the repo.
- Includes only published `v*` releases with a `manifest.json` asset.
- Sorts payload versions newest-first.
- Writes a fresh `generated_at` timestamp.
- Also powers the workflow's manual **index refresh only** path for repairing a
  stale GitHub Pages index without rebuilding or republishing payload zips.

### `preserve-current`

Available as an optional/manual mode when you explicitly want to reuse the
currently published
`https://fruitiecutiepie.github.io/taskclf/payload-index.json`.

- Fetches the current published copy.
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

Preserve the published copy manually:

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
