# scripts/

One-off utilities and experiments that are not part of the stable CLI surface area.

Examples:
- bulk export helpers
- quick sanity checks
- migration scripts for schema changes

## Invariants
- Scripts may be messy; keep core logic in `src/taskclf/` and import it.
- If a script becomes important, promote it into a proper CLI command.
