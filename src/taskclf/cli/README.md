# cli/

Human-facing command surface area (stable).

## Responsibilities
- Typer app entrypoint
- Commands call pipeline functions (thin wrapper)
- Provide consistent flags and defaults

## Invariants
- CLI should remain backward compatible whenever possible.
- Keep business logic out of CLI; delegate to packages.
