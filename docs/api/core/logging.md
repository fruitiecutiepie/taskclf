# core.logging

Sanitizing log filter that redacts sensitive payloads before they reach handlers.
`SanitizingFilter` is implemented as a dataclass-backed `logging.Filter`
with the same filtering behavior.

::: taskclf.core.logging
