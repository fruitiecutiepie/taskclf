"""Local wrapper around the TOML writer dependency."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any, Protocol, cast


class _TomliWriter(Protocol):
    def dumps(
        self,
        obj: Mapping[str, Any],
        /,
        *,
        multiline_strings: bool = False,
        indent: int = 4,
    ) -> str: ...


_TOMLI_W = cast(_TomliWriter, import_module("tomli_w"))


def dumps(
    obj: Mapping[str, Any],
    /,
    *,
    multiline_strings: bool = False,
    indent: int = 4,
) -> str:
    """Serialize a mapping as TOML text via the configured writer backend."""
    return _TOMLI_W.dumps(
        obj,
        multiline_strings=multiline_strings,
        indent=indent,
    )
