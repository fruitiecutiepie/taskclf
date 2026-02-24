"""Placeholder tests for features not yet implemented.

Each test is skipped with a reason pointing to the stub module that needs
implementation.  Remove the skip marker once the feature is built.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# TC-CORE-005: raw title opt-in policy (config gating)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: remove .skip once title_policy config is implemented in core/types.py")
def test_tc_core_005_raw_title_opt_in() -> None:
    """TC-CORE-005: allow raw title only when config title_policy=raw_opt_in."""
