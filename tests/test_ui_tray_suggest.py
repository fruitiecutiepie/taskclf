"""Tests for _LabelSuggester in the tray module.

Covers:
- UID-001: suggest() passes config-backed user_id to build_features_from_aw_events
- UID-003: fallback to default user_id when none is set
- INP-001: suggest() fetches input events and passes them to build_features_from_aw_events
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from taskclf.cli.main import app
from taskclf.features.build import generate_dummy_features

runner = CliRunner()


@pytest.fixture()
def trained_model_dir(tmp_path: Path) -> Path:
    models_dir = tmp_path / "models"
    result = runner.invoke(
        app,
        [
            "train",
            "lgbm",
            "--from",
            "2025-06-14",
            "--to",
            "2025-06-15",
            "--synthetic",
            "--models-dir",
            str(models_dir),
            "--num-boost-round",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    return next(models_dir.iterdir())


class TestLabelSuggesterUserId:
    def test_uid001_suggest_passes_config_user_id(
        self, trained_model_dir: Path
    ) -> None:
        """UID-001: suggest() passes the configured user_id to build_features_from_aw_events."""
        from taskclf.ui.tray import _LabelSuggester

        suggester = _LabelSuggester(trained_model_dir)
        stable_uid = "user-uuid-12345"
        suggester._user_id = stable_uid

        dummy_rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=1)
        start = dt.datetime(2025, 6, 15, 10, 0)
        end = dt.datetime(2025, 6, 15, 10, 1)

        with (
            patch(
                "taskclf.adapters.activitywatch.client.find_window_bucket_id",
                return_value="aw-test-bucket",
            ),
            patch(
                "taskclf.adapters.activitywatch.client.fetch_aw_events",
                return_value=[MagicMock()],
            ),
            patch(
                "taskclf.features.build.build_features_from_aw_events",
                return_value=dummy_rows,
            ) as mock_build,
        ):
            suggester.suggest(start, end)

        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args
        assert call_kwargs.kwargs.get("user_id") == stable_uid

    def test_uid003_default_user_id_fallback(self, trained_model_dir: Path) -> None:
        """UID-003: _user_id defaults to 'default-user' when not overridden."""
        from taskclf.ui.tray import _LabelSuggester

        suggester = _LabelSuggester(trained_model_dir)
        assert suggester._user_id == "default-user"


class TestLabelSuggesterInputEvents:
    def test_inp001_suggest_fetches_and_passes_input_events(
        self, trained_model_dir: Path
    ) -> None:
        """INP-001: suggest() fetches input events and passes them to build_features_from_aw_events."""
        from taskclf.adapters.activitywatch.types import AWInputEvent
        from taskclf.ui.tray import _LabelSuggester

        suggester = _LabelSuggester(trained_model_dir)

        dummy_rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=1)
        start = dt.datetime(2025, 6, 15, 10, 0)
        end = dt.datetime(2025, 6, 15, 10, 1)

        fake_input_events = [
            AWInputEvent(
                timestamp=start,
                duration_seconds=5.0,
                presses=10,
                clicks=2,
                delta_x=100,
                delta_y=50,
                scroll_x=0,
                scroll_y=0,
            )
        ]

        with (
            patch(
                "taskclf.adapters.activitywatch.client.find_window_bucket_id",
                return_value="aw-test-bucket",
            ),
            patch(
                "taskclf.adapters.activitywatch.client.fetch_aw_events",
                return_value=[MagicMock()],
            ),
            patch(
                "taskclf.adapters.activitywatch.client.find_input_bucket_id",
                return_value="aw-test-input",
            ),
            patch(
                "taskclf.adapters.activitywatch.client.fetch_aw_input_events",
                return_value=fake_input_events,
            ) as mock_fetch_input,
            patch(
                "taskclf.features.build.build_features_from_aw_events",
                return_value=dummy_rows,
            ) as mock_build,
        ):
            suggester.suggest(start, end)

        mock_fetch_input.assert_called_once_with(
            suggester._aw_host, "aw-test-input", start, end
        )
        call_kwargs = mock_build.call_args
        assert call_kwargs.kwargs.get("input_events") is fake_input_events
