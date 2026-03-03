"""Unit tests for WindowAPI methods.

Covers: bind, bind_label, bind_panel, show/hide label grid,
toggle state panel, reposition label, position panel.

All tests use mock window objects — no real GUI required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from taskclf.ui.window import WindowAPI


def _make_window(x: int = 200, y: int = 100) -> MagicMock:
    """Create a mock window with x, y, show, hide, move."""
    win = MagicMock()
    win.x = x
    win.y = y
    return win


# ── TC-UI-WIN-001: show_label_grid with bound label window ────────────────


class TestShowLabelGrid:
    def test_label_visible_and_show_called(self) -> None:
        """TC-UI-WIN-001: show_label_grid sets _label_visible and calls show()."""
        api = WindowAPI()
        main = _make_window()
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.show_label_grid()

        assert api._label_visible is True
        label.show.assert_called_once()

    def test_no_op_without_label_window(self) -> None:
        """TC-UI-WIN-002: show_label_grid with no label window is a no-op."""
        api = WindowAPI()
        main = _make_window()
        api.bind(main)

        api.show_label_grid()

        assert api._label_visible is False

    def test_no_op_without_main_window(self) -> None:
        """show_label_grid with no main window is a no-op."""
        api = WindowAPI()
        label = _make_window()
        api.bind_label(label)

        api.show_label_grid()

        assert api._label_visible is False
        label.show.assert_not_called()


# ── TC-UI-WIN-003: hide_label_grid ────────────────────────────────────────


class TestHideLabelGrid:
    def test_label_hidden_after_timer(self) -> None:
        """TC-UI-WIN-003: hide_label_grid schedules delayed hide."""
        api = WindowAPI()
        main = _make_window()
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.show_label_grid()
        assert api._label_visible is True

        api._do_hide_label()

        assert api._label_visible is False
        label.hide.assert_called_once()


# ── TC-UI-WIN-004/005: toggle_state_panel ─────────────────────────────────


class TestToggleStatePanel:
    def test_show_panel(self) -> None:
        """TC-UI-WIN-004: toggle hidden panel → visible, show() called."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.toggle_state_panel()

        assert api._panel_visible is True
        panel.show.assert_called_once()

    def test_hide_panel(self) -> None:
        """TC-UI-WIN-005: toggle visible panel → schedules hide."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.toggle_state_panel()
        assert api._panel_visible is True

        api._do_hide_panel()

        assert api._panel_visible is False
        panel.hide.assert_called_once()

    def test_no_op_without_panel_window(self) -> None:
        """toggle_state_panel with no panel window is a no-op."""
        api = WindowAPI()
        main = _make_window()
        api.bind(main)

        api.toggle_state_panel()

        assert api._panel_visible is False

    def test_no_op_without_main_window(self) -> None:
        """toggle_state_panel with no main window is a no-op."""
        api = WindowAPI()
        panel = _make_window()
        api.bind_panel(panel)

        api.toggle_state_panel()

        assert api._panel_visible is False
        panel.show.assert_not_called()


# ── TC-UI-WIN-006: _reposition_label ──────────────────────────────────────


class TestRepositionLabel:
    def test_label_positioned_below_pill(self) -> None:
        """TC-UI-WIN-006: label window move() with correct coordinates."""
        api = WindowAPI()
        main = _make_window(x=200, y=100)
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.show_label_grid()

        label.move.assert_called_once_with(
            200 + 150 - 280,  # x - 130
            100 + 30 + 4,     # y + 34
        )


# ── TC-UI-WIN-007/008: _position_panel ────────────────────────────────────


class TestPositionPanel:
    def test_panel_below_pill_label_hidden(self) -> None:
        """TC-UI-WIN-007: panel positioned below pill when label is hidden."""
        api = WindowAPI()
        main = _make_window(x=200, y=100)
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)
        assert api._label_visible is False

        api.toggle_state_panel()

        panel.move.assert_called_once_with(
            200 + 150 - 280,  # x - 130
            100 + 30 + 4,     # y + 34
        )

    def test_panel_below_label_when_visible(self) -> None:
        """TC-UI-WIN-008: panel positioned below label grid when it is visible."""
        api = WindowAPI()
        main = _make_window(x=200, y=100)
        label = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_label(label)
        api.bind_panel(panel)

        api.show_label_grid()
        api.toggle_state_panel()

        panel.move.assert_called_once_with(
            200 + 150 - 280,          # x - 130
            100 + 30 + 4 + 330 + 4,   # y + 368
        )


# ── TC-UI-WIN-009/010/011: bind methods ──────────────────────────────────


class TestBind:
    def test_bind_sets_window_and_event(self) -> None:
        """TC-UI-WIN-009: bind sets _window and subscribes to moved event."""
        api = WindowAPI()
        win = _make_window()
        moved_mock = win.events.moved

        api.bind(win)

        assert api._window is win
        moved_mock.__iadd__.assert_called_once_with(api._on_main_window_moved)

    def test_bind_label_sets_label_window(self) -> None:
        """TC-UI-WIN-010: bind_label sets _label_window."""
        api = WindowAPI()
        label = _make_window()

        api.bind_label(label)

        assert api._label_window is label

    def test_bind_panel_sets_panel_window(self) -> None:
        """TC-UI-WIN-011: bind_panel sets _panel_window."""
        api = WindowAPI()
        panel = _make_window()

        api.bind_panel(panel)

        assert api._panel_window is panel
