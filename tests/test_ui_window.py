"""Unit tests for WindowAPI methods.

Covers: bind, bind_label, bind_panel, show/hide label grid,
show/hide/toggle state panel (hover + pin), cancel panel hide,
reposition label, position panel.

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


# ── show_state_panel (hover show) ─────────────────────────────────────────


class TestShowStatePanel:
    def test_show_makes_visible(self) -> None:
        """show_state_panel on hidden panel → visible, show() called."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.show_state_panel()

        assert api._panel_visible is True
        assert api._panel_pinned is False
        panel.show.assert_called_once()

    def test_show_idempotent_when_already_visible(self) -> None:
        """Calling show_state_panel twice doesn't call show() again."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.show_state_panel()
        api.show_state_panel()

        panel.show.assert_called_once()

    def test_show_cancels_pending_hide(self) -> None:
        """Re-hovering cancels a pending hide timer."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.show_state_panel()
        api.hide_state_panel()
        assert api._panel_hide_timer is not None

        api.show_state_panel()
        assert api._panel_hide_timer is None

    def test_no_op_without_panel_window(self) -> None:
        api = WindowAPI()
        api.bind(_make_window())
        api.show_state_panel()
        assert api._panel_visible is False

    def test_no_op_without_main_window(self) -> None:
        api = WindowAPI()
        panel = _make_window()
        api.bind_panel(panel)
        api.show_state_panel()
        assert api._panel_visible is False
        panel.show.assert_not_called()


# ── hide_state_panel (hover hide) ────────────────────────────────────────


class TestHideStatePanel:
    def test_schedules_hide_when_not_pinned(self) -> None:
        """hide_state_panel schedules a delayed hide."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.show_state_panel()
        api.hide_state_panel()

        assert api._panel_hide_timer is not None

    def test_no_op_when_pinned(self) -> None:
        """hide_state_panel is a no-op when panel is pinned."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.show_state_panel()
        api.toggle_state_panel()  # pin
        assert api._panel_pinned is True

        api.hide_state_panel()

        assert api._panel_hide_timer is None
        assert api._panel_visible is True

    def test_do_hide_resets_state(self) -> None:
        """_do_hide_panel clears visible and pinned flags."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.show_state_panel()
        api._do_hide_panel()

        assert api._panel_visible is False
        assert api._panel_pinned is False
        panel.hide.assert_called_once()


# ── cancel_panel_hide ────────────────────────────────────────────────────


class TestCancelPanelHide:
    def test_cancels_pending_timer(self) -> None:
        """cancel_panel_hide cancels a scheduled hide."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.show_state_panel()
        api.hide_state_panel()
        assert api._panel_hide_timer is not None

        api.cancel_panel_hide()

        assert api._panel_hide_timer is None
        assert api._panel_visible is True

    def test_no_op_without_pending_timer(self) -> None:
        """cancel_panel_hide with no timer is a safe no-op."""
        api = WindowAPI()
        api.cancel_panel_hide()
        assert api._panel_hide_timer is None


# ── toggle_state_panel (click to pin/unpin) ──────────────────────────────


class TestToggleStatePanel:
    def test_click_from_hidden_shows_and_pins(self) -> None:
        """Click while hidden → show + pin."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.toggle_state_panel()

        assert api._panel_visible is True
        assert api._panel_pinned is True
        panel.show.assert_called_once()

    def test_click_while_hover_visible_pins(self) -> None:
        """Click while hover-visible (not pinned) → pin."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.show_state_panel()
        assert api._panel_visible is True
        assert api._panel_pinned is False

        api.toggle_state_panel()

        assert api._panel_visible is True
        assert api._panel_pinned is True
        panel.hide.assert_not_called()

    def test_click_while_pinned_hides(self) -> None:
        """Click while pinned → unpin + hide."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.toggle_state_panel()  # show + pin
        assert api._panel_pinned is True

        api.toggle_state_panel()  # unpin + hide

        assert api._panel_visible is False
        assert api._panel_pinned is False
        panel.hide.assert_called_once()

    def test_no_op_without_panel_window(self) -> None:
        api = WindowAPI()
        api.bind(_make_window())
        api.toggle_state_panel()
        assert api._panel_visible is False

    def test_no_op_without_main_window(self) -> None:
        api = WindowAPI()
        panel = _make_window()
        api.bind_panel(panel)
        api.toggle_state_panel()
        assert api._panel_visible is False
        panel.show.assert_not_called()


# ── Full state-machine scenarios ─────────────────────────────────────────


class TestPanelStateMachine:
    """End-to-end scenarios simulating the hover/click lifecycle."""

    def _setup(self) -> tuple[WindowAPI, MagicMock]:
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)
        return api, panel

    def test_hover_then_unhover(self) -> None:
        """Hover dot → show; leave dot → scheduled hide → hidden."""
        api, panel = self._setup()

        api.show_state_panel()
        assert api._panel_visible is True

        api.hide_state_panel()
        assert api._panel_hide_timer is not None

        api._do_hide_panel()  # timer fires
        assert api._panel_visible is False
        panel.hide.assert_called_once()

    def test_hover_dot_then_enter_panel(self) -> None:
        """Hover dot → show; leave dot → start hide; enter panel → cancel hide."""
        api, panel = self._setup()

        api.show_state_panel()
        api.hide_state_panel()
        assert api._panel_hide_timer is not None

        api.cancel_panel_hide()  # mouse entered panel window
        assert api._panel_hide_timer is None
        assert api._panel_visible is True
        panel.hide.assert_not_called()

    def test_hover_panel_then_leave_panel(self) -> None:
        """Enter panel (cancel hide); leave panel → schedule hide → hidden."""
        api, panel = self._setup()

        api.show_state_panel()
        api.hide_state_panel()
        api.cancel_panel_hide()

        api.hide_state_panel()  # mouse left panel
        assert api._panel_hide_timer is not None

        api._do_hide_panel()
        assert api._panel_visible is False

    def test_hover_then_click_pins(self) -> None:
        """Hover → visible; click → pinned; unhover → stays visible."""
        api, panel = self._setup()

        api.show_state_panel()
        api.toggle_state_panel()  # pin
        assert api._panel_pinned is True

        api.hide_state_panel()  # unhover — no-op because pinned
        assert api._panel_hide_timer is None
        assert api._panel_visible is True

    def test_pinned_then_click_hides(self) -> None:
        """Pinned panel → click → unpin + immediate hide."""
        api, panel = self._setup()

        api.show_state_panel()
        api.toggle_state_panel()  # pin
        api.toggle_state_panel()  # unpin + hide

        assert api._panel_visible is False
        assert api._panel_pinned is False
        panel.hide.assert_called_once()

    def test_click_without_hover(self) -> None:
        """Click from hidden → show + pin; click again → hide."""
        api, panel = self._setup()

        api.toggle_state_panel()  # show + pin
        assert api._panel_visible is True
        assert api._panel_pinned is True

        api.toggle_state_panel()  # unpin + hide
        assert api._panel_visible is False
        assert api._panel_pinned is False

    def test_unpin_then_hover_again(self) -> None:
        """After unpinning (hidden), hovering shows it again as non-pinned."""
        api, panel = self._setup()

        api.toggle_state_panel()  # pin
        api.toggle_state_panel()  # unpin + hide
        assert api._panel_visible is False

        api.show_state_panel()  # re-hover
        assert api._panel_visible is True
        assert api._panel_pinned is False


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
