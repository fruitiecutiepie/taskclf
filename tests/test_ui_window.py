"""Unit tests for WindowAPI methods.

Covers: bind, bind_label, bind_panel,
show/hide/toggle label grid (hover + pin), cancel label hide,
show/hide/toggle state panel (hover + pin), cancel panel hide,
reposition label, position panel.

All tests use mock window objects — no real GUI required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from taskclf.ui.window import WindowAPI


def _make_window(x: int = 200, y: int = 100) -> MagicMock:
    """Create a mock window with x, y, show, hide, move.

    The mock's x/y update when move() is called, so position-checking
    logic in WindowAPI (e.g. user-drag detection) works correctly.
    """
    win = MagicMock()
    win.x = x
    win.y = y

    def _track_move(new_x: int, new_y: int) -> None:
        win.x = new_x
        win.y = new_y

    win.move = MagicMock(side_effect=_track_move)
    return win


# ── label_grid_show (hover show) ──────────────────────────────────────────


class TestShowLabelGrid:
    def test_show_makes_visible(self) -> None:
        """label_grid_show on hidden label → visible, show() called."""
        api = WindowAPI()
        main = _make_window()
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_show()

        assert api._label.visible is True
        assert api._label.pinned is False
        label.show.assert_called_once()

    def test_show_idempotent_when_already_visible(self) -> None:
        """Calling label_grid_show twice doesn't call show() again."""
        api = WindowAPI()
        main = _make_window()
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_show()
        api.label_grid_show()

        label.show.assert_called_once()

    def test_show_cancels_pending_hide(self) -> None:
        """Re-hovering cancels a pending hide timer."""
        api = WindowAPI()
        main = _make_window()
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_show()
        api.label_grid_hide()
        assert api._label.hide_timer is not None

        api.label_grid_show()
        assert api._label.hide_timer is None

    def test_no_op_without_label_window(self) -> None:
        api = WindowAPI()
        api.bind(_make_window())
        api.label_grid_show()
        assert api._label.visible is False

    def test_no_op_without_main_window(self) -> None:
        api = WindowAPI()
        label = _make_window()
        api.bind_label(label)
        api.label_grid_show()
        assert api._label.visible is False
        label.show.assert_not_called()


# ── label_grid_hide (hover hide) ─────────────────────────────────────────


class TestHideLabelGrid:
    def test_schedules_hide_when_not_pinned(self) -> None:
        """label_grid_hide schedules a delayed hide."""
        api = WindowAPI()
        main = _make_window()
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_show()
        api.label_grid_hide()

        assert api._label.hide_timer is not None

    def test_no_op_when_pinned(self) -> None:
        """label_grid_hide is a no-op when label is pinned."""
        api = WindowAPI()
        main = _make_window()
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_show()
        api.label_grid_toggle()  # pin
        assert api._label.pinned is True

        api.label_grid_hide()

        assert api._label.hide_timer is None
        assert api._label.visible is True

    def test_visibility_off_resets_state(self) -> None:
        """visibility_off clears visible and pinned flags."""
        api = WindowAPI()
        main = _make_window()
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_show()
        api._label.visibility_off()

        assert api._label.visible is False
        assert api._label.pinned is False
        label.hide.assert_called_once()


# ── label_grid_cancel_hide ────────────────────────────────────────────────────


class TestCancelLabelHide:
    def test_cancels_pending_timer(self) -> None:
        """label_grid_cancel_hide cancels a scheduled hide."""
        api = WindowAPI()
        main = _make_window()
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_show()
        api.label_grid_hide()
        assert api._label.hide_timer is not None

        api.label_grid_cancel_hide()

        assert api._label.hide_timer is None
        assert api._label.visible is True

    def test_no_op_without_pending_timer(self) -> None:
        """label_grid_cancel_hide with no timer is a safe no-op."""
        api = WindowAPI()
        api.label_grid_cancel_hide()
        assert api._label.hide_timer is None


# ── label_grid_toggle (click to pin/unpin) ───────────────────────────────


class TestToggleLabelGrid:
    def test_click_from_hidden_shows_and_pins(self) -> None:
        """Click while hidden → show + pin."""
        api = WindowAPI()
        main = _make_window()
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_toggle()

        assert api._label.visible is True
        assert api._label.pinned is True
        label.show.assert_called_once()

    def test_click_while_hover_visible_pins(self) -> None:
        """Click while hover-visible (not pinned) → pin."""
        api = WindowAPI()
        main = _make_window()
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_show()
        assert api._label.visible is True
        assert api._label.pinned is False

        api.label_grid_toggle()

        assert api._label.visible is True
        assert api._label.pinned is True
        label.hide.assert_not_called()

    def test_click_while_pinned_hides(self) -> None:
        """Click while pinned → unpin + hide."""
        api = WindowAPI()
        main = _make_window()
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_toggle()  # show + pin
        assert api._label.pinned is True

        api.label_grid_toggle()  # unpin + hide

        assert api._label.visible is False
        assert api._label.pinned is False
        label.hide.assert_called_once()

    def test_no_op_without_label_window(self) -> None:
        api = WindowAPI()
        api.bind(_make_window())
        api.label_grid_toggle()
        assert api._label.visible is False

    def test_no_op_without_main_window(self) -> None:
        api = WindowAPI()
        label = _make_window()
        api.bind_label(label)
        api.label_grid_toggle()
        assert api._label.visible is False
        label.show.assert_not_called()


# ── Full label state-machine scenarios ───────────────────────────────────


class TestLabelStateMachine:
    """End-to-end scenarios simulating the label grid hover/click lifecycle."""

    def _setup(self) -> tuple[WindowAPI, MagicMock]:
        api = WindowAPI()
        main = _make_window()
        label = _make_window()
        api.bind(main)
        api.bind_label(label)
        return api, label

    def test_hover_then_unhover(self) -> None:
        """Hover badge → show; leave badge → scheduled hide → hidden."""
        api, label = self._setup()

        api.label_grid_show()
        assert api._label.visible is True

        api.label_grid_hide()
        assert api._label.hide_timer is not None

        api._label.visibility_off()
        assert api._label.visible is False
        label.hide.assert_called_once()

    def test_hover_badge_then_enter_label(self) -> None:
        """Hover badge → show; leave badge → start hide; enter label → cancel."""
        api, label = self._setup()

        api.label_grid_show()
        api.label_grid_hide()
        assert api._label.hide_timer is not None

        api.label_grid_cancel_hide()
        assert api._label.hide_timer is None
        assert api._label.visible is True
        label.hide.assert_not_called()

    def test_hover_label_then_leave_label(self) -> None:
        """Enter label (cancel hide); leave label → schedule hide → hidden."""
        api, label = self._setup()

        api.label_grid_show()
        api.label_grid_hide()
        api.label_grid_cancel_hide()

        api.label_grid_hide()
        assert api._label.hide_timer is not None

        api._label.visibility_off()
        assert api._label.visible is False

    def test_hover_then_click_pins(self) -> None:
        """Hover → visible; click → pinned; unhover → stays visible."""
        api, label = self._setup()

        api.label_grid_show()
        api.label_grid_toggle()  # pin
        assert api._label.pinned is True

        api.label_grid_hide()  # unhover — no-op because pinned
        assert api._label.hide_timer is None
        assert api._label.visible is True

    def test_pinned_then_click_hides(self) -> None:
        """Pinned label → click → unpin + immediate hide."""
        api, label = self._setup()

        api.label_grid_show()
        api.label_grid_toggle()  # pin
        api.label_grid_toggle()  # unpin + hide

        assert api._label.visible is False
        assert api._label.pinned is False
        label.hide.assert_called_once()

    def test_click_without_hover(self) -> None:
        """Click from hidden → show + pin; click again → hide."""
        api, label = self._setup()

        api.label_grid_toggle()
        assert api._label.visible is True
        assert api._label.pinned is True

        api.label_grid_toggle()
        assert api._label.visible is False
        assert api._label.pinned is False

    def test_unpin_then_hover_again(self) -> None:
        """After unpinning (hidden), hovering shows it again as non-pinned."""
        api, label = self._setup()

        api.label_grid_toggle()  # pin
        api.label_grid_toggle()  # unpin + hide
        assert api._label.visible is False

        api.label_grid_show()  # re-hover
        assert api._label.visible is True
        assert api._label.pinned is False


# ── state_panel_show (hover show) ─────────────────────────────────────────


class TestShowStatePanel:
    def test_show_makes_visible(self) -> None:
        """state_panel_show on hidden panel → visible, show() called."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.state_panel_show()

        assert api._panel.visible is True
        assert api._panel.pinned is False
        panel.show.assert_called_once()

    def test_show_idempotent_when_already_visible(self) -> None:
        """Calling state_panel_show twice doesn't call show() again."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.state_panel_show()
        api.state_panel_show()

        panel.show.assert_called_once()

    def test_show_cancels_pending_hide(self) -> None:
        """Re-hovering cancels a pending hide timer."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.state_panel_show()
        api.state_panel_hide()
        assert api._panel.hide_timer is not None

        api.state_panel_show()
        assert api._panel.hide_timer is None

    def test_no_op_without_panel_window(self) -> None:
        api = WindowAPI()
        api.bind(_make_window())
        api.state_panel_show()
        assert api._panel.visible is False

    def test_no_op_without_main_window(self) -> None:
        api = WindowAPI()
        panel = _make_window()
        api.bind_panel(panel)
        api.state_panel_show()
        assert api._panel.visible is False
        panel.show.assert_not_called()


# ── state_panel_hide (hover hide) ────────────────────────────────────────


class TestHideStatePanel:
    def test_schedules_hide_when_not_pinned(self) -> None:
        """state_panel_hide schedules a delayed hide."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.state_panel_show()
        api.state_panel_hide()

        assert api._panel.hide_timer is not None

    def test_no_op_when_pinned(self) -> None:
        """state_panel_hide is a no-op when panel is pinned."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.state_panel_show()
        api.state_panel_toggle()  # pin
        assert api._panel.pinned is True

        api.state_panel_hide()

        assert api._panel.hide_timer is None
        assert api._panel.visible is True

    def test_visibility_off_resets_state(self) -> None:
        """visibility_off clears visible and pinned flags."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.state_panel_show()
        api._panel.visibility_off()

        assert api._panel.visible is False
        assert api._panel.pinned is False
        panel.hide.assert_called_once()


# ── state_panel_cancel_hide ────────────────────────────────────────────────────


class TestCancelPanelHide:
    def test_cancels_pending_timer(self) -> None:
        """state_panel_cancel_hide cancels a scheduled hide."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.state_panel_show()
        api.state_panel_hide()
        assert api._panel.hide_timer is not None

        api.state_panel_cancel_hide()

        assert api._panel.hide_timer is None
        assert api._panel.visible is True

    def test_no_op_without_pending_timer(self) -> None:
        """state_panel_cancel_hide with no timer is a safe no-op."""
        api = WindowAPI()
        api.state_panel_cancel_hide()
        assert api._panel.hide_timer is None


# ── state_panel_toggle (click to pin/unpin) ──────────────────────────────


class TestToggleStatePanel:
    def test_click_from_hidden_shows_and_pins(self) -> None:
        """Click while hidden → show + pin."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.state_panel_toggle()

        assert api._panel.visible is True
        assert api._panel.pinned is True
        panel.show.assert_called_once()

    def test_click_while_hover_visible_pins(self) -> None:
        """Click while hover-visible (not pinned) → pin."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.state_panel_show()
        assert api._panel.visible is True
        assert api._panel.pinned is False

        api.state_panel_toggle()

        assert api._panel.visible is True
        assert api._panel.pinned is True
        panel.hide.assert_not_called()

    def test_click_while_pinned_hides(self) -> None:
        """Click while pinned → unpin + hide."""
        api = WindowAPI()
        main = _make_window()
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.state_panel_toggle()  # show + pin
        assert api._panel.pinned is True

        api.state_panel_toggle()  # unpin + hide

        assert api._panel.visible is False
        assert api._panel.pinned is False
        panel.hide.assert_called_once()

    def test_no_op_without_panel_window(self) -> None:
        api = WindowAPI()
        api.bind(_make_window())
        api.state_panel_toggle()
        assert api._panel.visible is False

    def test_no_op_without_main_window(self) -> None:
        api = WindowAPI()
        panel = _make_window()
        api.bind_panel(panel)
        api.state_panel_toggle()
        assert api._panel.visible is False
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

        api.state_panel_show()
        assert api._panel.visible is True

        api.state_panel_hide()
        assert api._panel.hide_timer is not None

        api._panel.visibility_off()  # timer fires
        assert api._panel.visible is False
        panel.hide.assert_called_once()

    def test_hover_dot_then_enter_panel(self) -> None:
        """Hover dot → show; leave dot → start hide; enter panel → cancel hide."""
        api, panel = self._setup()

        api.state_panel_show()
        api.state_panel_hide()
        assert api._panel.hide_timer is not None

        api.state_panel_cancel_hide()  # mouse entered panel window
        assert api._panel.hide_timer is None
        assert api._panel.visible is True
        panel.hide.assert_not_called()

    def test_hover_panel_then_leave_panel(self) -> None:
        """Enter panel (cancel hide); leave panel → schedule hide → hidden."""
        api, panel = self._setup()

        api.state_panel_show()
        api.state_panel_hide()
        api.state_panel_cancel_hide()

        api.state_panel_hide()  # mouse left panel
        assert api._panel.hide_timer is not None

        api._panel.visibility_off()
        assert api._panel.visible is False

    def test_hover_then_click_pins(self) -> None:
        """Hover → visible; click → pinned; unhover → stays visible."""
        api, panel = self._setup()

        api.state_panel_show()
        api.state_panel_toggle()  # pin
        assert api._panel.pinned is True

        api.state_panel_hide()  # unhover — no-op because pinned
        assert api._panel.hide_timer is None
        assert api._panel.visible is True

    def test_pinned_then_click_hides(self) -> None:
        """Pinned panel → click → unpin + immediate hide."""
        api, panel = self._setup()

        api.state_panel_show()
        api.state_panel_toggle()  # pin
        api.state_panel_toggle()  # unpin + hide

        assert api._panel.visible is False
        assert api._panel.pinned is False
        panel.hide.assert_called_once()

    def test_click_without_hover(self) -> None:
        """Click from hidden → show + pin; click again → hide."""
        api, panel = self._setup()

        api.state_panel_toggle()  # show + pin
        assert api._panel.visible is True
        assert api._panel.pinned is True

        api.state_panel_toggle()  # unpin + hide
        assert api._panel.visible is False
        assert api._panel.pinned is False

    def test_unpin_then_hover_again(self) -> None:
        """After unpinning (hidden), hovering shows it again as non-pinned."""
        api, panel = self._setup()

        api.state_panel_toggle()  # pin
        api.state_panel_toggle()  # unpin + hide
        assert api._panel.visible is False

        api.state_panel_show()  # re-hover
        assert api._panel.visible is True
        assert api._panel.pinned is False


# ── TC-UI-WIN-006: _reposition_label ──────────────────────────────────────


class TestRepositionLabel:
    def test_label_positioned_below_pill(self) -> None:
        """TC-UI-WIN-006: label window move() with correct coordinates."""
        api = WindowAPI()
        main = _make_window(x=200, y=100)
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_show()

        label.move.assert_called_once_with(
            200 + 150 - 280,  # x - 130
            100 + 30 + 4,  # y + 34
        )

    def test_expected_pos_set_after_reposition(self) -> None:
        """Programmatic move records expected position for drag detection."""
        api = WindowAPI()
        main = _make_window(x=200, y=100)
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_show()

        assert api._label.expected_pos == (70, 134)

    def test_expected_pos_cleared_on_hide(self) -> None:
        """Hiding the label resets expected position for next show."""
        api = WindowAPI()
        main = _make_window(x=200, y=100)
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_show()
        assert api._label.expected_pos is not None

        api._label.visibility_off()
        assert api._label.expected_pos is None

    def test_label_follows_main_when_not_user_dragged(self) -> None:
        """Label repositions when main window moves and label hasn't been dragged."""
        api = WindowAPI()
        main = _make_window(x=200, y=100)
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_show()
        assert label.move.call_count == 1

        main.x = 300
        api._on_main_window_moved()

        assert label.move.call_count == 2
        label.move.assert_called_with(300 + 150 - 280, 100 + 30 + 4)

    def test_label_not_repositioned_after_user_drag(self) -> None:
        """After user drags label away, main window move does not snap it back."""
        api = WindowAPI()
        main = _make_window(x=200, y=100)
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_show()
        assert label.move.call_count == 1

        label.x = 500
        label.y = 500

        main.x = 300
        api._on_main_window_moved()

        assert label.move.call_count == 1

    def test_label_reanchors_after_hide_show_cycle(self) -> None:
        """After hide+show, label re-anchors to pill (expected_pos was reset)."""
        api = WindowAPI()
        main = _make_window(x=200, y=100)
        label = _make_window()
        api.bind(main)
        api.bind_label(label)

        api.label_grid_show()
        label.x = 500
        label.y = 500
        api._label.visibility_off()

        main.x = 400
        api.label_grid_show()

        label.move.assert_called_with(400 + 150 - 280, 100 + 30 + 4)


# ── TC-UI-WIN-007/008: _position_panel ────────────────────────────────────


class TestPositionPanel:
    def test_panel_below_pill_label_hidden(self) -> None:
        """TC-UI-WIN-007: panel positioned below pill when label is hidden."""
        api = WindowAPI()
        main = _make_window(x=200, y=100)
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)
        assert api._label.visible is False

        api.state_panel_toggle()

        panel.move.assert_called_once_with(
            200 + 150 - 280,  # x - 130
            100 + 30 + 4,  # y + 34
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

        api.label_grid_show()
        api.state_panel_toggle()

        panel.move.assert_called_once_with(
            200 + 150 - 280,  # x - 130
            100 + 30 + 4 + 330 + 4,  # y + 368
        )

    def test_panel_expected_pos_set(self) -> None:
        """Panel records expected position after programmatic move."""
        api = WindowAPI()
        main = _make_window(x=200, y=100)
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.state_panel_toggle()

        assert api._panel.expected_pos == (70, 134)

    def test_panel_expected_pos_cleared_on_hide(self) -> None:
        """Hiding the panel resets expected position."""
        api = WindowAPI()
        main = _make_window(x=200, y=100)
        panel = _make_window()
        api.bind(main)
        api.bind_panel(panel)

        api.state_panel_toggle()
        assert api._panel.expected_pos is not None

        api._panel.visibility_off()
        assert api._panel.expected_pos is None


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
        """TC-UI-WIN-010: bind_label sets _label.window."""
        api = WindowAPI()
        label = _make_window()

        api.bind_label(label)

        assert api._label.window is label

    def test_bind_panel_sets_panel_window(self) -> None:
        """TC-UI-WIN-011: bind_panel sets _panel.window."""
        api = WindowAPI()
        panel = _make_window()

        api.bind_panel(panel)

        assert api._panel.window is panel
