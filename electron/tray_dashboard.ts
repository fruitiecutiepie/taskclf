/**
 * Pure helpers for Electron tray interactions that affect dashboard visibility.
 *
 * Primary tray clicks should only reveal/focus the dashboard. Explicit menu
 * actions keep toggle semantics so the user can still hide the shell on demand.
 */

export type TrayDashboardInteraction = "icon-click" | "menu-toggle";
export type DashboardWindowAction = "show" | "toggle";

export function dashboardWindowActionForTrayInteraction(
  interaction: TrayDashboardInteraction,
): DashboardWindowAction {
  if (interaction === "icon-click") {
    return "show";
  }
  return "toggle";
}
