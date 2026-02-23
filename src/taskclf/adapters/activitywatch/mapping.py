"""App-name normalization and browser/editor/terminal classification.

ActivityWatch reports the foreground application as a human-readable
name (e.g. ``"Firefox"``, ``"Code"``).  This module maps those names
to reverse-domain identifiers and boolean flags consumed by
:class:`~taskclf.core.types.FeatureRow`.
"""

from __future__ import annotations

from typing import Final

AppInfo = tuple[str, bool, bool, bool]  # (app_id, is_browser, is_editor, is_terminal)

# ---- known application registry ------------------------------------------------
# (reverse_domain_id, is_browser, is_editor, is_terminal)

_BROWSERS: Final[dict[str, AppInfo]] = {
    "firefox": ("org.mozilla.firefox", True, False, False),
    "google chrome": ("com.google.Chrome", True, False, False),
    "google-chrome": ("com.google.Chrome", True, False, False),
    "chrome": ("com.google.Chrome", True, False, False),
    "chromium": ("org.chromium.Chromium", True, False, False),
    "chromium-browser": ("org.chromium.Chromium", True, False, False),
    "safari": ("com.apple.Safari", True, False, False),
    "arc": ("company.thebrowser.Browser", True, False, False),
    "brave browser": ("com.brave.Browser", True, False, False),
    "brave-browser": ("com.brave.Browser", True, False, False),
    "microsoft edge": ("com.microsoft.edgemac", True, False, False),
    "msedge": ("com.microsoft.edgemac", True, False, False),
    "vivaldi": ("com.vivaldi.Vivaldi", True, False, False),
    "opera": ("com.operasoftware.Opera", True, False, False),
    "zen browser": ("io.github.nicothin.zen", True, False, False),
}

_EDITORS: Final[dict[str, AppInfo]] = {
    "code": ("com.microsoft.VSCode", False, True, False),
    "visual studio code": ("com.microsoft.VSCode", False, True, False),
    "code - insiders": ("com.microsoft.VSCodeInsiders", False, True, False),
    "cursor": ("com.todesktop.cursor", False, True, False),
    "intellij idea": ("com.jetbrains.intellij", False, True, False),
    "idea": ("com.jetbrains.intellij", False, True, False),
    "pycharm": ("com.jetbrains.pycharm", False, True, False),
    "webstorm": ("com.jetbrains.webstorm", False, True, False),
    "goland": ("com.jetbrains.goland", False, True, False),
    "clion": ("com.jetbrains.clion", False, True, False),
    "rustrover": ("com.jetbrains.rustrover", False, True, False),
    "sublime text": ("com.sublimetext.4", False, True, False),
    "sublime_text": ("com.sublimetext.4", False, True, False),
    "vim": ("org.vim.Vim", False, True, False),
    "gvim": ("org.vim.Vim", False, True, False),
    "neovim": ("io.neovim.nvim", False, True, False),
    "nvim": ("io.neovim.nvim", False, True, False),
    "emacs": ("org.gnu.emacs", False, True, False),
    "xcode": ("com.apple.dt.Xcode", False, True, False),
    "android studio": ("com.google.android.studio", False, True, False),
    "zed": ("dev.zed.Zed", False, True, False),
}

_TERMINALS: Final[dict[str, AppInfo]] = {
    "terminal": ("com.apple.Terminal", False, False, True),
    "iterm2": ("com.googlecode.iterm2", False, False, True),
    "iterm": ("com.googlecode.iterm2", False, False, True),
    "alacritty": ("org.alacritty", False, False, True),
    "kitty": ("net.kovidgoyal.kitty", False, False, True),
    "wezterm": ("org.wezfurlong.wezterm", False, False, True),
    "wezterm-gui": ("org.wezfurlong.wezterm", False, False, True),
    "gnome-terminal": ("org.gnome.Terminal", False, False, True),
    "konsole": ("org.kde.konsole", False, False, True),
    "hyper": ("co.zeit.hyper", False, False, True),
    "ghostty": ("com.mitchellh.ghostty", False, False, True),
    "rio": ("io.raphamorim.rio", False, False, True),
    "warp": ("dev.warp.Warp", False, False, True),
    "tabby": ("org.tabby.Tabby", False, False, True),
}

_OTHER: Final[dict[str, AppInfo]] = {
    "mail": ("com.apple.mail", False, False, False),
    "thunderbird": ("org.mozilla.thunderbird", False, False, False),
    "outlook": ("com.microsoft.Outlook", False, False, False),
    "zoom": ("us.zoom.xos", False, False, False),
    "zoom.us": ("us.zoom.xos", False, False, False),
    "slack": ("com.tinyspeck.slackmacgap", False, False, False),
    "discord": ("com.discordapp.Discord", False, False, False),
    "microsoft teams": ("com.microsoft.teams2", False, False, False),
    "teams": ("com.microsoft.teams2", False, False, False),
    "finder": ("com.apple.finder", False, False, False),
    "nautilus": ("org.gnome.Nautilus", False, False, False),
    "files": ("org.gnome.Nautilus", False, False, False),
    "notes": ("com.apple.Notes", False, False, False),
    "obsidian": ("md.obsidian", False, False, False),
    "notion": ("notion.id", False, False, False),
    "spotify": ("com.spotify.client", False, False, False),
    "preview": ("com.apple.Preview", False, False, False),
    "system preferences": ("com.apple.systempreferences", False, False, False),
    "system settings": ("com.apple.systempreferences", False, False, False),
    "activity monitor": ("com.apple.ActivityMonitor", False, False, False),
    "messages": ("com.apple.MobileSMS", False, False, False),
    "facetime": ("com.apple.FaceTime", False, False, False),
    "figma": ("com.figma.Desktop", False, False, False),
    "postman": ("com.postmanlabs.mac", False, False, False),
    "docker desktop": ("com.docker.docker", False, False, False),
    "1password": ("com.1password.1password", False, False, False),
    "bitwarden": ("com.bitwarden.desktop", False, False, False),
    "linear": ("com.linear", False, False, False),
}

KNOWN_APPS: Final[dict[str, AppInfo]] = {
    **_BROWSERS,
    **_EDITORS,
    **_TERMINALS,
    **_OTHER,
}


def normalize_app(app_name: str) -> AppInfo:
    """Map an AW application name to a reverse-domain ID and boolean flags.

    Performs a case-insensitive lookup in :data:`KNOWN_APPS`.  Unknown
    applications fall back to ``"unknown.<sanitized_name>"`` with all
    flags set to ``False``.

    Args:
        app_name: Application name as reported by ActivityWatch
            (e.g. ``"Firefox"``).

    Returns:
        A ``(app_id, is_browser, is_editor, is_terminal)`` tuple.
    """
    key = app_name.strip().lower()
    if key in KNOWN_APPS:
        return KNOWN_APPS[key]
    sanitized = key.replace(" ", "_").replace("/", "_")
    return (f"unknown.{sanitized}", False, False, False)
