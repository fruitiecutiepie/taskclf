"""App-name normalization, classification, and category assignment.

ActivityWatch reports the foreground application as a human-readable
name (e.g. ``"Firefox"``, ``"Code"``).  This module maps those names
to reverse-domain identifiers, boolean flags, and a semantic category
consumed by :class:`~taskclf.core.types.FeatureRow`.
"""

from __future__ import annotations

from typing import Final

AppInfo = tuple[str, bool, bool, bool, str]
# (app_id, is_browser, is_editor, is_terminal, app_category)

# ---- known application registry ------------------------------------------------
# (reverse_domain_id, is_browser, is_editor, is_terminal, app_category)

_BROWSERS: Final[dict[str, AppInfo]] = {
    "firefox": ("org.mozilla.firefox", True, False, False, "browser"),
    "google chrome": ("com.google.Chrome", True, False, False, "browser"),
    "google-chrome": ("com.google.Chrome", True, False, False, "browser"),
    "chrome": ("com.google.Chrome", True, False, False, "browser"),
    "chromium": ("org.chromium.Chromium", True, False, False, "browser"),
    "chromium-browser": ("org.chromium.Chromium", True, False, False, "browser"),
    "safari": ("com.apple.Safari", True, False, False, "browser"),
    "arc": ("company.thebrowser.Browser", True, False, False, "browser"),
    "brave browser": ("com.brave.Browser", True, False, False, "browser"),
    "brave-browser": ("com.brave.Browser", True, False, False, "browser"),
    "microsoft edge": ("com.microsoft.edgemac", True, False, False, "browser"),
    "msedge": ("com.microsoft.edgemac", True, False, False, "browser"),
    "vivaldi": ("com.vivaldi.Vivaldi", True, False, False, "browser"),
    "opera": ("com.operasoftware.Opera", True, False, False, "browser"),
    "zen browser": ("io.github.nicothin.zen", True, False, False, "browser"),
}

_EDITORS: Final[dict[str, AppInfo]] = {
    "code": ("com.microsoft.VSCode", False, True, False, "editor"),
    "visual studio code": ("com.microsoft.VSCode", False, True, False, "editor"),
    "code - insiders": ("com.microsoft.VSCodeInsiders", False, True, False, "editor"),
    "cursor": ("com.todesktop.cursor", False, True, False, "editor"),
    "intellij idea": ("com.jetbrains.intellij", False, True, False, "editor"),
    "idea": ("com.jetbrains.intellij", False, True, False, "editor"),
    "pycharm": ("com.jetbrains.pycharm", False, True, False, "editor"),
    "webstorm": ("com.jetbrains.webstorm", False, True, False, "editor"),
    "goland": ("com.jetbrains.goland", False, True, False, "editor"),
    "clion": ("com.jetbrains.clion", False, True, False, "editor"),
    "rustrover": ("com.jetbrains.rustrover", False, True, False, "editor"),
    "sublime text": ("com.sublimetext.4", False, True, False, "editor"),
    "sublime_text": ("com.sublimetext.4", False, True, False, "editor"),
    "vim": ("org.vim.Vim", False, True, False, "editor"),
    "gvim": ("org.vim.Vim", False, True, False, "editor"),
    "neovim": ("io.neovim.nvim", False, True, False, "editor"),
    "nvim": ("io.neovim.nvim", False, True, False, "editor"),
    "emacs": ("org.gnu.emacs", False, True, False, "editor"),
    "xcode": ("com.apple.dt.Xcode", False, True, False, "editor"),
    "android studio": ("com.google.android.studio", False, True, False, "editor"),
    "zed": ("dev.zed.Zed", False, True, False, "editor"),
}

_TERMINALS: Final[dict[str, AppInfo]] = {
    "terminal": ("com.apple.Terminal", False, False, True, "terminal"),
    "iterm2": ("com.googlecode.iterm2", False, False, True, "terminal"),
    "iterm": ("com.googlecode.iterm2", False, False, True, "terminal"),
    "alacritty": ("org.alacritty", False, False, True, "terminal"),
    "kitty": ("net.kovidgoyal.kitty", False, False, True, "terminal"),
    "wezterm": ("org.wezfurlong.wezterm", False, False, True, "terminal"),
    "wezterm-gui": ("org.wezfurlong.wezterm", False, False, True, "terminal"),
    "gnome-terminal": ("org.gnome.Terminal", False, False, True, "terminal"),
    "konsole": ("org.kde.konsole", False, False, True, "terminal"),
    "hyper": ("co.zeit.hyper", False, False, True, "terminal"),
    "ghostty": ("com.mitchellh.ghostty", False, False, True, "terminal"),
    "rio": ("io.raphamorim.rio", False, False, True, "terminal"),
    "warp": ("dev.warp.Warp", False, False, True, "terminal"),
    "tabby": ("org.tabby.Tabby", False, False, True, "terminal"),
}

_EMAIL: Final[dict[str, AppInfo]] = {
    "mail": ("com.apple.mail", False, False, False, "email"),
    "thunderbird": ("org.mozilla.thunderbird", False, False, False, "email"),
    "outlook": ("com.microsoft.Outlook", False, False, False, "email"),
}

_CHAT: Final[dict[str, AppInfo]] = {
    "slack": ("com.tinyspeck.slackmacgap", False, False, False, "chat"),
    "discord": ("com.discordapp.Discord", False, False, False, "chat"),
    "microsoft teams": ("com.microsoft.teams2", False, False, False, "chat"),
    "teams": ("com.microsoft.teams2", False, False, False, "chat"),
    "messages": ("com.apple.MobileSMS", False, False, False, "chat"),
}

_MEETING: Final[dict[str, AppInfo]] = {
    "zoom": ("us.zoom.xos", False, False, False, "meeting"),
    "zoom.us": ("us.zoom.xos", False, False, False, "meeting"),
    "facetime": ("com.apple.FaceTime", False, False, False, "meeting"),
}

_DOCS: Final[dict[str, AppInfo]] = {
    "notes": ("com.apple.Notes", False, False, False, "docs"),
    "obsidian": ("md.obsidian", False, False, False, "docs"),
    "notion": ("notion.id", False, False, False, "docs"),
}

_DESIGN: Final[dict[str, AppInfo]] = {
    "figma": ("com.figma.Desktop", False, False, False, "design"),
}

_DEVTOOLS: Final[dict[str, AppInfo]] = {
    "postman": ("com.postmanlabs.mac", False, False, False, "devtools"),
    "docker desktop": ("com.docker.docker", False, False, False, "devtools"),
}

_MEDIA: Final[dict[str, AppInfo]] = {
    "spotify": ("com.spotify.client", False, False, False, "media"),
}

_FILE_MANAGER: Final[dict[str, AppInfo]] = {
    "finder": ("com.apple.finder", False, False, False, "file_manager"),
    "nautilus": ("org.gnome.Nautilus", False, False, False, "file_manager"),
    "files": ("org.gnome.Nautilus", False, False, False, "file_manager"),
}

_UTILITIES: Final[dict[str, AppInfo]] = {
    "preview": ("com.apple.Preview", False, False, False, "utilities"),
    "system preferences": ("com.apple.systempreferences", False, False, False, "utilities"),
    "system settings": ("com.apple.systempreferences", False, False, False, "utilities"),
    "activity monitor": ("com.apple.ActivityMonitor", False, False, False, "utilities"),
    "1password": ("com.1password.1password", False, False, False, "utilities"),
    "bitwarden": ("com.bitwarden.desktop", False, False, False, "utilities"),
}

_PROJECT_MGMT: Final[dict[str, AppInfo]] = {
    "linear": ("com.linear", False, False, False, "project_mgmt"),
}

KNOWN_APPS: Final[dict[str, AppInfo]] = {
    **_BROWSERS,
    **_EDITORS,
    **_TERMINALS,
    **_EMAIL,
    **_CHAT,
    **_MEETING,
    **_DOCS,
    **_DESIGN,
    **_DEVTOOLS,
    **_MEDIA,
    **_FILE_MANAGER,
    **_UTILITIES,
    **_PROJECT_MGMT,
}

APP_CATEGORIES: Final[frozenset[str]] = frozenset({
    "browser", "editor", "terminal", "email", "chat", "meeting",
    "docs", "design", "devtools", "media", "file_manager",
    "utilities", "project_mgmt", "other",
})


def normalize_app(app_name: str) -> AppInfo:
    """Map an AW application name to a reverse-domain ID, flags, and category.

    Performs a case-insensitive lookup in :data:`KNOWN_APPS`.  Unknown
    applications fall back to ``"unknown.<sanitized_name>"`` with all
    flags set to ``False`` and category ``"other"``.

    Args:
        app_name: Application name as reported by ActivityWatch
            (e.g. ``"Firefox"``).

    Returns:
        A ``(app_id, is_browser, is_editor, is_terminal, app_category)``
        tuple.
    """
    key = app_name.strip().lower()
    if key in KNOWN_APPS:
        return KNOWN_APPS[key]
    sanitized = key.replace(" ", "_").replace("/", "_")
    return (f"unknown.{sanitized}", False, False, False, "other")
