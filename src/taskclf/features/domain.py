"""Privacy-preserving browser domain classification.

Maps eTLD+1 domains (e.g. ``"github.com"``) to semantic categories
without storing full URLs, paths, or query strings.  Only the domain
category string is persisted — never the raw domain or URL.

When no domain information is available (e.g. no ``aw-watcher-web``
integration), the classifier falls back to ``"unknown"`` for browser
apps and ``"non_browser"`` for non-browser apps.

See ``docs/guide/privacy.md`` §3.4 for the data-handling contract.
"""

from __future__ import annotations

from typing import Final

DOMAIN_CATEGORIES: Final[tuple[str, ...]] = (
    "search",
    "docs",
    "social",
    "video",
    "code_hosting",
    "news",
    "email_web",
    "productivity",
    "chat",
    "design",
    "other",
    "unknown",
    "non_browser",
)

_DOMAIN_RULES: Final[dict[str, str]] = {
    "google.com": "search",
    "bing.com": "search",
    "duckduckgo.com": "search",
    "baidu.com": "search",
    "yahoo.com": "search",
    "yandex.com": "search",
    "github.com": "code_hosting",
    "gitlab.com": "code_hosting",
    "bitbucket.org": "code_hosting",
    "codeberg.org": "code_hosting",
    "sourcehut.org": "code_hosting",
    "stackoverflow.com": "docs",
    "stackexchange.com": "docs",
    "developer.mozilla.org": "docs",
    "docs.python.org": "docs",
    "docs.rs": "docs",
    "readthedocs.io": "docs",
    "devdocs.io": "docs",
    "man7.org": "docs",
    "cppreference.com": "docs",
    "learn.microsoft.com": "docs",
    "wiki.archlinux.org": "docs",
    "wikipedia.org": "docs",
    "medium.com": "docs",
    "dev.to": "docs",
    "youtube.com": "video",
    "vimeo.com": "video",
    "twitch.tv": "video",
    "netflix.com": "video",
    "twitter.com": "social",
    "x.com": "social",
    "facebook.com": "social",
    "instagram.com": "social",
    "reddit.com": "social",
    "linkedin.com": "social",
    "mastodon.social": "social",
    "threads.net": "social",
    "bsky.app": "social",
    "news.ycombinator.com": "news",
    "techcrunch.com": "news",
    "arstechnica.com": "news",
    "bbc.com": "news",
    "cnn.com": "news",
    "reuters.com": "news",
    "theverge.com": "news",
    "gmail.com": "email_web",
    "mail.google.com": "email_web",
    "outlook.live.com": "email_web",
    "outlook.office.com": "email_web",
    "protonmail.com": "email_web",
    "mail.yahoo.com": "email_web",
    "notion.so": "productivity",
    "trello.com": "productivity",
    "asana.com": "productivity",
    "linear.app": "productivity",
    "jira.atlassian.com": "productivity",
    "confluence.atlassian.com": "productivity",
    "airtable.com": "productivity",
    "miro.com": "productivity",
    "docs.google.com": "productivity",
    "sheets.google.com": "productivity",
    "slides.google.com": "productivity",
    "drive.google.com": "productivity",
    "slack.com": "chat",
    "discord.com": "chat",
    "teams.microsoft.com": "chat",
    "web.telegram.org": "chat",
    "web.whatsapp.com": "chat",
    "figma.com": "design",
    "canva.com": "design",
    "dribbble.com": "design",
}


def classify_domain(domain: str | None, *, is_browser: bool = True) -> str:
    """Map a domain string to a privacy-safe category.

    Args:
        domain: An eTLD+1 or subdomain string (e.g. ``"github.com"``).
            ``None`` when domain information is unavailable.
        is_browser: Whether the foreground app is a browser.

    Returns:
        One of :data:`DOMAIN_CATEGORIES`.
    """
    if not is_browser:
        return "non_browser"
    if domain is None:
        return "unknown"
    domain = domain.lower().strip()
    if not domain:
        return "unknown"

    if domain in _DOMAIN_RULES:
        return _DOMAIN_RULES[domain]

    # Try parent domain (e.g. "mail.google.com" -> "google.com")
    parts = domain.split(".")
    if len(parts) > 2:
        parent = ".".join(parts[-2:])
        if parent in _DOMAIN_RULES:
            return _DOMAIN_RULES[parent]

    return "other"
