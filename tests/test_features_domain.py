"""Tests for privacy-preserving browser domain classification."""

from __future__ import annotations

from taskclf.features.domain import DOMAIN_CATEGORIES, classify_domain


class TestClassifyDomain:
    def test_non_browser_returns_non_browser(self) -> None:
        assert classify_domain("github.com", is_browser=False) == "non_browser"
        assert classify_domain(None, is_browser=False) == "non_browser"

    def test_none_domain_returns_unknown(self) -> None:
        assert classify_domain(None, is_browser=True) == "unknown"

    def test_empty_domain_returns_unknown(self) -> None:
        assert classify_domain("", is_browser=True) == "unknown"
        assert classify_domain("  ", is_browser=True) == "unknown"

    def test_known_domain_classified(self) -> None:
        assert classify_domain("github.com") == "code_hosting"
        assert classify_domain("google.com") == "search"
        assert classify_domain("youtube.com") == "video"
        assert classify_domain("twitter.com") == "social"
        assert classify_domain("stackoverflow.com") == "docs"
        assert classify_domain("gmail.com") == "email_web"
        assert classify_domain("notion.so") == "productivity"
        assert classify_domain("slack.com") == "chat"
        assert classify_domain("figma.com") == "design"

    def test_subdomain_matches_parent(self) -> None:
        assert classify_domain("mail.google.com") == "email_web"
        assert classify_domain("api.github.com") == "code_hosting"
        assert classify_domain("m.youtube.com") == "video"

    def test_unknown_domain_returns_other(self) -> None:
        assert classify_domain("randomsite.xyz") == "other"
        assert classify_domain("mycompany.internal") == "other"

    def test_case_insensitive(self) -> None:
        assert classify_domain("GitHub.COM") == "code_hosting"
        assert classify_domain("GOOGLE.COM") == "search"

    def test_all_categories_are_valid(self) -> None:
        for domain in ["github.com", "google.com", "youtube.com",
                        "twitter.com", "bbc.com", "gmail.com",
                        "notion.so", "slack.com", "figma.com",
                        "randomsite.xyz", None, ""]:
            result = classify_domain(domain)
            assert result in DOMAIN_CATEGORIES

    def test_news_sites(self) -> None:
        assert classify_domain("news.ycombinator.com") == "news"
        assert classify_domain("techcrunch.com") == "news"
        assert classify_domain("bbc.com") == "news"
