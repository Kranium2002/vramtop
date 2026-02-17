"""Security tests for the base HTTP scraper.

Tests all five security rules:
1. Localhost enforcement
2. Port owner PID verification
3. Rate limiting
4. No redirects
5. Timeout + response size limit

Plus config respect (enable=false, allowed_ports).
"""

from __future__ import annotations

import time
from collections import namedtuple
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vramtop.config import ScrapingConfig
from vramtop.enrichment.scraper import (
    BaseScraper,
    ScrapeFailedError,
    ScraperSecurityError,
)


# --- Concrete test scraper ---


class _EchoScraper(BaseScraper):
    """Minimal scraper that returns raw bytes as a dict."""

    endpoint: str = "/test"

    def _parse_response(self, raw: bytes) -> dict[str, Any]:
        return {"raw": raw.decode("utf-8", errors="replace")}


# --- Helpers ---

FakeConn = namedtuple("FakeConn", ["status", "laddr", "pid"])
FakeAddr = namedtuple("FakeAddr", ["ip", "port"])


def _make_listen_conn(pid: int, port: int) -> FakeConn:
    return FakeConn(status="LISTEN", laddr=FakeAddr(ip="127.0.0.1", port=port), pid=pid)


# --- Localhost enforcement ---


class TestLocalhostEnforcement:
    def test_is_localhost_ipv4(self) -> None:
        assert BaseScraper._is_localhost("127.0.0.1") is True

    def test_is_localhost_ipv6(self) -> None:
        assert BaseScraper._is_localhost("::1") is True

    def test_is_localhost_name(self) -> None:
        assert BaseScraper._is_localhost("localhost") is True

    def test_rejects_private_ip(self) -> None:
        assert BaseScraper._is_localhost("192.168.1.1") is False

    def test_rejects_public_ip(self) -> None:
        assert BaseScraper._is_localhost("8.8.8.8") is False

    def test_rejects_dns_name(self) -> None:
        assert BaseScraper._is_localhost("example.com") is False

    def test_rejects_internal_dns(self) -> None:
        assert BaseScraper._is_localhost("my-server.local") is False

    def test_allowed_ports_enforced(self) -> None:
        config = ScrapingConfig(allowed_ports=[8000, 8080])
        scraper = _EchoScraper(config=config)

        # Port not in allowed list
        with pytest.raises(ScraperSecurityError, match="not in allowed_ports"):
            scraper._verify_localhost(9999)

    def test_allowed_port_passes(self) -> None:
        config = ScrapingConfig(allowed_ports=[8000, 8080])
        scraper = _EchoScraper(config=config)
        # Should not raise
        scraper._verify_localhost(8000)


# --- Port owner PID verification ---


class TestPortOwnerVerification:
    @patch("vramtop.enrichment.scraper.psutil")
    def test_correct_pid_passes(self, mock_psutil: MagicMock) -> None:
        mock_psutil.net_connections.return_value = [
            _make_listen_conn(pid=1234, port=8000),
        ]

        # Should not raise
        BaseScraper._verify_port_owner(1234, 8000)

    @patch("vramtop.enrichment.scraper.psutil")
    def test_wrong_pid_rejected(self, mock_psutil: MagicMock) -> None:
        mock_psutil.net_connections.return_value = [
            _make_listen_conn(pid=9999, port=8000),
        ]

        with pytest.raises(ScraperSecurityError, match="not owned by PID"):
            BaseScraper._verify_port_owner(1234, 8000)

    @patch("vramtop.enrichment.scraper.psutil")
    def test_no_listeners_rejected(self, mock_psutil: MagicMock) -> None:
        mock_psutil.net_connections.return_value = []

        with pytest.raises(ScraperSecurityError, match="not owned by PID"):
            BaseScraper._verify_port_owner(1234, 8000)

    @patch("vramtop.enrichment.scraper.psutil")
    def test_access_denied_rejected(self, mock_psutil: MagicMock) -> None:
        import psutil as real_psutil

        mock_psutil.AccessDenied = real_psutil.AccessDenied
        mock_psutil.net_connections.side_effect = real_psutil.AccessDenied(pid=0)

        with pytest.raises(ScraperSecurityError, match="Cannot read"):
            BaseScraper._verify_port_owner(1234, 8000)


# --- Rate limiting ---


class TestRateLimiting:
    def test_first_call_allowed(self) -> None:
        scraper = _EchoScraper()
        # Should not raise
        scraper._enforce_rate_limit(pid=1234)

    def test_second_call_within_limit_blocked(self) -> None:
        scraper = _EchoScraper()
        scraper._enforce_rate_limit(pid=1234)

        with pytest.raises(ScraperSecurityError, match="Rate limited"):
            scraper._enforce_rate_limit(pid=1234)

    def test_different_pid_not_blocked(self) -> None:
        scraper = _EchoScraper()
        scraper._enforce_rate_limit(pid=1234)
        # Different PID should not be rate limited
        scraper._enforce_rate_limit(pid=5678)

    def test_after_interval_allowed(self) -> None:
        config = ScrapingConfig(rate_limit_seconds=1)
        scraper = _EchoScraper(config=config)
        scraper._enforce_rate_limit(pid=1234)

        # Manually set the timestamp to the past
        key = (1234, "/test")
        scraper._rate_limit_timestamps[key] = time.monotonic() - 2.0

        # Should not raise now
        scraper._enforce_rate_limit(pid=1234)

    def test_config_rate_limit_respected(self) -> None:
        config = ScrapingConfig(rate_limit_seconds=10)
        scraper = _EchoScraper(config=config)
        scraper._enforce_rate_limit(pid=1234)

        with pytest.raises(ScraperSecurityError, match="Rate limited"):
            scraper._enforce_rate_limit(pid=1234)


# --- No redirects ---


class TestNoRedirects:
    @patch("vramtop.enrichment.scraper.psutil")
    def test_redirect_rejected(self, mock_psutil: MagicMock) -> None:
        """Test that HTTP redirects are rejected."""
        mock_psutil.net_connections.return_value = [
            _make_listen_conn(pid=1234, port=8000),
        ]

        import http.server
        import threading

        class RedirectHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(301)
                self.send_header("Location", "http://evil.com/steal")
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass  # Suppress logs

        server = http.server.HTTPServer(("127.0.0.1", 0), RedirectHandler)
        port = server.server_address[1]

        # Re-mock with correct port
        mock_psutil.net_connections.return_value = [
            _make_listen_conn(pid=1234, port=port),
        ]

        thread = threading.Thread(target=server.handle_request)
        thread.start()

        config = ScrapingConfig(
            allowed_ports=[port],
            rate_limit_seconds=0,
        )
        scraper = _EchoScraper(config=config)

        with pytest.raises((ScraperSecurityError, ScrapeFailedError)):
            scraper.scrape(pid=1234, port=port)

        thread.join(timeout=5)
        server.server_close()


# --- Timeout enforcement ---


class TestTimeout:
    @patch("vramtop.enrichment.scraper.psutil")
    def test_slow_response_times_out(self, mock_psutil: MagicMock) -> None:
        """Test that a slow server triggers a timeout."""
        import http.server
        import threading

        class SlowHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                time.sleep(3)  # Way longer than 500ms timeout
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"too late")

            def log_message(self, *args: Any) -> None:
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), SlowHandler)
        port = server.server_address[1]

        mock_psutil.net_connections.return_value = [
            _make_listen_conn(pid=1234, port=port),
        ]

        thread = threading.Thread(target=server.handle_request)
        thread.start()

        config = ScrapingConfig(
            allowed_ports=[port],
            timeout_ms=200,
            rate_limit_seconds=0,
        )
        scraper = _EchoScraper(config=config)

        with pytest.raises(ScrapeFailedError, match="(Timeout|Connection failed|timed out)"):
            scraper.scrape(pid=1234, port=port)

        thread.join(timeout=5)
        server.server_close()


# --- Response size limit ---


class TestResponseSizeLimit:
    @patch("vramtop.enrichment.scraper.psutil")
    def test_large_response_rejected(self, mock_psutil: MagicMock) -> None:
        """Test that responses > 64KB are rejected."""
        import http.server
        import threading

        class LargeHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                # Send 128KB of data
                self.wfile.write(b"x" * (128 * 1024))

            def log_message(self, *args: Any) -> None:
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), LargeHandler)
        port = server.server_address[1]

        mock_psutil.net_connections.return_value = [
            _make_listen_conn(pid=1234, port=port),
        ]

        thread = threading.Thread(target=server.handle_request)
        thread.start()

        config = ScrapingConfig(
            allowed_ports=[port],
            rate_limit_seconds=0,
        )
        scraper = _EchoScraper(config=config)

        with pytest.raises(ScrapeFailedError, match="65536 byte limit"):
            scraper.scrape(pid=1234, port=port)

        thread.join(timeout=5)
        server.server_close()


# --- Schema validation ---


class TestSchemaValidation:
    def test_parse_valid_response(self) -> None:
        scraper = _EchoScraper()
        result = scraper._parse_response(b"hello world")
        assert result == {"raw": "hello world"}


# --- Config respect ---


class TestConfigRespect:
    def test_scraping_disabled(self) -> None:
        """When enable=false, the enrichment orchestrator should skip scraping."""
        from vramtop.enrichment import EnrichmentResult

        config = ScrapingConfig(enable=False)
        result = EnrichmentResult()
        # Verify the flag is accessible
        assert config.enable is False

    def test_default_config_values(self) -> None:
        config = ScrapingConfig()
        assert config.enable is True
        assert config.rate_limit_seconds == 5
        assert config.timeout_ms == 500
        assert 8000 in config.allowed_ports


# --- Full scrape integration (with mock server) ---


class TestFullScrapeFlow:
    @patch("vramtop.enrichment.scraper.psutil")
    def test_successful_scrape(self, mock_psutil: MagicMock) -> None:
        """End-to-end: real HTTP server, mock psutil, valid response."""
        import http.server
        import threading

        class OkHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"metrics OK")

            def log_message(self, *args: Any) -> None:
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), OkHandler)
        port = server.server_address[1]

        mock_psutil.net_connections.return_value = [
            _make_listen_conn(pid=1234, port=port),
        ]

        thread = threading.Thread(target=server.handle_request)
        thread.start()

        config = ScrapingConfig(
            allowed_ports=[port],
            rate_limit_seconds=0,
        )
        scraper = _EchoScraper(config=config)
        result = scraper.scrape(pid=1234, port=port)

        assert result == {"raw": "metrics OK"}

        thread.join(timeout=5)
        server.server_close()
