"""Base HTTP scraper with security rules.

Security guarantees:
1. Localhost only (127.0.0.1 / ::1) — no DNS names, no non-localhost IPs
2. Port owner PID verification via psutil
3. Configurable per-endpoint, per-PID rate limiting (default 5s)
4. No redirect following
5. 500ms timeout, 64KB response size limit
"""

from __future__ import annotations

import ipaddress
import logging
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import psutil  # type: ignore[import-untyped]

from vramtop.backends.base import VramtopError

if TYPE_CHECKING:
    from vramtop.config import ScrapingConfig

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 0.5
_DEFAULT_RATE_LIMIT_S = 5
_MAX_RESPONSE_BYTES = 64 * 1024  # 64KB


# --- Exceptions ---


class ScraperSecurityError(VramtopError):
    """A scraping request was blocked by a security rule."""


class ScrapeFailedError(VramtopError):
    """A scraping request failed (timeout, bad response, etc.)."""


# --- No-redirect handler ---


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """HTTP handler that rejects all redirects."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        raise ScraperSecurityError(
            f"Redirect rejected: {code} -> {newurl}"
        )


# --- Base scraper ---


class BaseScraper(ABC):
    """Template-method scraper enforcing all security rules."""

    # Subclass should override with the endpoint path (e.g. "/metrics")
    endpoint: str = "/"

    def __init__(self, config: ScrapingConfig | None = None) -> None:
        self._config = config
        self._rate_limit_timestamps: dict[tuple[int, str], float] = {}

    @property
    def _timeout_s(self) -> float:
        if self._config is not None:
            return self._config.timeout_ms / 1000.0
        return _DEFAULT_TIMEOUT_S

    @property
    def _rate_limit_s(self) -> float:
        if self._config is not None:
            return float(self._config.rate_limit_seconds)
        return _DEFAULT_RATE_LIMIT_S

    @property
    def _allowed_ports(self) -> list[int] | None:
        if self._config is not None:
            return self._config.allowed_ports
        return None

    # --- Public API ---

    def scrape(self, pid: int, port: int) -> dict[str, Any]:
        """Scrape metrics from a local inference server.

        Enforces all five security rules before making the request.
        """
        self._verify_localhost(port)
        self._verify_port_owner(pid, port)
        self._enforce_rate_limit(pid)

        url = f"http://127.0.0.1:{port}{self.endpoint}"
        try:
            raw = self._http_get(url)
        except Exception:
            # On network failure, clear the rate-limit timestamp so
            # the next attempt isn't throttled by a failed request.
            self._rate_limit_timestamps.pop((pid, self.endpoint), None)
            raise
        return self._parse_response(raw)

    # --- Security rules ---

    def _verify_localhost(self, port: int) -> None:
        """Rule 1: Only allow localhost connections.

        Also enforce allowed_ports from config.
        """
        allowed = self._allowed_ports
        if allowed is not None and port not in allowed:
            raise ScraperSecurityError(
                f"Port {port} not in allowed_ports: {allowed}"
            )

    @staticmethod
    def _is_localhost(host: str) -> bool:
        """Check whether a host string resolves to localhost."""
        if host in ("127.0.0.1", "::1", "localhost"):
            return True
        try:
            addr = ipaddress.ip_address(host)
            return addr.is_loopback
        except ValueError:
            # DNS name that isn't 'localhost' — reject
            return False

    @staticmethod
    def _verify_port_owner(pid: int, port: int) -> None:
        """Rule 2: Verify the target port is owned by the expected PID."""
        try:
            connections = psutil.net_connections(kind="inet")
        except (psutil.AccessDenied, OSError) as exc:
            raise ScraperSecurityError(
                f"Cannot read network connections: {exc}"
            ) from exc

        for conn in connections:
            if (
                conn.status == "LISTEN"
                and conn.laddr is not None
                and conn.laddr.port == port
                and conn.pid == pid
            ):
                return

        raise ScraperSecurityError(
            f"Port {port} is not owned by PID {pid}"
        )

    def _enforce_rate_limit(self, pid: int) -> None:
        """Rule 3: Per-endpoint, per-PID rate limiting."""
        key = (pid, self.endpoint)
        now = time.monotonic()
        last = self._rate_limit_timestamps.get(key)

        if last is not None and (now - last) < self._rate_limit_s:
            raise ScraperSecurityError(
                f"Rate limited: last scrape was {now - last:.1f}s ago "
                f"(limit: {self._rate_limit_s}s)"
            )

        self._rate_limit_timestamps[key] = now

    def _http_get(self, url: str) -> bytes:
        """Rule 4+5: HTTP GET with no redirects, timeout, and size limit."""
        opener = urllib.request.build_opener(_NoRedirectHandler)

        try:
            req = urllib.request.Request(url, method="GET")
            resp = opener.open(req, timeout=self._timeout_s)
        except ScraperSecurityError:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                raise ScraperSecurityError(
                    f"Redirect rejected: HTTP {exc.code}"
                ) from exc
            raise ScrapeFailedError(
                f"HTTP error {exc.code} from {url}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ScrapeFailedError(
                f"Connection failed to {url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise ScrapeFailedError(
                f"Timeout connecting to {url}"
            ) from exc
        except OSError as exc:
            raise ScrapeFailedError(
                f"Connection failed to {url}: {exc}"
            ) from exc

        # Read with size limit
        data = resp.read(_MAX_RESPONSE_BYTES + 1)
        if len(data) > _MAX_RESPONSE_BYTES:
            raise ScrapeFailedError(
                f"Response exceeds {_MAX_RESPONSE_BYTES} byte limit"
            )

        return bytes(data)

    @abstractmethod
    def _parse_response(self, raw: bytes) -> dict[str, Any]:
        """Parse the raw HTTP response bytes into a metrics dict.

        Subclasses implement framework-specific parsing with
        Pydantic validation where applicable.
        """
