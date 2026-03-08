"""Copilot token manager — exchange GitHub token for short-lived Copilot JWT."""

from __future__ import annotations

import time

import httpx
from rich.console import Console

from copilotx.auth.storage import AuthStorage, Credentials
from copilotx.config import (
    COPILOT_API_BASE_FALLBACK,
    COPILOT_HEADERS,
    GITHUB_COPILOT_TOKEN_URL,
    TOKEN_REFRESH_BUFFER,
)

console = Console()


class TokenError(Exception):
    """Raised when token operations fail."""


class TokenManager:
    """Manages the two-layer token lifecycle:
    - github_token  (long-lived) → stored
    - copilot_token (short-lived, ~30 min) → auto-refreshed
    """

    def __init__(self, storage: AuthStorage | None = None) -> None:
        self.storage = storage or AuthStorage()
        self._creds: Credentials | None = None

    # ── public ──────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load credentials from disk.  Returns True if valid creds exist."""
        self._creds = self.storage.load()
        return self._creds is not None

    def save_github_token(self, github_token: str) -> None:
        """Store a new GitHub OAuth token (from device flow or --token flag)."""
        self._creds = Credentials(github_token=github_token)
        self.storage.save(self._creds)

    def logout(self) -> bool:
        """Clear stored credentials."""
        self._creds = None
        return self.storage.delete()

    @property
    def is_authenticated(self) -> bool:
        if self._creds is None:
            self.load()
        return self._creds is not None and bool(self._creds.github_token)

    @property
    def copilot_token_valid(self) -> bool:
        if self._creds is None:
            return False
        return (
            bool(self._creds.copilot_token)
            and self._creds.expires_at > time.time() + TOKEN_REFRESH_BUFFER
        )

    @property
    def expires_in_seconds(self) -> int:
        if self._creds is None or self._creds.expires_at == 0:
            return 0
        remaining = int(self._creds.expires_at - time.time())
        return max(remaining, 0)

    @property
    def api_base_url(self) -> str:
        """Return the dynamic API base URL from token response, or fallback."""
        if self._creds and self._creds.api_base_url:
            return self._creds.api_base_url.rstrip("/")
        return COPILOT_API_BASE_FALLBACK

    async def ensure_copilot_token(self) -> str:
        """Return a valid Copilot JWT, refreshing if needed."""
        if self._creds is None:
            self.load()
        if self._creds is None:
            raise TokenError("Not authenticated. Run `copilotx auth login` first.")

        if self.copilot_token_valid:
            return self._creds.copilot_token

        # Refresh
        new_token, expires_at, api_base = await _fetch_copilot_token(
            self._creds.github_token
        )
        self._creds.copilot_token = new_token
        self._creds.expires_at = expires_at
        if api_base:
            self._creds.api_base_url = api_base
        self.storage.save(self._creds)
        return new_token

    def get_status(self) -> dict:
        """Return a status dict for CLI display."""
        if not self.is_authenticated:
            return {"authenticated": False}
        return {
            "authenticated": True,
            "copilot_token_valid": self.copilot_token_valid,
            "expires_in": self.expires_in_seconds,
            "api_base_url": self.api_base_url,
        }


# ── helpers ─────────────────────────────────────────────────────────


async def fetch_copilot_token(github_token: str) -> tuple[str, float, str]:
    """Exchange a GitHub OAuth token for a short-lived Copilot JWT.
    Returns (copilot_jwt, expires_at_unix, api_base_url).
    """
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        **COPILOT_HEADERS,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(GITHUB_COPILOT_TOKEN_URL, headers=headers)

        if resp.status_code == 401:
            raise TokenError(
                "GitHub token is invalid or expired. Run `copilotx auth login` again."
            )
        if resp.status_code == 403:
            raise TokenError(
                "GitHub Copilot is not enabled for this account. "
                "Make sure you have a Copilot subscription."
            )
        resp.raise_for_status()

        data = resp.json()
        token = data["token"]
        expires_at = float(data["expires_at"])

        # Extract dynamic API base URL from endpoints.api
        endpoints = data.get("endpoints", {})
        api_base_url = endpoints.get("api", "")

        return token, expires_at, api_base_url


_fetch_copilot_token = fetch_copilot_token
