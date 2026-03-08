"""Global configuration constants."""

import os
from pathlib import Path

# ── Copilot OAuth ──────────────────────────────────────────────────
# This is the same client_id used by the official VS Code Copilot Chat extension.
GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"
GITHUB_SCOPE = "read:user"
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"

# ── Copilot API ────────────────────────────────────────────────────
# The correct API base URL is read dynamically from the token response
# (endpoints.api field). This fallback is only used if that field is missing.
COPILOT_API_BASE_FALLBACK = "https://api.githubcopilot.com"
# Path suffixes (appended to dynamic api_base_url)
COPILOT_CHAT_COMPLETIONS_PATH = "/chat/completions"
COPILOT_MODELS_PATH = "/models"
COPILOT_RESPONSES_PATH = "/responses"

# Headers to mimic the official VS Code Copilot extension (v0.36.1)
COPILOT_HEADERS = {
    "Editor-Version": "vscode/1.108.0",
    "Editor-Plugin-Version": "copilot-chat/0.36.1",
    "User-Agent": "GitHubCopilotChat/0.36.1",
    "Copilot-Integration-Id": "vscode-chat",
    "X-GitHub-Api-Version": "2025-10-01",
    "openai-intent": "conversation-panel",
    "x-vscode-user-agent-library-version": "electron-fetch",
}

# ── Server ─────────────────────────────────────────────────────────
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 24680
REQUEST_TIMEOUT = 120  # seconds

# ── Security ───────────────────────────────────────────────────────
# Set COPILOTX_API_KEY env var to enable API key protection.
# Set COPILOTX_TRUST_LOCALHOST=0 to require auth even from localhost.
# When set: remote requests require Bearer token.
# When unset: all requests are allowed (backward compatible).
LOCALHOST_ADDRS = {"127.0.0.1", "::1", "localhost"}
# Paths that are always accessible without API key (health checks, etc.)
PUBLIC_PATHS = {"/health", "/"}

_FALSEY_ENV_VALUES = {"0", "false", "no", "off"}


def get_copilotx_api_key() -> str:
    """Return the configured API key for incoming client authentication."""
    return os.environ.get("COPILOTX_API_KEY", "").strip()


def trust_localhost() -> bool:
    """Return whether localhost callers bypass API key checks."""
    raw = os.environ.get("COPILOTX_TRUST_LOCALHOST", "1").strip().lower()
    if not raw:
        return True
    return raw not in _FALSEY_ENV_VALUES

# ── Token ──────────────────────────────────────────────────────────
TOKEN_REFRESH_BUFFER = 60  # refresh token 60s before expiry
DEVICE_CODE_POLL_INTERVAL = 5  # seconds
DEVICE_CODE_TIMEOUT = 900  # 15 minutes

# ── Models Cache ───────────────────────────────────────────────────
MODELS_CACHE_TTL = 300  # 5 minutes

# ── Storage ────────────────────────────────────────────────────────
COPILOTX_DIR = Path.home() / ".copilotx"
AUTH_FILE = COPILOTX_DIR / "auth.json"
ACCOUNTS_DB_FILE = COPILOTX_DIR / "accounts.db"
SERVER_FILE = COPILOTX_DIR / "server.json"

# ── Multi-Account Rotation ─────────────────────────────────────────
DEFAULT_ROTATION_STRATEGY = "fill-first"
ROTATION_STRATEGIES = {"fill-first", "round-robin"}
POOL_SYNC_INTERVAL = 5  # seconds
POOL_429_COOLDOWN = 60  # seconds
POOL_FAILURE_COOLDOWN = 15  # seconds
POOL_MAX_RETRY_ATTEMPTS = 3
