"""Auth package — GitHub OAuth + Copilot token management."""

from copilotx.auth.accounts import AccountRecord, AccountRepository
from copilotx.auth.pool import TokenPool
from copilotx.auth.storage import AuthStorage, Credentials
from copilotx.auth.token import TokenManager

__all__ = [
    "AccountRecord",
    "AccountRepository",
    "AuthStorage",
    "Credentials",
    "TokenManager",
    "TokenPool",
]
