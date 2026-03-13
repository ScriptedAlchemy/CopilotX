"""Multi-account persistence backed by SQLite."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from copilotx.auth.storage import AuthStorage
from copilotx.config import (
    ACCOUNTS_DB_FILE,
    AUTH_FILE,
    COPILOTX_DIR,
    DEFAULT_ROTATION_STRATEGY,
    ROTATION_STRATEGIES,
)


@dataclass(slots=True)
class AccountRecord:
    """Stored upstream GitHub/Copilot account."""

    account_id: str
    github_login: str
    github_user_id: str
    label: str
    github_token: str
    copilot_token: str = ""
    expires_at: float = 0.0
    api_base_url: str = ""
    enabled: bool = True
    reauth_required: bool = False
    priority: int = 0
    model_ids: list[str] | None = None
    last_used_at: float = 0.0
    last_error: str = ""
    last_error_at: float = 0.0
    cooldown_until: float = 0.0
    last_rate_limited_at: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def display_name(self) -> str:
        return self.label or self.github_login or self.account_id


class AccountRepository:
    """Persistent registry for all configured upstream accounts."""

    def __init__(
        self,
        path: Path = ACCOUNTS_DB_FILE,
        legacy_auth_path: Path = AUTH_FILE,
    ) -> None:
        self.path = path
        self.legacy_auth_path = legacy_auth_path
        self._ensure_schema()
        self._migrate_legacy_auth()

    # ── Public API ──────────────────────────────────────────────────

    def has_accounts(self) -> bool:
        return self.count_accounts() > 0

    def count_accounts(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM accounts").fetchone()
        return int(row["count"]) if row else 0

    def list_accounts(self, *, enabled_only: bool = False) -> list[AccountRecord]:
        query = (
            "SELECT * FROM accounts WHERE enabled = 1 "
            if enabled_only
            else "SELECT * FROM accounts "
        )
        query += "ORDER BY priority ASC, created_at ASC, account_id ASC"
        with self._connect() as conn:
            rows = conn.execute(query).fetchall()
        return [self._row_to_account(row) for row in rows]

    def get_account(self, selector: str) -> AccountRecord | None:
        selector = selector.strip()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM accounts
                WHERE account_id = ?
                   OR github_login = ?
                   OR label = ?
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
                """,
                (selector, selector, selector),
            ).fetchone()
        return self._row_to_account(row) if row else None

    def upsert_account(self, account: AccountRecord) -> AccountRecord:
        existing = self.get_account(account.account_id)
        now = time.time()
        if existing is None:
            account.created_at = account.created_at or now
            account.updated_at = now
            account.priority = self._next_priority()
            account.label = self._unique_label(account.label or account.github_login)
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO accounts (
                        account_id, github_login, github_user_id, label, github_token,
                        copilot_token, expires_at, api_base_url, enabled, reauth_required,
                        priority, model_ids_json, last_used_at, last_error, last_error_at,
                        cooldown_until, last_rate_limited_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        account.account_id,
                        account.github_login,
                        account.github_user_id,
                        account.label,
                        account.github_token,
                        account.copilot_token,
                        account.expires_at,
                        account.api_base_url,
                        int(account.enabled),
                        int(account.reauth_required),
                        account.priority,
                        json.dumps(account.model_ids or []),
                        account.last_used_at,
                        account.last_error,
                        account.last_error_at,
                        account.cooldown_until,
                        account.last_rate_limited_at,
                        account.created_at,
                        account.updated_at,
                    ),
                )
            self._ensure_default_account(account.account_id)
            self.sync_legacy_auth_file()
            return self.get_account(account.account_id) or account

        label = account.label or existing.label or existing.github_login
        if label != existing.label:
            label = self._unique_label(label, exclude_account_id=existing.account_id)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE accounts
                SET github_login = ?,
                    github_user_id = ?,
                    label = ?,
                    github_token = ?,
                    copilot_token = ?,
                    expires_at = ?,
                    api_base_url = ?,
                    enabled = ?,
                    reauth_required = ?,
                    model_ids_json = ?,
                    last_used_at = ?,
                    last_error = ?,
                    last_error_at = ?,
                    cooldown_until = ?,
                    last_rate_limited_at = ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (
                    account.github_login or existing.github_login,
                    account.github_user_id or existing.github_user_id,
                    label,
                    account.github_token or existing.github_token,
                    account.copilot_token,
                    account.expires_at,
                    account.api_base_url,
                    int(account.enabled),
                    int(account.reauth_required),
                    json.dumps(account.model_ids or existing.model_ids or []),
                    account.last_used_at or existing.last_used_at,
                    account.last_error,
                    account.last_error_at,
                    account.cooldown_until or existing.cooldown_until,
                    account.last_rate_limited_at or existing.last_rate_limited_at,
                    now,
                    existing.account_id,
                ),
            )
        self.sync_legacy_auth_file()
        return self.get_account(existing.account_id) or account

    def delete_account(self, selector: str) -> bool:
        account = self.get_account(selector)
        if account is None:
            return False
        with self._connect() as conn:
            conn.execute("DELETE FROM accounts WHERE account_id = ?", (account.account_id,))
        if self.get_default_account_id() == account.account_id:
            next_default = self.list_accounts(enabled_only=True)
            self.set_default_account_id(next_default[0].account_id if next_default else "")
        self.sync_legacy_auth_file()
        return True

    def clear_accounts(self) -> int:
        count = self.count_accounts()
        with self._connect() as conn:
            conn.execute("DELETE FROM accounts")
            conn.execute("DELETE FROM settings WHERE key = 'default_account_id'")
        self.sync_legacy_auth_file()
        return count

    def set_account_enabled(self, selector: str, enabled: bool) -> AccountRecord | None:
        account = self.get_account(selector)
        if account is None:
            return None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE accounts
                SET enabled = ?, reauth_required = ?, updated_at = ?
                WHERE account_id = ?
                """,
                (
                    int(enabled),
                    0 if enabled else int(account.reauth_required),
                    time.time(),
                    account.account_id,
                ),
            )
        default_account_id = self.get_default_account_id()
        if enabled:
            if not default_account_id:
                self.set_default_account_id(account.account_id)
        elif default_account_id == account.account_id:
            enabled_accounts = self.list_accounts(enabled_only=True)
            next_default = enabled_accounts[0].account_id if enabled_accounts else ""
            self.set_default_account_id(next_default)
        self.sync_legacy_auth_file()
        return self.get_account(account.account_id)

    def set_account_priority(self, selector: str, priority: int) -> AccountRecord | None:
        account = self.get_account(selector)
        if account is None:
            return None
        with self._connect() as conn:
            conn.execute(
                "UPDATE accounts SET priority = ?, updated_at = ? WHERE account_id = ?",
                (priority, time.time(), account.account_id),
            )
        self.sync_legacy_auth_file()
        return self.get_account(account.account_id)

    def update_tokens(
        self,
        account_id: str,
        *,
        copilot_token: str,
        expires_at: float,
        api_base_url: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE accounts
                SET copilot_token = ?,
                    expires_at = ?,
                    api_base_url = ?,
                    reauth_required = 0,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (copilot_token, expires_at, api_base_url, time.time(), account_id),
            )
        self.sync_legacy_auth_file()

    def update_models(self, account_id: str, model_ids: list[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE accounts SET model_ids_json = ?, updated_at = ? WHERE account_id = ?",
                (json.dumps(model_ids), time.time(), account_id),
            )

    def mark_account(
        self,
        account_id: str,
        *,
        reauth_required: bool | None = None,
        last_used_at: float | None = None,
        last_error: str | None = None,
        last_error_at: float | None = None,
        cooldown_until: float | None = None,
        last_rate_limited_at: float | None = None,
    ) -> None:
        account = self.get_account(account_id)
        if account is None:
            return
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE accounts
                SET reauth_required = ?,
                    last_used_at = ?,
                    last_error = ?,
                    last_error_at = ?,
                    cooldown_until = ?,
                    last_rate_limited_at = ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (
                    int(account.reauth_required if reauth_required is None else reauth_required),
                    account.last_used_at if last_used_at is None else last_used_at,
                    account.last_error if last_error is None else last_error,
                    account.last_error_at if last_error_at is None else last_error_at,
                    account.cooldown_until if cooldown_until is None else cooldown_until,
                    (
                        account.last_rate_limited_at
                        if last_rate_limited_at is None
                        else last_rate_limited_at
                    ),
                    time.time(),
                    account_id,
                ),
            )
        self.sync_legacy_auth_file()

    def get_rotation_strategy(self) -> str:
        value = self.get_setting("rotation_strategy", DEFAULT_ROTATION_STRATEGY)
        if value not in ROTATION_STRATEGIES:
            return DEFAULT_ROTATION_STRATEGY
        return value

    def set_rotation_strategy(self, strategy: str) -> str:
        if strategy not in ROTATION_STRATEGIES:
            raise ValueError(f"Unknown rotation strategy: {strategy}")
        self.set_setting("rotation_strategy", strategy)
        return strategy

    def get_default_account_id(self) -> str:
        return self.get_setting("default_account_id", "")

    def set_default_account_id(self, account_id: str) -> None:
        if account_id:
            self.set_setting("default_account_id", account_id)
        else:
            with self._connect() as conn:
                conn.execute("DELETE FROM settings WHERE key = 'default_account_id'")

    def get_setting(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?",
                (key,),
            ).fetchone()
        return str(row["value"]) if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, time.time()),
            )

    def sync_legacy_auth_file(self) -> None:
        storage = AuthStorage(self.legacy_auth_path)
        default_account = self._get_default_account()
        if default_account is None:
            storage.delete()
            return

        from copilotx.auth.storage import Credentials

        storage.save(
            Credentials(
                github_token=default_account.github_token,
                copilot_token=default_account.copilot_token,
                expires_at=default_account.expires_at,
                api_base_url=default_account.api_base_url,
            )
        )

    # ── Internal Helpers ────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        self._ensure_dir()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    github_login TEXT NOT NULL,
                    github_user_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    github_token TEXT NOT NULL,
                    copilot_token TEXT NOT NULL DEFAULT '',
                    expires_at REAL NOT NULL DEFAULT 0,
                    api_base_url TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    reauth_required INTEGER NOT NULL DEFAULT 0,
                    priority INTEGER NOT NULL DEFAULT 0,
                    model_ids_json TEXT NOT NULL DEFAULT '[]',
                    last_used_at REAL NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    last_error_at REAL NOT NULL DEFAULT 0,
                    cooldown_until REAL NOT NULL DEFAULT 0,
                    last_rate_limited_at REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._ensure_column(
                conn,
                "accounts",
                "cooldown_until",
                "REAL NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                "accounts",
                "last_rate_limited_at",
                "REAL NOT NULL DEFAULT 0",
            )

    def _migrate_legacy_auth(self) -> None:
        if self.count_accounts() > 0:
            return
        storage = AuthStorage(self.legacy_auth_path)
        creds = storage.load()
        if creds is None:
            return

        now = time.time()
        legacy_account = AccountRecord(
            account_id=f"legacy-{uuid.uuid4().hex[:8]}",
            github_login="legacy",
            github_user_id="legacy",
            label="default",
            github_token=creds.github_token,
            copilot_token=creds.copilot_token,
            expires_at=creds.expires_at,
            api_base_url=creds.api_base_url,
            enabled=True,
            reauth_required=False,
            priority=0,
            model_ids=[],
            created_at=now,
            updated_at=now,
        )
        self.upsert_account(legacy_account)
        self.set_default_account_id(legacy_account.account_id)

    def _get_default_account(self) -> AccountRecord | None:
        default_account_id = self.get_default_account_id()
        if default_account_id:
            account = self.get_account(default_account_id)
            if account is not None and account.enabled:
                return account
        enabled_accounts = self.list_accounts(enabled_only=True)
        return enabled_accounts[0] if enabled_accounts else None

    def _ensure_default_account(self, account_id: str) -> None:
        if not self.get_default_account_id():
            self.set_default_account_id(account_id)

    def _next_priority(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(priority), -1) + 1 AS next_priority "
                "FROM accounts"
            ).fetchone()
        return int(row["next_priority"]) if row else 0

    def _unique_label(self, label: str, *, exclude_account_id: str | None = None) -> str:
        label = (label or "account").strip() or "account"
        candidate = label
        suffix = 2
        while True:
            with self._connect() as conn:
                if exclude_account_id:
                    row = conn.execute(
                        "SELECT account_id FROM accounts WHERE label = ? AND account_id != ?",
                        (candidate, exclude_account_id),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT account_id FROM accounts WHERE label = ?",
                        (candidate,),
                    ).fetchone()
            if row is None:
                return candidate
            candidate = f"{label}-{suffix}"
            suffix += 1

    @staticmethod
    def _row_to_account(row: sqlite3.Row | None) -> AccountRecord | None:
        if row is None:
            return None
        return AccountRecord(
            account_id=str(row["account_id"]),
            github_login=str(row["github_login"]),
            github_user_id=str(row["github_user_id"]),
            label=str(row["label"]),
            github_token=str(row["github_token"]),
            copilot_token=str(row["copilot_token"]),
            expires_at=float(row["expires_at"]),
            api_base_url=str(row["api_base_url"]),
            enabled=bool(row["enabled"]),
            reauth_required=bool(row["reauth_required"]),
            priority=int(row["priority"]),
            model_ids=list(json.loads(row["model_ids_json"] or "[]")),
            last_used_at=float(row["last_used_at"]),
            last_error=str(row["last_error"]),
            last_error_at=float(row["last_error_at"]),
            cooldown_until=float(row["cooldown_until"]),
            last_rate_limited_at=float(row["last_rate_limited_at"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        column_sql: str,
    ) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        if any(str(row["name"]) == column for row in rows):
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_sql}")

    def _ensure_dir(self) -> None:
        COPILOTX_DIR.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            COPILOTX_DIR.chmod(stat.S_IRWXU)
        if not self.path.exists():
            self.path.touch()
        if os.name != "nt":
            self.path.chmod(stat.S_IRUSR | stat.S_IWUSR)
