"""Runtime token pool with automatic account rotation."""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

from copilotx.auth.accounts import AccountRecord, AccountRepository
from copilotx.auth.token import TokenError, fetch_copilot_token
from copilotx.config import (
    POOL_429_COOLDOWN,
    POOL_FAILURE_COOLDOWN,
    POOL_MAX_RETRY_ATTEMPTS,
    POOL_SINGLE_ACCOUNT_429_COOLDOWN,
    POOL_SYNC_INTERVAL,
    ROTATION_STRATEGIES,
    TOKEN_REFRESH_BUFFER,
)
from copilotx.proxy.client import CopilotClient

logger = logging.getLogger(__name__)


class PoolError(Exception):
    """Raised when the multi-account pool cannot satisfy a request."""


class ModelUnavailableError(PoolError):
    """Raised when no account can serve a requested model."""


@dataclass
class RetryDecision:
    """How the pool should react to a request failure."""

    retry_same_account: bool = False
    retry_other_account: bool = False
    reauth_required: bool = False
    cooldown_seconds: float = 0.0


@dataclass
class AccountLease:
    """A reserved upstream account for one request lifecycle."""

    pool: "TokenPool"
    entry: "PoolEntry"
    is_stream: bool
    released: bool = False
    force_refreshed: bool = False

    @property
    def account_id(self) -> str:
        return self.entry.account.account_id

    @property
    def display_name(self) -> str:
        return self.entry.account.display_name

    @property
    def client(self) -> CopilotClient:
        if self.entry.client is None:
            raise PoolError("Upstream client not initialized.")
        return self.entry.client

    async def release(self) -> None:
        if self.released:
            return
        self.released = True
        await self.pool.release(self)


@dataclass
class PoolEntry:
    """Runtime state for one configured account."""

    account: AccountRecord
    client_factory: Callable[[str, str], CopilotClient]
    token_fetcher: Callable[[str], Awaitable[tuple[str, float, str]]]
    client: CopilotClient | None = None
    refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    client_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_requests: int = 0
    active_streams: int = 0
    cooldown_until: float = 0.0
    error_streak: int = 0
    known_models: list[dict[str, Any]] = field(default_factory=list)
    removed: bool = False

    @property
    def known_model_ids(self) -> set[str]:
        if self.known_models:
            return {str(model.get("id")) for model in self.known_models if model.get("id")}
        return set(self.account.model_ids or [])

    @property
    def token_valid(self) -> bool:
        return bool(
            self.account.copilot_token
            and self.account.expires_at > time.time() + TOKEN_REFRESH_BUFFER
        )

    async def ensure_client(self) -> CopilotClient:
        async with self.client_lock:
            if self.client is None:
                self.client = self.client_factory(
                    self.account.copilot_token,
                    self.account.api_base_url,
                )
                await self.client.__aenter__()
            else:
                self.client.update_token(self.account.copilot_token)
                self.client.update_api_base(self.account.api_base_url)
        return self.client

    async def close(self) -> None:
        async with self.client_lock:
            if self.client is not None:
                await self.client.__aexit__(None, None, None)
                self.client = None


class TokenPool:
    """Schedules requests across many upstream Copilot accounts."""

    def __init__(
        self,
        repository: AccountRepository,
        *,
        client_factory: Callable[[str, str], CopilotClient] = CopilotClient,
        token_fetcher: Callable[[str], Awaitable[tuple[str, float, str]]] = fetch_copilot_token,
    ) -> None:
        self.repository = repository
        self.client_factory = client_factory
        self.token_fetcher = token_fetcher
        self.entries: dict[str, PoolEntry] = {}
        self._selection_lock = asyncio.Lock()
        self._round_robin_cursor = 0
        self._last_sync = 0.0

    async def __aenter__(self) -> "TokenPool":
        await self.sync_accounts(force=True)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        for entry in list(self.entries.values()):
            await entry.close()

    # ── Public API ──────────────────────────────────────────────────

    async def sync_accounts(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_sync) < POOL_SYNC_INTERVAL:
            return

        records = {account.account_id: account for account in self.repository.list_accounts()}
        seen_ids = set(records)
        for account_id, account in records.items():
            if account_id in self.entries:
                entry = self.entries[account_id]
                entry.account = account
                entry.cooldown_until = max(entry.cooldown_until, account.cooldown_until)
                entry.removed = False
            else:
                self.entries[account_id] = PoolEntry(
                    account=account,
                    client_factory=self.client_factory,
                    token_fetcher=self.token_fetcher,
                )

        for account_id, entry in list(self.entries.items()):
            if account_id in seen_ids:
                continue
            entry.removed = True
            if entry.active_requests == 0:
                await entry.close()
                del self.entries[account_id]

        self._last_sync = now

    async def acquire(
        self,
        *,
        model: str | None,
        is_stream: bool,
        exclude_account_ids: set[str] | None = None,
    ) -> AccountLease:
        exclude_account_ids = exclude_account_ids or set()
        await self.sync_accounts()
        last_error: Exception | None = None

        for entry in self._candidate_entries(model=model, exclude_account_ids=exclude_account_ids):
            entry_error: Exception | None = None
            if not entry.account.enabled or entry.account.reauth_required or entry.removed:
                continue
            if self._cooldown_active(entry):
                continue

            async with self._selection_lock:
                if entry.removed or not entry.account.enabled or entry.account.reauth_required:
                    continue
                if self._cooldown_active(entry):
                    continue
                entry.active_requests += 1
                if is_stream:
                    entry.active_streams += 1

            try:
                await self._prepare_entry(entry, model=model)
                return AccountLease(pool=self, entry=entry, is_stream=is_stream)
            except ModelUnavailableError as exc:
                entry_error = exc
                last_error = exc
                await self._cooldown(entry, 0, str(exc))
            except Exception as exc:
                entry_error = exc
                last_error = exc
                await self._handle_prepare_error(entry, exc)
            finally:
                if entry_error is not None:
                    await self._release_entry(entry, is_stream=is_stream)

        if last_error is not None:
            raise last_error
        raise PoolError("No healthy Copilot accounts are available.")

    async def release(self, lease: AccountLease) -> None:
        await self._release_entry(lease.entry, is_stream=lease.is_stream)

    async def list_models(self) -> list[dict[str, Any]]:
        await self.sync_accounts(force=True)
        merged: dict[str, dict[str, Any]] = {}

        for entry in self._candidate_entries(model=None, exclude_account_ids=set()):
            if not entry.account.enabled or entry.account.reauth_required or entry.removed:
                continue
            if self._cooldown_active(entry):
                continue
            try:
                await self._prepare_entry(entry, model=None)
                client = await entry.ensure_client()
                models = await client.list_models()
                entry.known_models = models
                self.repository.update_models(
                    entry.account.account_id,
                    [str(model.get("id")) for model in models if model.get("id")],
                )
                for model in models:
                    model_id = str(model.get("id"))
                    if model_id and model_id not in merged:
                        merged[model_id] = model
            except Exception as exc:
                await self._handle_prepare_error(entry, exc)

        return list(merged.values())

    async def health_snapshot(self) -> dict[str, Any]:
        await self.sync_accounts()
        now = time.time()
        enabled_entries = [
            entry for entry in self.entries.values() if entry.account.enabled and not entry.removed
        ]
        valid_expiries = [
            max(int(entry.account.expires_at - now), 0)
            for entry in enabled_entries
            if entry.token_valid
        ]
        healthy = [
            entry
            for entry in enabled_entries
            if not entry.account.reauth_required and not self._cooldown_active(entry, now=now)
        ]
        cooling_down = [
            entry
            for entry in enabled_entries
            if self._cooldown_active(entry, now=now) and not entry.account.reauth_required
        ]
        reauth = [entry for entry in enabled_entries if entry.account.reauth_required]
        return {
            "authenticated": bool(enabled_entries),
            "token_valid": bool(valid_expiries),
            "token_expires_in": max(valid_expiries) if valid_expiries else 0,
            "accounts_total": len(self.entries),
            "accounts_enabled": len(enabled_entries),
            "accounts_healthy": len(healthy),
            "accounts_cooling_down": len(cooling_down),
            "accounts_reauth_required": len(reauth),
            "strategy": self.repository.get_rotation_strategy(),
        }

    async def execute(
        self,
        *,
        model: str | None,
        operation: Callable[[CopilotClient], Awaitable[Any]],
    ) -> Any:
        await self.sync_accounts()
        tried: set[str] = set()
        last_error: Exception | None = None
        max_attempts = max(1, min(POOL_MAX_RETRY_ATTEMPTS, max(len(self.entries), 1)))

        for _ in range(max_attempts):
            lease = await self.acquire(
                model=model,
                is_stream=False,
                exclude_account_ids=tried,
            )
            try:
                result = await operation(lease.client)
                await self._mark_success(lease.entry)
                return result
            except Exception as exc:
                decision = await self._handle_request_error(lease, exc)
                last_error = exc
                if decision.retry_same_account and not lease.force_refreshed:
                    lease.force_refreshed = True
                    try:
                        await self._force_refresh_entry(lease.entry)
                        result = await operation(lease.client)
                        await self._mark_success(lease.entry)
                        return result
                    except Exception as retry_exc:
                        last_error = retry_exc
                        decision = await self._handle_request_error(lease, retry_exc)

                tried.add(lease.account_id)
                if not decision.retry_other_account:
                    raise last_error
            finally:
                await lease.release()

        raise last_error or PoolError("No upstream account could satisfy the request.")

    async def probe(
        self,
        *,
        model: str | None,
        operation: Callable[[CopilotClient], Awaitable[Any]],
    ) -> Any:
        """Run a best-effort operation without mutating pool health on failure."""
        await self.sync_accounts()
        lease = await self.acquire(
            model=model,
            is_stream=False,
            exclude_account_ids=set(),
        )
        try:
            result = await operation(lease.client)
            await self._mark_success(lease.entry)
            return result
        finally:
            await lease.release()

    async def stream(
        self,
        *,
        model: str | None,
        operation: Callable[[CopilotClient], AsyncIterator[bytes]],
    ) -> AsyncIterator[bytes]:
        await self.sync_accounts()
        tried: set[str] = set()
        max_attempts = max(1, min(POOL_MAX_RETRY_ATTEMPTS, max(len(self.entries), 1)))
        attempts = 0

        while attempts < max_attempts:
            attempts += 1
            lease = await self.acquire(
                model=model,
                is_stream=True,
                exclude_account_ids=tried,
            )
            yielded = False
            try:
                async for chunk in operation(lease.client):
                    yielded = True
                    yield chunk
                await self._mark_success(lease.entry)
                return
            except Exception as exc:
                decision = await self._handle_request_error(lease, exc)
                if yielded:
                    raise
                if decision.retry_same_account and not lease.force_refreshed:
                    lease.force_refreshed = True
                    try:
                        await self._force_refresh_entry(lease.entry)
                        async for chunk in operation(lease.client):
                            yielded = True
                            yield chunk
                        await self._mark_success(lease.entry)
                        return
                    except Exception as retry_exc:
                        decision = await self._handle_request_error(lease, retry_exc)
                        if yielded or not decision.retry_other_account:
                            raise
                if not decision.retry_other_account:
                    raise
                tried.add(lease.account_id)
            finally:
                await lease.release()

        raise PoolError("No upstream account could open the requested stream.")

    # ── Internal Helpers ────────────────────────────────────────────

    def _candidate_entries(
        self,
        *,
        model: str | None,
        exclude_account_ids: set[str],
    ) -> list[PoolEntry]:
        entries = [
            entry
            for entry in self.entries.values()
            if entry.account.account_id not in exclude_account_ids
        ]
        entries.sort(
            key=lambda entry: (
                entry.account.priority,
                entry.account.created_at,
                entry.account.account_id,
            )
        )

        strategy = self.repository.get_rotation_strategy()
        if strategy not in ROTATION_STRATEGIES:
            strategy = "fill-first"

        if strategy == "round-robin" and entries:
            cursor = self._round_robin_cursor % len(entries)
            ordered = entries[cursor:] + entries[:cursor]
            self._round_robin_cursor = (cursor + 1) % len(entries)
            return ordered

        return entries

    async def _prepare_entry(self, entry: PoolEntry, *, model: str | None) -> None:
        if not entry.token_valid:
            await self._force_refresh_entry(entry)
        client = await entry.ensure_client()
        if model and entry.known_model_ids and model not in entry.known_model_ids:
            models = await client.list_models()
            entry.known_models = models
            model_ids = [str(item.get("id")) for item in models if item.get("id")]
            self.repository.update_models(entry.account.account_id, model_ids)
            if model not in model_ids:
                raise ModelUnavailableError(
                    f"Account '{entry.account.display_name}' does not expose model '{model}'."
                )

    async def _force_refresh_entry(self, entry: PoolEntry) -> None:
        async with entry.refresh_lock:
            if entry.token_valid:
                await entry.ensure_client()
                return
            token, expires_at, api_base_url = await entry.token_fetcher(entry.account.github_token)
            entry.account.copilot_token = token
            entry.account.expires_at = expires_at
            if api_base_url:
                entry.account.api_base_url = api_base_url
            entry.account.reauth_required = False
            self.repository.update_tokens(
                entry.account.account_id,
                copilot_token=token,
                expires_at=expires_at,
                api_base_url=entry.account.api_base_url,
            )
            await entry.ensure_client()

    async def _release_entry(self, entry: PoolEntry, *, is_stream: bool) -> None:
        async with self._selection_lock:
            entry.active_requests = max(entry.active_requests - 1, 0)
            if is_stream:
                entry.active_streams = max(entry.active_streams - 1, 0)
            should_remove = entry.removed and entry.active_requests == 0
        if should_remove:
            await entry.close()
            self.entries.pop(entry.account.account_id, None)

    async def _handle_prepare_error(self, entry: PoolEntry, exc: Exception) -> None:
        message = str(exc)
        if isinstance(exc, TokenError):
            lower = message.lower()
            if "invalid or expired" in lower:
                entry.account.reauth_required = True
                self.repository.mark_account(
                    entry.account.account_id,
                    reauth_required=True,
                    last_error=message,
                    last_error_at=time.time(),
                )
                return
        await self._cooldown(entry, POOL_FAILURE_COOLDOWN, message)

    async def _handle_request_error(self, lease: AccountLease, exc: Exception) -> RetryDecision:
        entry = lease.entry
        now = time.time()
        if isinstance(exc, TokenError):
            lower = str(exc).lower()
            if "invalid or expired" in lower:
                entry.account.reauth_required = True
                self.repository.mark_account(
                    entry.account.account_id,
                    reauth_required=True,
                    last_error=str(exc),
                    last_error_at=now,
                )
                return RetryDecision(retry_other_account=True, reauth_required=True)

        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            if status_code == 401:
                if not lease.force_refreshed:
                    return RetryDecision(retry_same_account=True)
                entry.account.reauth_required = True
                self.repository.mark_account(
                    entry.account.account_id,
                    reauth_required=True,
                    last_error=str(exc),
                    last_error_at=now,
                )
                return RetryDecision(retry_other_account=True, reauth_required=True)
            if status_code == 429:
                retry_after = exc.response.headers.get("retry-after", "").strip()
                cooldown = self._rate_limit_cooldown(retry_after)
                await self._cooldown(entry, cooldown, str(exc))
                return RetryDecision(retry_other_account=True, cooldown_seconds=cooldown)
            if status_code in {500, 502, 503, 504}:
                await self._cooldown(entry, POOL_FAILURE_COOLDOWN, str(exc))
                return RetryDecision(
                    retry_other_account=True,
                    cooldown_seconds=POOL_FAILURE_COOLDOWN,
                )
            if status_code == 403:
                await self._cooldown(entry, POOL_429_COOLDOWN, str(exc))
                return RetryDecision(retry_other_account=True, cooldown_seconds=POOL_429_COOLDOWN)

        if isinstance(exc, httpx.RequestError):
            await self._cooldown(entry, POOL_FAILURE_COOLDOWN, str(exc))
            return RetryDecision(retry_other_account=True, cooldown_seconds=POOL_FAILURE_COOLDOWN)

        self.repository.mark_account(
            entry.account.account_id,
            last_error=str(exc),
            last_error_at=now,
        )
        return RetryDecision()

    async def _mark_success(self, entry: PoolEntry) -> None:
        entry.error_streak = 0
        entry.cooldown_until = 0.0
        entry.account.cooldown_until = 0.0
        self.repository.mark_account(
            entry.account.account_id,
            reauth_required=False,
            last_used_at=time.time(),
            last_error="",
            last_error_at=0.0,
            cooldown_until=0.0,
        )

    async def _cooldown(self, entry: PoolEntry, seconds: float, message: str) -> None:
        entry.error_streak += 1
        now = time.time()
        cooldown_until = now + max(seconds, 0)
        entry.cooldown_until = cooldown_until
        entry.account.cooldown_until = cooldown_until
        last_rate_limited_at = entry.account.last_rate_limited_at
        if self._looks_rate_limited(message):
            last_rate_limited_at = now
            entry.account.last_rate_limited_at = now
        self.repository.mark_account(
            entry.account.account_id,
            last_error=message,
            last_error_at=now,
            cooldown_until=cooldown_until,
            last_rate_limited_at=last_rate_limited_at,
        )

    def _cooldown_active(self, entry: PoolEntry, *, now: float | None = None) -> bool:
        current_time = time.time() if now is None else now
        cooldown_until = max(entry.cooldown_until, entry.account.cooldown_until)
        entry.cooldown_until = cooldown_until
        return cooldown_until > current_time

    def _rate_limit_cooldown(self, retry_after: str) -> float:
        retry_after = retry_after.strip()
        if retry_after:
            try:
                cooldown = float(retry_after)
            except ValueError:
                pass
            else:
                if math.isfinite(cooldown):
                    return max(cooldown, 0.0)
            try:
                retry_at = parsedate_to_datetime(retry_after)
            except (TypeError, ValueError, IndexError, OverflowError):
                retry_at = None
            if retry_at is not None:
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(retry_at.timestamp() - time.time(), 0.0)
        if self._healthy_enabled_account_count() <= 1:
            return float(POOL_SINGLE_ACCOUNT_429_COOLDOWN)
        return float(POOL_429_COOLDOWN)

    def _healthy_enabled_account_count(self) -> int:
        return sum(
            1
            for entry in self.entries.values()
            if entry.account.enabled and not entry.account.reauth_required and not entry.removed
        )

    @staticmethod
    def _looks_rate_limited(message: str) -> bool:
        lowered = message.lower()
        return any(
            marker in lowered
            for marker in (
                " 429",
                "rate limit",
                "rate-limit",
                "rate_limited",
                "too many requests",
                "token usage",
            )
        )


def clone_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy request payloads before retries mutate them."""

    return copy.deepcopy(payload)
