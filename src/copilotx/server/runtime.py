"""Unified server runtime and adaptive model routing hints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol, runtime_checkable

from copilotx.auth.pool import TokenPool
from copilotx.auth.token import TokenManager
from copilotx.proxy.client import CopilotClient

CHAT_COMPLETIONS_API = "chat_completions"
RESPONSES_API = "responses"


@dataclass(slots=True)
class ModelRoutingState:
    """Learned and inferred endpoint support for one model."""

    vendor: str = ""
    preferred_api: str | None = None
    chat_supported: bool | None = None
    responses_supported: bool | None = None


class ModelRoutingRegistry:
    """Route models to the most likely compatible upstream API surface."""

    def __init__(self) -> None:
        self._states: dict[str, ModelRoutingState] = {}

    def observe_models(self, models: list[dict[str, Any]]) -> None:
        for model in models:
            model_id = str(model.get("id", "")).strip()
            if not model_id:
                continue
            vendor = str(model.get("vendor", "")).strip()
            self._state_for(model_id, vendor=vendor)

    def preferred_api(self, model: str | None, requested_api: str) -> str:
        if not model:
            return requested_api

        state = self._state_for(model)
        alternate_api = _alternate_api(requested_api)
        requested_supported = _supports_api(state, requested_api)
        alternate_supported = _supports_api(state, alternate_api)

        if requested_supported is True:
            return requested_api
        if requested_supported is False and alternate_supported is not False:
            return alternate_api
        if alternate_supported is True:
            return alternate_api
        if alternate_supported is False:
            return requested_api
        if state.preferred_api == alternate_api:
            return alternate_api
        return requested_api

    def mark_api_success(self, model: str | None, api: str) -> None:
        if not model:
            return
        state = self._state_for(model)
        _set_api_support(state, api, True)

    def mark_api_unsupported(self, model: str | None, api: str) -> None:
        if not model:
            return
        state = self._state_for(model)
        _set_api_support(state, api, False)

    def _state_for(self, model: str, *, vendor: str = "") -> ModelRoutingState:
        state = self._states.setdefault(model, ModelRoutingState())
        if vendor and not state.vendor:
            state.vendor = vendor
        preferred_api = _infer_preferred_api(model, state.vendor or vendor)
        if preferred_api and state.preferred_api is None:
            state.preferred_api = preferred_api
        return state


def _alternate_api(api: str) -> str:
    return RESPONSES_API if api == CHAT_COMPLETIONS_API else CHAT_COMPLETIONS_API


def _supports_api(state: ModelRoutingState, api: str) -> bool | None:
    if api == CHAT_COMPLETIONS_API:
        return state.chat_supported
    return state.responses_supported


def _set_api_support(state: ModelRoutingState, api: str, supported: bool) -> None:
    if api == CHAT_COMPLETIONS_API:
        state.chat_supported = supported
    else:
        state.responses_supported = supported


def _infer_preferred_api(model: str, vendor: str = "") -> str | None:
    model_lower = model.lower()
    vendor_lower = vendor.lower()

    if model_lower.startswith(("claude-", "gemini-")):
        return CHAT_COMPLETIONS_API
    if model_lower.startswith("gpt-5") or "codex" in model_lower:
        return RESPONSES_API

    if vendor_lower in {"anthropic", "google", "google-deepmind"}:
        return CHAT_COMPLETIONS_API
    if vendor_lower == "openai" and (
        model_lower.startswith("gpt-5") or "codex" in model_lower
    ):
        return RESPONSES_API

    return None


@runtime_checkable
class AppRuntime(Protocol):
    """Shared server runtime interface for single-account and pooled modes."""

    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def execute(
        self,
        *,
        model: str | None,
        operation: Callable[[CopilotClient], Awaitable[Any]],
    ) -> Any: ...

    async def probe(
        self,
        *,
        model: str | None,
        operation: Callable[[CopilotClient], Awaitable[Any]],
    ) -> Any: ...

    async def stream(
        self,
        *,
        model: str | None,
        operation: Callable[[CopilotClient], AsyncIterator[bytes]],
    ) -> AsyncIterator[bytes]: ...

    async def list_models(self) -> list[dict[str, Any]]: ...

    async def health_snapshot(self) -> dict[str, Any]: ...

    def preferred_api_surface(self, model: str | None, requested_api: str) -> str: ...

    def mark_api_success(self, model: str | None, api: str) -> None: ...

    def mark_api_unsupported(self, model: str | None, api: str) -> None: ...


class LegacyRuntime:
    """Single-account runtime backed by TokenManager and one CopilotClient."""

    def __init__(self, token_manager: TokenManager) -> None:
        self.token_manager = token_manager
        self.client: CopilotClient | None = None
        self.routing = ModelRoutingRegistry()

    async def startup(self) -> None:
        token = await self.token_manager.ensure_copilot_token()
        self.client = CopilotClient(token, api_base_url=self.token_manager.api_base_url)
        await self.client.__aenter__()

    async def shutdown(self) -> None:
        if self.client is not None:
            await self.client.__aexit__(None, None, None)
            self.client = None

    async def execute(
        self,
        *,
        model: str | None,
        operation: Callable[[CopilotClient], Awaitable[Any]],
    ) -> Any:
        client = await self._get_ready_client()
        return await operation(client)

    async def probe(
        self,
        *,
        model: str | None,
        operation: Callable[[CopilotClient], Awaitable[Any]],
    ) -> Any:
        return await self.execute(model=model, operation=operation)

    async def stream(
        self,
        *,
        model: str | None,
        operation: Callable[[CopilotClient], AsyncIterator[bytes]],
    ) -> AsyncIterator[bytes]:
        client = await self._get_ready_client()
        async for chunk in operation(client):
            yield chunk

    async def list_models(self) -> list[dict[str, Any]]:
        client = await self._get_ready_client()
        models = await client.list_models()
        self.routing.observe_models(models)
        return models

    async def health_snapshot(self) -> dict[str, Any]:
        return {
            "authenticated": self.token_manager.is_authenticated,
            "token_valid": self.token_manager.copilot_token_valid,
            "token_expires_in": self.token_manager.expires_in_seconds,
        }

    def preferred_api_surface(self, model: str | None, requested_api: str) -> str:
        return self.routing.preferred_api(model, requested_api)

    def mark_api_success(self, model: str | None, api: str) -> None:
        self.routing.mark_api_success(model, api)

    def mark_api_unsupported(self, model: str | None, api: str) -> None:
        self.routing.mark_api_unsupported(model, api)

    async def _get_ready_client(self) -> CopilotClient:
        if self.client is None:
            raise RuntimeError("Server runtime is not started.")

        token = await self.token_manager.ensure_copilot_token()
        self.client.update_token(token)
        self.client.update_api_base(self.token_manager.api_base_url)
        return self.client


class PoolRuntime:
    """Multi-account runtime backed by TokenPool."""

    def __init__(self, pool: TokenPool) -> None:
        self.pool = pool
        self.routing = ModelRoutingRegistry()

    async def startup(self) -> None:
        await self.pool.__aenter__()

    async def shutdown(self) -> None:
        await self.pool.__aexit__(None, None, None)

    async def execute(
        self,
        *,
        model: str | None,
        operation: Callable[[CopilotClient], Awaitable[Any]],
    ) -> Any:
        return await self.pool.execute(model=model, operation=operation)

    async def probe(
        self,
        *,
        model: str | None,
        operation: Callable[[CopilotClient], Awaitable[Any]],
    ) -> Any:
        return await self.pool.probe(model=model, operation=operation)

    async def stream(
        self,
        *,
        model: str | None,
        operation: Callable[[CopilotClient], AsyncIterator[bytes]],
    ) -> AsyncIterator[bytes]:
        async for chunk in self.pool.stream(model=model, operation=operation):
            yield chunk

    async def list_models(self) -> list[dict[str, Any]]:
        models = await self.pool.list_models()
        self.routing.observe_models(models)
        return models

    async def health_snapshot(self) -> dict[str, Any]:
        return await self.pool.health_snapshot()

    def preferred_api_surface(self, model: str | None, requested_api: str) -> str:
        return self.routing.preferred_api(model, requested_api)

    def mark_api_success(self, model: str | None, api: str) -> None:
        self.routing.mark_api_success(model, api)

    def mark_api_unsupported(self, model: str | None, api: str) -> None:
        self.routing.mark_api_unsupported(model, api)


def coerce_runtime(runtime: TokenManager | TokenPool | AppRuntime) -> AppRuntime:
    """Wrap legacy runtime objects into the shared AppRuntime interface."""
    if isinstance(runtime, TokenPool):
        return PoolRuntime(runtime)
    if isinstance(runtime, TokenManager):
        return LegacyRuntime(runtime)
    return runtime
