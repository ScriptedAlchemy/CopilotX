from __future__ import annotations

import asyncio
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from rich.console import Console

import copilotx.cli as cli_module
from copilotx.auth.accounts import AccountRecord, AccountRepository
from copilotx.auth.pool import TokenPool
from copilotx.proxy.client import merge_forced_models
from copilotx.proxy.translator import openai_responses_to_chat_request
from copilotx.server.app import create_app
from copilotx.server.runtime import ModelRoutingRegistry

RESPONSES_ONLY_ERROR = (
    '{"error":{"message":"model \\"%s\\" is not accessible via '
    'the /chat/completions endpoint","code":"unsupported_api_for_model"}}'
)
CHAT_ONLY_ERROR = (
    '{"error":{"message":"model \\"%s\\" '
    'does not support Responses API.","code":"unsupported_api_for_model"}}'
)


def make_account(
    account_id: str,
    label: str,
    github_token: str,
    copilot_token: str,
    *,
    enabled: bool = True,
    model_ids: list[str] | None = None,
) -> AccountRecord:
    now = time.time()
    return AccountRecord(
        account_id=account_id,
        github_login=label,
        github_user_id=label,
        label=label,
        github_token=github_token,
        copilot_token=copilot_token,
        expires_at=now + 3600,
        api_base_url="https://upstream.test",
        enabled=enabled,
        model_ids=model_ids or [],
        created_at=now,
        updated_at=now,
    )


def make_repo(tmp_path: Path) -> tuple[AccountRepository, Path]:
    auth_path = tmp_path / "auth.json"
    repo = AccountRepository(
        path=tmp_path / "accounts.db",
        legacy_auth_path=auth_path,
    )
    return repo, auth_path


def make_http_error(
    status_code: int,
    message: str,
    *,
    headers: dict[str, str] | None = None,
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://upstream.test/v1/chat/completions")
    response = httpx.Response(status_code, request=request, text=message, headers=headers)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}: {message}",
        request=request,
        response=response,
    )


class FakeCopilotClient:
    def __init__(
        self,
        token: str,
        api_base_url: str,
        state: dict[str, dict],
    ) -> None:
        self.token = token
        self.api_base_url = api_base_url
        self.state = state

    async def __aenter__(self) -> "FakeCopilotClient":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    def update_token(self, token: str) -> None:
        self.token = token

    def update_api_base(self, api_base_url: str) -> None:
        self.api_base_url = api_base_url

    async def list_models(self) -> list[dict]:
        account = self.state[self.token]
        return [{"id": model_id, "vendor": "test"} for model_id in account["models"]]

    async def chat_completions(self, payload: dict) -> dict:
        account = self.state[self.token]
        account["chat_calls"] += 1
        if account["errors"]:
            raise account["errors"].pop(0)
        if "chat_result" in account:
            return copy.deepcopy(account["chat_result"])
        return {
            "id": f"chatcmpl-{account['name']}",
            "object": "chat.completion",
            "model": payload.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"ok from {account['name']}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "account": account["name"],
        }

    async def chat_completions_stream(self, payload: dict):
        account = self.state[self.token]
        account["stream_calls"] = account.get("stream_calls", 0) + 1
        if account.get("stream_errors"):
            raise account["stream_errors"].pop(0)
        yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def responses(
        self,
        payload: dict,
        *,
        vision: bool = False,
        initiator: str = "user",
    ) -> dict:
        account = self.state[self.token]
        account["responses_calls"] = account.get("responses_calls", 0) + 1
        if account.get("responses_errors"):
            raise account["responses_errors"].pop(0)
        result = account.get("responses_result", {})
        return copy.deepcopy(result)

    async def responses_stream(
        self,
        payload: dict,
        *,
        vision: bool = False,
        initiator: str = "user",
    ):
        account = self.state[self.token]
        account["responses_stream_calls"] = account.get("responses_stream_calls", 0) + 1
        if account.get("responses_stream_errors"):
            raise account["responses_stream_errors"].pop(0)
        for chunk in account.get("responses_stream_chunks", []):
            yield chunk


class HiddenModelCopilotClient(FakeCopilotClient):
    async def list_models(self) -> list[dict]:
        return copy.deepcopy(self.state[self.token]["models"])


class ForceAwareCopilotClient(HiddenModelCopilotClient):
    async def list_models(self) -> list[dict]:
        raw_models = copy.deepcopy(self.state[self.token]["models"])
        return merge_forced_models(raw_models)


def test_models_route_includes_hidden_models_from_upstream_catalog(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": [
                {"id": "gpt-5.4", "vendor": "OpenAI", "model_picker_enabled": False},
                {
                    "id": "claude-opus-4.6",
                    "vendor": "Anthropic",
                    "model_picker_enabled": False,
                },
            ],
            "errors": [],
            "chat_calls": 0,
        }
    }

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: HiddenModelCopilotClient(
            token,
            api_base,
            state,
        ),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    ids = {model["id"] for model in response.json()["data"]}
    assert "gpt-5.4" in ids
    assert "claude-opus-4.6" in ids


def test_token_pool_can_use_hidden_models_when_requested_explicitly(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": [
                {"id": "gpt-5.4", "vendor": "OpenAI", "model_picker_enabled": False},
            ],
            "errors": [],
            "chat_calls": 0,
            "responses_result": {
                "id": "resp-hidden-1",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hidden ok"}],
                    }
                ],
            },
        }
    }

    async def run() -> dict:
        pool = TokenPool(
            repo,
            client_factory=lambda token, api_base: HiddenModelCopilotClient(
                token,
                api_base,
                state,
            ),
        )
        try:
            await pool.sync_accounts(force=True)
            return await pool.execute(
                model="gpt-5.4",
                operation=lambda client: client.responses(
                    {
                        "model": "gpt-5.4",
                        "input": "Say hello.",
                        "max_output_tokens": 16,
                    }
                ),
            )
        finally:
            await pool.__aexit__(None, None, None)

    result = asyncio.run(run())

    assert result["model"] == "gpt-5.4"
    assert state["token-1"]["responses_calls"] == 1


def test_merge_forced_models_appends_missing_manual_ids(monkeypatch) -> None:
    monkeypatch.setenv("COPILOTX_FORCE_MODELS", "gpt-5.4, claude-opus-4.6")

    merged = merge_forced_models(
        [
            {
                "id": "gpt-4o",
                "vendor": "Azure OpenAI",
                "model_picker_enabled": True,
            }
        ]
    )

    by_id = {model["id"]: model for model in merged}
    assert by_id["gpt-4o"]["vendor"] == "Azure OpenAI"
    assert by_id["gpt-5.4"]["vendor"] == "OpenAI"
    assert by_id["gpt-5.4"]["copilotx_forced"] is True
    assert by_id["gpt-5.4"]["model_picker_enabled"] is False
    assert by_id["claude-opus-4.6"]["vendor"] == "Anthropic"


def test_models_route_includes_forced_models_missing_from_upstream_catalog(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))
    monkeypatch.setenv("COPILOTX_FORCE_MODELS", "gpt-5.4, claude-opus-4.6")

    state = {
        "token-1": {
            "name": "alpha",
            "models": [
                {
                    "id": "gpt-4o",
                    "vendor": "Azure OpenAI",
                    "model_picker_enabled": True,
                }
            ],
            "errors": [],
            "chat_calls": 0,
        }
    }

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: ForceAwareCopilotClient(
            token,
            api_base,
            state,
        ),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    ids = {model["id"] for model in response.json()["data"]}
    assert "gpt-4o" in ids
    assert "gpt-5.4" in ids
    assert "claude-opus-4.6" in ids


def test_token_pool_can_use_forced_model_missing_from_upstream_catalog(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(
        make_account(
            "acct-1",
            "alpha",
            "gh-1",
            "token-1",
            model_ids=["gpt-4o"],
        )
    )
    monkeypatch.setenv("COPILOTX_FORCE_MODELS", "gpt-5.4")

    state = {
        "token-1": {
            "name": "alpha",
            "models": [
                {
                    "id": "gpt-4o",
                    "vendor": "Azure OpenAI",
                    "model_picker_enabled": True,
                }
            ],
            "errors": [],
            "chat_calls": 0,
            "responses_result": {
                "id": "resp-forced-1",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "forced ok"}],
                    }
                ],
            },
        }
    }

    async def run() -> dict:
        pool = TokenPool(
            repo,
            client_factory=lambda token, api_base: ForceAwareCopilotClient(
                token,
                api_base,
                state,
            ),
        )
        try:
            await pool.sync_accounts(force=True)
            return await pool.execute(
                model="gpt-5.4",
                operation=lambda client: client.responses(
                    {
                        "model": "gpt-5.4",
                        "input": "Say hello.",
                        "max_output_tokens": 16,
                    }
                ),
            )
        finally:
            await pool.__aexit__(None, None, None)

    result = asyncio.run(run())

    assert result["model"] == "gpt-5.4"
    assert state["token-1"]["responses_calls"] == 1


def test_token_pool_retries_other_account_for_forced_model_when_first_account_rejects_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(
        make_account(
            "acct-1",
            "alpha",
            "gh-1",
            "token-1",
            model_ids=["gpt-4o"],
        )
    )
    repo.upsert_account(
        make_account(
            "acct-2",
            "beta",
            "gh-2",
            "token-2",
            model_ids=["gpt-4o"],
        )
    )
    monkeypatch.setenv("COPILOTX_FORCE_MODELS", "gpt-5.4")

    state = {
        "token-1": {
            "name": "alpha",
            "models": [{"id": "gpt-4o", "vendor": "Azure OpenAI"}],
            "errors": [],
            "chat_calls": 0,
            "responses_errors": [make_http_error(400, CHAT_ONLY_ERROR % "gpt-5.4")],
        },
        "token-2": {
            "name": "beta",
            "models": [{"id": "gpt-4o", "vendor": "Azure OpenAI"}],
            "errors": [],
            "chat_calls": 0,
            "responses_result": {
                "id": "resp-forced-2",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "fallback ok"}],
                    }
                ],
            },
        },
    }

    async def run() -> dict:
        pool = TokenPool(
            repo,
            client_factory=lambda token, api_base: ForceAwareCopilotClient(
                token,
                api_base,
                state,
            ),
        )
        try:
            await pool.sync_accounts(force=True)
            return await pool.execute(
                model="gpt-5.4",
                operation=lambda client: client.responses(
                    {
                        "model": "gpt-5.4",
                        "input": "Say hello.",
                        "max_output_tokens": 16,
                    }
                ),
            )
        finally:
            await pool.__aexit__(None, None, None)

    result = asyncio.run(run())

    assert result["model"] == "gpt-5.4"
    assert state["token-1"]["responses_calls"] == 1
    assert state["token-2"]["responses_calls"] == 1


def test_token_pool_retries_all_accounts_for_forced_model_support_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, _ = make_repo(tmp_path)
    for index, token in enumerate(("token-1", "token-2", "token-3", "token-4"), start=1):
        repo.upsert_account(
            make_account(
                f"acct-{index}",
                f"acct-{index}",
                f"gh-{index}",
                token,
                model_ids=["gpt-4o"],
            )
        )
    monkeypatch.setenv("COPILOTX_FORCE_MODELS", "gpt-5.4")

    state = {
        "token-1": {
            "name": "alpha",
            "models": [{"id": "gpt-4o", "vendor": "Azure OpenAI"}],
            "errors": [],
            "chat_calls": 0,
            "responses_errors": [make_http_error(400, CHAT_ONLY_ERROR % "gpt-5.4")],
        },
        "token-2": {
            "name": "beta",
            "models": [{"id": "gpt-4o", "vendor": "Azure OpenAI"}],
            "errors": [],
            "chat_calls": 0,
            "responses_errors": [make_http_error(400, CHAT_ONLY_ERROR % "gpt-5.4")],
        },
        "token-3": {
            "name": "gamma",
            "models": [{"id": "gpt-4o", "vendor": "Azure OpenAI"}],
            "errors": [],
            "chat_calls": 0,
            "responses_errors": [make_http_error(400, CHAT_ONLY_ERROR % "gpt-5.4")],
        },
        "token-4": {
            "name": "delta",
            "models": [{"id": "gpt-4o", "vendor": "Azure OpenAI"}],
            "errors": [],
            "chat_calls": 0,
            "responses_result": {
                "id": "resp-forced-4",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "fourth account ok"}],
                    }
                ],
            },
        },
    }

    async def run() -> dict:
        pool = TokenPool(
            repo,
            client_factory=lambda token, api_base: ForceAwareCopilotClient(
                token,
                api_base,
                state,
            ),
        )
        try:
            await pool.sync_accounts(force=True)
            return await pool.execute(
                model="gpt-5.4",
                operation=lambda client: client.responses(
                    {
                        "model": "gpt-5.4",
                        "input": "Say hello.",
                        "max_output_tokens": 16,
                    }
                ),
            )
        finally:
            await pool.__aexit__(None, None, None)

    result = asyncio.run(run())

    assert result["model"] == "gpt-5.4"
    assert state["token-1"]["responses_calls"] == 1
    assert state["token-2"]["responses_calls"] == 1
    assert state["token-3"]["responses_calls"] == 1
    assert state["token-4"]["responses_calls"] == 1


def test_responses_stream_route_falls_back_after_single_account_forced_model_rejection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(
        make_account(
            "acct-1",
            "alpha",
            "gh-1",
            "token-1",
            model_ids=["gpt-4o"],
        )
    )
    monkeypatch.setenv("COPILOTX_FORCE_MODELS", "gpt-5.4")

    state = {
        "token-1": {
            "name": "alpha",
            "models": [{"id": "gpt-4o", "vendor": "Azure OpenAI"}],
            "errors": [],
            "chat_calls": 0,
            "responses_stream_errors": [make_http_error(400, CHAT_ONLY_ERROR % "gpt-5.4")],
            "chat_result": {
                "id": "chat-fallback-1",
                "object": "chat.completion",
                "created": 1772973005,
                "model": "gpt-5.4",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "stream fallback ok",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                },
            },
        }
    }

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: ForceAwareCopilotClient(
            token,
            api_base,
            state,
        ),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5.4",
                "input": "Say hello.",
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "stream fallback ok" in response.text
    assert state["token-1"]["responses_stream_calls"] == 1
    assert state["token-1"]["chat_calls"] == 1


def test_disabling_last_enabled_account_removes_legacy_auth(tmp_path: Path) -> None:
    repo, auth_path = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "cp-1"))

    assert auth_path.exists()

    updated = repo.set_account_enabled("acct-1", False)

    assert updated is not None
    assert updated.enabled is False
    assert repo.get_default_account_id() == ""
    assert not auth_path.exists()


def test_disabling_default_account_promotes_next_enabled_account(tmp_path: Path) -> None:
    repo, auth_path = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "cp-1"))
    repo.upsert_account(make_account("acct-2", "beta", "gh-2", "cp-2"))

    updated = repo.set_account_enabled("acct-1", False)

    assert updated is not None
    assert updated.enabled is False
    assert repo.get_default_account_id() == "acct-2"
    assert json.loads(auth_path.read_text())["github_token"] == "gh-2"


def test_token_pool_execute_retries_other_account_before_initial_sync(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))
    repo.upsert_account(make_account("acct-2", "beta", "gh-2", "token-2"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["gpt-5.4"],
            "errors": [make_http_error(429, "rate limited")],
            "chat_calls": 0,
        },
        "token-2": {
            "name": "beta",
            "models": ["gpt-5.4"],
            "errors": [],
            "chat_calls": 0,
        },
    }

    async def run() -> dict:
        pool = TokenPool(
            repo,
            client_factory=lambda token, api_base: FakeCopilotClient(
                token,
                api_base,
                state,
            ),
        )
        try:
            return await pool.execute(
                model="gpt-5.4",
                operation=lambda client: client.chat_completions({"model": "gpt-5.4"}),
            )
        finally:
            await pool.__aexit__(None, None, None)

    result = asyncio.run(run())

    assert result["account"] == "beta"
    assert state["token-1"]["chat_calls"] == 1
    assert state["token-2"]["chat_calls"] == 1


def test_token_pool_acquire_preserves_successful_lease_after_previous_error(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(
        make_account(
            "acct-1",
            "alpha",
            "gh-1",
            "token-1",
            model_ids=["claude-opus-4.6"],
        )
    )
    repo.upsert_account(
        make_account(
            "acct-2",
            "beta",
            "gh-2",
            "token-2",
            model_ids=["gpt-5.4"],
        )
    )

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["claude-opus-4.6"],
            "errors": [],
            "chat_calls": 0,
        },
        "token-2": {
            "name": "beta",
            "models": ["gpt-5.4"],
            "errors": [],
            "chat_calls": 0,
        },
    }

    async def run() -> None:
        pool = TokenPool(
            repo,
            client_factory=lambda token, api_base: FakeCopilotClient(
                token,
                api_base,
                state,
            ),
        )
        await pool.sync_accounts(force=True)
        lease = await pool.acquire(model="gpt-5.4", is_stream=False)
        try:
            assert lease.account_id == "acct-2"
            assert lease.entry.active_requests == 1
        finally:
            await lease.release()
            await pool.__aexit__(None, None, None)

    asyncio.run(run())


def test_openai_route_uses_token_pool_runtime(tmp_path: Path) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["gemini-3.1-pro"],
            "errors": [],
            "chat_calls": 0,
        }
    }

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: FakeCopilotClient(
            token,
            api_base,
            state,
        ),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["accounts_healthy"] == 1

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-3.1-pro",
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        )

    assert response.status_code == 200
    assert response.json()["account"] == "alpha"
    assert state["token-1"]["chat_calls"] == 1


def test_api_key_middleware_trusts_localhost_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["gpt-5.4"],
            "errors": [],
            "chat_calls": 0,
        }
    }

    monkeypatch.setenv("COPILOTX_API_KEY", "secret")
    monkeypatch.delenv("COPILOTX_TRUST_LOCALHOST", raising=False)

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: FakeCopilotClient(
            token,
            api_base,
            state,
        ),
    )
    app = create_app(pool)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "gpt-5.4"


def test_api_key_middleware_can_require_auth_on_localhost(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["gpt-5.4"],
            "errors": [],
            "chat_calls": 0,
        }
    }

    monkeypatch.setenv("COPILOTX_API_KEY", "secret")
    monkeypatch.setenv("COPILOTX_TRUST_LOCALHOST", "0")

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: FakeCopilotClient(
            token,
            api_base,
            state,
        ),
    )
    app = create_app(pool)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        unauthorized = client.get("/v1/models")
        authorized = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer secret"},
        )

    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["type"] == "authentication_error"
    assert authorized.status_code == 200
    assert authorized.json()["data"][0]["id"] == "gpt-5.4"


def test_anthropic_streaming_error_returns_json_response(tmp_path: Path) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["claude-sonnet-4.6"],
            "errors": [],
            "stream_errors": [
                make_http_error(
                    400,
                    '{"error":{"message":"model not accessible via chat/completions"}}',
                )
            ],
            "chat_calls": 0,
        }
    }

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: FakeCopilotClient(
            token,
            api_base,
            state,
        ),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4.6",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "model not accessible via chat/completions"


def test_anthropic_non_stream_falls_back_to_responses_for_responses_only_models(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["mystery-responses-model"],
            "errors": [
                make_http_error(
                    400,
                    RESPONSES_ONLY_ERROR % "mystery-responses-model",
                )
            ],
            "responses_result": {
                "model": "mystery-responses-model",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "gpt54 ok"}
                        ],
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 8},
            },
            "chat_calls": 0,
        }
    }

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: FakeCopilotClient(
            token,
            api_base,
            state,
        ),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "mystery-responses-model",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        )

    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "gpt54 ok"
    assert state["token-1"]["chat_calls"] == 1
    assert state["token-1"]["responses_calls"] == 1


def test_anthropic_stream_falls_back_to_buffered_responses_for_responses_only_models(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["mystery-responses-model"],
            "errors": [],
            "stream_errors": [
                make_http_error(
                    400,
                    RESPONSES_ONLY_ERROR % "mystery-responses-model",
                )
            ],
            "responses_result": {
                "model": "mystery-responses-model",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "gpt54 ok"}
                        ],
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 8},
            },
            "chat_calls": 0,
        }
    }

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: FakeCopilotClient(
            token,
            api_base,
            state,
        ),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "mystery-responses-model",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        )

    assert response.status_code == 200
    assert "event: message_start" in response.text
    assert "gpt54 ok" in response.text
    assert state["token-1"]["responses_calls"] == 1


def test_openai_non_stream_falls_back_to_responses_for_responses_only_models(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["mystery-responses-model"],
            "errors": [
                make_http_error(
                    400,
                    RESPONSES_ONLY_ERROR % "mystery-responses-model",
                )
            ],
            "responses_result": {
                "id": "resp-openai-1",
                "model": "mystery-responses-model",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "gpt54 chat ok"}
                        ],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4},
            },
            "chat_calls": 0,
        }
    }

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: FakeCopilotClient(
            token,
            api_base,
            state,
        ),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "mystery-responses-model",
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "gpt54 chat ok"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert state["token-1"]["chat_calls"] == 1
    assert state["token-1"]["responses_calls"] == 1


def test_openai_stream_falls_back_to_responses_for_responses_only_models(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["mystery-responses-model"],
            "errors": [],
            "stream_errors": [
                make_http_error(
                    400,
                    RESPONSES_ONLY_ERROR % "mystery-responses-model",
                )
            ],
            "responses_result": {
                "id": "resp-openai-2",
                "model": "mystery-responses-model",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "gpt54 stream ok"}
                        ],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4},
            },
            "chat_calls": 0,
        }
    }

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: FakeCopilotClient(
            token,
            api_base,
            state,
        ),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "mystery-responses-model",
                "stream": True,
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        )

    assert response.status_code == 200
    assert "gpt54 stream ok" in response.text
    assert "data: [DONE]" in response.text
    assert state["token-1"]["responses_calls"] == 1


def test_openai_responses_to_chat_request_preserves_string_input() -> None:
    payload = openai_responses_to_chat_request(
        {
            "model": "claude-sonnet-4.6",
            "input": "Say hello.",
        }
    )

    assert payload["messages"] == [{"role": "user", "content": "Say hello."}]


def test_openai_responses_to_chat_request_maps_instructions_to_system_message() -> None:
    payload = openai_responses_to_chat_request(
        {
            "model": "claude-sonnet-4.6",
            "instructions": "Reply tersely.",
            "input": "Say hello.",
        }
    )

    assert payload["messages"][0] == {
        "role": "system",
        "content": "Reply tersely.",
    }
    assert payload["messages"][1] == {"role": "user", "content": "Say hello."}


def test_openai_fallback_to_responses_preserves_vision_flag(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    captured: dict[str, object] = {}
    state = {
        "token-1": {
            "name": "alpha",
            "models": ["gpt-5.4"],
            "errors": [
                make_http_error(
                    400,
                    RESPONSES_ONLY_ERROR % "gpt-5.4",
                )
            ],
            "responses_result": {
                "id": "resp-openai-vision-1",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "vision ok"}
                        ],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4},
            },
            "chat_calls": 0,
        }
    }

    class CapturingClient(FakeCopilotClient):
        async def responses(
            self,
            payload: dict,
            *,
            vision: bool = False,
            initiator: str = "user",
        ) -> dict:
            captured["payload"] = copy.deepcopy(payload)
            captured["vision"] = vision
            return await super().responses(
                payload,
                vision=vision,
                initiator=initiator,
            )

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: CapturingClient(
            token,
            api_base,
            state,
        ),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.4",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image."},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/cat.png"},
                            },
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "vision ok"
    assert captured["vision"] is True
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["input"][0]["content"][1]["type"] == "input_image"


def test_model_routing_registry_learns_and_infers_model_surfaces() -> None:
    registry = ModelRoutingRegistry()

    assert registry.preferred_api("gpt-5.4", "chat_completions") == "responses"
    assert registry.preferred_api("claude-sonnet-4.6", "responses") == "chat_completions"

    assert registry.preferred_api("mystery-model", "chat_completions") == "chat_completions"
    registry.mark_api_unsupported("mystery-model", "chat_completions")
    assert registry.preferred_api("mystery-model", "chat_completions") == "responses"


def test_openai_route_prefers_responses_for_gpt5_without_chat_probe(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["gpt-5.4"],
            "errors": [],
            "responses_result": {
                "id": "resp-direct-1",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "direct responses ok"}],
                    }
                ],
                "usage": {"input_tokens": 8, "output_tokens": 4},
            },
            "chat_calls": 0,
        }
    }

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: FakeCopilotClient(token, api_base, state),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "direct responses ok"
    assert state["token-1"]["chat_calls"] == 0
    assert state["token-1"]["responses_calls"] == 1


def test_openai_route_falls_back_to_requested_chat_surface_when_preferred_responses_fails(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["gpt-5.4"],
            "errors": [],
            "responses_errors": [make_http_error(500, "temporary responses failure")],
            "chat_result": {
                "id": "chat-after-responses-failure-1",
                "object": "chat.completion",
                "created": 1772973003,
                "model": "gpt-5.4",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "chat fallback ok",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 3,
                    "total_tokens": 11,
                },
            },
            "chat_calls": 0,
        }
    }

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: FakeCopilotClient(token, api_base, state),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "chat fallback ok"
    assert state["token-1"]["responses_calls"] == 1
    assert state["token-1"]["chat_calls"] == 1


def test_responses_route_prefers_chat_for_claude_without_responses_probe(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["claude-sonnet-4.6"],
            "errors": [],
            "chat_result": {
                "id": "chat-direct-1",
                "object": "chat.completion",
                "created": 1772973002,
                "model": "claude-sonnet-4.6",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "direct chat ok",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 3,
                    "total_tokens": 12,
                },
            },
            "chat_calls": 0,
        }
    }

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: FakeCopilotClient(token, api_base, state),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": "claude-sonnet-4.6",
                "input": "Say hello.",
            },
        )

    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == "direct chat ok"
    assert state["token-1"].get("responses_calls", 0) == 0
    assert state["token-1"]["chat_calls"] == 1


def test_responses_route_preserves_apply_patch_tool_on_chat_surface(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    captured: dict[str, dict] = {}
    state = {
        "token-1": {
            "name": "alpha",
            "models": ["claude-sonnet-4.6"],
            "errors": [],
            "chat_result": {
                "id": "chat-apply-patch-1",
                "object": "chat.completion",
                "created": 1772973004,
                "model": "claude-sonnet-4.6",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "tool ok",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 2,
                    "total_tokens": 9,
                },
            },
            "chat_calls": 0,
        }
    }

    class CapturingClient(FakeCopilotClient):
        async def chat_completions(self, payload: dict) -> dict:
            captured["payload"] = copy.deepcopy(payload)
            return await super().chat_completions(payload)

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: CapturingClient(token, api_base, state),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": "claude-sonnet-4.6",
                "input": "Apply this patch.",
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert captured["payload"]["tools"][0]["type"] == "function"
    assert captured["payload"]["tools"][0]["function"]["name"] == "apply_patch"
    assert state["token-1"]["chat_calls"] == 1


def test_openai_route_learns_preferred_surface_after_first_fallback(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["mystery-model"],
            "errors": [
                make_http_error(
                    400,
                    (
                        '{"error":{"message":"model \\"mystery-model\\" is not accessible via '
                        'the /chat/completions endpoint","code":"unsupported_api_for_model"}}'
                    ),
                )
            ],
            "responses_result": {
                "id": "resp-learned-1",
                "model": "mystery-model",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "learned ok"}],
                    }
                ],
                "usage": {"input_tokens": 6, "output_tokens": 2},
            },
            "chat_calls": 0,
        }
    }

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: FakeCopilotClient(token, api_base, state),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            json={
                "model": "mystery-model",
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        )
        second = client.post(
            "/v1/chat/completions",
            json={
                "model": "mystery-model",
                "messages": [{"role": "user", "content": "Say hello again."}],
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert state["token-1"]["chat_calls"] == 1
    assert state["token-1"]["responses_calls"] == 2


def test_responses_non_stream_falls_back_to_chat_completions_for_chat_only_models(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["mystery-chat-model"],
            "errors": [],
            "responses_errors": [
                make_http_error(
                    400,
                    CHAT_ONLY_ERROR % "mystery-chat-model",
                )
            ],
            "chat_result": {
                "id": "chat-fallback-1",
                "object": "chat.completion",
                "created": 1772973000,
                "model": "mystery-chat-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "sonnet via chat ok",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 5,
                    "total_tokens": 17,
                },
            },
            "chat_calls": 0,
        }
    }

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: FakeCopilotClient(
            token,
            api_base,
            state,
        ),
    )
    app = create_app(pool)

    with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "mystery-chat-model",
                    "input": [
                        {
                            "role": "user",
                        "content": [{"type": "input_text", "text": "Say hello."}],
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "Read",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"file_path": {"type": "string"}},
                        },
                    }
                ],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["output"][0]["content"][0]["text"] == "sonnet via chat ok"
    assert payload["tools"][0]["name"] == "Read"
    assert state["token-1"]["responses_calls"] == 1
    assert state["token-1"]["chat_calls"] == 1


def test_responses_stream_falls_back_to_chat_completions_for_chat_only_models(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["mystery-chat-model"],
            "errors": [],
            "responses_stream_errors": [
                make_http_error(
                    400,
                    CHAT_ONLY_ERROR % "mystery-chat-model",
                )
            ],
            "chat_result": {
                "id": "chat-fallback-2",
                "object": "chat.completion",
                "created": 1772973001,
                "model": "mystery-chat-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "gemini via chat ok",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 6,
                    "total_tokens": 21,
                },
            },
            "chat_calls": 0,
        }
    }

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: FakeCopilotClient(
            token,
            api_base,
            state,
        ),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": "mystery-chat-model",
                "stream": True,
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Say hello."}],
                    }
                ],
                "text": {"verbosity": "low"},
            },
        )

    assert response.status_code == 200
    assert "event: response.completed" in response.text
    assert "gemini via chat ok" in response.text
    assert state["token-1"]["responses_stream_calls"] == 1
    assert state["token-1"]["chat_calls"] == 1


def test_responses_route_strips_phase_metadata_from_input_items(tmp_path: Path) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    captured: dict[str, dict] = {}
    state = {
        "token-1": {
            "name": "alpha",
            "models": ["mystery-responses-model"],
            "errors": [],
            "responses_result": {
                "id": "resp-phase-1",
                "object": "response",
                "model": "mystery-responses-model",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }
    }

    class CapturingClient(FakeCopilotClient):
        async def responses(
            self,
            payload: dict,
            *,
            vision: bool = False,
            initiator: str = "user",
        ) -> dict:
            captured["payload"] = copy.deepcopy(payload)
            return await super().responses(
                payload, vision=vision, initiator=initiator
            )

    pool = TokenPool(
        repo,
        client_factory=lambda token, api_base: CapturingClient(
            token,
            api_base,
            state,
        ),
    )
    app = create_app(pool)

    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": "mystery-responses-model",
                "input": [
                    {
                        "role": "user",
                        "phase": "final_answer",
                        "content": [{"type": "input_text", "text": "Say hello."}],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert "payload" in captured
    assert captured["payload"]["input"][0]["role"] == "user"
    assert "phase" not in captured["payload"]["input"][0]


def test_render_account_table_shows_cooldown_and_last_429(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))
    repo.mark_account(
        "acct-1",
        last_error="HTTP 429: rate limited",
        last_error_at=997.0,
        cooldown_until=1025.0,
        last_rate_limited_at=997.0,
    )

    test_console = Console(record=True, width=160)
    monkeypatch.setattr(cli_module, "console", test_console)
    monkeypatch.setattr(cli_module.time, "time", lambda: 1000.0)

    cli_module._render_account_table(repo)
    output = test_console.export_text()

    assert "cooling down" in output
    assert "25s" in output
    assert "3s ago" in output


def test_single_account_429_uses_shorter_fallback_cooldown(tmp_path: Path) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["gpt-5.4"],
            "errors": [make_http_error(429, "rate limited")],
            "chat_calls": 0,
        }
    }

    async def run() -> None:
        pool = TokenPool(
            repo,
            client_factory=lambda token, api_base: FakeCopilotClient(
                token,
                api_base,
                state,
            ),
        )
        try:
            await pool.execute(
                model="gpt-5.4",
                operation=lambda client: client.chat_completions({"model": "gpt-5.4"}),
            )
        finally:
            await pool.__aexit__(None, None, None)

    try:
        asyncio.run(run())
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 429
    else:
        raise AssertionError("Expected a 429 error for the single-account pool")

    saved = repo.get_account("acct-1")
    assert saved is not None
    assert saved.last_rate_limited_at > 0
    remaining = saved.cooldown_until - time.time()
    assert remaining > 0
    assert remaining <= 8.5


def test_single_account_429_honors_http_date_retry_after(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(time, "time", lambda: 1000.0)

    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))
    retry_after = datetime.fromtimestamp(1015, tz=timezone.utc).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["gpt-5.4"],
            "errors": [
                make_http_error(
                    429,
                    "rate limited",
                    headers={"Retry-After": retry_after},
                )
            ],
            "chat_calls": 0,
        }
    }

    async def run() -> None:
        pool = TokenPool(
            repo,
            client_factory=lambda token, api_base: FakeCopilotClient(
                token,
                api_base,
                state,
            ),
        )
        try:
            await pool.execute(
                model="gpt-5.4",
                operation=lambda client: client.chat_completions({"model": "gpt-5.4"}),
            )
        finally:
            await pool.__aexit__(None, None, None)

    try:
        asyncio.run(run())
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 429
    else:
        raise AssertionError("Expected a 429 error for the single-account pool")

    saved = repo.get_account("acct-1")
    assert saved is not None
    remaining = saved.cooldown_until - time.time()
    assert 14.5 <= remaining <= 15.0


def test_single_account_429_ignores_non_finite_retry_after(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(time, "time", lambda: 1000.0)

    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["gpt-5.4"],
            "errors": [
                make_http_error(
                    429,
                    "rate limited",
                    headers={"Retry-After": "NaN"},
                )
            ],
            "chat_calls": 0,
        }
    }

    async def run() -> None:
        pool = TokenPool(
            repo,
            client_factory=lambda token, api_base: FakeCopilotClient(
                token,
                api_base,
                state,
            ),
        )
        try:
            await pool.execute(
                model="gpt-5.4",
                operation=lambda client: client.chat_completions({"model": "gpt-5.4"}),
            )
        finally:
            await pool.__aexit__(None, None, None)

    try:
        asyncio.run(run())
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 429
    else:
        raise AssertionError("Expected a 429 error for the single-account pool")

    saved = repo.get_account("acct-1")
    assert saved is not None
    remaining = saved.cooldown_until - time.time()
    assert remaining > 0
    assert remaining <= 8.5


def test_single_account_429_ignores_negative_retry_after(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(time, "time", lambda: 1000.0)

    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["gpt-5.4"],
            "errors": [
                make_http_error(
                    429,
                    "rate limited",
                    headers={"Retry-After": "-1"},
                )
            ],
            "chat_calls": 0,
        }
    }

    async def run() -> None:
        pool = TokenPool(
            repo,
            client_factory=lambda token, api_base: FakeCopilotClient(
                token,
                api_base,
                state,
            ),
        )
        try:
            await pool.execute(
                model="gpt-5.4",
                operation=lambda client: client.chat_completions({"model": "gpt-5.4"}),
            )
        finally:
            await pool.__aexit__(None, None, None)

    try:
        asyncio.run(run())
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 429
    else:
        raise AssertionError("Expected a 429 error for the single-account pool")

    saved = repo.get_account("acct-1")
    assert saved is not None
    remaining = saved.cooldown_until - time.time()
    assert remaining > 0
    assert remaining <= 8.5
