from __future__ import annotations

import asyncio
import copy
import json
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from copilotx.auth.accounts import AccountRecord, AccountRepository
from copilotx.auth.pool import TokenPool
from copilotx.server.app import create_app


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


def make_http_error(status_code: int, message: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://upstream.test/v1/chat/completions")
    response = httpx.Response(status_code, request=request, text=message)
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
            "models": ["gpt-5.4", "gemini-3.1-pro"],
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
                "model": "gpt-5.4",
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
            "models": ["gpt-5.4"],
            "errors": [
                make_http_error(
                    400,
                    (
                        '{"error":{"message":"model \\"gpt-5.4\\" is not accessible via '
                        'the /chat/completions endpoint","code":"unsupported_api_for_model"}}'
                    ),
                )
            ],
            "responses_result": {
                "model": "gpt-5.4",
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
                "model": "gpt-5.4",
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
            "models": ["gpt-5.4"],
            "errors": [],
            "stream_errors": [
                make_http_error(
                    400,
                    (
                        '{"error":{"message":"model \\"gpt-5.4\\" is not accessible via '
                        'the /chat/completions endpoint","code":"unsupported_api_for_model"}}'
                    ),
                )
            ],
            "responses_result": {
                "model": "gpt-5.4",
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
                "model": "gpt-5.4",
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
            "models": ["gpt-5.4"],
            "errors": [
                make_http_error(
                    400,
                    (
                        '{"error":{"message":"model \\"gpt-5.4\\" is not accessible via '
                        'the /chat/completions endpoint","code":"unsupported_api_for_model"}}'
                    ),
                )
            ],
            "responses_result": {
                "id": "resp-openai-1",
                "model": "gpt-5.4",
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
                "model": "gpt-5.4",
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
            "models": ["gpt-5.4"],
            "errors": [],
            "stream_errors": [
                make_http_error(
                    400,
                    (
                        '{"error":{"message":"model \\"gpt-5.4\\" is not accessible via '
                        'the /chat/completions endpoint","code":"unsupported_api_for_model"}}'
                    ),
                )
            ],
            "responses_result": {
                "id": "resp-openai-2",
                "model": "gpt-5.4",
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
                "model": "gpt-5.4",
                "stream": True,
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        )

    assert response.status_code == 200
    assert "gpt54 stream ok" in response.text
    assert "data: [DONE]" in response.text
    assert state["token-1"]["responses_calls"] == 1


def test_responses_non_stream_falls_back_to_chat_completions_for_chat_only_models(
    tmp_path: Path,
) -> None:
    repo, _ = make_repo(tmp_path)
    repo.upsert_account(make_account("acct-1", "alpha", "gh-1", "token-1"))

    state = {
        "token-1": {
            "name": "alpha",
            "models": ["claude-sonnet-4.6"],
            "errors": [],
            "responses_errors": [
                make_http_error(
                    400,
                    (
                        '{"error":{"message":"model \\"claude-sonnet-4.6\\" '
                        'does not support Responses API.","code":"unsupported_api_for_model"}}'
                    ),
                )
            ],
            "chat_result": {
                "id": "chat-fallback-1",
                "object": "chat.completion",
                "created": 1772973000,
                "model": "claude-sonnet-4.6",
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
                "model": "claude-sonnet-4.6",
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
            "models": ["gemini-3.1-pro-preview"],
            "errors": [],
            "responses_stream_errors": [
                make_http_error(
                    400,
                    (
                        '{"error":{"message":"model \\"gemini-3.1-pro-preview\\" '
                        'does not support Responses API.","code":"unsupported_api_for_model"}}'
                    ),
                )
            ],
            "chat_result": {
                "id": "chat-fallback-2",
                "object": "chat.completion",
                "created": 1772973001,
                "model": "gemini-3.1-pro-preview",
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
                "model": "gemini-3.1-pro-preview",
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
            "models": ["oswe-vscode-prime"],
            "errors": [],
            "responses_result": {
                "id": "resp-phase-1",
                "object": "response",
                "model": "oswe-vscode-prime",
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
                "model": "oswe-vscode-prime",
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
