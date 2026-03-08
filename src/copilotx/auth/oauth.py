"""GitHub OAuth Device Flow — get a GitHub access token interactively."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from rich.console import Console

from copilotx.config import (
    DEVICE_CODE_POLL_INTERVAL,
    DEVICE_CODE_TIMEOUT,
    GITHUB_ACCESS_TOKEN_URL,
    GITHUB_CLIENT_ID,
    GITHUB_DEVICE_CODE_URL,
    GITHUB_SCOPE,
)

console = Console()


@dataclass
class DeviceCodeResponse:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


@dataclass
class GitHubUser:
    """Authenticated GitHub user identity."""

    user_id: str
    login: str


class OAuthError(Exception):
    """Raised when the OAuth flow fails."""


async def request_device_code(client: httpx.AsyncClient) -> DeviceCodeResponse:
    """Step 1: Request a device code from GitHub."""
    resp = await client.post(
        GITHUB_DEVICE_CODE_URL,
        json={"client_id": GITHUB_CLIENT_ID, "scope": GITHUB_SCOPE},
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    return DeviceCodeResponse(
        device_code=data["device_code"],
        user_code=data["user_code"],
        verification_uri=data["verification_uri"],
        expires_in=data["expires_in"],
        interval=data.get("interval", DEVICE_CODE_POLL_INTERVAL),
    )


async def poll_for_access_token(
    client: httpx.AsyncClient,
    device_code: str,
    interval: int = DEVICE_CODE_POLL_INTERVAL,
    timeout: int = DEVICE_CODE_TIMEOUT,
) -> str:
    """Step 2-3: Poll GitHub until user authorizes, return the OAuth access token."""
    elapsed = 0
    poll_interval = interval

    while elapsed < timeout:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

        resp = await client.post(
            GITHUB_ACCESS_TOKEN_URL,
            json={
                "client_id": GITHUB_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        if "access_token" in data:
            return data["access_token"]

        error = data.get("error", "")
        if error == "authorization_pending":
            continue
        elif error == "slow_down":
            poll_interval += 5  # back off as requested
            continue
        elif error == "expired_token":
            raise OAuthError("Device code expired. Please try again.")
        elif error == "access_denied":
            raise OAuthError("Authorization was denied by the user.")
        else:
            raise OAuthError(f"Unexpected OAuth error: {error}")

    raise OAuthError(f"Timed out waiting for authorization ({timeout}s).")


async def device_flow_login() -> str:
    """Run the full OAuth Device Flow interactively.  Returns a GitHub access token."""
    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1: request device code
        dc = await request_device_code(client)

        # Show instructions
        console.print()
        console.print("[bold cyan]🔐 GitHub Authorization Required[/]")
        console.print()
        console.print(f"  1. Open: [bold link={dc.verification_uri}]{dc.verification_uri}[/]")
        console.print(f"  2. Enter code: [bold yellow]{dc.user_code}[/]")
        console.print()
        console.print("[dim]Waiting for authorization...[/]")

        # Step 2-3: poll until authorized
        token = await poll_for_access_token(
            client,
            dc.device_code,
            interval=dc.interval,
        )

        console.print("[bold green]✅ GitHub authorization successful![/]")
        return token


async def fetch_github_user(github_token: str) -> GitHubUser:
    """Fetch the GitHub identity bound to an access token."""
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get("https://api.github.com/user", headers=headers)
        if resp.status_code == 401:
            raise OAuthError("GitHub token is invalid or expired.")
        resp.raise_for_status()
        data = resp.json()
        return GitHubUser(
            user_id=str(data.get("id", "")),
            login=str(data.get("login", "")),
        )
