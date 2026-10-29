"""Login for the assistant UI, using the ERP's own accounts.

The browser sends username/password to /auth/login. We forward them to the ERP's /login/ and keep the ERP
access token ONLY on the server; the browser gets an opaque session token instead. ERP tool calls then run
with that user's own ERP token, so the ERP's role checks apply to everything the assistant does.

We deliberately do NOT send APP_CLIENT_KEY here: roles the ERP restricts to its desktop app
(APP_ONLY_ROLES, e.g. technician / distributor) are refused by the ERP and stay refused here.
"""
import os
import secrets
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

ERP_BASE_URL = os.getenv("ERP_BASE_URL", "https://acerbiomedical.in").rstrip("/")
TTL = int(os.getenv("AUTH_TTL_HOURS", "12")) * 3600
TIMEOUT = float(os.getenv("ERP_TIMEOUT", "60"))

_sessions: dict = {}


class AuthExpired(Exception):
    """The ERP rejected the user's token (expired / invalid): the user must log in again."""


class LoginFailed(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


async def login(username: str, password: str) -> dict:
    if not username or not password:
        raise LoginFailed(400, "Username aur password dono chahiye.")
    try:
        async with httpx.AsyncClient(base_url=ERP_BASE_URL, timeout=TIMEOUT) as client:
            r = await client.post("/login/", json={"username": username, "password": password})
    except httpx.HTTPError as e:
        raise LoginFailed(503, f"ERP server not reachable ({e.__class__.__name__}). ERP_BASE_URL check karo.")
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code != 200 or not body.get("access_token"):
        detail = body.get("detail") if isinstance(body, dict) else None
        raise LoginFailed(r.status_code if r.status_code != 200 else 502, detail or "Login failed")

    now = time.time()
    for k in [k for k, v in _sessions.items() if v["expires"] < now]:   # drop expired sessions
        _sessions.pop(k, None)
    session_token = secrets.token_urlsafe(32)
    _sessions[session_token] = {
        "session": session_token,
        "username": username,
        "role": body.get("role", ""),
        "allowed_pages": body.get("allowed_pages"),
        "erp_token": body["access_token"],
        "expires": now + TTL,
    }
    return {"token": session_token, "username": username, "role": body.get("role", "")}


def get(session_token: str) -> dict | None:
    s = _sessions.get(session_token or "")
    if not s:
        return None
    if s["expires"] < time.time():
        _sessions.pop(session_token, None)
        return None
    return s


def drop(session_token: str):
    _sessions.pop(session_token or "", None)