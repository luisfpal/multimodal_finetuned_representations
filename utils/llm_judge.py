"""
Orfeo JWT acquisition and LLM judge credential setup.

Ensures OPENAI_API_KEY / OPENAI_API_URL / MODEL_VERSION are set for
lmms-eval tasks that use GPT-based scoring (mmbench, mmvet, llava_in_the_wild).

Workflow:
  1. Load .env from the project root (via python-dotenv, if available).
  2. If OPENAI_API_KEY is already set → nothing to do.
  3. Otherwise, read ORFEO_* credentials from .env → HTTP POST to get a JWT.
  4. Export OPENAI_API_KEY, OPENAI_API_URL, MODEL_VERSION into os.environ.

Works transparently whether the script is launched from a shell wrapper
(which may already export the env vars) or run directly with ``python``.
"""

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


def _load_project_env() -> None:
    """Load .env from the project root (one level above utils/)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        # override=False keeps any vars the shell already exported
        load_dotenv(env_file, override=False)


def _acquire_orfeo_jwt() -> str | None:
    """POST to the Orfeo OAuth endpoint and return the access_token (or None)."""
    auth_url = os.environ.get("ORFEO_AUTH_URL")
    client_id = os.environ.get("ORFEO_CLIENT_ID")
    username = os.environ.get("ORFEO_USERNAME")
    password = os.environ.get("ORFEO_CLIENT_SECRET")

    if not all([auth_url, client_id, username, password]):
        return None

    data = urllib.parse.urlencode(
        {
            "grant_type": "password",
            "client_id": client_id,
            "username": username,
            "password": password,
            "scope": "profile",
        }
    ).encode()

    try:
        req = urllib.request.Request(
            auth_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        token = body.get("access_token")
        if token and token != "null":
            return token
    except Exception:
        pass

    return None


def ensure_llm_judge_env() -> bool:
    """Make sure the three env vars lmms-eval needs for GPT judging are set.

    Returns True when credentials are available, False otherwise.
    """
    _load_project_env()

    # Already configured (e.g. by a shell wrapper)?
    if os.environ.get("OPENAI_API_KEY"):
        return True

    # Try the Orfeo OAuth flow
    token = _acquire_orfeo_jwt()
    if not token:
        return False

    os.environ["OPENAI_API_KEY"] = token

    orfeo_url = os.environ.get("ORFEO_LLM_URL")
    if orfeo_url and not os.environ.get("OPENAI_API_URL"):
        os.environ["OPENAI_API_URL"] = orfeo_url

    orfeo_model = os.environ.get("ORFEO_MODEL")
    if orfeo_model and not os.environ.get("MODEL_VERSION"):
        os.environ["MODEL_VERSION"] = orfeo_model

    return True
