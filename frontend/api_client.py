import os
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class APIError(Exception):
    """Raised when the backend returns an unsuccessful HTTP response."""


class APIConnectionError(APIError):
    """Raised when the backend cannot be reached at all."""


if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parent / ".env")


def get_base_url() -> str:
    base_url = os.environ.get("API_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "API_BASE_URL is not set. Copy frontend/.env.example to frontend/.env "
            "and set it to the backend address, e.g. http://localhost:8000"
        )
    return base_url.rstrip("/")


def check_health(timeout: float = 5.0) -> dict:
    try:
        response = requests.get(f"{get_base_url()}/health", timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise APIConnectionError("The backend is unreachable") from exc
    if response.status_code != 200:
        raise APIError(f"Health check failed with status {response.status_code}")
    return response.json()


def ask_question(question: str, timeout: float = 180.0) -> dict:
    """POST a question to the backend and return {'answer': str, 'sources': list[str]}."""
    try:
        response = requests.post(
            f"{get_base_url()}/query",
            json={"question": question},
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError as exc:
        raise APIConnectionError("The backend is unreachable") from exc
    except requests.exceptions.Timeout as exc:
        raise APIConnectionError("The backend did not answer in time") from exc
    if response.status_code != 200:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        raise APIError(f"Backend returned {response.status_code}: {detail}")
    payload = response.json()
    return {"answer": payload["answer"], "sources": payload["sources"]}
