"""
Thin wrapper around the FastAPI backend. Streamlit's app.py should never
build a requests.post(...) call directly — every backend interaction
goes through a function here, so the UI code stays focused on layout.
"""
from __future__ import annotations
import requests

BACKEND_URL = "https://ai-data-analyst-clbq.onrender.com"
TIMEOUT_SECONDS = 30


class APIError(Exception):
    """Raised when the backend returns a non-2xx response, with the detail message."""
    pass


def _handle_response(response: requests.Response) -> dict:
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise APIError(detail)
    return response.json()


def upload_file(file_bytes: bytes, filename: str) -> dict:
    files = {"file": (filename, file_bytes)}
    response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
    return _handle_response(response)


def clean_data(session_id: str, rules: list[dict], drop_duplicates: bool, dtype_overrides: dict | None = None) -> dict:
    payload = {
        "session_id": session_id,
        "rules": rules,
        "drop_duplicates": drop_duplicates,
        "dtype_overrides": dtype_overrides or {},
    }
    response = requests.post(f"{BACKEND_URL}/clean", json=payload, timeout=TIMEOUT_SECONDS)
    return _handle_response(response)


def run_eda(session_id: str) -> dict:
    response = requests.post(f"{BACKEND_URL}/eda", json={"session_id": session_id}, timeout=TIMEOUT_SECONDS)
    return _handle_response(response)


def check_backend_alive() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/", timeout=60)
        return r.status_code == 200
    except Exception:
        return False
    
def train_models(session_id: str, target_column: str, excluded_columns: list[str] | None = None) -> dict:
    payload = {
        "session_id": session_id,
        "target_column": target_column,
        "excluded_columns": excluded_columns or [],
    }
    response = requests.post(f"{BACKEND_URL}/train", json=payload, timeout=120)
    return _handle_response(response)

def explain_model(session_id: str) -> dict:
    response = requests.post(f"{BACKEND_URL}/explain", json={"session_id": session_id}, timeout=120)
    return _handle_response(response)

def chat_with_dataset(session_id: str, question: str) -> dict:
    payload = {"session_id": session_id, "question": question}
    response = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=30)
    return _handle_response(response)

def generate_report(session_id: str) -> bytes:
    response = requests.post(f"{BACKEND_URL}/report", json={"session_id": session_id}, timeout=60)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise APIError(detail)
    return response.content