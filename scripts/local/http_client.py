"""Small stdlib-only HTTP helpers for smoke tests."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .errors import SmokeError


def checked_json_request(method: str, url: str, payload: dict | None, timeout: float) -> dict:
    status, body = json_request(method, url, payload, timeout)
    if not (200 <= status < 300):
        raise SmokeError(f"{method} {url} returned HTTP {status}: {body}")
    return body


def json_request(method: str, url: str, payload: dict | None, timeout: float) -> tuple[int, dict]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    return raw_request(method, url, data, headers, timeout)


def raw_request(method: str, url: str, data: bytes | None, headers: dict[str, str], timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = int(response.status)
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read()
    except urllib.error.URLError as exc:
        raise SmokeError(f"{method} {url} failed: {exc}") from exc
    text = raw.decode("utf-8", errors="replace")
    if not text:
        return status, {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return status, parsed
        return status, {"value": parsed}
    except json.JSONDecodeError:
        return status, {"raw": text}


def raw_bytes_request(method: str, url: str, data: bytes | None, headers: dict[str, str], timeout: float) -> tuple[int, bytes, dict[str, str]]:
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return int(response.status), response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(), dict(exc.headers.items())
    except urllib.error.URLError as exc:
        raise SmokeError(f"{method} {url} failed: {exc}") from exc


def encode_multipart(boundary: str, fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> bytes:
    chunks: list[bytes] = []
    marker = f"--{boundary}\r\n".encode("utf-8")
    for name, value in fields.items():
        chunks.append(marker)
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    for name, (filename, content, content_type) in files.items():
        chunks.append(marker)
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        chunks.append(content)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks)
