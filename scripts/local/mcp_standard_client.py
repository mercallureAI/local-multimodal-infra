"""Validate the standard MCP Streamable HTTP controller endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_ADMIN_URL = "http://127.0.0.1:17892/mcp/admin"
DEFAULT_INFER_URL = "http://127.0.0.1:17892/mcp/infer"
YOLO_MODEL_ID = "yolo11n.onnx"
QWEN_ASR_MODEL_ID = "qwen3-asr-0.6b-onnx"
INDEXTTS_MODEL_ID = "indextts-1.5-onnx"
INFER_TOOLS = {
    "create_task",
    "start_task",
    "get_task",
    "wait_task",
    "run_task",
    "sign_assets",
    "sign_asset_urls",
    "asr_transcribe",
    "object_detect",
    "tts_synthesize",
    "text_embed",
    "text_rerank",
}
ADMIN_TOOLS = {
    "list_models",
    "get_model",
    "add_model",
    "upsert_model",
    "download_model",
    "enable_model",
    "disable_model",
    "list_nodes",
    "get_cluster_status",
    "list_assets",
    "search_assets",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate local standard MCP Streamable HTTP endpoint.")
    parser.add_argument("--admin-url", default=DEFAULT_ADMIN_URL, help=f"Admin MCP URL (default: {DEFAULT_ADMIN_URL})")
    parser.add_argument("--infer-url", default=DEFAULT_INFER_URL, help=f"Inference MCP URL (default: {DEFAULT_INFER_URL})")
    parser.add_argument("--admin-token", default=os.environ.get("LOCAL_ADMIN_TOKEN"), help="Required admin MCP token; defaults to LOCAL_ADMIN_TOKEN.")
    parser.add_argument("--infer-token", default=None, help="Optional inference MCP token when the server requires one.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run standard MCP admin/catalog/assets plus generic and direct inference coverage where local resources are available.",
    )
    parser.add_argument("--sample-image", type=Path, default=None, help="JPEG/PNG sample for MCP object detection generic/direct coverage.")
    parser.add_argument("--sample-audio", type=Path, default=None, help="WAV sample for MCP ASR direct coverage.")
    parser.add_argument("--reference-audio", type=Path, default=None, help="WAV reference sample for MCP TTS direct coverage.")
    parser.add_argument("--text", default="你好，这是本地 IndexTTS 冒烟测试。", help="Text for MCP TTS direct coverage.")
    parser.add_argument(
        "--indextts-artifacts-ready",
        action="store_true",
        help="Hint that local IndexTTS artifacts are present; otherwise TTS direct coverage is reported as skipped.",
    )
    parser.add_argument("--timeout", type=float, default=1800.0, help="Task wait/upload timeout seconds.")
    args = parser.parse_args(argv)
    try:
        summary = asyncio.run(
            run(
                args.admin_url,
                args.infer_url,
                admin_token=args.admin_token,
                infer_token=args.infer_token,
                timeout=args.timeout,
                full=args.full,
                sample_image=args.sample_image,
                sample_audio=args.sample_audio,
                reference_audio=args.reference_audio,
                text=args.text,
                indextts_artifacts_ready=args.indextts_artifacts_ready,
            )
        )
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "mcp":
            print(
                json.dumps(
                    {
                        "ok": False,
                        "admin_url": args.admin_url,
                        "infer_url": args.infer_url,
                        "error": "Python package `mcp` is not installed in this interpreter",
                        "hint": "Install the official MCP Python SDK, e.g. `pip install mcp`, or run with an environment that already has it.",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1
        raise
    except Exception as exc:
        print(json.dumps({"ok": False, "admin_url": args.admin_url, "infer_url": args.infer_url, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0 if summary.get("ok") else 1


async def run(
    admin_url: str,
    infer_url: str,
    timeout: float = 1800.0,
    *,
    admin_token: str | None,
    infer_token: str | None = None,
    full: bool = False,
    sample_image: Path | None = None,
    sample_audio: Path | None = None,
    reference_audio: Path | None = None,
    text: str = "你好，这是本地 IndexTTS 冒烟测试。",
    indextts_artifacts_ready: bool = False,
) -> dict[str, Any]:
    from mcp import ClientSession

    try:
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
    except ImportError:
        from mcp.client.streamable_http import create_mcp_http_client
        from mcp.client.streamable_http import streamablehttp_client as streamable_http_client

    if not admin_token:
        raise RuntimeError("admin token is required (--admin-token or LOCAL_ADMIN_TOKEN)")
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    infer_headers = {"Authorization": f"Bearer {infer_token}"} if infer_token else {}
    async with create_mcp_http_client(admin_headers) as admin_http:
        async with streamable_http_client(admin_url, http_client=admin_http) as (admin_read, admin_write, *_):
            async with ClientSession(admin_read, admin_write) as admin_session:
                admin_initialize = await admin_session.initialize()
                admin_tools_result = await admin_session.list_tools()
                admin_tools = sorted(tool.name for tool in admin_tools_result.tools)
                admin_missing = sorted(ADMIN_TOOLS - set(admin_tools))
                if admin_missing:
                    return {
                        "ok": False,
                        "admin_url": admin_url,
                        "initialized": to_jsonable(admin_initialize),
                        "tools": admin_tools,
                        "missing_tools": admin_missing,
                    }
                leaked_admin = sorted(set(admin_tools) & INFER_TOOLS)
                if leaked_admin:
                    raise RuntimeError(f"admin MCP leaked inference tools: {leaked_admin}")
                list_models_result = await call_tool_checked(admin_session, "list_models", {})
                cluster_status_result = await call_tool_checked(admin_session, "get_cluster_status", {})
                list_nodes_result = await call_tool_checked(admin_session, "list_nodes", {})
                list_assets_result = await call_tool_checked(admin_session, "list_assets", {})
                models = list_models_result if isinstance(list_models_result, list) else []
                async with create_mcp_http_client(infer_headers) as infer_http:
                    async with streamable_http_client(infer_url, http_client=infer_http) as (infer_read, infer_write, *_):
                        async with ClientSession(infer_read, infer_write) as infer_session:
                            infer_initialize = await infer_session.initialize()
                            infer_tools_result = await infer_session.list_tools()
                            infer_tools = sorted(tool.name for tool in infer_tools_result.tools)
                            infer_missing = sorted(INFER_TOOLS - set(infer_tools))
                            if infer_missing:
                                return {
                                    "ok": False,
                                    "infer_url": infer_url,
                                    "initialized": to_jsonable(infer_initialize),
                                    "tools": infer_tools,
                                    "missing_tools": infer_missing,
                                }
                            leaked_infer = sorted(set(infer_tools) & ADMIN_TOOLS)
                            if leaked_infer:
                                raise RuntimeError(f"inference MCP leaked admin tools: {leaked_infer}")
                            summary = {
                                "ok": True,
                                "admin_url": admin_url,
                                "infer_url": infer_url,
                                "admin_initialized": to_jsonable(admin_initialize),
                                "infer_initialized": to_jsonable(infer_initialize),
                                "admin_tools": admin_tools,
                                "infer_tools": infer_tools,
                                "list_models": list_models_result,
                                "get_cluster_status": cluster_status_result,
                                "list_nodes": list_nodes_result,
                                "list_assets": list_assets_result,
                            }
                            if full:
                                summary["assets"] = await run_assets_smoke(admin_session, infer_session, timeout)
                                summary["generic_tasks"] = await run_generic_smoke(
                                    infer_session,
                                    sample_image,
                                    sample_audio,
                                    reference_audio,
                                    text,
                                    timeout,
                                    models,
                                    indextts_artifacts_ready=indextts_artifacts_ready,
                                )
                                summary["direct_inference"] = await run_direct_smoke(
                                    admin_session,
                                    infer_session,
                                    sample_image,
                                    sample_audio,
                                    reference_audio,
                                    text,
                                    timeout,
                                    models,
                                    indextts_artifacts_ready=indextts_artifacts_ready,
                                )
                            return summary


async def run_assets_smoke(admin_session: Any, infer_session: Any, timeout: float) -> dict[str, Any]:
    marker = f"{int(time.time() * 1000)}"
    body = f"hello standard mcp assets {marker}\n".encode("utf-8")
    path = f"smoke/mcp/{marker}/hello.txt"
    requests = [
        {
            "operation": "upload",
            "kind": "material",
            "path": path,
            "content_type": "text/plain; charset=utf-8",
            "expires": "never",
            "url_ttl_sec": 600,
        },
        {
            "action": "download",
            "uri": f"assets://material/{path}",
            "url_ttl_sec": 600,
        },
    ]
    signed = await call_tool_checked(infer_session, "sign_asset_urls", {"items": requests})
    signed_items = signed.get("items") if isinstance(signed, dict) else None
    if not isinstance(signed_items, list) or len(signed_items) != 2:
        raise RuntimeError(f"MCP sign_asset_urls returned malformed batch: {signed}")
    if signed_items[0].get("method") != "POST" or signed_items[1].get("method") != "GET":
        raise RuntimeError(f"MCP sign_asset_urls did not preserve upload/download methods: {signed}")
    alias_signed = await call_tool_checked(infer_session, "sign_assets", {"requests": requests})
    alias_items = alias_signed.get("items") if isinstance(alias_signed, dict) else None
    if not isinstance(alias_items, list) or [item.get("method") for item in alias_items] != ["POST", "GET"]:
        raise RuntimeError(f"MCP sign_assets alias returned malformed batch: {alias_signed}")

    upload_url = signed_items[0].get("signed_url")
    download_url = signed_items[1].get("signed_url")
    if not upload_url or not download_url:
        raise RuntimeError(f"MCP signed URLs missing upload/download URLs: {signed}")
    upload_payload = upload_bytes(upload_url, body, "text/plain; charset=utf-8", timeout)
    uri = upload_payload.get("uri")
    if uri != f"assets://material/{path}":
        raise RuntimeError(f"MCP asset upload returned unexpected uri: {upload_payload}")
    list_payload = await call_tool_checked(admin_session, "list_assets", {"kind": "material", "contains": marker})
    assets = list_payload.get("assets") if isinstance(list_payload, dict) else []
    if not any(isinstance(asset, dict) and asset.get("uri") == uri for asset in assets or []):
        raise RuntimeError(f"MCP list_assets did not find uploaded asset {uri}: {list_payload}")
    downloaded = download_bytes(download_url, timeout)
    if downloaded != body:
        raise RuntimeError(f"MCP signed asset download mismatch: {downloaded!r}")
    return {
        "status": "passed",
        "uri": uri,
        "list_count": len(assets or []),
        "signed_batch_methods": [item.get("method") for item in signed_items],
    }


async def run_generic_smoke(
    session: Any,
    sample_image: Path | None,
    sample_audio: Path | None,
    reference_audio: Path | None,
    text: str,
    timeout: float,
    models: list[Any],
    *,
    indextts_artifacts_ready: bool,
) -> dict[str, Any]:
    return {
        "object_detect": await generic_object_detect(session, sample_image, timeout, models),
        "asr_transcribe": await generic_asr_transcribe(session, sample_audio, timeout, models),
        "tts_synthesize": await generic_tts_synthesize(
            session,
            reference_audio,
            text,
            timeout,
            models,
            artifacts_ready=indextts_artifacts_ready,
        ),
    }


async def generic_object_detect(session: Any, sample_image: Path | None, timeout: float, models: list[Any]) -> dict[str, Any]:
    if sample_image is None:
        return skipped("no --sample-image supplied for generic object detection task")
    if not sample_image.exists():
        return skipped(f"sample image does not exist: {sample_image}")
    if not model_enabled(models, YOLO_MODEL_ID):
        return skipped(f"{YOLO_MODEL_ID} is not enabled")
    create = await call_tool_checked(
        session,
        "create_task",
        {
            "task_kind": "object.detect",
            "model": YOLO_MODEL_ID,
            "files": [{"name": sample_image.name, "mime": image_mime(sample_image), "role": "image", "required": True}],
            "params": {},
        },
    )
    upload = first_upload(create, "image")
    upload_file(upload["upload_url"], sample_image, image_mime(sample_image), timeout)
    task_id = require_task_id(create, "MCP generic object.detect")
    start = await call_tool_checked(session, "start_task", {"task_id": task_id, "wait": False})
    wait = await call_tool_checked(session, "wait_task", {"task_id": task_id, "timeout_sec": int(timeout)})
    get = await call_tool_checked(session, "get_task", {"task_id": task_id})
    validate_task_output(wait, "object_detections", "MCP generic object.detect")
    object_count, car_count = object_detection_counts(wait.get("output"), "MCP generic object.detect")
    return {"status": "passed", "input_image": str(sample_image), "object_count": object_count, "car_count": car_count, "create_task": create, "start_task": start, "wait_task": wait, "get_task": get}


async def generic_asr_transcribe(session: Any, sample_audio: Path | None, timeout: float, models: list[Any]) -> dict[str, Any]:
    if sample_audio is None:
        return skipped("no --sample-audio supplied for generic ASR task")
    if not sample_audio.exists():
        return skipped(f"sample audio does not exist: {sample_audio}")
    if not model_enabled(models, QWEN_ASR_MODEL_ID):
        return skipped(f"{QWEN_ASR_MODEL_ID} is not enabled")
    create = await call_tool_checked(
        session,
        "create_task",
        {
            "task_kind": "asr.transcribe",
            "model": QWEN_ASR_MODEL_ID,
            "files": [{"name": sample_audio.name, "mime": "audio/wav", "role": "audio", "required": True}],
            "params": {},
        },
    )
    upload_file(first_upload(create, "audio")["upload_url"], sample_audio, "audio/wav", timeout)
    task_id = require_task_id(create, "MCP generic asr.transcribe")
    start = await call_tool_checked(session, "start_task", {"task_id": task_id, "wait": False})
    wait = await call_tool_checked(session, "wait_task", {"task_id": task_id, "timeout_sec": int(timeout)})
    get = await call_tool_checked(session, "get_task", {"task_id": task_id})
    validate_task_output(wait, "asr_transcription", "MCP generic asr.transcribe")
    asr_text = extract_direct_asr_text(wait.get("output"))
    if not asr_text.strip():
        raise RuntimeError(f"MCP generic asr.transcribe returned empty text: {wait}")
    return {"status": "passed", "input_audio": str(sample_audio), "asr_text": asr_text, "asr_text_length": len(asr_text), "create_task": create, "start_task": start, "wait_task": wait, "get_task": get}


async def generic_tts_synthesize(
    session: Any,
    reference_audio: Path | None,
    text: str,
    timeout: float,
    models: list[Any],
    *,
    artifacts_ready: bool,
) -> dict[str, Any]:
    if not artifacts_ready:
        return skipped("missing IndexTTS artifacts; no download attempted")
    if reference_audio is None:
        return skipped("no --reference-audio supplied for generic TTS task")
    if not reference_audio.exists():
        return skipped(f"reference audio does not exist: {reference_audio}")
    if not model_enabled(models, INDEXTTS_MODEL_ID):
        return skipped(f"{INDEXTTS_MODEL_ID} is not enabled")
    create = await call_tool_checked(
        session,
        "create_task",
        {
            "task_kind": "tts.synthesize",
            "model": INDEXTTS_MODEL_ID,
            "files": [{"name": reference_audio.name, "mime": "audio/wav", "role": "reference_audio", "required": False}],
            "params": {"text": text},
        },
    )
    upload_file(first_upload(create, "reference_audio")["upload_url"], reference_audio, "audio/wav", timeout)
    task_id = require_task_id(create, "MCP generic tts.synthesize")
    start = await call_tool_checked(session, "start_task", {"task_id": task_id, "wait": False})
    wait = await call_tool_checked(session, "wait_task", {"task_id": task_id, "timeout_sec": int(timeout)})
    get = await call_tool_checked(session, "get_task", {"task_id": task_id})
    validate_task_output(wait, "tts_audio", "MCP generic tts.synthesize")
    return {"status": "passed", "create_task": create, "start_task": start, "wait_task": wait, "get_task": get}


async def run_direct_smoke(
    admin_session: Any,
    infer_session: Any,
    sample_image: Path | None,
    sample_audio: Path | None,
    reference_audio: Path | None,
    text: str,
    timeout: float,
    models: list[Any],
    *,
    indextts_artifacts_ready: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    results["object_detect"] = await direct_object_detect(infer_session, sample_image, models)
    results["asr_transcribe"] = await direct_asr_transcribe(infer_session, sample_audio, models)
    results["tts_synthesize"] = await direct_tts_synthesize(
        admin_session,
        infer_session,
        reference_audio,
        text,
        models,
        timeout,
        artifacts_ready=indextts_artifacts_ready,
        asr_enabled=model_enabled(models, QWEN_ASR_MODEL_ID),
    )
    return results


async def direct_object_detect(session: Any, image: Path | None, models: list[Any]) -> dict[str, Any]:
    if image is None:
        return skipped("no --sample-image supplied")
    if not image.exists():
        return skipped(f"sample image does not exist: {image}")
    if not model_enabled(models, YOLO_MODEL_ID):
        return skipped(f"{YOLO_MODEL_ID} is not enabled")
    payload = await call_tool_checked(
        session,
        "object_detect",
        {"model": YOLO_MODEL_ID, "image": {"path": str(image), "mime": image_mime(image)}},
    )
    validate_direct_output(payload, "object_detections", "MCP direct object_detect")
    object_count, car_count = object_detection_counts(payload, "MCP direct object_detect")
    return {
        "status": "passed",
        "input_image": str(image),
        "object_count": object_count,
        "car_count": car_count,
        "result": payload,
    }


async def direct_asr_transcribe(session: Any, audio: Path | None, models: list[Any]) -> dict[str, Any]:
    if audio is None:
        return skipped("no --sample-audio supplied")
    if not audio.exists():
        return skipped(f"sample audio does not exist: {audio}")
    if not model_enabled(models, QWEN_ASR_MODEL_ID):
        return skipped(f"{QWEN_ASR_MODEL_ID} is not enabled")
    payload = await call_tool_checked(
        session,
        "asr_transcribe",
        {"model": QWEN_ASR_MODEL_ID, "audio": {"path": str(audio), "mime": "audio/wav"}},
    )
    validate_direct_output(payload, "asr_transcription", "MCP direct asr_transcribe")
    asr_text = extract_direct_asr_text(payload)
    if not asr_text.strip():
        raise RuntimeError(f"MCP direct asr_transcribe returned empty text: {payload}")
    return {
        "status": "passed",
        "input_audio": str(audio),
        "asr_text": asr_text,
        "asr_text_length": len(asr_text),
        "result": payload,
    }


async def direct_tts_synthesize(
    admin_session: Any,
    infer_session: Any,
    reference_audio: Path | None,
    text: str,
    models: list[Any],
    timeout: float,
    *,
    artifacts_ready: bool,
    asr_enabled: bool,
) -> dict[str, Any]:
    if not artifacts_ready:
        return skipped("missing IndexTTS artifacts; no download attempted")
    if reference_audio is None:
        return skipped("no --reference-audio supplied")
    if not reference_audio.exists():
        return skipped(f"reference audio does not exist: {reference_audio}")
    if not model_enabled(models, INDEXTTS_MODEL_ID):
        await call_tool_checked(admin_session, "enable_model", {"id": INDEXTTS_MODEL_ID})
    payload = await call_tool_checked(
        infer_session,
        "tts_synthesize",
        {
            "model": INDEXTTS_MODEL_ID,
            "text": text,
            "reference_audio": {"path": str(reference_audio), "mime": "audio/wav"},
        },
    )
    validate_direct_output(payload, "tts_audio", "MCP direct tts_synthesize")
    result: dict[str, Any] = {"status": "passed", "result": payload}
    if not asr_enabled:
        result["tts_asr_cross_check"] = skipped(f"{QWEN_ASR_MODEL_ID} is not enabled")
        return result
    audio_path, cleanup_path = resolve_tts_audio_for_asr(payload, timeout)
    try:
        if audio_path is None:
            result["tts_asr_cross_check"] = skipped("tts_synthesize returned no local or downloadable audio path")
        else:
            asr_payload = await call_tool_checked(
                infer_session,
                "asr_transcribe",
                {"model": QWEN_ASR_MODEL_ID, "audio": {"path": str(audio_path), "mime": "audio/wav"}},
            )
            validate_direct_output(asr_payload, "asr_transcription", "MCP direct tts_synthesize ASR cross-check")
            asr_text = extract_direct_asr_text(asr_payload)
            if not asr_text.strip():
                raise RuntimeError(f"MCP direct tts_synthesize ASR cross-check returned empty text: {asr_payload}")
            result["tts_asr_cross_check"] = {"status": "passed", "input_audio": str(audio_path), "result": asr_payload}
            result["tts_asr_text"] = asr_text
            result["tts_asr_text_length"] = len(asr_text)
    finally:
        if cleanup_path is not None:
            try:
                cleanup_path.unlink()
            except OSError:
                pass
    return result



async def call_tool_checked(session: Any, name: str, arguments: dict[str, Any]) -> Any:
    result = await session.call_tool(name, arguments)
    if bool(getattr(result, "isError", False)) or bool(getattr(result, "is_error", False)):
        raise RuntimeError(f"MCP tool {name} returned error: {to_jsonable(getattr(result, 'content', []))}")
    return parse_tool_payload(result)


def require_task_id(payload: dict[str, Any], label: str) -> str:
    task_id = payload.get("task_id") if isinstance(payload, dict) else None
    if not task_id:
        raise RuntimeError(f"{label} returned no task_id: {payload}")
    return str(task_id)


def first_upload(status: dict[str, Any], role: str) -> dict[str, Any]:
    uploads = status.get("uploads") if isinstance(status, dict) else None
    for upload in uploads or []:
        if isinstance(upload, dict) and (upload.get("role") == role or upload.get("slot") == role):
            return upload
    raise RuntimeError(f"no upload slot with role {role}: {status}")


def upload_file(upload_url: str, path: Path, content_type: str, timeout: float) -> dict[str, Any]:
    return upload_bytes(upload_url, path.read_bytes(), content_type, timeout)


def upload_bytes(upload_url: str, body: bytes, content_type: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        upload_url,
        data=body,
        headers={"Content-Type": content_type, "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read()
    if not (200 <= status < 300):
        raise RuntimeError(f"upload returned HTTP {status}: {raw.decode('utf-8', errors='replace')}")
    if not raw:
        return {}
    parsed = json.loads(raw.decode("utf-8", errors="replace"))
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def download_bytes(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"Accept": "*/*"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read()
    if not (200 <= status < 300):
        raise RuntimeError(f"download returned HTTP {status}: {raw.decode('utf-8', errors='replace')}")
    return raw


def image_mime(path: Path | str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".bmp":
        return "image/bmp"
    return "application/octet-stream"


def skipped(reason: str) -> dict[str, Any]:
    return {"status": "skipped", "reason": reason}


def model_enabled(models: list[Any], model_id: str) -> bool:
    model = next((item for item in models if isinstance(item, dict) and item.get("id") == model_id), None)
    return bool(model and model.get("enabled") is True)


def object_detection_counts(payload: Any, label: str) -> tuple[int, int]:
    validate_direct_output(payload, "object_detections", label)
    objects = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(objects, list) or not objects:
        raise RuntimeError(f"{label} returned no objects: {payload}")
    labels = [object_label(item) for item in objects if isinstance(item, dict)]
    car_count = sum(1 for item_label in labels if is_car_like_label(item_label))
    if car_count <= 0:
        raise RuntimeError(f"{label} returned no car-like labels: labels={labels!r}")
    return len(objects), car_count


def object_label(item: dict[str, Any]) -> str:
    for key in ("label", "class", "name", "class_name"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


def is_car_like_label(label: str) -> bool:
    normalized = "".join(ch for ch in label.lower() if ch.isalnum())
    return normalized in {"car", "cars", "automobile"} or "car" in normalized


def extract_direct_asr_text(payload: Any) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        return payload["text"]
    return ""


def resolve_tts_audio_for_asr(payload: Any, timeout: float) -> tuple[Path | None, Path | None]:
    audio = payload.get("audio") if isinstance(payload, dict) else None
    if not isinstance(audio, dict):
        return None, None
    raw_path = audio.get("path")
    if isinstance(raw_path, str) and raw_path:
        path = Path(raw_path)
        path = path if path.is_absolute() else path.resolve()
        if path.exists():
            return path, None
    audio_url = audio.get("url")
    if isinstance(audio_url, str) and audio_url:
        downloaded = download_bytes(audio_url, timeout)
        if not downloaded:
            raise RuntimeError("MCP direct tts_synthesize returned an empty downloadable audio file")
        with tempfile.NamedTemporaryFile(prefix="local-mcp-tts-", suffix=".wav", delete=False) as handle:
            handle.write(downloaded)
            return Path(handle.name), Path(handle.name)
    return None, None


def validate_task_output(payload: Any, expected_type: str, label: str) -> None:
    if not isinstance(payload, dict) or payload.get("state") != "succeeded":
        raise RuntimeError(f"{label} task did not succeed: {payload}")
    validate_direct_output(payload.get("output"), expected_type, label)


def validate_direct_output(payload: Any, expected_type: str, label: str) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} output is malformed: {payload}")
    if payload.get("type") != expected_type:
        raise RuntimeError(f"{label} output type mismatch: {payload}")


def parse_tool_payload(result: Any) -> Any:
    if bool(getattr(result, "isError", False)) or bool(getattr(result, "is_error", False)):
        return {"is_error": True, "content": to_jsonable(getattr(result, "content", []))}
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return to_jsonable(structured)
    content = getattr(result, "content", [])
    if not content:
        return None
    first = content[0]
    text = getattr(first, "text", None)
    if text is None:
        return to_jsonable(first)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
