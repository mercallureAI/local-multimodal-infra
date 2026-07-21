"""Python-only API/legacy-RPC/MCP smoke harness for local controller and worker.

The harness intentionally uses only Python subprocess management for services so
that smoke tests can run consistently from shells and agent environments without
PowerShell service wrappers or handwritten curl snippets.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from collections.abc import Iterable
from difflib import SequenceMatcher

from .errors import SmokeError
from .http_client import checked_json_request, encode_multipart, json_request, raw_bytes_request, raw_request
from .paths import repo_root, resolve_cli_path
from .processes import ManagedProcess, cleanup_processes, locate_bin, run_build, start_service, wait_ports_closed


CONTROLLER_URL = "http://127.0.0.1:17890"
WORKER_URL = "http://127.0.0.1:17891"
REGISTRATION_TOKEN = "local-smoke-registration-token"
MCP_ADMIN_URL = "http://127.0.0.1:17892/mcp/admin"
MCP_INFER_URL = "http://127.0.0.1:17892/mcp/infer"
PORTS = (17890, 17891, 17892)
ASSET_DIR = repo_root() / "scripts" / "assets"
DEFAULT_YOLO_IMAGE = ASSET_DIR / "yolo-input.jpg"
DEFAULT_SENSEVOICE_ASR_AUDIO = ASSET_DIR / "tts-input-mon3tr.wav"
TEST_ALIASES = {
    "all",
    "rpc",
    "mcp",
    "assets",
    "yolo",
    "sensevoice-asr",
    "indextts",
    "indextts_asr",
    "embedding",
    "rerank",
    "text",
    "mcp_standard",
}
RPC_TESTS = {"assets", "yolo", "sensevoice-asr", "indextts", "indextts_asr", "embedding", "rerank"}
MCP_TESTS = {"mcp_standard"}
INDEXTTS_MODEL_ID = "indextts-1.5-onnx"
EMBEDDING_MODEL_ID = "multilingual-e5-small-onnx"
RERANK_MODEL_ID = "mmarco-minilm-l12-onnx"
ADMIN_TOKEN = "local-smoke-admin-token"
INFER_TOKEN = "local-smoke-infer-token"
INDEXTTS_REQUIRED = [
    "IndexTTS_A.onnx",
    "IndexTTS_B.onnx",
    "IndexTTS_C.onnx",
    "IndexTTS_D.onnx",
    "IndexTTS_E.onnx",
    "IndexTTS_F.onnx",
    "bpe.model",
]

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    workdir = resolve_cli_path(args.workdir, root)
    model_dir = resolve_cli_path(args.model_dir, root) if args.model_dir else (workdir / "models").resolve()
    yolo_image = resolve_cli_path(args.yolo_image, root) if args.yolo_image else DEFAULT_YOLO_IMAGE
    sensevoice_asr_audio = resolve_cli_path(args.sensevoice_asr_audio, root) if args.sensevoice_asr_audio else DEFAULT_SENSEVOICE_ASR_AUDIO
    indextts_reference = resolve_cli_path(args.indextts_reference, root) if args.indextts_reference else sensevoice_asr_audio
    data_dir = workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    try:
        requested_tests = selected_tests(args)
    except SmokeError as exc:
        print(f"[smoke] FAIL: {exc}", file=sys.stderr)
        return 1
    launched: list[ManagedProcess] = []
    failures: list[str] = []

    try:
        indextts_artifacts = inspect_indextts_artifacts(model_dir)
        if args.build:
            run_build(root, release=args.release)

        controller_bin = locate_bin(root, "controller", args.controller_bin, release=args.release)
        worker_bin = locate_bin(root, "worker", args.worker_bin, release=args.release)

        print(f"[smoke] root={root}")
        print(f"[smoke] workdir={workdir}")
        print(f"[smoke] model_dir={model_dir}")
        print(f"[smoke] data_dir={data_dir}")
        print("[smoke] default assets=scripts/assets/yolo-input.jpg, tts-input-mon3tr.wav")
        print(f"[smoke] yolo_image={yolo_image}")
        print(f"[smoke] sensevoice_asr_audio={sensevoice_asr_audio}")
        print(f"[smoke] indextts_reference={indextts_reference}")
        print(f"[smoke] requested_tests={','.join(sorted(requested_tests)) or '<none>'}")

        env = os.environ.copy()
        env["LOCAL_DATA_DIR"] = str(data_dir)
        env["LOCAL_WORKER_REGISTRATION_TOKEN"] = REGISTRATION_TOKEN
        env["LOCAL_ADMIN_TOKEN"] = ADMIN_TOKEN
        env["LOCAL_MCP_INFER_TOKENS"] = f"unused-{INFER_TOKEN},{INFER_TOKEN}"
        if "LOCAL_INDEXTTS_MODEL_DIR" not in env:
            env["LOCAL_INDEXTTS_MODEL_DIR"] = str(model_dir / INDEXTTS_MODEL_ID)
        # Smoke runs must be deterministic even if the caller has an experimental
        # frontend in their shell (notably pinyin_explicit).  official-python
        # mode injects oracle token ids, but keep service env official_like too
        # so any diagnostics/fallbacks are not misleading.
        env["LOCAL_INDEXTTS_TEXT_FRONTEND"] = "official_like"
        if {"indextts", "indextts_asr"} & requested_tests:
            print("[smoke] LOCAL_INDEXTTS_TEXT_FRONTEND=official_like (forced by smoke harness)")

        controller_args = [
            str(controller_bin),
            "configs/controller.yaml",
            "--workdir",
            str(workdir),
            "--model-dir",
            str(model_dir),
            "--worker-registration-token",
            REGISTRATION_TOKEN,
            "--public-base-url",
            CONTROLLER_URL,
            "--admin-token",
            ADMIN_TOKEN,
            "--mcp-bind",
            "127.0.0.1:17892",
        ]
        controller = start_service("controller", controller_args, root, data_dir, timestamp, env)
        launched.append(controller)
        wait_health("controller", f"{CONTROLLER_URL}/health", controller, args.ready_timeout, args.request_timeout, data_dir)

        if ({"indextts", "indextts_asr", "mcp_standard"} & requested_tests) and indextts_artifacts["ready"]:
            print("[smoke] enabling IndexTTS before worker starts so worker registry snapshot can serve it")
            rpc_enable_indextts(args.request_timeout)

        worker_args = [
            str(worker_bin),
            "configs/worker.yaml",
            "--workdir",
            str(workdir),
            "--model-dir",
            str(model_dir),
            "--registration-token",
            REGISTRATION_TOKEN,
        ]
        worker = start_service("worker", worker_args, root, data_dir, timestamp, env)
        launched.append(worker)
        wait_health("worker", f"{WORKER_URL}/health", worker, args.ready_timeout, args.request_timeout, data_dir)

        health_payload = {
            "controller": checked_json_request("GET", f"{CONTROLLER_URL}/health", None, args.request_timeout),
            "worker": checked_json_request("GET", f"{WORKER_URL}/health", None, args.request_timeout),
        }
        try:
            health_payload["route_policy"] = check_route_policy(args.request_timeout)
            health_payload["auth_policy"] = check_auth_policy(args.request_timeout)
        except SmokeError as exc:
            failures.append(f"route/auth policy: {exc}")
        save_json(data_dir / f"smoke-health-{timestamp}.json", health_payload)

        if "assets" in requested_tests:
            try:
                run_assets(data_dir, timestamp, args.request_timeout)
            except SmokeError as exc:
                failures.append(f"assets: {exc}")

        if "mcp_standard" in requested_tests:
            try:
                run_mcp_standard(
                    data_dir,
                    timestamp,
                    args.request_timeout,
                    None if args.skip_yolo else yolo_image,
                    None if args.skip_sensevoice_asr else sensevoice_asr_audio,
                    None if args.skip_indextts else indextts_reference,
                    args.indextts_text,
                    {"ready": False, "reason": "--skip-indextts"} if args.skip_indextts else indextts_artifacts,
                )
            except SmokeError as exc:
                failures.append(f"mcp_standard: {exc}")

        if "yolo" in requested_tests:
            try:
                run_yolo(yolo_image, data_dir, timestamp, args.request_timeout)
            except SmokeError as exc:
                failures.append(f"yolo: {exc}")

        if "sensevoice-asr" in requested_tests:
            try:
                run_sensevoice_asr(sensevoice_asr_audio, data_dir, timestamp, args.request_timeout)
            except SmokeError as exc:
                failures.append(f"sensevoice-asr: {exc}")

        if "indextts" in requested_tests:
            try:
                run_indextts(
                    indextts_artifacts,
                    indextts_reference,
                    args.indextts_text,
                    args.indextts_frontend,
                    data_dir,
                    timestamp,
                    args.request_timeout,
                )
            except SmokeError as exc:
                failures.append(f"indextts: {exc}")

        if "indextts_asr" in requested_tests:
            try:
                run_indextts_asr(
                    indextts_artifacts,
                    indextts_reference,
                    args.indextts_text,
                    args.indextts_frontend,
                    data_dir,
                    timestamp,
                    args.request_timeout,
                )
            except SmokeError as exc:
                failures.append(f"indextts_asr: {exc}")

        if "embedding" in requested_tests:
            try:
                run_embedding(data_dir, timestamp, args.request_timeout)
            except SmokeError as exc:
                failures.append(f"embedding: {exc}")

        if "rerank" in requested_tests:
            try:
                run_rerank(data_dir, timestamp, args.request_timeout)
            except SmokeError as exc:
                failures.append(f"rerank: {exc}")

    except KeyboardInterrupt:
        print("[smoke] interrupted; cleaning up launched services", file=sys.stderr)
        failures.append("interrupted")
    except SmokeError as exc:
        failures.append(str(exc))
    finally:
        cleanup_processes(launched)
        try:
            wait_ports_closed(PORTS, timeout=8.0)
        except SmokeError as exc:
            failures.append(str(exc))

    if failures:
        print("[smoke] FAIL", file=sys.stderr)
        for failure in failures:
            print(f"[smoke] - {failure}", file=sys.stderr)
        return 1

    print("[smoke] PASS")
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local controller/worker API, legacy RPC, and MCP smoke tests. "
            "Default media assets are local files under scripts/assets; this harness has no network-download sample args."
        )
    )
    parser.add_argument("--workdir", type=Path, default=Path("./workdir"))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--controller-bin", type=Path, default=None)
    parser.add_argument("--worker-bin", type=Path, default=None)
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run cargo build before locating binaries (default; kept for compatibility).",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Build/use target/release/controller(.exe) and target/release/worker(.exe). Overrides still take precedence.",
    )
    parser.add_argument("--skip-build", action="store_true", help="Do not build; use existing binaries.")
    parser.add_argument(
        "--tests",
        default="assets,yolo,sensevoice-asr,indextts",
        help=(
            "Comma-separated smoke tests or groups: rpc,mcp,all,assets,yolo,sensevoice-asr,indextts,"
            "indextts_asr,embedding,rerank,text,mcp_standard. "
            "rpc expands to legacy JSON-RPC coverage on /rpc/admin and /rpc/infer. "
            "mcp expands to standard MCP SDK coverage on /mcp/admin and /mcp/infer. "
            "all runs both groups. OCR smoke aliases were removed after withdrawal."
        ),
    )
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument(
        "--skip-sensevoice-asr",
        dest="skip_sensevoice_asr",
        action="store_true",
        help="Skip SenseVoice ASR smoke coverage.",
    )
    parser.add_argument("--skip-indextts", action="store_true")
    parser.add_argument(
        "--indextts-asr-check",
        action="store_true",
        help="Also synthesize IndexTTS audio and transcribe it with SenseVoice ASR via generic legacy RPC tasks.",
    )
    parser.add_argument(
        "--yolo-image",
        type=Path,
        default=None,
        help="Local YOLO image override. Default: scripts/assets/yolo-input.jpg.",
    )
    parser.add_argument(
        "--sensevoice-asr-audio",
        dest="sensevoice_asr_audio",
        type=Path,
        default=None,
        help="Local SenseVoice ASR WAV override. Default: scripts/assets/tts-input-mon3tr.wav.",
    )
    parser.add_argument(
        "--indextts-reference",
        type=Path,
        default=None,
        help="Local IndexTTS reference WAV override. Default: scripts/assets/tts-input-mon3tr.wav.",
    )
    parser.add_argument("--indextts-text", default="你好，这是本地 IndexTTS 冒烟测试。")
    parser.add_argument(
        "--indextts-frontend",
        choices=("auto", "rust", "official-python", "rust-fallback"),
        default="auto",
        help="IndexTTS text frontend for smoke tasks: auto/rust use the Rust runtime frontend; official-python injects oracle token ids; rust-fallback is a backward-compatible alias for rust.",
    )
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=15.0)
    args = parser.parse_args(argv)
    if args.build and args.skip_build:
        parser.error("--build and --skip-build cannot be used together")
    args.build = not args.skip_build
    return args


def selected_tests(args: argparse.Namespace) -> set[str]:
    raw_tests = {item.strip().lower() for item in args.tests.split(",") if item.strip()}
    unknown = raw_tests - TEST_ALIASES
    if unknown:
        raise SmokeError(f"unknown --tests entries: {', '.join(sorted(unknown))}")
    tests = set(raw_tests)
    if "all" in raw_tests:
        tests.update(RPC_TESTS)
        tests.update(MCP_TESTS)
    if "rpc" in raw_tests:
        tests.update(RPC_TESTS)
    if "mcp" in raw_tests:
        tests.update(MCP_TESTS)
    if "text" in raw_tests:
        tests.update({"embedding", "rerank"})
    tests.difference_update({"all", "rpc", "mcp", "text"})
    if args.skip_yolo:
        tests.discard("yolo")
    if args.skip_sensevoice_asr:
        tests.discard("sensevoice-asr")
    if args.skip_indextts:
        tests.discard("indextts")
        tests.discard("indextts_asr")
    if args.indextts_asr_check:
        tests.add("indextts_asr")
    return tests


def wait_health(
    name: str,
    url: str,
    process: ManagedProcess,
    ready_timeout: float,
    request_timeout: float,
    data_dir: Path,
) -> dict:
    deadline = time.monotonic() + ready_timeout
    last_error = "not checked"
    while time.monotonic() < deadline:
        code = process.proc.poll()
        if code is not None:
            logs = collect_recent_logs(data_dir, [process.stdout_path, process.stderr_path])
            raise SmokeError(
                f"{name} exited before ready with code {code}; stdout={process.stdout_path}; "
                f"stderr={process.stderr_path}; recent_logs={logs}"
            )
        try:
            status, payload = json_request("GET", url, None, request_timeout)
            if status == 200:
                print(f"[smoke] {name} ready: {payload}")
                return payload
            last_error = f"HTTP {status}: {payload}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5)
    logs = collect_recent_logs(data_dir, [process.stdout_path, process.stderr_path])
    raise SmokeError(f"{name} was not ready after {ready_timeout:.1f}s: {last_error}; recent_logs={logs}")


def collect_recent_logs(data_dir: Path, preferred: Iterable[Path], max_files: int = 4, tail_bytes: int = 4096) -> dict[str, str]:
    candidates = [path for path in preferred if path.exists()]
    candidates.extend(
        path
        for path in sorted(data_dir.glob("*.log"), key=lambda item: item.stat().st_mtime, reverse=True)
        if path not in candidates
    )
    logs: dict[str, str] = {}
    for path in candidates[:max_files]:
        try:
            data = path.read_bytes()[-tail_bytes:]
            logs[str(path)] = data.decode("utf-8", errors="replace")
        except OSError as exc:
            logs[str(path)] = f"<failed to read log: {exc}>"
    return logs


def check_route_policy(timeout: float) -> dict:
    """Assert legacy RPC and standard MCP routes stay separate."""
    checks: list[dict] = []
    for path in ("/mcp/admin", "/mcp/infer"):
        url = f"{CONTROLLER_URL}{path}"
        for method, body_data, headers in (
            ("GET", None, {"Accept": "*/*"}),
            (
                "POST",
                json.dumps({"jsonrpc": "2.0", "id": "route-policy", "method": "list_models", "params": {}}).encode(
                    "utf-8"
                ),
                {"Accept": "application/json", "Content-Type": "application/json"},
            ),
        ):
            status, body, _headers = raw_bytes_request(method, url, body_data, headers, timeout)
            check = {"method": method, "url": url, "expected_status": 404, "status": status, "ok": status == 404}
            checks.append(check)
            if status != 404:
                preview = body[:200].decode("utf-8", errors="replace")
                raise SmokeError(f"{method} {path} must remain unavailable on controller; got HTTP {status}: {preview}")
    print("[smoke] route policy ok: /mcp/admin and /mcp/infer returned 404 for GET and POST")
    return {"ok": True, "checks": checks}


def check_auth_policy(timeout: float) -> dict:
    """Assert all configured admin/inference RPC and MCP routes reject missing credentials."""
    rpc_body = json.dumps({"jsonrpc": "2.0", "id": "auth-policy", "method": "get_task", "params": {}}).encode(
        "utf-8"
    )
    checks: list[dict] = []
    for url in (
        f"{CONTROLLER_URL}/rpc/admin",
        f"{CONTROLLER_URL}/rpc/infer",
        MCP_ADMIN_URL,
        MCP_INFER_URL,
    ):
        status, body, _headers = raw_bytes_request(
            "POST",
            url,
            rpc_body,
            {"Accept": "application/json", "Content-Type": "application/json"},
            timeout,
        )
        check = {"method": "POST", "url": url, "expected_status": 401, "status": status, "ok": status == 401}
        checks.append(check)
        if status != 401:
            preview = body[:200].decode("utf-8", errors="replace")
            raise SmokeError(f"POST {url} without credentials must return 401; got HTTP {status}: {preview}")
    print("[smoke] auth policy ok: configured admin/inference RPC and MCP routes rejected missing credentials")
    return {"ok": True, "checks": checks}


def rpc_infer(method: str, params: dict, request_id: str, timeout: float) -> tuple[int, dict]:
    payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return raw_request(
        "POST",
        f"{CONTROLLER_URL}/rpc/infer",
        data,
        {"Accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {INFER_TOKEN}"},
        timeout,
    )


def rpc_admin(method: str, params: dict, request_id: str, timeout: float) -> tuple[int, dict]:
    payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return raw_request(
        "POST",
        f"{CONTROLLER_URL}/rpc/admin",
        data,
        {"Accept": "application/json", "Content-Type": "application/json", "x-local-admin-token": ADMIN_TOKEN},
        timeout,
    )

def run_mcp_standard(
    data_dir: Path,
    timestamp: str,
    timeout: float,
    sample_image: Path | None,
    sample_audio: Path | None,
    reference_audio: Path | None,
    text: str,
    indextts_artifacts: dict,
) -> None:
    print("[smoke] validating standard admin and inference MCP endpoints")
    output_path = data_dir / f"smoke-mcp-standard-{timestamp}.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.local.mcp_standard_client",
        "--admin-url",
        MCP_ADMIN_URL,
        "--infer-url",
        MCP_INFER_URL,
        "--admin-token",
        ADMIN_TOKEN,
        "--infer-token",
        INFER_TOKEN,
        "--full",
        "--text",
        text,
        "--timeout",
        str(int(timeout)),
    ]
    if sample_image is not None:
        cmd.extend(["--sample-image", str(sample_image)])
    if sample_audio is not None:
        cmd.extend(["--sample-audio", str(sample_audio)])
    if reference_audio is not None:
        cmd.extend(["--reference-audio", str(reference_audio)])
    if indextts_artifacts.get("ready"):
        cmd.append("--indextts-artifacts-ready")
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(repo_root()),
            text=True,
            capture_output=True,
            timeout=max(10.0, timeout + 15.0),
        )
    except subprocess.TimeoutExpired as exc:
        raise SmokeError(f"standard MCP client timed out after {exc.timeout}s") from exc
    output_path.write_text(completed.stdout or "{}", encoding="utf-8")
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr)
    if completed.returncode != 0:
        raise SmokeError(f"standard MCP client failed with exit code {completed.returncode}; summary={output_path}")
    print(f"[smoke] standard MCP summary={output_path}")


def run_assets(data_dir: Path, timestamp: str, timeout: float) -> None:
    body = f"hello assets {timestamp}\n".encode("utf-8")
    path = f"smoke/{timestamp}/hello.txt"
    sign_request = {
        "requests": [
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
    }
    http_sign_payload = checked_json_request("POST", f"{CONTROLLER_URL}/assets/sign", sign_request, timeout)
    sign_status, rpc_sign_payload = rpc_infer(
        "sign_asset_urls",
        {"items": sign_request["requests"]},
        f"smoke-assets-rpc-sign-{timestamp}",
        timeout,
    )
    assert_rpc_success(sign_status, rpc_sign_payload, "legacy RPC sign_asset_urls")
    signed = http_sign_payload.get("items") or []
    rpc_signed = (rpc_sign_payload.get("result") or {}).get("items") or []
    if (
        len(signed) != 2
        or signed[0].get("method") != "POST"
        or signed[1].get("method") != "GET"
        or len(rpc_signed) != 2
        or rpc_signed[0].get("method") != "POST"
        or rpc_signed[1].get("method") != "GET"
    ):
        raise SmokeError(f"sign_assets did not preserve batch order/methods: http={signed} legacy_rpc={rpc_signed}")
    upload_url = signed[0].get("signed_url")
    download_url = signed[1].get("signed_url")
    status, payload = raw_request("POST", upload_url, body, {"Content-Type": "text/plain; charset=utf-8"}, timeout)
    if status != 200:
        raise SmokeError(f"asset upload returned HTTP {status}: {payload}")
    uri = payload.get("uri")
    if uri != f"assets://material/{path}":
        raise SmokeError(f"unexpected asset uri after upload: {payload}")
    list_payload = checked_json_request("GET", f"{CONTROLLER_URL}/assets?kind=material&contains={timestamp}", None, timeout)
    assets = list_payload.get("assets") or []
    if not any(asset.get("uri") == uri for asset in assets):
        raise SmokeError(f"uploaded asset not found in list/search: {list_payload}")
    task = rpc_create_task(
        {
            "task_kind": "asr.transcribe",
            "files": [
                {
                    "name": "duplicate.txt",
                    "mime": "text/plain; charset=utf-8",
                    "role": "audio",
                    "required": True,
                }
            ],
            "params": {},
        },
        f"smoke-assets-dedup-create-{timestamp}",
        timeout,
    )
    task_upload = first_upload(task, "audio")
    requested_uri = task_upload.get("asset_uri") or ""
    status, dedup_payload = raw_request(
        "POST",
        task_upload["upload_url"],
        body,
        {"Content-Type": "text/plain; charset=utf-8"},
        timeout,
    )
    if status != 200 or dedup_payload.get("uri") != uri:
        raise SmokeError(f"task material upload was not hash-reused status={status}: {dedup_payload}")
    task_status_code, task_status_payload = rpc_infer(
        "get_task",
        {"task_id": task["task_id"]},
        f"smoke-assets-dedup-get-{timestamp}",
        timeout,
    )
    assert_rpc_success(task_status_code, task_status_payload, "legacy RPC get deduplicated task")
    task_status = task_status_payload.get("result") or {}
    reused_upload = first_upload(task_status, "audio")
    if task_status.get("state") != "ready" or not reused_upload.get("uploaded") or reused_upload.get("asset_uri") != uri:
        raise SmokeError(f"deduplicated task slot was not retargeted to reused material: {task_status}")
    requested_prefix = "assets://material/"
    if requested_uri.startswith(requested_prefix):
        duplicate_payload_path = data_dir / "assets" / "material" / requested_uri[len(requested_prefix) :]
        if duplicate_payload_path.exists():
            raise SmokeError(f"deduplicated task upload still wrote a second payload: {duplicate_payload_path}")
    status, _downloaded, _headers = raw_bytes_request("GET", f"{CONTROLLER_URL}/assets/material/{path}", None, {}, timeout)
    if status == 200:
        raise SmokeError("unsigned asset download unexpectedly succeeded")
    status, downloaded, _headers = raw_bytes_request("GET", download_url, None, {}, timeout)
    if status != 200 or downloaded != body:
        raise SmokeError(f"asset download mismatch status={status} bytes={downloaded!r}")
    status, delete_payload = raw_request(
        "DELETE",
        f"{CONTROLLER_URL}/assets/material/{path}",
        None,
        {"x-local-admin-token": ADMIN_TOKEN},
        timeout,
    )
    if status != 200 or not delete_payload.get("deleted"):
        raise SmokeError(f"asset delete failed HTTP {status}: {delete_payload}")
    status, _downloaded, _headers = raw_bytes_request("GET", download_url, None, {}, timeout)
    if status != 404:
        raise SmokeError(f"deleted asset remained downloadable: HTTP {status}")
    out = data_dir / f"smoke-assets-{timestamp}.json"
    save_json(
        out,
        {
            "uri": uri,
            "list_count": len(assets),
            "deleted": True,
            "signed_batch": signed,
            "deduplicated_task_id": task["task_id"],
            "deduplicated_requested_uri": requested_uri,
        },
    )
    print(f"[smoke] assets saved {out}")


def assert_rpc_success(status: int, payload: dict, label: str) -> None:
    if status != 200:
        raise SmokeError(f"{label} returned HTTP {status}: {payload}")
    if payload.get("error"):
        raise SmokeError(f"{label} returned JSON-RPC error: {payload['error']}")


def image_mime(path: Path | str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".bmp":
        return "image/bmp"
    return "application/octet-stream"

def run_embedding(data_dir: Path, timestamp: str, timeout: float) -> None:
    payload = checked_json_request(
        "POST",
        f"{CONTROLLER_URL}/v1/embeddings",
        {
            "model": EMBEDDING_MODEL_ID,
            "input": ["今天天气很好。", "Local ONNX inference works."],
            "input_type": "passage",
            "encoding_format": "float",
        },
        timeout,
    )
    if payload.get("object") != "list" or payload.get("model") != EMBEDDING_MODEL_ID:
        raise SmokeError(f"OpenAI embeddings response envelope mismatch: {payload}")
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != 2:
        raise SmokeError(f"OpenAI embeddings response must preserve batch size/order: {payload}")
    for index, item in enumerate(data):
        embedding = item.get("embedding") if isinstance(item, dict) else None
        if item.get("index") != index or item.get("object") != "embedding":
            raise SmokeError(f"OpenAI embedding item {index} has wrong metadata: {item}")
        if not isinstance(embedding, list) or len(embedding) != 384:
            raise SmokeError(f"OpenAI embedding item {index} is not 384-dimensional")
        norm = sum(float(value) ** 2 for value in embedding) ** 0.5
        if abs(norm - 1.0) > 1e-3:
            raise SmokeError(f"OpenAI embedding item {index} is not L2-normalized: norm={norm}")
    usage = payload.get("usage", {})
    if not isinstance(usage.get("prompt_tokens"), int) or usage["prompt_tokens"] <= 0:
        raise SmokeError(f"OpenAI embeddings usage is missing prompt token count: {payload}")
    save_json(data_dir / f"smoke-embedding-{timestamp}.json", payload)


def run_rerank(data_dir: Path, timestamp: str, timeout: float) -> None:
    request = {
        "model": RERANK_MODEL_ID,
        "query": "法国的首都是什么？",
        "documents": [
            "巴西的首都是巴西利亚。",
            "法国的首都是巴黎。",
            "马和牛都是动物。",
        ],
        "top_n": 2,
    }
    payload = checked_json_request(
        "POST", f"{CONTROLLER_URL}/v1/rerank", request, timeout
    )
    if payload.get("model") != RERANK_MODEL_ID or not str(payload.get("id", "")).startswith("rerank-"):
        raise SmokeError(f"vLLM rerank response envelope mismatch: {payload}")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 2:
        raise SmokeError(f"vLLM rerank top_n was not honored: {payload}")
    if results[0].get("index") != 1 or results[0].get("document", {}).get("text") != request["documents"][1]:
        raise SmokeError(f"vLLM rerank did not rank the relevant document first: {payload}")
    scores = [result.get("relevance_score") for result in results]
    if not all(isinstance(score, (int, float)) and 0.0 <= score <= 1.0 for score in scores):
        raise SmokeError(f"vLLM rerank scores must be activated probabilities: {payload}")
    if scores != sorted(scores, reverse=True):
        raise SmokeError(f"vLLM rerank results are not score-sorted: {payload}")
    usage = payload.get("usage", {})
    if not isinstance(usage.get("total_tokens"), int) or usage["total_tokens"] <= 0:
        raise SmokeError(f"vLLM rerank usage is missing total token count: {payload}")
    save_json(data_dir / f"smoke-rerank-{timestamp}.json", payload)


def run_yolo(image: Path, data_dir: Path, timestamp: str, timeout: float) -> None:
    if not image.exists():
        raise SmokeError(f"YOLO image does not exist: {image}")
    content_type = image_mime(image)
    create = rpc_create_task(
        {
            "task_kind": "object.detect",
            "model": "yolo11n.onnx",
            "files": [{"name": image.name, "mime": content_type, "role": "image", "required": True}],
            "params": {},
        },
        f"smoke-yolo-create-{timestamp}",
        timeout,
    )
    upload = first_upload(create, "image")
    payload = upload_file(upload["upload_url"] + "&with_start_task=true", image, content_type, timeout)
    if payload.get("uri", "").startswith("assets://"):
        payload = rpc_start_task(create["task_id"], f"smoke-yolo-start-{timestamp}", timeout)
    object_count, car_count = validate_yolo_payload(payload)
    out = data_dir / f"smoke-yolo-{timestamp}.json"
    save_json(
        out,
        {
            "input_image": str(image),
            "object_count": object_count,
            "car_count": car_count,
            "task": payload,
        },
    )
    print(f"[smoke] yolo car_count={car_count} object_count={object_count} saved {out}")


def run_sensevoice_asr(audio: Path, data_dir: Path, timestamp: str, timeout: float) -> None:
    if not audio.exists():
        raise SmokeError(f"SenseVoice ASR audio does not exist: {audio}")
    out = data_dir / f"smoke-sensevoice-asr-{timestamp}.json"
    create = rpc_create_task(
        {
            "task_kind": "asr.transcribe",
            "model": "sensevoice-small-onnx",
            "files": [{"name": audio.name, "mime": "audio/wav", "role": "audio", "required": True}],
            "params": {},
        },
        f"smoke-sensevoice-asr-create-{timestamp}",
        timeout,
    )
    upload_file(first_upload(create, "audio")["upload_url"], audio, "audio/wav", timeout)
    payload = rpc_start_task(create["task_id"], f"smoke-sensevoice-asr-start-{timestamp}", timeout)
    if payload.get("state") != "succeeded":
        raise SmokeError(f"SenseVoice ASR generic task did not succeed: {payload}")
    asr_text = extract_asr_text(payload)
    if not asr_text.strip():
        raise SmokeError(f"SenseVoice ASR generic task returned empty text: {payload}")
    output = payload.get("output") or {}
    segments = output.get("segments") if isinstance(output, dict) else None
    speakers = output.get("speakers") if isinstance(output, dict) else None
    timestamped_text = output.get("timestamped_text") if isinstance(output, dict) else None
    if not isinstance(timestamped_text, str) or not timestamped_text.startswith("["):
        raise SmokeError(f"SenseVoice ASR did not return default timestamped_text: {payload}")
    if not isinstance(segments, list) or not segments:
        raise SmokeError(f"SenseVoice ASR did not return default timeline segments: {payload}")
    if not isinstance(speakers, list) or not speakers:
        raise SmokeError(f"SenseVoice ASR did not return default speaker diarization: {payload}")
    for segment in segments:
        if (
            not isinstance(segment, dict)
            or not isinstance(segment.get("start_ms"), int)
            or not isinstance(segment.get("end_ms"), int)
            or segment["end_ms"] <= segment["start_ms"]
            or segment["end_ms"] - segment["start_ms"] > 15_000
            or not str(segment.get("speaker", "")).startswith("speaker_")
            or bool(segment.get("tokens"))
        ):
            raise SmokeError(f"SenseVoice ASR returned an invalid timeline/speaker segment: {segment}")
    save_json(
        out,
        {
            "input_audio": str(audio),
            "asr_text": asr_text,
            "asr_text_length": len(asr_text),
            "segment_count": len(segments),
            "speaker_count": len(speakers),
            "task": payload,
        },
    )
    print(
        f"[smoke] sensevoice-asr text prefix={asr_text[:120]!r} length={len(asr_text)} "
        f"segments={len(segments)} speakers={len(speakers)} saved {out}"
    )



def post_openai_transcription_multipart(audio: Path, timeout: float) -> tuple[int, dict]:
    boundary = f"----local-smoke-{int(time.time() * 1000)}"
    body = encode_multipart(
        boundary,
        fields={"model": "sensevoice-small-onnx"},
        files={"file": (audio.name, audio.read_bytes(), "audio/wav")},
    )
    headers = {
        "Accept": "application/json",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    return raw_request("POST", f"{CONTROLLER_URL}/v1/audio/transcriptions", body, headers, timeout)


def inspect_indextts_artifacts(model_dir: Path) -> dict:
    env_root = os.environ.get("LOCAL_INDEXTTS_MODEL_DIR")
    root = Path(env_root).resolve() if env_root else (model_dir / INDEXTTS_MODEL_ID).resolve()
    missing = [str(root / name) for name in INDEXTTS_REQUIRED if not (root / name).exists()]
    return {
        "ready": not missing,
        "root": str(root),
        "source": "LOCAL_INDEXTTS_MODEL_DIR" if env_root else "--model-dir/default",
        "required": INDEXTTS_REQUIRED,
        "missing": missing,
    }



def rpc_enable_indextts(timeout: float) -> None:
    status, payload = rpc_admin("enable_model", {"id": INDEXTTS_MODEL_ID}, "smoke-enable-indextts", timeout)
    assert_rpc_success(status, payload, "IndexTTS enable_model")


def run_indextts(
    artifacts: dict,
    reference: Path,
    text: str,
    frontend: str,
    data_dir: Path,
    timestamp: str,
    timeout: float,
) -> None:
    out = data_dir / f"smoke-indextts-{timestamp}.json"
    if not artifacts["ready"]:
        payload = {
            "status": "skipped",
            "reason": "missing IndexTTS artifacts; no download attempted",
            "artifacts": artifacts,
            **indextts_frontend_report(frontend),
            "token_ids_source": "none",
        }
        save_json(out, payload)
        print(f"[smoke] indextts skipped; details saved {out}")
        return
    if not reference.exists():
        raise SmokeError(f"IndexTTS reference audio does not exist: {reference}")
    frontend_params, frontend_info = prepare_indextts_frontend_params(
        text,
        frontend,
        data_dir,
        f"{timestamp}-tts",
        Path(artifacts["root"]),
    )
    payload = synthesize_indextts(reference, text, timestamp, timeout, frontend_params)
    payload["frontend"] = frontend_info
    save_json(out, payload)
    if payload.get("state") != "succeeded":
        raise SmokeError(f"IndexTTS generic task did not succeed: {payload}")
    print(f"[smoke] indextts saved {out}")


def synthesize_indextts(reference: Path, text: str, timestamp: str, timeout: float, extra_params: dict | None = None) -> dict:
    task_params = {"text": text}
    if extra_params:
        task_params.update(extra_params)
    create = rpc_create_task(
        {
            "task_kind": "tts.synthesize",
            "model": INDEXTTS_MODEL_ID,
            "files": [
                {"name": reference.name, "mime": "audio/wav", "role": "reference_audio", "required": False}
            ],
            "params": task_params,
        },
        f"smoke-indextts-create-{timestamp}",
        timeout,
    )
    upload_file(first_upload(create, "reference_audio")["upload_url"], reference, "audio/wav", timeout)
    return rpc_start_task(create["task_id"], f"smoke-indextts-start-{timestamp}", timeout)


def run_indextts_asr(
    artifacts: dict,
    reference: Path,
    text: str,
    frontend: str,
    data_dir: Path,
    timestamp: str,
    timeout: float,
) -> None:
    out = data_dir / f"smoke-indextts-asr-{timestamp}.json"
    if not artifacts["ready"]:
        payload = {
            "status": "skipped",
            "reason": "missing IndexTTS artifacts; no download attempted",
            "artifacts": artifacts,
            "source_text": text,
            **indextts_frontend_report(frontend),
            "token_ids_source": "none",
        }
        save_json(out, payload)
        print(f"[smoke] indextts_asr skipped; details saved {out}")
        return
    if not reference.exists():
        raise SmokeError(f"IndexTTS reference audio does not exist: {reference}")

    frontend_params, frontend_info = prepare_indextts_frontend_params(
        text,
        frontend,
        data_dir,
        f"{timestamp}-asr",
        Path(artifacts["root"]),
    )
    tts_payload = synthesize_indextts(reference, text, f"{timestamp}-asr", timeout, frontend_params)
    if tts_payload.get("state") != "succeeded":
        raise SmokeError(f"IndexTTS ASR cross-check synthesis did not succeed: {tts_payload}")
    audio_ref = extract_tts_audio_ref(tts_payload)
    tts_wav = resolve_audio_path(audio_ref)
    if tts_wav is None or not tts_wav.exists():
        raise SmokeError(f"IndexTTS ASR cross-check could not resolve synthesized wav: {audio_ref}")

    asr_payload = transcribe_audio_generic(tts_wav, timestamp, timeout)
    if asr_payload.get("state") != "succeeded":
        raise SmokeError(f"IndexTTS ASR cross-check transcription did not succeed: {asr_payload}")
    asr_text = extract_asr_text(asr_payload)
    if not asr_text.strip():
        raise SmokeError(f"IndexTTS ASR cross-check transcription returned empty text: {asr_payload}")
    expected_text = normalize_expected_text(text)
    comparison = compare_text(expected_text, normalize_expected_text(asr_text))
    payload = {
        "source_text": text,
        "frontend_mode": frontend_info["frontend_mode"],
        "token_ids_source": frontend_info["token_ids_source"],
        "frontend_report": frontend_info.get("report_path"),
        "normalized_expected_text": expected_text,
        "tts_wav": {
            "path": str(tts_wav),
            "url": audio_ref.get("url"),
        },
        "asr_text": asr_text,
        "asr_text_length": len(asr_text),
        "similarity": comparison["similarity"],
        "char_coverage": comparison["char_coverage"],
        "missing_chars": comparison["missing_chars"],
        "extra_chars": comparison["extra_chars"],
        "tts_task": tts_payload,
        "asr_task": asr_payload,
    }
    save_json(out, payload)
    print(f"[smoke] indextts_asr saved {out}")


def prepare_indextts_frontend_params(
    text: str,
    frontend: str,
    data_dir: Path,
    timestamp: str,
    artifact_root: Path,
) -> tuple[dict, dict]:
    if frontend in {"auto", "rust", "rust-fallback"}:
        return {}, indextts_frontend_report(frontend)
    try:
        from . import indextts_text_parity

        root = repo_root()
        report = indextts_text_parity.build_report(
            [text],
            indextts_text_parity.default_source_root(root),
            indextts_bridge_bpe_model(root, artifact_root),
            data_dir / "indextts-official-tagger-cache",
            compare_rust_fallback=False,
            compare_rust_frontend=True,
        )
        report_path = data_dir / f"indextts-text-parity-{timestamp}.json"
        save_json(report_path, report)
        first = (report.get("texts") or [{}])[0]
        token_ids = first.get("token_ids")
        if isinstance(token_ids, list) and token_ids and all(isinstance(item, int) for item in token_ids):
            return {"indextts_text_token_ids": token_ids}, {
                "frontend_mode": "official-python",
                "requested_frontend": frontend,
                "token_ids_source": "official_python",
                "rust_frontend_env": "official_like",
                "report_path": str(report_path),
            }
        raise SmokeError(f"official IndexTTS token ids unavailable; see {report_path}")
    except SmokeError:
        raise
    except Exception as exc:
        raise SmokeError(f"official IndexTTS frontend bridge failed: {exc}") from exc


def indextts_frontend_report(frontend: str) -> dict:
    if frontend in {"auto", "rust", "rust-fallback"}:
        return {
            "frontend_mode": "rust",
            "requested_frontend": frontend,
            "token_ids_source": "rust_runtime",
            "rust_frontend_env": "official_like",
        }
    return {
        "frontend_mode": "official-python",
        "requested_frontend": frontend,
        "token_ids_source": "official_python",
        "rust_frontend_env": "official_like",
    }


def indextts_bridge_bpe_model(root: Path, artifact_root: Path) -> Path | None:
    bpe_model = artifact_root / "bpe.model"
    if bpe_model.exists():
        return bpe_model
    from . import indextts_text_parity

    return indextts_text_parity.default_bpe_model(root)


def transcribe_audio_generic(audio: Path, timestamp: str, timeout: float) -> dict:
    create = rpc_create_task(
        {
            "task_kind": "asr.transcribe",
            "model": "sensevoice-small-onnx",
            "files": [{"name": audio.name, "mime": "audio/wav", "role": "audio", "required": True}],
            "params": {},
        },
        f"smoke-indextts-asr-sensevoice-create-{timestamp}",
        timeout,
    )
    upload_file(first_upload(create, "audio")["upload_url"], audio, "audio/wav", timeout)
    return rpc_start_task(create["task_id"], f"smoke-indextts-asr-sensevoice-start-{timestamp}", timeout)


def extract_tts_audio_ref(payload: dict) -> dict:
    output = payload.get("output") or {}
    if isinstance(output, dict) and isinstance(output.get("audio"), dict):
        return output["audio"]
    for file_ref in payload.get("files") or []:
        if isinstance(file_ref, dict) and file_ref.get("mime") == "audio/wav":
            return file_ref
    return {}


def resolve_audio_path(file_ref: dict) -> Path | None:
    raw_path = file_ref.get("path")
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else (repo_root() / path).resolve()


def validate_yolo_payload(payload: dict) -> tuple[int, int]:
    if payload.get("state") != "succeeded":
        raise SmokeError(f"YOLO generic task did not succeed: {payload}")
    output = payload.get("output") or {}
    if not isinstance(output, dict) or output.get("type") != "object_detections":
        raise SmokeError(f"YOLO output type mismatch: {payload}")
    objects = output.get("objects")
    if not isinstance(objects, list) or not objects:
        raise SmokeError(f"YOLO output did not contain any objects: {payload}")
    labels = [object_label(item) for item in objects if isinstance(item, dict)]
    car_count = sum(1 for label in labels if is_car_like_label(label))
    if car_count <= 0:
        raise SmokeError(f"YOLO output did not contain a car-like label: labels={labels!r}")
    return len(objects), car_count


def object_label(item: dict) -> str:
    for key in ("label", "class", "name", "class_name"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


def is_car_like_label(label: str) -> bool:
    normalized = "".join(ch for ch in label.lower() if ch.isalnum())
    return normalized in {"car", "cars", "automobile"} or "car" in normalized


def extract_asr_text(payload: dict) -> str:
    output = payload.get("output") or {}
    if isinstance(output, dict) and isinstance(output.get("text"), str):
        return output["text"]
    return ""


def normalize_expected_text(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def compare_text(expected: str, actual: str) -> dict:
    similarity = SequenceMatcher(None, expected, actual).ratio() if (expected or actual) else 1.0
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    missing = expected_counts - actual_counts
    extra = actual_counts - expected_counts
    covered = max(0, len(expected) - sum(missing.values()))
    coverage = covered / len(expected) if expected else 1.0
    return {
        "similarity": similarity,
        "char_coverage": coverage,
        "missing_chars": dict(sorted(missing.items())),
        "extra_chars": dict(sorted(extra.items())),
    }


def rpc_create_task(params: dict, request_id: str, timeout: float) -> dict:
    status, payload = rpc_infer("create_task", params, request_id, timeout)
    assert_rpc_success(status, payload, "legacy RPC create_task")
    result = payload.get("result")
    if not isinstance(result, dict) or not result.get("task_id"):
        raise SmokeError(f"create_task returned malformed result: {payload}")
    return result


def rpc_start_task(task_id: str, request_id: str, timeout: float) -> dict:
    status, payload = rpc_infer("start_task", {"task_id": task_id, "wait": True, "timeout_sec": int(timeout)}, request_id, timeout)
    assert_rpc_success(status, payload, "legacy RPC start_task")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise SmokeError(f"start_task returned malformed result: {payload}")
    return result


def first_upload(status: dict, role: str) -> dict:
    uploads = status.get("uploads") or []
    for upload in uploads:
        if upload.get("role") == role or upload.get("slot") == role:
            return upload
    raise SmokeError(f"no upload slot with role {role}: {status}")


def upload_file(upload_url: str, path: Path, content_type: str, timeout: float) -> dict:
    status, payload = raw_request("POST", upload_url, path.read_bytes(), {"Content-Type": content_type}, timeout)
    if not (200 <= status < 300):
        raise SmokeError(f"upload {path} returned HTTP {status}: {payload}")
    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
