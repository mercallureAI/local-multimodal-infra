"""Python-only API/MCP smoke harness for local controller and worker.

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
REGISTRATION_TOKEN = "lcoal-smoke-registration-token"
MCP_STANDARD_URL = "http://127.0.0.1:17892/mcp"
PORTS = (17890, 17891, 17892)
TEST_ALIASES = {"assets", "yolo", "qwen", "indextts", "indextts_asr", "mcp_standard"}
INDEXTTS_MODEL_ID = "indextts-1.5-onnx"
ADMIN_TOKEN = "lcoal-smoke-admin-token"
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
    yolo_image = resolve_cli_path(args.yolo_image, root) if args.yolo_image else workdir / "data" / "samples" / "cars.jpg"
    qwen_audio = resolve_cli_path(args.qwen_audio, root) if args.qwen_audio else workdir / "data" / "samples" / "0_jackson_0.wav"
    indextts_reference = resolve_cli_path(args.indextts_reference, root) if args.indextts_reference else qwen_audio
    data_dir = workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    try:
        requested_tests = selected_tests(args)
    except SmokeError as exc:
        print(f"[smoke] FAIL: {exc}", file=sys.stderr)
        return 1
    indextts_artifacts = inspect_indextts_artifacts(model_dir)
    launched: list[ManagedProcess] = []
    failures: list[str] = []

    try:
        if args.build:
            run_build(root)

        controller_bin = locate_bin(root, "controller", args.controller_bin)
        worker_bin = locate_bin(root, "worker", args.worker_bin)

        print(f"[smoke] root={root}")
        print(f"[smoke] workdir={workdir}")
        print(f"[smoke] model_dir={model_dir}")
        print(f"[smoke] data_dir={data_dir}")
        print(f"[smoke] requested_tests={','.join(sorted(requested_tests)) or '<none>'}")

        env = os.environ.copy()
        env["LCOAL_DATA_DIR"] = str(data_dir)
        env["LCOAL_WORKER_REGISTRATION_TOKEN"] = REGISTRATION_TOKEN
        env["LCOAL_ADMIN_TOKEN"] = ADMIN_TOKEN
        if "LCOAL_INDEXTTS_MODEL_DIR" not in env:
            env["LCOAL_INDEXTTS_MODEL_DIR"] = str(model_dir / INDEXTTS_MODEL_ID)
        # Smoke runs must be deterministic even if the caller has an experimental
        # frontend in their shell (notably pinyin_explicit).  official-python
        # mode injects oracle token ids, but keep service env official_like too
        # so any diagnostics/fallbacks are not misleading.
        env["LCOAL_INDEXTTS_TEXT_FRONTEND"] = "official_like"
        if {"indextts", "indextts_asr"} & requested_tests:
            print("[smoke] LCOAL_INDEXTTS_TEXT_FRONTEND=official_like (forced by smoke harness)")

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

        if ({"indextts", "indextts_asr"} & requested_tests) and indextts_artifacts["ready"]:
            print("[smoke] enabling IndexTTS before worker starts so worker registry snapshot can serve it")
            enable_indextts(args.request_timeout)

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
        save_json(data_dir / f"smoke-health-{timestamp}.json", health_payload)

        if "assets" in requested_tests:
            try:
                run_assets(data_dir, timestamp, args.request_timeout)
            except SmokeError as exc:
                failures.append(f"assets: {exc}")

        if "mcp_standard" in requested_tests:
            try:
                run_mcp_standard(data_dir, timestamp, args.request_timeout)
            except SmokeError as exc:
                failures.append(f"mcp_standard: {exc}")

        if "yolo" in requested_tests:
            try:
                run_yolo(yolo_image, data_dir, timestamp, args.request_timeout)
            except SmokeError as exc:
                failures.append(f"yolo: {exc}")

        if "qwen" in requested_tests:
            try:
                run_qwen(qwen_audio, data_dir, timestamp, args.request_timeout)
            except SmokeError as exc:
                failures.append(f"qwen: {exc}")

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
    parser = argparse.ArgumentParser(description="Run local controller/worker API and MCP smoke tests.")
    parser.add_argument("--workdir", type=Path, default=Path("./workdir"))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--controller-bin", type=Path, default=None)
    parser.add_argument("--worker-bin", type=Path, default=None)
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run `cargo build --bins` before locating binaries (default; kept for compatibility).",
    )
    parser.add_argument("--skip-build", action="store_true", help="Do not build; use existing binaries.")
    parser.add_argument(
        "--tests",
        default="assets,yolo,qwen,indextts",
        help="Comma-separated smoke tests: assets,yolo,qwen,indextts,indextts_asr,mcp_standard",
    )
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-indextts", action="store_true")
    parser.add_argument(
        "--indextts-asr-check",
        action="store_true",
        help="Also synthesize IndexTTS audio and transcribe it with Qwen ASR via generic MCP tasks.",
    )
    parser.add_argument("--yolo-image", type=Path, default=None, help="Default: <workdir>/data/samples/cars.jpg")
    parser.add_argument("--qwen-audio", type=Path, default=None, help="Default: <workdir>/data/samples/0_jackson_0.wav")
    parser.add_argument("--indextts-reference", type=Path, default=None)
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
    tests = {item.strip().lower() for item in args.tests.split(",") if item.strip()}
    unknown = tests - TEST_ALIASES
    if unknown:
        raise SmokeError(f"unknown --tests entries: {', '.join(sorted(unknown))}")
    if args.skip_yolo:
        tests.discard("yolo")
    if args.skip_qwen:
        tests.discard("qwen")
    if args.skip_indextts:
        tests.discard("indextts")
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


def mcp_infer(method: str, params: dict, request_id: str, timeout: float) -> tuple[int, dict]:
    payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    return json_request("POST", f"{CONTROLLER_URL}/mcp/infer", payload, timeout)


def mcp_admin(method: str, params: dict, request_id: str, timeout: float) -> tuple[int, dict]:
    payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    return json_request("POST", f"{CONTROLLER_URL}/mcp/admin", payload, timeout)


def run_mcp_standard(data_dir: Path, timestamp: str, timeout: float) -> None:
    print("[smoke] validating standard MCP endpoint")
    output_path = data_dir / f"smoke-mcp-standard-{timestamp}.json"
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "scripts.lcoal.mcp_standard_client", "--url", MCP_STANDARD_URL],
            cwd=str(repo_root()),
            text=True,
            capture_output=True,
            timeout=max(10.0, timeout + 5.0),
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
    sign_status, mcp_sign_payload = mcp_infer(
        "sign_asset_urls",
        {"items": sign_request["requests"]},
        f"smoke-assets-mcp-sign-{timestamp}",
        timeout,
    )
    assert_mcp_success(sign_status, mcp_sign_payload, "MCP sign_asset_urls")
    signed = http_sign_payload.get("items") or []
    mcp_signed = (mcp_sign_payload.get("result") or {}).get("items") or []
    if (
        len(signed) != 2
        or signed[0].get("method") != "POST"
        or signed[1].get("method") != "GET"
        or len(mcp_signed) != 2
        or mcp_signed[0].get("method") != "POST"
        or mcp_signed[1].get("method") != "GET"
    ):
        raise SmokeError(f"sign_assets did not preserve batch order/methods: http={signed} mcp={mcp_signed}")
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
        {"x-lcoal-admin-token": ADMIN_TOKEN},
        timeout,
    )
    if status != 200 or not delete_payload.get("deleted"):
        raise SmokeError(f"asset delete failed HTTP {status}: {delete_payload}")
    status, _downloaded, _headers = raw_bytes_request("GET", download_url, None, {}, timeout)
    if status != 404:
        raise SmokeError(f"deleted asset remained downloadable: HTTP {status}")
    out = data_dir / f"smoke-assets-{timestamp}.json"
    save_json(out, {"uri": uri, "list_count": len(assets), "deleted": True, "signed_batch": signed})
    print(f"[smoke] assets saved {out}")


def assert_mcp_success(status: int, payload: dict, label: str) -> None:
    if status != 200:
        raise SmokeError(f"{label} returned HTTP {status}: {payload}")
    if payload.get("error"):
        raise SmokeError(f"{label} returned JSON-RPC error: {payload['error']}")


def run_yolo(image: Path, data_dir: Path, timestamp: str, timeout: float) -> None:
    if not image.exists():
        raise SmokeError(f"YOLO image does not exist: {image}")
    create = create_task(
        {
            "task_kind": "object.detect",
            "model": "yolo11n.onnx",
            "files": [{"name": image.name, "mime": "image/jpeg", "role": "image", "required": True}],
            "params": {},
        },
        f"smoke-yolo-create-{timestamp}",
        timeout,
    )
    upload = first_upload(create, "image")
    payload = upload_file(upload["upload_url"] + "&with_start_task=true", image, "image/jpeg", timeout)
    if payload.get("uri", "").startswith("assets://"):
        payload = start_task(create["task_id"], f"smoke-yolo-start-{timestamp}", timeout)
    out = data_dir / f"smoke-yolo-{timestamp}.json"
    save_json(out, payload)
    if payload.get("state") != "succeeded":
        raise SmokeError(f"YOLO generic task did not succeed: {payload}")
    print(f"[smoke] yolo saved {out}")


def run_qwen(audio: Path, data_dir: Path, timestamp: str, timeout: float) -> None:
    if not audio.exists():
        raise SmokeError(f"Qwen ASR audio does not exist: {audio}")
    out = data_dir / f"smoke-qwen-asr-{timestamp}.json"
    create = create_task(
        {
            "task_kind": "asr.transcribe",
            "model": "qwen3-asr-0.6b-onnx",
            "files": [{"name": audio.name, "mime": "audio/wav", "role": "audio", "required": True}],
            "params": {},
        },
        f"smoke-qwen-create-{timestamp}",
        timeout,
    )
    upload_file(first_upload(create, "audio")["upload_url"], audio, "audio/wav", timeout)
    payload = start_task(create["task_id"], f"smoke-qwen-start-{timestamp}", timeout)
    save_json(out, payload)
    if payload.get("state") != "succeeded":
        raise SmokeError(f"Qwen generic task did not succeed: {payload}")
    print(f"[smoke] qwen generic MCP saved {out}")


def post_openai_transcription_multipart(audio: Path, timeout: float) -> tuple[int, dict]:
    boundary = f"----lcoal-smoke-{int(time.time() * 1000)}"
    body = encode_multipart(
        boundary,
        fields={"model": "qwen3-asr-0.6b-onnx"},
        files={"file": (audio.name, audio.read_bytes(), "audio/wav")},
    )
    headers = {
        "Accept": "application/json",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    return raw_request("POST", f"{CONTROLLER_URL}/v1/audio/transcriptions", body, headers, timeout)


def inspect_indextts_artifacts(model_dir: Path) -> dict:
    env_root = os.environ.get("LCOAL_INDEXTTS_MODEL_DIR")
    root = Path(env_root).resolve() if env_root else (model_dir / INDEXTTS_MODEL_ID).resolve()
    missing = [str(root / name) for name in INDEXTTS_REQUIRED if not (root / name).exists()]
    return {
        "ready": not missing,
        "root": str(root),
        "source": "LCOAL_INDEXTTS_MODEL_DIR" if env_root else "--model-dir/default",
        "required": INDEXTTS_REQUIRED,
        "missing": missing,
    }


def enable_indextts(timeout: float) -> None:
    status, payload = mcp_admin("enable_model", {"id": INDEXTTS_MODEL_ID}, "smoke-enable-indextts", timeout)
    assert_mcp_success(status, payload, "IndexTTS enable_model")


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
    create = create_task(
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
    return start_task(create["task_id"], f"smoke-indextts-start-{timestamp}", timeout)


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
    create = create_task(
        {
            "task_kind": "asr.transcribe",
            "model": "qwen3-asr-0.6b-onnx",
            "files": [{"name": audio.name, "mime": "audio/wav", "role": "audio", "required": True}],
            "params": {},
        },
        f"smoke-indextts-asr-qwen-create-{timestamp}",
        timeout,
    )
    upload_file(first_upload(create, "audio")["upload_url"], audio, "audio/wav", timeout)
    return start_task(create["task_id"], f"smoke-indextts-asr-qwen-start-{timestamp}", timeout)


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


def create_task(params: dict, request_id: str, timeout: float) -> dict:
    status, payload = mcp_infer("create_task", params, request_id, timeout)
    assert_mcp_success(status, payload, "MCP create_task")
    result = payload.get("result")
    if not isinstance(result, dict) or not result.get("task_id"):
        raise SmokeError(f"create_task returned malformed result: {payload}")
    return result


def start_task(task_id: str, request_id: str, timeout: float) -> dict:
    status, payload = mcp_infer("start_task", {"task_id": task_id, "wait": True, "timeout_sec": int(timeout)}, request_id, timeout)
    assert_mcp_success(status, payload, "MCP start_task")
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
