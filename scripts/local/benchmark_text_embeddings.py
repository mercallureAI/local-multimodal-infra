"""End-to-end OpenAI embeddings batch/length benchmark.

The benchmark starts a release controller/worker pair using the same managed
process helpers as the smoke suite, measures HTTP end-to-end latency, saves the
first vector for CPU/GPU parity checks, and always cleans up its services.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

from .errors import SmokeError
from .http_client import checked_json_request
from .paths import repo_root, resolve_cli_path
from .processes import (
    ManagedProcess,
    cleanup_processes,
    locate_bin,
    port_listening,
    start_service,
    wait_ports_closed,
)


CONTROLLER_URL = "http://127.0.0.1:17890"
WORKER_URL = "http://127.0.0.1:17891"
PORTS = (17890, 17891, 17892)
REGISTRATION_TOKEN = "local-text-benchmark-registration-token"
ADMIN_TOKEN = "local-text-benchmark-admin-token"
INFER_TOKEN = "local-text-benchmark-infer-token"
INFER_AUTH_HEADERS = {"Authorization": f"Bearer {INFER_TOKEN}"}
MODEL_ID = "multilingual-e5-small-onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("cpu", "gpu"))
    parser.add_argument("--controller-bin", type=Path)
    parser.add_argument("--worker-bin", type=Path)
    parser.add_argument("--workdir", type=Path, default=Path("workdir"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batches", default="1,8,32,128")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--ready-timeout", type=float, default=60.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    return parser.parse_args()


def wait_health(name: str, url: str, process: ManagedProcess, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process.proc.poll() is not None:
            raise SmokeError(f"{name} exited early with code {process.proc.returncode}")
        try:
            payload = checked_json_request("GET", url, None, 2.0)
            if payload.get("status") == "ok":
                return
            last_error = f"unexpected health payload: {payload}"
        except Exception as error:  # bounded readiness polling
            last_error = str(error)
        time.sleep(0.25)
    raise SmokeError(f"{name} did not become ready: {last_error}")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def text_batch(length: str, batch: int) -> list[str]:
    if length == "short":
        base = "本地 ONNX 检索服务需要准确且快速地生成多语言向量。"
    else:
        paragraph = (
            "Multilingual retrieval systems map queries and passages into a shared vector space. "
            "多语言检索系统需要处理中文、English、日本語和代码片段，并保持语义一致性。"
        )
        base = paragraph * 80
    return [f"{base} sample-{index}" for index in range(batch)]


def run_case(
    length: str,
    batch: int,
    warmups: int,
    iterations: int,
    timeout: float,
) -> dict:
    request = {
        "model": MODEL_ID,
        "input": text_batch(length, batch),
        "input_type": "passage",
        "encoding_format": "float",
    }
    payload: dict | None = None
    for _ in range(warmups):
        payload = checked_json_request(
            "POST", f"{CONTROLLER_URL}/v1/embeddings", request, timeout, INFER_AUTH_HEADERS
        )
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        payload = checked_json_request(
            "POST", f"{CONTROLLER_URL}/v1/embeddings", request, timeout, INFER_AUTH_HEADERS
        )
        samples.append((time.perf_counter() - started) * 1000.0)

    assert payload is not None
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != batch:
        raise SmokeError(f"embedding response batch mismatch for {length}/{batch}")
    vector = data[0].get("embedding") if isinstance(data[0], dict) else None
    if not isinstance(vector, list) or len(vector) != 384:
        raise SmokeError(f"embedding dimension mismatch for {length}/{batch}")
    norm = sum(float(value) ** 2 for value in vector) ** 0.5
    if abs(norm - 1.0) > 1.0e-3:
        raise SmokeError(f"embedding norm mismatch for {length}/{batch}: {norm}")
    usage = payload.get("usage", {})
    return {
        "length": length,
        "batch": batch,
        "prompt_tokens": usage.get("prompt_tokens"),
        "tokens_per_item": (
            usage.get("prompt_tokens") / batch
            if isinstance(usage.get("prompt_tokens"), int)
            else None
        ),
        "avg_ms": round(statistics.fmean(samples), 3),
        "p50_ms": round(statistics.median(samples), 3),
        "p95_ms": round(percentile(samples, 0.95), 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
        "throughput_items_per_sec": round(batch * 1000.0 / statistics.fmean(samples), 3),
        "vector_norm": round(norm, 9),
        "first_vector": vector,
        "samples_ms": [round(sample, 3) for sample in samples],
    }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = sum(float(value) ** 2 for value in left) ** 0.5
    right_norm = sum(float(value) ** 2 for value in right) ** 0.5
    return dot / (left_norm * right_norm)


def write_comparison(data_dir: Path) -> Path | None:
    cpu_path = data_dir / "text-embedding-matrix-cpu.json"
    gpu_path = data_dir / "text-embedding-matrix-gpu.json"
    if not cpu_path.is_file() or not gpu_path.is_file():
        return None
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
    gpu_cases = {
        (case["length"], case["batch"]): case for case in gpu["cases"]
    }
    cases = []
    for cpu_case in cpu["cases"]:
        key = (cpu_case["length"], cpu_case["batch"])
        gpu_case = gpu_cases[key]
        cases.append(
            {
                "length": key[0],
                "batch": key[1],
                "tokens_per_item": cpu_case["tokens_per_item"],
                "cpu_avg_ms": cpu_case["avg_ms"],
                "gpu_avg_ms": gpu_case["avg_ms"],
                "avg_speedup": round(cpu_case["avg_ms"] / gpu_case["avg_ms"], 4),
                "cpu_p50_ms": cpu_case["p50_ms"],
                "gpu_p50_ms": gpu_case["p50_ms"],
                "p50_speedup": round(cpu_case["p50_ms"] / gpu_case["p50_ms"], 4),
                "cpu_items_per_sec": cpu_case["throughput_items_per_sec"],
                "gpu_items_per_sec": gpu_case["throughput_items_per_sec"],
                "embedding_cosine": round(
                    cosine_similarity(
                        cpu_case["first_vector"], gpu_case["first_vector"]
                    ),
                    9,
                ),
            }
        )
    comparison = {
        "profile": "release",
        "timing": "OpenAI HTTP end-to-end milliseconds",
        "warmups": cpu["warmups"],
        "iterations": cpu["iterations"],
        "cpu_report": str(cpu_path),
        "gpu_report": str(gpu_path),
        "cases": cases,
    }
    output = data_dir / "text-embedding-matrix-comparison.json"
    output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> int:
    args = parse_args()
    root = repo_root()
    workdir = resolve_cli_path(args.workdir, root)
    model_dir = (
        resolve_cli_path(args.model_dir, root)
        if args.model_dir
        else (workdir / "models").resolve()
    )
    data_dir = workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output = (
        resolve_cli_path(args.output, root)
        if args.output
        else data_dir / f"text-embedding-matrix-{args.mode}.json"
    )
    batches = [int(value) for value in args.batches.split(",")]
    if any(batch <= 0 for batch in batches):
        raise SmokeError("all batch sizes must be positive")
    if args.warmups < 1 or args.iterations < 1:
        raise SmokeError("warmups and iterations must be positive")
    occupied = [port for port in PORTS if port_listening(port)]
    if occupied:
        raise SmokeError(f"benchmark ports are already occupied: {occupied}")

    controller_bin = locate_bin(root, "controller", args.controller_bin, release=True)
    worker_bin = locate_bin(root, "worker", args.worker_bin, release=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    env = os.environ.copy()
    env["LOCAL_DATA_DIR"] = str(data_dir)
    env["LOCAL_WORKER_REGISTRATION_TOKEN"] = REGISTRATION_TOKEN
    env["LOCAL_ADMIN_TOKEN"] = ADMIN_TOKEN
    env["LOCAL_MCP_INFER_TOKENS"] = INFER_TOKEN
    env["RUST_LOG"] = "local_runtime=info,local_adapter_e5_embedding=info,ort=warn"
    launched: list[ManagedProcess] = []
    started = time.time()
    try:
        controller = start_service(
            f"benchmark-{args.mode}-controller",
            [
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
            ],
            root,
            data_dir,
            timestamp,
            env,
        )
        launched.append(controller)
        wait_health("controller", f"{CONTROLLER_URL}/health", controller, args.ready_timeout)
        worker = start_service(
            f"benchmark-{args.mode}-worker",
            [
                str(worker_bin),
                "configs/worker.yaml",
                "--workdir",
                str(workdir),
                "--model-dir",
                str(model_dir),
                "--registration-token",
                REGISTRATION_TOKEN,
            ],
            root,
            data_dir,
            timestamp,
            env,
        )
        launched.append(worker)
        wait_health("worker", f"{WORKER_URL}/health", worker, args.ready_timeout)
        time.sleep(1.0)

        cases = []
        for length in ("short", "long"):
            for batch in batches:
                print(f"[benchmark] mode={args.mode} length={length} batch={batch}")
                cases.append(
                    run_case(
                        length,
                        batch,
                        args.warmups,
                        args.iterations,
                        args.request_timeout,
                    )
                )
        report = {
            "mode": args.mode,
            "profile": "release",
            "model": MODEL_ID,
            "timing": "OpenAI HTTP end-to-end milliseconds",
            "warmups": args.warmups,
            "iterations": args.iterations,
            "started_at_unix": started,
            "cases": cases,
            "worker_stdout": str(worker.stdout_path),
            "worker_stderr": str(worker.stderr_path),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[benchmark] wrote {output}")
        comparison = write_comparison(data_dir)
        if comparison is not None:
            print(f"[benchmark] wrote {comparison}")
        return 0
    finally:
        cleanup_processes(launched)
        wait_ports_closed(PORTS, 8.0)


if __name__ == "__main__":
    raise SystemExit(main())
