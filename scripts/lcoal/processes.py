"""Process and port helpers for Python-only service smoke tests."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

from .errors import SmokeError


class ManagedProcess:
    def __init__(self, name: str, proc: subprocess.Popen, stdout_file, stderr_file, stdout_path: Path, stderr_path: Path):
        self.name = name
        self.proc = proc
        self.stdout_file = stdout_file
        self.stderr_file = stderr_file
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path

    @property
    def pid(self) -> int:
        return int(self.proc.pid)

    def close_logs(self) -> None:
        for handle in (self.stdout_file, self.stderr_file):
            try:
                handle.close()
            except Exception:
                pass


def run_build(root: Path, *, release: bool = False) -> None:
    profile_args = ["--release"] if release else []
    command = ["cargo", "build", *profile_args, "--bins"]
    print(f"[smoke] building binaries with {' '.join(command)}")
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise SmokeError(f"{' '.join(command)} timed out after 600s") from exc
    if completed.returncode != 0:
        raise SmokeError(f"{' '.join(command)} failed with exit code {completed.returncode}")


def locate_bin(root: Path, name: str, override: Path | None, *, release: bool = False) -> Path:
    if override is not None:
        path = override.resolve()
        if not path.exists():
            raise SmokeError(f"{name} binary override does not exist: {path}")
        return path
    profile = "release" if release else "debug"
    candidates = [root / "target" / profile / f"{name}.exe", root / "target" / profile / name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(path) for path in candidates)
    raise SmokeError(f"could not locate {name} binary; tried {joined}. Run without --skip-build, use the matching --release flag, or pass --{name}-bin")


def start_service(name: str, cmd: list[str], root: Path, data_dir: Path, timestamp: str, env: dict[str, str]) -> ManagedProcess:
    stdout_path = data_dir / f"{name}-{timestamp}.stdout.log"
    stderr_path = data_dir / f"{name}-{timestamp}.stderr.log"
    stdout_file = stdout_path.open("ab", buffering=0)
    stderr_file = stderr_path.open("ab", buffering=0)
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdout=stdout_file,
            stderr=stderr_file,
            stdin=subprocess.DEVNULL,
            env=env,
            creationflags=creationflags,
        )
    except Exception:
        stdout_file.close()
        stderr_file.close()
        raise
    managed = ManagedProcess(name, proc, stdout_file, stderr_file, stdout_path, stderr_path)
    print(f"[smoke] started {name} pid={managed.pid}")
    print(f"[smoke] {name} stdout={stdout_path}")
    print(f"[smoke] {name} stderr={stderr_path}")
    return managed


def cleanup_processes(processes: list[ManagedProcess]) -> None:
    for managed in reversed(processes):
        if managed.proc.poll() is None:
            print(f"[smoke] stopping {managed.name} pid={managed.pid}")
            try:
                managed.proc.terminate()
            except Exception as exc:
                print(f"[smoke] terminate failed for {managed.name}: {exc}", file=sys.stderr)
    deadline = time.monotonic() + 5.0
    for managed in reversed(processes):
        remaining = max(0.1, deadline - time.monotonic())
        try:
            managed.proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            if managed.proc.poll() is None:
                print(f"[smoke] killing {managed.name} pid={managed.pid}")
                managed.proc.kill()
                try:
                    managed.proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    print(f"[smoke] WARNING: {managed.name} pid={managed.pid} did not exit", file=sys.stderr)
        finally:
            managed.close_logs()


def port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_ports_closed(ports: tuple[int, ...], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        open_ports = [port for port in ports if port_listening(port)]
        if not open_ports:
            print(f"[smoke] confirmed ports closed: {', '.join(str(port) for port in ports)}")
            return
        time.sleep(0.5)
    open_ports = [port for port in ports if port_listening(port)]
    if open_ports:
        raise SmokeError(
            "ports still listening after cleanup "
            f"(not killing unknown processes): {', '.join(str(port) for port in open_ports)}"
        )
