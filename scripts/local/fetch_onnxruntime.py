"""Fetches the pinned ONNX Runtime the services load at run time.

The worker loads ONNX Runtime dynamically (ort `load-dynamic`): the library
must sit beside the binaries or be named by ORT_DYLIB_PATH. This downloads
the pinned 1.31.0 build for this platform, checks its SHA-256 and copies the
libraries into each destination (default: target/release and target/debug).

- The official GitHub release archives (the Docker images use the same Linux
  ones). The CUDA builds are the CUDA 12 ones, built with CUDA 12.8: on
  Windows put a CUDA 12.8+ runtime (cudart, cuBLAS) and cuDNN 9 on PATH. (The
  onnxruntime-gpu 1.31 wheel on PyPI is a CUDA 13 build.)

    python -m scripts.local.fetch_onnxruntime [--cpu] [--dest DIR ...] [--cache DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import platform
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .paths import repo_root

ORT_VERSION = "1.31.0"


@dataclass(frozen=True)
class Package:
    url: str
    sha256: str
    # Member path suffixes to extract (matched against archive names).
    members: tuple[str, ...]


GITHUB = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}"
PACKAGES = {
    ("Windows", "cuda"): Package(
        f"{GITHUB}/onnxruntime-win-x64-gpu_cuda12-{ORT_VERSION}.zip",
        "fa79097fa478961cbd31c297671ac169eb0d3f1dfdc9e8481e091850f8c4810b",
        (
            "/lib/onnxruntime.dll",
            "/lib/onnxruntime_providers_shared.dll",
            "/lib/onnxruntime_providers_cuda.dll",
        ),
    ),
    ("Windows", "cpu"): Package(
        f"{GITHUB}/onnxruntime-win-x64-{ORT_VERSION}.zip",
        "3958d8a44984160692c1b4ffd0de3cb48f8c39f9d34bb31e433db06e30b82195",
        ("/lib/onnxruntime.dll", "/lib/onnxruntime_providers_shared.dll"),
    ),
    ("Linux", "cuda"): Package(
        f"{GITHUB}/onnxruntime-linux-x64-gpu_cuda12-{ORT_VERSION}.tgz",
        "62d3af5269af2143b5d37494337573f1ffb8885f3ad7477ae0cb1cc79c8dedfd",
        (
            f"/lib/libonnxruntime.so.{ORT_VERSION}",
            "/lib/libonnxruntime_providers_shared.so",
            "/lib/libonnxruntime_providers_cuda.so",
        ),
    ),
    ("Linux", "cpu"): Package(
        f"{GITHUB}/onnxruntime-linux-x64-{ORT_VERSION}.tgz",
        "cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57",
        (f"/lib/libonnxruntime.so.{ORT_VERSION}",),
    ),
}


def download(package: Package, cache: Path) -> bytes:
    path = cache / package.url.rsplit("/", 1)[1]
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == package.sha256:
        return path.read_bytes()
    cache.mkdir(parents=True, exist_ok=True)
    url = package.url
    data = b""
    attempts = 6
    for attempt in range(1, attempts + 1):
        print(f"[ort] downloading {url}")
        try:
            with urllib.request.urlopen(url, timeout=600) as response:
                data = response.read()
            break
        except OSError as exc:
            if attempt == attempts:
                raise SystemExit(f"[ort] download failed: {exc}") from exc
            print(f"[ort] retrying in {5 * attempt} s after: {exc}")
            time.sleep(5 * attempt)
    digest = hashlib.sha256(data).hexdigest()
    if digest != package.sha256:
        raise SystemExit(f"[ort] SHA-256 mismatch for {url}: {digest} != {package.sha256}")
    path.write_bytes(data)
    return data


def extract(package: Package, data: bytes) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    if package.url.endswith((".zip", ".whl")):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in archive.namelist():
                if any(name.endswith(member) for member in package.members):
                    files[Path(name).name] = archive.read(name)
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile() and any(member.name.endswith(m) for m in package.members):
                    files[Path(member.name).name] = archive.extractfile(member).read()
    missing = [m for m in package.members if Path(m).name not in files]
    if missing:
        raise SystemExit(f"[ort] {package.url} lacks {missing}")
    return files


def install(files: dict[str, bytes], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        target = dest / name
        # Never write through a link (e.g. one into a download cache).
        if target.is_symlink() or target.exists():
            target.unlink()
        target.write_bytes(data)
        print(f"[ort] {target}")
    if f"libonnxruntime.so.{ORT_VERSION}" in files:
        # The loader asks for libonnxruntime.so.
        for alias in ("libonnxruntime.so", "libonnxruntime.so.1"):
            link = dest / alias
            if link.is_symlink() or link.exists():
                link.unlink()
            shutil.copyfile(dest / f"libonnxruntime.so.{ORT_VERSION}", link)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu", action="store_true", help="the CPU-only build")
    parser.add_argument(
        "--dest",
        action="append",
        type=Path,
        help="directory to copy the libraries into (repeatable; default target/release and target/debug)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="download cache (default target/ort-cache); an archive already there is used if its digest matches",
    )
    args = parser.parse_args(argv)
    system = platform.system()
    key = (system, "cpu" if args.cpu else "cuda")
    if key not in PACKAGES or platform.machine().lower() not in ("amd64", "x86_64"):
        print(f"[ort] no pinned ONNX Runtime for {system} {platform.machine()}", file=sys.stderr)
        return 1
    root = repo_root()
    package = PACKAGES[key]
    files = extract(package, download(package, args.cache or root / "target" / "ort-cache"))
    for dest in args.dest or [root / "target" / "release", root / "target" / "debug"]:
        install(files, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
