"""Fetches the pinned ONNX Runtime the services load at run time.

The worker loads ONNX Runtime dynamically (ort `load-dynamic`): the library
must sit beside the binaries or be named by ORT_DYLIB_PATH. This downloads
the pinned 1.30.0 build for this platform, checks its SHA-256 and copies the
libraries into each destination (default: target/release and target/debug).

- Windows, CUDA: the DLLs of the onnxruntime-gpu wheel on PyPI (a zip; no
  Python install is used). They run with CUDA 12.4+ and cuDNN 9 on PATH; the
  official CUDA 12 zip on GitHub needs a newer CUDA 12 runtime.
- Windows CPU and Linux: the official GitHub release archives (the Docker
  images use the same Linux archives).

    python -m scripts.local.fetch_onnxruntime [--cpu] [--dest DIR ...]
        [--pypi-mirror https://pypi.tuna.tsinghua.edu.cn]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import platform
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .paths import repo_root

ORT_VERSION = "1.30.0"


@dataclass(frozen=True)
class Package:
    url: str
    sha256: str
    # Member path suffixes to extract (matched against archive names).
    members: tuple[str, ...]


PYPI_FILES = "https://files.pythonhosted.org"
GITHUB = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}"
PACKAGES = {
    ("Windows", "cuda"): Package(
        f"{PYPI_FILES}/packages/19/ef/e15db03ea1fce9bc60349add2604df8c39abac10d28d4eef27b621845139/"
        f"onnxruntime_gpu-{ORT_VERSION}-cp312-cp312-win_amd64.whl",
        "882da66f936e1f560b155118c5953dc45268cd92893abb8d31a940f4d1effb36",
        (
            "onnxruntime/capi/onnxruntime.dll",
            "onnxruntime/capi/onnxruntime_providers_shared.dll",
            "onnxruntime/capi/onnxruntime_providers_cuda.dll",
        ),
    ),
    ("Windows", "cpu"): Package(
        f"{GITHUB}/onnxruntime-win-x64-{ORT_VERSION}.zip",
        "c6ba983baf5681af108599675d2a89c2d145512d02de28aed0bff177cd0ba949",
        ("/lib/onnxruntime.dll", "/lib/onnxruntime_providers_shared.dll"),
    ),
    ("Linux", "cuda"): Package(
        f"{GITHUB}/onnxruntime-linux-x64-gpu_cuda12-{ORT_VERSION}.tgz",
        "f9886932ee7bb0b4d3fcab736a392d4ff5efaa0672b47f19f0cec03437cf64f1",
        (
            f"/lib/libonnxruntime.so.{ORT_VERSION}",
            "/lib/libonnxruntime_providers_shared.so",
            "/lib/libonnxruntime_providers_cuda.so",
        ),
    ),
    ("Linux", "cpu"): Package(
        f"{GITHUB}/onnxruntime-linux-x64-{ORT_VERSION}.tgz",
        "a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd",
        (f"/lib/libonnxruntime.so.{ORT_VERSION}",),
    ),
}


def download(package: Package, cache: Path, pypi_mirror: str | None = None) -> bytes:
    path = cache / package.url.rsplit("/", 1)[1]
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == package.sha256:
        return path.read_bytes()
    cache.mkdir(parents=True, exist_ok=True)
    url = package.url
    if pypi_mirror and url.startswith(PYPI_FILES):
        url = pypi_mirror.rstrip("/") + url[len(PYPI_FILES) :]
    data = b""
    for attempt in range(1, 4):
        print(f"[ort] downloading {url}")
        try:
            with urllib.request.urlopen(url, timeout=600) as response:
                data = response.read()
            break
        except OSError as exc:
            if attempt == 3:
                raise SystemExit(f"[ort] download failed: {exc}") from exc
            print(f"[ort] retrying after: {exc}")
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
    parser.add_argument("--cache", type=Path, default=None, help="download cache (default target/ort-cache)")
    parser.add_argument(
        "--pypi-mirror",
        default=None,
        help="a PyPI mirror serving /packages/... like files.pythonhosted.org (the digest is still checked)",
    )
    args = parser.parse_args(argv)
    system = platform.system()
    key = (system, "cpu" if args.cpu else "cuda")
    if key not in PACKAGES or platform.machine().lower() not in ("amd64", "x86_64"):
        print(f"[ort] no pinned ONNX Runtime for {system} {platform.machine()}", file=sys.stderr)
        return 1
    root = repo_root()
    package = PACKAGES[key]
    files = extract(package, download(package, args.cache or root / "target" / "ort-cache", args.pypi_mirror))
    for dest in args.dest or [root / "target" / "release", root / "target" / "debug"]:
        install(files, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
