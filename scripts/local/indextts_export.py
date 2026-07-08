"""Export/package IndexTTS ONNX artifacts for LOCAL.

This module keeps the public ``scripts/indextts_export.py`` entrypoint thin while
providing two preparation paths:

* package an already-exported IndexTTS_A.onnx ... IndexTTS_F.onnx layout; or
* export the official local IndexTTS-1.5 PyTorch checkpoint into the A-F graph
  contract consumed by ``crates/adapter-index-tts``.

The A-F split is project-local code. It follows the adapter contract and the
official IndexTTS model APIs, but does not vendor or copy third-party exporter
implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


MODEL_FILES = [f"IndexTTS_{stage}.onnx" for stage in "ABCDEF"]
SUPPORTED_PRECISION = "cpu-fp32"
SAMPLE_RATE = 24_000
START_TOKEN = 8192
STOP_TOKEN = 8193
MAX_GENERATE_LENGTH = 800

RAW_EXPORT_REQUIRED_MODULES = [
    ("torch", "torch", "pip install torch --index-url https://download.pytorch.org/whl/cpu"),
    ("torchaudio", "torchaudio", "pip install torchaudio --index-url https://download.pytorch.org/whl/cpu"),
    ("onnx", "onnx", "pip install onnx"),
    ("onnxruntime", "onnxruntime", "pip install onnxruntime"),
    ("sentencepiece", "sentencepiece", "pip install sentencepiece"),
    ("omegaconf", "omegaconf", "pip install omegaconf"),
    ("soundfile", "soundfile", "pip install soundfile"),
    ("pydub", "pydub", "pip install pydub"),
    ("transformers", "transformers", "pip install transformers"),
    ("tqdm", "tqdm", "pip install tqdm"),
    # Transitive imports used by the official local IndexTTS code during model construction.
    ("yaml", "PyYAML", "pip install PyYAML"),
    ("einops", "einops", "pip install einops"),
    ("packaging", "packaging", "pip install packaging"),
    ("scipy", "scipy", "pip install scipy"),
    ("matplotlib", "matplotlib", "pip install matplotlib"),
]


@dataclass(frozen=True)
class PackageResult:
    ready: bool


class UnsupportedExport(RuntimeError):
    """Raised when this wrapper cannot produce a ready A-F layout."""


class DependencyError(RuntimeError):
    """Raised when raw export dependencies are missing."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LOCAL IndexTTS A-F ONNX artifacts")
    parser.add_argument(
        "--index-tts-project",
        required=True,
        type=Path,
        help="Path to an index-tts project checkout; imported locally and never modified",
    )
    parser.add_argument(
        "--source-model-dir",
        required=True,
        type=Path,
        help="Directory containing local IndexTTS weights/tokenizer or pre-exported A-F ONNX files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output artifact directory, e.g. workdir/models/indextts-1.5-onnx",
    )
    parser.add_argument(
        "--precision",
        choices=[SUPPORTED_PRECISION],
        default=SUPPORTED_PRECISION,
        help="IndexTTS artifacts are FP32-only; cpu-q4 and gpu-fp16 export/runtime paths are no longer supported",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--mode",
        choices=["auto", "package", "raw-export", "local-export"],
        default="auto",
        help=(
            "auto packages existing A-F files first, then tries raw PyTorch export; "
            "package never imports the project; raw-export only traces local PyTorch weights; "
            "local-export only invokes explicit user-provided --local-export-entry values"
        ),
    )
    parser.add_argument(
        "--local-export-entry",
        action="append",
        default=[],
        metavar="MODULE:FUNCTION",
        help=(
            "Optional exporter callable inside --index-tts-project. It may accept keyword args "
            "index_tts_project, source_model_dir, output_dir, precision (always cpu-fp32), device, and stages. Can be repeated."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files in the output directory")
    parser.add_argument("--check-deps", action="store_true", help="Check raw-export Python dependencies and exit")
    parser.add_argument("--opset", type=int, default=17, help="torch.onnx.export opset version; default: 17")
    parser.add_argument(
        "--audio-length",
        type=int,
        default=SAMPLE_RATE,
        help="Dummy reference audio sample count used while tracing IndexTTS_A; default: 24000",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=16,
        help="Dummy text/decode sequence length used for dynamic ONNX tracing; default: 16",
    )
    parser.add_argument(
        "--external-data-threshold-mb",
        type=int,
        default=512,
        help="Requested external-data threshold note for large ONNX exports; PyTorch support is version-dependent",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    notes: list[str] = []
    validate_args(args)

    if args.check_deps:
        return dependency_check_command(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    notes.append(
        "A-F graph contract: A(reference audio conds), B(text ids), C(GPT token embedding), "
        "D(concat embeddings), E autoregressive KV loop, F BigVGAN vocoder."
    )

    status = "unsupported"
    try:
        ready = False
        if args.mode in {"auto", "package"}:
            package_result = package_existing_onnx(args, notes)
            ready = package_result.ready
        if not ready and args.mode in {"auto", "raw-export"}:
            ready = raw_export(args, notes)
        if not ready and args.mode == "local-export":
            ready = invoke_local_exporter(args, notes)
        if not ready:
            raise UnsupportedExport(
                "No complete IndexTTS_A.onnx ... IndexTTS_F.onnx set was produced. "
                "Use --mode raw-export for official local PyTorch weights, provide pre-exported A-F ONNX files, "
                "or run --mode local-export with an explicit --local-export-entry MODULE:FUNCTION."
            )
        apply_precision(args, notes)
        status = "ready"
    except (UnsupportedExport, DependencyError) as exc:
        notes.append(str(exc))
        status = "unsupported"

    copy_bpe(args, notes)
    manifest = build_manifest(args, status, notes)
    write_yaml(args.output_dir / "manifest.yaml", manifest)
    write_json(args.output_dir / "manifest.json", manifest)
    validate_layout(args.output_dir, require_onnx=(status == "ready"))

    if status == "unsupported":
        print("IndexTTS export is unsupported with the provided inputs; manifest written with details.", file=sys.stderr)
        return 2
    print(f"IndexTTS artifacts prepared under {args.output_dir}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if not args.index_tts_project.is_dir():
        raise SystemExit(f"--index-tts-project is not a directory: {args.index_tts_project}")
    if not args.source_model_dir.is_dir():
        raise SystemExit(f"--source-model-dir is not a directory: {args.source_model_dir}")
    if args.precision != SUPPORTED_PRECISION:
        raise SystemExit("IndexTTS currently supports --precision cpu-fp32 only")
    if args.device == "cuda":
        print("warning: CUDA device requested; IndexTTS packaging remains FP32-only and will not create or select fp16 artifacts", file=sys.stderr)
    if args.opset < 17:
        raise SystemExit("--opset must be 17 or newer for the IndexTTS split exporter")
    if args.audio_length <= 0:
        raise SystemExit("--audio-length must be positive")
    if args.max_seq_len <= 0:
        raise SystemExit("--max-seq-len must be positive")


def dependency_check_command(args: argparse.Namespace) -> int:
    report = dependency_report(args.index_tts_project)
    print(format_dependency_report(report))
    return 0 if not report["missing_required"] else 2


def dependency_report(index_tts_project: Path) -> dict[str, Any]:
    missing_required = []
    present_required = []
    for module, package, suggestion in RAW_EXPORT_REQUIRED_MODULES:
        if importlib.util.find_spec(module) is None:
            missing_required.append({"module": module, "package": package, "install": suggestion})
        else:
            present_required.append(module)

    text_norm_modules = text_normalization_modules()
    missing_text_norm = []
    present_text_norm = []
    for module, package, suggestion in text_norm_modules:
        if importlib.util.find_spec(module) is None:
            missing_text_norm.append({"module": module, "package": package, "install": suggestion})
        else:
            present_text_norm.append(module)

    project_importable = (index_tts_project / "indextts" / "infer.py").exists()
    return {
        "present_required": present_required,
        "missing_required": missing_required,
        "present_text_normalization": present_text_norm,
        "missing_text_normalization": missing_text_norm,
        "project_importable": project_importable,
    }


def text_normalization_modules() -> list[tuple[str, str, str]]:
    if platform.system() == "Darwin":
        return [("wetext", "WeTextProcessing", "pip install WeTextProcessing")]
    return [("tn", "WeTextProcessing", "pip install WeTextProcessing")]


def format_dependency_report(report: dict[str, Any]) -> str:
    lines = ["IndexTTS raw-export dependency check:"]
    if report["project_importable"]:
        lines.append("  official project: OK (indextts/infer.py found)")
    else:
        lines.append("  official project: MISSING indextts/infer.py under --index-tts-project")
    if report["present_required"]:
        lines.append("  present required modules: " + ", ".join(report["present_required"]))
    if report["missing_required"]:
        lines.append("  missing required modules:")
        for item in report["missing_required"]:
            lines.append(f"    - {item['module']} ({item['package']}): {item['install']}")
    else:
        lines.append("  required modules: OK")
    if report["missing_text_normalization"]:
        lines.append("  optional official text-normalization modules missing (not used by raw export; needed by official inference):")
        for item in report["missing_text_normalization"]:
            lines.append(f"    - {item['module']} ({item['package']}): {item['install']}")
    elif report["present_text_normalization"]:
        lines.append("  official text-normalization modules: " + ", ".join(report["present_text_normalization"]))
    return "\n".join(lines)


def require_raw_export_dependencies(args: argparse.Namespace) -> None:
    report = dependency_report(args.index_tts_project)
    problems = []
    if not report["project_importable"]:
        problems.append(f"official project is missing indextts/infer.py: {args.index_tts_project}")
    if report["missing_required"]:
        installs = "; ".join(f"{item['module']}: {item['install']}" for item in report["missing_required"])
        problems.append("missing Python modules required for raw export: " + installs)
    if problems:
        raise DependencyError("IndexTTS raw export dependency check failed. " + " | ".join(problems))


def package_existing_onnx(args: argparse.Namespace, notes: list[str]) -> PackageResult:
    candidates: list[Path] = [args.source_model_dir]
    found = next(
        (root for root in candidates if all((root / name).exists() for name in MODEL_FILES)),
        None,
    )
    if found is None:
        return PackageResult(False)
    source_root = found
    for filename in MODEL_FILES:
        copy_one(source_root / filename, args.output_dir / filename, args.overwrite)
    for data_file in source_root.glob("*.onnx.data"):
        copy_one(data_file, args.output_dir / data_file.name, args.overwrite)
    notes.append(f"packaged pre-exported FP32 ONNX files from {source_root}")
    return PackageResult(True)


def raw_export(args: argparse.Namespace, notes: list[str]) -> bool:
    require_raw_export_dependencies(args)
    try:
        exporter = RawIndexTtsExporter(args, notes)
        exporter.export_all()
    except Exception as exc:  # pragma: no cover - real model/export dependent
        raise UnsupportedExport(f"raw PyTorch export failed: {exc}") from exc
    complete = all((args.output_dir / name).exists() for name in MODEL_FILES)
    if complete:
        notes.append("raw PyTorch export produced IndexTTS_A.onnx ... IndexTTS_F.onnx")
    return complete


class RawIndexTtsExporter:
    def __init__(self, args: argparse.Namespace, notes: list[str]):
        self.args = args
        self.notes = notes
        self._bigvgan_resampler_pad_mode = "replicate"

    def export_all(self) -> None:
        self._check_targets()
        old_path = list(sys.path)
        sys.path.insert(0, str(self.args.index_tts_project.resolve()))
        try:
            import torch

            tts = self._load_official_index_tts()
            self._freeze(tts)
            self._patch_bigvgan_alias_free_resamplers(torch, tts.bigvgan, pad_mode="replicate")
            self._export_graphs(torch, tts)
        finally:
            sys.path[:] = old_path

    def _check_targets(self) -> None:
        existing = [name for name in MODEL_FILES if (self.args.output_dir / name).exists()]
        if existing and not self.args.overwrite:
            raise UnsupportedExport(
                "raw export target files already exist and --overwrite was not supplied: " + ", ".join(existing)
            )

    def _load_official_index_tts(self):
        from indextts.infer import IndexTTS
        from indextts.utils import front as front_module

        cfg_path = self.args.source_model_dir / "config.yaml"
        if not cfg_path.exists():
            raise UnsupportedExport(f"source config.yaml is missing: {cfg_path}")
        use_half = False
        original_load = front_module.TextNormalizer.load

        def export_noop_text_normalizer_load(self):  # noqa: ANN001 - monkeypatching official class
            # Avoid building official tagger_cache under the source checkout. Raw export does not tokenize text.
            self.zh_normalizer = None
            self.en_normalizer = None

        front_module.TextNormalizer.load = export_noop_text_normalizer_load
        try:
            signature = inspect.signature(IndexTTS.__init__)
            supported = signature.parameters
            init_kwargs: dict[str, Any] = {}
            if "cfg_path" in supported:
                init_kwargs["cfg_path"] = str(cfg_path)
            if "model_dir" in supported:
                init_kwargs["model_dir"] = str(self.args.source_model_dir)
            if "is_fp16" in supported:
                init_kwargs["is_fp16"] = use_half
            elif "use_fp16" in supported:
                init_kwargs["use_fp16"] = use_half
            if "device" in supported:
                init_kwargs["device"] = self.args.device
            if "use_cuda_kernel" in supported:
                init_kwargs["use_cuda_kernel"] = False
            tts = IndexTTS(**init_kwargs)
        finally:
            front_module.TextNormalizer.load = original_load
        self.notes.append(
            f"loaded official IndexTTS from {self.args.index_tts_project} with cfg_path={cfg_path}, "
            f"device={self.args.device}, fp16={use_half}, use_cuda_kernel=False where supported"
        )
        return tts

    def _freeze(self, tts: Any) -> None:
        for module_name in ["gpt", "bigvgan"]:
            module = getattr(tts, module_name)
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def _patch_bigvgan_alias_free_resamplers(self, torch: Any, bigvgan: Any, pad_mode: str) -> None:
        patched = 0
        for module in bigvgan.modules():
            if not is_alias_free_activation(module):
                continue
            channels = infer_activation_channels(module)
            if channels is None:
                continue
            module.upsample = StaticGroupedUpSample1d(torch, module.upsample, channels, pad_mode=pad_mode)
            module.downsample = StaticGroupedDownSample1d(torch, module.downsample, channels, pad_mode=pad_mode)
            patched += 1
        self._bigvgan_resampler_pad_mode = pad_mode
        if patched:
            self.notes.append(
                f"patched {patched} BigVGAN alias-free Activation1d resampler pairs in memory for ONNX export: "
                f"pre-expanded grouped convolution filters with static channel count and {pad_mode} padding; "
                "official source files were not modified"
            )

    def _set_bigvgan_resampler_pad_mode(self, bigvgan: Any, pad_mode: str) -> int:
        changed = 0
        for module in bigvgan.modules():
            for attr in ["upsample", "downsample"]:
                resampler = getattr(module, attr, None)
                if hasattr(resampler, "pad_mode"):
                    resampler.pad_mode = pad_mode
                    changed += 1
        self._bigvgan_resampler_pad_mode = pad_mode
        return changed

    def _export_graphs(self, torch: Any, tts: Any) -> None:
        gpt = tts.gpt
        bigvgan = tts.bigvgan
        device = torch.device(self.args.device)
        dtype = torch.float32
        hidden = int(gpt.model_dim)
        heads = int(gpt.heads)
        head_dim = hidden // heads
        text_len = min(max(2, self.args.max_seq_len), int(gpt.max_text_tokens))
        mel_len = min(max(2, self.args.max_seq_len), int(gpt.max_mel_tokens))
        cond_len = int(getattr(gpt, "cond_num", 32))
        conds_latent = torch.zeros((1, cond_len, hidden), dtype=dtype, device=device)
        text_hidden = torch.zeros((1, text_len + 2, hidden), dtype=dtype, device=device)
        gpt_hidden = torch.zeros((1, 1, hidden), dtype=dtype, device=device)

        self._export_a(torch, tts, dtype)
        self._export_b(torch, gpt, text_len)
        self._export_c(torch, gpt)
        self._export_d(torch, conds_latent, text_hidden, gpt_hidden)
        self._export_e(torch, gpt, heads, head_dim, hidden, mel_len, dtype)
        self._export_f(torch, bigvgan, hidden, mel_len, dtype)

    def _export_a(self, torch: Any, tts: Any, dtype: Any) -> None:
        import torchaudio

        mel_frontend = OnnxSafeMelSpectrogram(torch, torchaudio, self.args.device)
        self.notes.append(
            "IndexTTS_A uses a project-local ONNX-safe conv1d mel frontend mirroring official "
            "MelSpectrogramFeatures(sample_rate=24000,n_fft=1024,hop=256,win=1024,n_mels=100,power=1,center=True)."
        )
        module = IndexTtsA(torch, mel_frontend, tts.gpt, tts.bigvgan).to(self.args.device)
        audio = torch.zeros((1, 1, self.args.audio_length), dtype=torch.int16, device=self.args.device)
        save_names = [f"save_bigvgan_conds_{idx}" for idx in range(bigvgan_condition_count(tts.bigvgan))]
        output_names = [*save_names, "bigvgan_cond_layer_speaker_embedding", "conds_latent"]
        dynamic_axes = {"audio": {2: "num_audio_samples"}, "conds_latent": {1: "num_condition_latents"}}
        self._export_onnx(module, (audio,), "IndexTTS_A.onnx", ["audio"], output_names, dynamic_axes)

    def _export_b(self, torch: Any, gpt: Any, text_len: int) -> None:
        module = IndexTtsB(gpt).to(self.args.device)
        text_ids = torch.zeros((1, text_len), dtype=torch.int32, device=self.args.device)
        dynamic_axes = {"text_ids": {1: "text_tokens"}, "text_hidden_state": {1: "text_tokens_plus_special"}}
        self._export_onnx(module, (text_ids,), "IndexTTS_B.onnx", ["text_ids"], ["text_hidden_state"], dynamic_axes)

    def _export_c(self, torch: Any, gpt: Any) -> None:
        module = IndexTtsC(gpt).to(self.args.device)
        gpt_ids = torch.tensor([[START_TOKEN]], dtype=torch.int32, device=self.args.device)
        gen_len = torch.zeros((1,), dtype=torch.int64, device=self.args.device)
        dynamic_axes = {"gpt_hidden_state": {1: "one_token"}}
        self._export_onnx(
            module,
            (gpt_ids, gen_len),
            "IndexTTS_C.onnx",
            ["gpt_ids", "gen_len"],
            ["gpt_hidden_state", "next_gen_len"],
            dynamic_axes,
        )

    def _export_d(self, torch: Any, conds_latent: Any, text_hidden: Any, gpt_hidden: Any) -> None:
        module = IndexTtsD().to(self.args.device)
        dynamic_axes = {
            "embed_x": {1: "cond_len"},
            "embed_y": {1: "text_len"},
            "embed_z": {1: "mel_len"},
            "concat_hidden_state": {1: "concat_len"},
        }
        self._export_onnx(
            module,
            (conds_latent, text_hidden, gpt_hidden),
            "IndexTTS_D.onnx",
            ["embed_x", "embed_y", "embed_z"],
            ["concat_hidden_state", "concat_len"],
            dynamic_axes,
        )

    def _export_e(self, torch: Any, gpt: Any, heads: int, head_dim: int, hidden: int, mel_len: int, dtype: Any) -> None:
        layers = int(gpt.layers)
        module = IndexTtsE(gpt, layers).to(self.args.device)
        inputs: list[Any] = []
        input_names: list[str] = []
        dynamic_axes: dict[str, dict[int, str]] = {}
        for idx in range(layers):
            key_name = f"in_key_{idx}"
            value_name = f"in_value_{idx}"
            inputs.extend(
                [
                    torch.zeros((1, heads, 1, head_dim), dtype=dtype, device=self.args.device),
                    torch.zeros((1, heads, 1, head_dim), dtype=dtype, device=self.args.device),
                ]
            )
            input_names.extend([key_name, value_name])
            dynamic_axes[key_name] = {2: "past_len"}
            dynamic_axes[value_name] = {2: "past_len"}

        inputs.extend(
            [
                torch.ones((1,), dtype=torch.int64, device=self.args.device),
                torch.ones((1, int(gpt.number_mel_codes)), dtype=dtype, device=self.args.device),
                torch.tensor([mel_len], dtype=torch.int64, device=self.args.device),
                torch.zeros((1, mel_len, hidden), dtype=dtype, device=self.args.device),
                torch.ones((1, mel_len + 1), dtype=torch.int64, device=self.args.device),
            ]
        )
        input_names.extend(["history_len", "repeat_penality", "ids_len", "hidden_state", "attention_mask"])
        dynamic_axes["repeat_penality"] = {1: "mel_code_size"}
        dynamic_axes["hidden_state"] = {1: "ids_len"}
        dynamic_axes["attention_mask"] = {1: "total_seq_len"}

        output_names: list[str] = []
        for idx in range(layers):
            output_names.extend([f"out_key_{idx}", f"out_value_{idx}"])
            dynamic_axes[f"out_key_{idx}"] = {2: "next_past_len"}
            dynamic_axes[f"out_value_{idx}"] = {2: "next_past_len"}
        output_names.extend(["kv_seq_len", "last_hidden_state", "max_logit_id"])
        dynamic_axes["last_hidden_state"] = {1: "one_or_ids"}
        self._export_onnx(module, tuple(inputs), "IndexTTS_E.onnx", input_names, output_names, dynamic_axes)

    def _export_f(self, torch: Any, bigvgan: Any, hidden: int, mel_len: int, dtype: Any) -> None:
        cond_count = bigvgan_condition_count(bigvgan)
        module = IndexTtsF(bigvgan).to(self.args.device)
        save_hidden_state = torch.zeros((mel_len, hidden), dtype=dtype, device=self.args.device)
        cond_layer = torch.zeros(
            (1, int(bigvgan.h.upsample_initial_channel), 1), dtype=dtype, device=self.args.device
        )
        conds = []
        for idx in range(cond_count):
            channels = int(bigvgan.h.upsample_initial_channel) // (2 ** (idx + 1))
            conds.append(torch.zeros((1, channels, 1), dtype=dtype, device=self.args.device))
        input_names = ["save_hidden_state", "bigvgan_cond_layer_speaker_embedding"] + [
            f"save_bigvgan_conds_{idx}" for idx in range(cond_count)
        ]
        dynamic_axes = {"save_hidden_state": {0: "decode_tokens"}, "generated_wav": {1: "num_samples"}}
        export_args = (save_hidden_state, cond_layer, *conds)
        try:
            self._export_onnx(
                module,
                export_args,
                "IndexTTS_F.onnx",
                input_names,
                ["generated_wav"],
                dynamic_axes,
            )
        except UnsupportedExport:
            if self._bigvgan_resampler_pad_mode == "constant":
                raise
            changed = self._set_bigvgan_resampler_pad_mode(bigvgan, "constant")
            if not changed:
                raise
            for partial in self.args.output_dir.glob("IndexTTS_F.onnx*"):
                if partial.is_file():
                    partial.unlink()
            self.notes.append(
                "IndexTTS_F export with replicate padding failed; retrying with export-only constant-pad "
                "fallback on patched BigVGAN alias-free resamplers. Official source files remain unchanged."
            )
            self._export_onnx(
                module,
                export_args,
                "IndexTTS_F.onnx",
                input_names,
                ["generated_wav"],
                dynamic_axes,
            )

    def _export_onnx(
        self,
        module: Any,
        args: tuple[Any, ...],
        filename: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        dynamic_axes: dict[str, dict[int, str]],
    ) -> None:
        import torch

        target = self.args.output_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        module.eval()
        export_kwargs = {
            "input_names": list(input_names),
            "output_names": list(output_names),
            "dynamic_axes": dynamic_axes,
            "opset_version": self.args.opset,
            "do_constant_folding": True,
        }
        signature = inspect.signature(torch.onnx.export)
        if "dynamo" in signature.parameters:
            export_kwargs["dynamo"] = False
        if "external_data" in signature.parameters:
            export_kwargs["external_data"] = True
        elif "use_external_data_format" in signature.parameters:
            export_kwargs["use_external_data_format"] = True
        self.notes.append(
            f"exporting {filename} with opset={self.args.opset}, external_data_threshold_mb="
            f"{self.args.external_data_threshold_mb} (PyTorch handles threshold support by version)"
        )
        try:
            with torch.no_grad():
                torch.onnx.export(module, args, str(target), **export_kwargs)
        except Exception as exc:
            hint = ""
            if filename == "IndexTTS_A.onnx":
                hint = " IndexTTS_A uses a conv1d STFT/mel frontend to avoid torch.stft ONNX lowering; check torch/onnx operator support for Conv/Pad/Sqrt/MatMul."
            raise UnsupportedExport(f"export {filename} failed: {exc}.{hint}") from exc


def bigvgan_condition_count(bigvgan: Any) -> int:
    return len(getattr(bigvgan, "conds", [])) if getattr(bigvgan, "cond_in_each_up_layer", False) else 0


def is_alias_free_activation(module: Any) -> bool:
    return all(hasattr(module, attr) for attr in ["act", "upsample", "downsample"])


def infer_activation_channels(module: Any) -> int | None:
    act = getattr(module, "act", None)
    in_features = getattr(act, "in_features", None)
    if in_features is not None:
        try:
            return int(in_features)
        except (TypeError, ValueError):
            pass
    alpha = getattr(act, "alpha", None)
    shape = getattr(alpha, "shape", None)
    if shape:
        try:
            return int(shape[0])
        except (TypeError, ValueError, IndexError):
            return None
    return None


class StaticGroupedUpSample1d:  # torch.nn.Module at runtime; avoids importing torch for --help
    def __new__(cls, torch: Any, source: Any, channels: int, pad_mode: str):
        import torch.nn.functional as F

        class _StaticGroupedUpSample1d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ratio = int(getattr(source, "ratio"))
                self.kernel_size = int(getattr(source, "kernel_size"))
                self.stride = int(getattr(source, "stride", self.ratio))
                self.pad = int(getattr(source, "pad"))
                self.pad_left = int(getattr(source, "pad_left"))
                self.pad_right = int(getattr(source, "pad_right"))
                self.channels = int(channels)
                self.pad_mode = pad_mode
                weight = source.filter.detach().clone().expand(self.channels, -1, -1).contiguous()
                self.register_buffer("weight", weight)

            def forward(self, x):
                if self.pad > 0:
                    if self.pad_mode == "constant":
                        x = F.pad(x, (self.pad, self.pad), mode="constant", value=0.0)
                    else:
                        x = F.pad(x, (self.pad, self.pad), mode="replicate")
                x = self.ratio * F.conv_transpose1d(
                    x,
                    self.weight.to(dtype=x.dtype),
                    stride=self.stride,
                    groups=self.channels,
                )
                right = -self.pad_right if self.pad_right > 0 else None
                return x[..., self.pad_left:right]

        return _StaticGroupedUpSample1d()


class StaticGroupedDownSample1d:  # torch.nn.Module at runtime; avoids importing torch for --help
    def __new__(cls, torch: Any, source: Any, channels: int, pad_mode: str):
        import torch.nn.functional as F

        class _StaticGroupedDownSample1d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ratio = int(getattr(source, "ratio"))
                lowpass = getattr(source, "lowpass")
                self.kernel_size = int(getattr(source, "kernel_size", getattr(lowpass, "kernel_size")))
                self.stride = int(getattr(lowpass, "stride", self.ratio))
                self.padding = bool(getattr(lowpass, "padding", True))
                self.pad_left = int(getattr(lowpass, "pad_left"))
                self.pad_right = int(getattr(lowpass, "pad_right"))
                self.channels = int(channels)
                self.pad_mode = pad_mode
                weight = lowpass.filter.detach().clone().expand(self.channels, -1, -1).contiguous()
                self.register_buffer("weight", weight)

            def forward(self, x):
                if self.padding:
                    if self.pad_mode == "constant":
                        x = F.pad(x, (self.pad_left, self.pad_right), mode="constant", value=0.0)
                    else:
                        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
                return F.conv1d(
                    x,
                    self.weight.to(dtype=x.dtype),
                    stride=self.stride,
                    groups=self.channels,
                )

        return _StaticGroupedDownSample1d()


class OnnxSafeMelSpectrogram:  # torch.nn.Module at runtime; avoids importing torch for --help
    def __new__(
        cls,
        torch: Any,
        torchaudio: Any,
        device: str,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 100,
        f_min: float = 0.0,
        f_max: float | None = None,
    ):
        import torch.nn.functional as F

        class _OnnxSafeMelSpectrogram(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n_fft = n_fft
                self.hop_length = hop_length
                self.win_length = win_length
                self.n_freqs = n_fft // 2 + 1
                self.center_pad = n_fft // 2
                self.register_buffer("stft_basis", self._build_stft_basis())
                self.register_buffer("mel_filter", self._build_mel_filter())

            def _build_stft_basis(self):
                window = torch.hann_window(self.win_length, periodic=True, dtype=torch.float32)
                if self.win_length < self.n_fft:
                    left = (self.n_fft - self.win_length) // 2
                    right = self.n_fft - self.win_length - left
                    window = F.pad(window, (left, right))
                freq = torch.arange(0, self.n_freqs, dtype=torch.float32).unsqueeze(1)
                time_index = torch.arange(0, self.n_fft, dtype=torch.float32).unsqueeze(0)
                angle = 2.0 * math.pi * freq * time_index / float(self.n_fft)
                real = torch.cos(angle) * window.unsqueeze(0)
                imag = -torch.sin(angle) * window.unsqueeze(0)
                return torch.cat([real, imag], dim=0).unsqueeze(1).to(device)

            def _build_mel_filter(self):
                max_freq = float(f_max if f_max is not None else sample_rate // 2)
                try:
                    mel_fbanks = torchaudio.functional.melscale_fbanks(
                        n_freqs=self.n_freqs,
                        f_min=f_min,
                        f_max=max_freq,
                        n_mels=n_mels,
                        sample_rate=sample_rate,
                        norm=None,
                        mel_scale="htk",
                    )
                except TypeError:
                    mel_fbanks = torchaudio.functional.melscale_fbanks(
                        self.n_freqs,
                        f_min,
                        max_freq,
                        n_mels,
                        sample_rate,
                        None,
                        "htk",
                    )
                return mel_fbanks.transpose(0, 1).contiguous().to(dtype=torch.float32, device=device)

            def forward(self, audio):
                # audio arrives as [B, N] float waveform from IndexTtsA. Use only ONNX-lowerable ops.
                wav = audio.unsqueeze(1)
                wav = F.pad(wav, (self.center_pad, self.center_pad), mode="reflect")
                spec = F.conv1d(wav, self.stft_basis, stride=self.hop_length)
                real = spec[:, : self.n_freqs, :]
                imag = spec[:, self.n_freqs :, :]
                magnitude = torch.sqrt(torch.clamp(real * real + imag * imag, min=1e-12))
                mel = torch.matmul(self.mel_filter.unsqueeze(0), magnitude)
                return torch.log(torch.clamp(mel, min=1e-7))

        return _OnnxSafeMelSpectrogram()


class IndexTtsA:  # torch.nn.Module at runtime; avoids importing torch for --help
    def __new__(cls, torch: Any, *args: Any, **kwargs: Any):
        class _IndexTtsA(torch.nn.Module):
            def __init__(self, mel_features: Any, gpt: Any, bigvgan: Any):
                super().__init__()
                self.mel_features = mel_features
                self.gpt = gpt
                self.bigvgan = bigvgan

            def forward(self, audio):
                wav = audio.to(torch.float32).squeeze(1) / 32768.0
                mel = self.mel_features(wav)
                cond_mel_lengths = torch._shape_as_tensor(mel)[-1:].to(torch.long)
                conds_latent = self.gpt.get_conditioning(mel, cond_mel_lengths)
                speaker_embedding = self.bigvgan.speaker_encoder(mel.transpose(1, 2), None)
                speaker_embedding = speaker_embedding.transpose(1, 2)
                save_conds = []
                if getattr(self.bigvgan, "cond_in_each_up_layer", False):
                    for cond in self.bigvgan.conds:
                        save_conds.append(cond(speaker_embedding))
                cond_layer = self.bigvgan.cond_layer(speaker_embedding)
                return (*save_conds, cond_layer, conds_latent)

        return _IndexTtsA(*args, **kwargs)


class IndexTtsB:
    def __new__(cls, gpt: Any):
        import torch
        import torch.nn.functional as F

        class _IndexTtsB(torch.nn.Module):
            def __init__(self, gpt_model: Any):
                super().__init__()
                self.gpt = gpt_model

            def forward(self, text_ids):
                text = text_ids.to(torch.long)
                text = F.pad(text, (0, 1), value=int(self.gpt.stop_text_token))
                text = F.pad(text, (1, 0), value=int(self.gpt.start_text_token))
                return self.gpt.text_embedding(text) + self.gpt.text_pos_embedding(text)

        return _IndexTtsB(gpt)


class IndexTtsC:
    def __new__(cls, gpt: Any):
        import torch

        class _IndexTtsC(torch.nn.Module):
            def __init__(self, gpt_model: Any):
                super().__init__()
                self.gpt = gpt_model

            def forward(self, gpt_ids, gen_len):
                ids = gpt_ids.to(torch.long)
                pos = gen_len.to(torch.long).clamp(min=0, max=int(self.gpt.max_mel_tokens) + 1)
                pos_emb = self.gpt.mel_pos_embedding.emb(pos).unsqueeze(1)
                hidden = self.gpt.mel_embedding(ids) + pos_emb
                return hidden, gen_len.to(torch.long) + 1

        return _IndexTtsC(gpt)


class IndexTtsD:
    def __new__(cls):
        import torch

        class _IndexTtsD(torch.nn.Module):
            def forward(self, embed_x, embed_y, embed_z):
                hidden = torch.cat([embed_x, embed_y, embed_z], dim=1)
                sx = torch._shape_as_tensor(embed_x)[1]
                sy = torch._shape_as_tensor(embed_y)[1]
                sz = torch._shape_as_tensor(embed_z)[1]
                concat_len = (sx + sy + sz).reshape(1).to(torch.long)
                return hidden, concat_len

        return _IndexTtsD()


class IndexTtsE:
    def __new__(cls, gpt: Any, layers: int):
        import torch

        class _IndexTtsE(torch.nn.Module):
            def __init__(self, gpt_model: Any, layer_count: int):
                super().__init__()
                self.gpt = gpt_model
                self.layer_count = layer_count

            def forward(self, *inputs):
                past_inputs = inputs[: self.layer_count * 2]
                history_len, repeat_penality, ids_len, hidden_state, attention_mask = inputs[self.layer_count * 2 :]
                past = tuple(
                    (past_inputs[idx * 2], past_inputs[idx * 2 + 1]) for idx in range(self.layer_count)
                )
                transformer_outputs = self.gpt.gpt(
                    inputs_embeds=hidden_state,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                present = transformer_outputs.past_key_values
                last_hidden = self.gpt.final_norm(transformer_outputs.last_hidden_state[:, -1:, :])
                logits = self.gpt.mel_head(last_hidden)
                penalties = repeat_penality.to(logits.dtype).unsqueeze(1)
                max_logit_id = torch.argmax(logits * penalties, dim=-1).to(torch.int32).reshape(1)
                kv_seq_len = history_len.to(torch.long).reshape(1) + ids_len.to(torch.long).reshape(1)
                flat_present = []
                for key, value in present:
                    flat_present.extend([key, value])
                return (*flat_present, kv_seq_len, last_hidden, max_logit_id)

        return _IndexTtsE(gpt, layers)


class IndexTtsF:
    def __new__(cls, bigvgan: Any):
        import torch

        class _IndexTtsF(torch.nn.Module):
            def __init__(self, vocoder: Any):
                super().__init__()
                self.bigvgan = vocoder

            def forward(self, save_hidden_state, bigvgan_cond_layer_speaker_embedding, *save_bigvgan_conds):
                x = save_hidden_state.unsqueeze(0) if save_hidden_state.dim() == 2 else save_hidden_state
                if self.bigvgan.feat_upsample:
                    x = torch.nn.functional.interpolate(x.transpose(1, 2), scale_factor=[4], mode="linear").squeeze(1)
                else:
                    x = x.transpose(1, 2)
                x = self.bigvgan.conv_pre(x)
                x = x + bigvgan_cond_layer_speaker_embedding
                for idx in range(self.bigvgan.num_upsamples):
                    for i_up in range(len(self.bigvgan.ups[idx])):
                        x = self.bigvgan.ups[idx][i_up](x)
                    if getattr(self.bigvgan, "cond_in_each_up_layer", False):
                        x = x + save_bigvgan_conds[idx]
                    xs = None
                    for kernel_idx in range(self.bigvgan.num_kernels):
                        block = self.bigvgan.resblocks[idx * self.bigvgan.num_kernels + kernel_idx]
                        xs = block(x) if xs is None else xs + block(x)
                    x = xs / self.bigvgan.num_kernels
                x = self.bigvgan.activation_post(x)
                x = self.bigvgan.conv_post(x)
                x = torch.tanh(x).squeeze(1)
                return torch.clamp(x * 32767.0, -32768.0, 32767.0).round().to(torch.int16)

        return _IndexTtsF(bigvgan)


def invoke_local_exporter(args: argparse.Namespace, notes: list[str]) -> bool:
    """Invoke a project-local exporter without vendoring upstream code."""
    entries = list(args.local_export_entry)
    if not entries:
        notes.append(
            "local-export mode requires at least one explicit --local-export-entry MODULE:FUNCTION; "
            "no default exporter entries are imported to avoid side effects in the official checkout"
        )
        return False
    old_path = list(sys.path)
    sys.path.insert(0, str(args.index_tts_project.resolve()))
    try:
        for entry in entries:
            try:
                func = load_export_callable(entry)
            except Exception as exc:  # pragma: no cover - project-dependent imports
                notes.append(f"local exporter {entry} unavailable: {exc}")
                continue
            try:
                notes.append(f"invoking local exporter {entry}")
                result = call_exporter(func, args)
                source = Path(result) if result else args.output_dir
                if source != args.output_dir and all((source / name).exists() for name in MODEL_FILES):
                    for filename in MODEL_FILES:
                        copy_one(source / filename, args.output_dir / filename, args.overwrite)
                    for data_file in source.glob("*.onnx.data"):
                        copy_one(data_file, args.output_dir / data_file.name, args.overwrite)
                if all((args.output_dir / name).exists() for name in MODEL_FILES):
                    notes.append(f"local exporter {entry} produced A-F ONNX files")
                    return True
                notes.append(f"local exporter {entry} returned but A-F ONNX files were incomplete")
            except Exception as exc:  # pragma: no cover - project-dependent exports
                notes.append(f"local exporter {entry} failed: {exc}")
        return False
    finally:
        sys.path[:] = old_path


def load_export_callable(entry: str):
    if ":" not in entry:
        raise ValueError("entry must be MODULE:FUNCTION")
    module_name, function_name = entry.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, function_name)
    if not callable(func):
        raise TypeError(f"{entry} is not callable")
    return func


def call_exporter(func: Any, args: argparse.Namespace):
    kwargs = {
        "index_tts_project": args.index_tts_project,
        "source_model_dir": args.source_model_dir,
        "output_dir": args.output_dir,
        "precision": SUPPORTED_PRECISION,
        "precision_policy": "fp32-only; cpu-q4 and gpu-fp16 paths are deprecated and ignored",
        "device": args.device,
        "stages": tuple("ABCDEF"),
    }
    signature = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return func(**kwargs)
    filtered = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return func(**filtered)


def apply_precision(args: argparse.Namespace, notes: list[str]) -> None:
    if args.precision != SUPPORTED_PRECISION:
        raise UnsupportedExport("IndexTTS currently supports cpu-fp32 artifacts only")
    notes.append("cpu-fp32 selected; no quantization or fp16 transform applied")

def copy_bpe(args: argparse.Namespace, notes: list[str]) -> None:
    bpe = first_existing(args.source_model_dir.rglob("bpe.model"))
    if bpe is None:
        notes.append("bpe.model was not found in source-model-dir; runtime validation will fail until it is copied")
        return
    copy_one(bpe, args.output_dir / "bpe.model", args.overwrite)
    notes.append(f"copied tokenizer {bpe}")


def copy_one(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def index_tts_config_metadata(cfg_path: Path) -> dict[str, Any]:
    cfg = read_config_yaml_best_effort(cfg_path) if cfg_path.exists() else {}
    gpt = cfg.get("gpt", {}) if isinstance(cfg, dict) else {}
    if not isinstance(gpt, dict):
        gpt = {}
    number_mel_codes = positive_int(gpt.get("number_mel_codes")) or (STOP_TOKEN + 1)
    number_text_tokens = positive_int(gpt.get("number_text_tokens"))
    return {
        "start_token": positive_int(gpt.get("start_mel_token")) or START_TOKEN,
        "stop_token": positive_int(gpt.get("stop_mel_token")) or STOP_TOKEN,
        "max_generate_length": positive_int(gpt.get("max_mel_tokens")) or MAX_GENERATE_LENGTH,
        "mel_code_size": number_mel_codes,
        # The Rust adapter uses vocab_size as a fallback for repeat_penality width, so keep it aligned
        # with mel code size and expose text_vocab_size separately when present.
        "vocab_size": number_mel_codes,
        "text_vocab_size": number_text_tokens,
        "config_source": str(cfg_path) if cfg_path.exists() else None,
    }


def read_config_yaml_best_effort(cfg_path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return parse_simple_yaml_mapping(cfg_path.read_text(encoding="utf-8"))


def parse_simple_yaml_mapping(text: str) -> dict[str, Any]:
    """Tiny fallback parser for the simple one-level config fields needed in missing-deps envs."""
    result: dict[str, Any] = {}
    current_top: str | None = None
    for raw_line in text.splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip() or ":" not in line_without_comment:
            continue
        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        key, value = line_without_comment.strip().split(":", 1)
        value = value.strip()
        if indent == 0:
            current_top = key
            result[key] = {} if not value else parse_yaml_value(value)
        elif current_top:
            section = result.setdefault(current_top, {})
            if isinstance(section, dict):
                section[key] = parse_yaml_value(value)
    return result


def parse_yaml_value(value: str) -> Any:
    if value == "":
        return {}
    stripped = value.strip().strip('"').strip("'")
    lower = stripped.lower()
    if lower in {"null", "none", "~"}:
        return None
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def build_manifest(args: argparse.Namespace, status: str, notes: list[str]) -> dict[str, Any]:
    artifacts = artifact_metadata(args.output_dir)
    optional = []
    optimized_root = args.output_dir / "optimized"
    if optimized_root.exists():
        optional.extend(str(path.relative_to(args.output_dir)) for path in sorted(optimized_root.glob("*")))
    optional.extend(path.name for path in sorted(args.output_dir.glob("*.onnx.data")))
    cfg_path = args.source_model_dir / "config.yaml"
    cfg_meta = index_tts_config_metadata(cfg_path)
    return {
        "model_family": "index_tts",
        "adapter": "index_tts",
        "sample_rate": SAMPLE_RATE,
        "start_token": cfg_meta["start_token"],
        "stop_token": cfg_meta["stop_token"],
        "max_generate_length": cfg_meta["max_generate_length"],
        "mel_code_size": cfg_meta["mel_code_size"],
        "vocab_size": cfg_meta["vocab_size"],
        "text_vocab_size": cfg_meta["text_vocab_size"],
        "precision": SUPPORTED_PRECISION,
        "precision_policy": "fp32-only; cpu-q4 and gpu-fp16 paths are deprecated and ignored",
        "device": args.device,
        "source_model_dir": str(args.source_model_dir),
        "index_tts_project": str(args.index_tts_project),
        "source_config": cfg_meta["config_source"],
        "files": [*MODEL_FILES, "bpe.model", "manifest.yaml", "manifest.json"],
        "optional_files": optional,
        "artifacts": artifacts,
        "export_provenance": {
            "script": "scripts/local/indextts_export.py",
            "entrypoint": "scripts/indextts_export.py",
            "mode": args.mode,
            "opset": args.opset,
            "audio_length": args.audio_length,
            "max_seq_len": args.max_seq_len,
            "created_unix": int(time.time()),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "status": status,
        "notes": notes,
    }


def artifact_metadata(root: Path) -> list[dict[str, Any]]:
    names = [*MODEL_FILES, "bpe.model"]
    records = []
    for name in names:
        path = root / name
        if path.exists():
            records.append(
                {
                    "path": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for data_file in sorted(root.glob("*.onnx.data")):
        records.append(
            {
                "path": data_file.name,
                "size_bytes": data_file.stat().st_size,
                "sha256": sha256_file(data_file),
            }
        )
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(to_yaml(data), encoding="utf-8")


def to_yaml(value: Any, indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(to_yaml(item, indent + 2).rstrip("\n"))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
    elif isinstance(value, list):
        if not value:
            lines.append(f"{prefix}[]")
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(to_yaml(item, indent + 2).rstrip("\n"))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
    else:
        lines.append(f"{prefix}{yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def validate_layout(output_dir: Path, require_onnx: bool) -> None:
    missing = [name for name in MODEL_FILES if not (output_dir / name).exists()]
    if require_onnx and missing:
        raise SystemExit(f"export reported ready but required files are missing: {missing}")
    if not (output_dir / "bpe.model").exists():
        print("warning: bpe.model missing; runtime adapter will reject this layout", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
