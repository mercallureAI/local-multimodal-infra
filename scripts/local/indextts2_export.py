"""Export/package IndexTTS-2.5 ONNX artifacts for LOCAL.

The graph split comes from DakeQQ's Text-to-Speech-TTS-ONNX ``Index_TTS/v2``
exporter, which is driven here as an external checkout (not vendored):

* ``export``   runs the upstream exporter against the official 2.5 checkpoint
  into a raw FP32-weight package (FP16 KV cache);
* ``optimize`` runs the upstream optimizer with this project's precision plan
  (``fp16`` for NVIDIA GPUs, ``fp32`` as the numerical reference);
* ``package``  writes the ``tokenizer.json`` / ``manifest.json`` contract read by
  ``crates/adapter-index-tts2`` next to an optimized package.

All three must run in a Python 3.11 environment built from the official
index-tts ``uv.lock`` plus ``onnx onnxruntime-gpu onnxslim pydub soundfile``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ADAPTER = "index_tts2"
MODEL_VERSION = "2.5"
MANIFEST_SCHEMA = "local.index_tts2.package.v1"
TOKENIZER_SCHEMA = "local.index_tts2.tokenizer.v1"
TIKTOKEN_FILE = "multilingual_zh_ja_yue_char_del.tiktoken"
# Graphs the Rust adapter loads. The Qwen emotion-text graphs are exported by
# upstream but intentionally not shipped: emotion arrives as an 8-float vector.
RUNTIME_GRAPH_KEYS = (
    "model_file_name_reference_preprocess",
    "model_file_name_conditioning",
    "model_file_name_target_prefill_sampling",
    "model_file_name_decode_step_sampling",
    "model_file_name_synthesis",
    "model_file_name_cfm_estimator",
    "model_file_name_decoder",
    "model_file_name_metadata",
)
RUNTIME_INT_KEYS = (
    "in_sample_rate",
    "out_sample_rate",
    "cfm_steps",
    "max_signal_length",
    "max_text_tokens",
    "mel_code_size",
    "stop_mel_token",
)
PRECISIONS = ("fp16", "fp32")
ONNX_ELEMENTS = {1: "f32", 10: "f16", 6: "i32", 7: "i64", 9: "bool", 3: "i8", 5: "i16"}
# Graphs kept FP32 in the fp16 package by default.
FP16_KEEP_F32_DEFAULT: tuple[str, ...] = ()
# Graphs converted only under these node-name prefixes. The reference graph's
# fbank/STFT front end (`feature/`) and speaker/mel path (`reference/`) lose
# frames and accuracy in half; only the w2v-BERT encoder (`semantic/`, 1.5 GiB
# of weights) is converted.
FP16_PARTIAL = {"IndexTTS2_ReferencePreprocess": ("semantic/",)}
# Ops kept in float32 on top of ORT's default block list: sampling and
# normalization arithmetic that overflows or loses resolution in half.
# CFM DiT geometry (config.yaml s2mel.DiT: hidden_dim 512, num_heads 8).
DIT_HIDDEN_SIZE = 512
DIT_ATTENTION_HEADS = 8
FP16_OP_BLOCK_EXTRA = {"RandomUniform", "RandomNormal", "Multinomial", "Softmax", "ReduceSum", "Pow", "Exp", "Log"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dakeqq-repo", required=True, type=Path, help="Text-to-Speech-TTS-ONNX checkout")

    export = sub.add_parser("export", help="run the upstream raw export")
    common(export)
    export.add_argument("--index-tts-project", required=True, type=Path, help="official index-tts checkout")
    export.add_argument("--source-model-dir", required=True, type=Path, help="IndexTeam/IndexTTS-2.5 snapshot")
    export.add_argument("--raw-dir", required=True, type=Path)

    optimize = sub.add_parser("optimize", help="build a runtime-only fp16/fp32 package from a raw export")
    common(optimize)
    optimize.add_argument("--raw-dir", required=True, type=Path)
    optimize.add_argument("--output-dir", required=True, type=Path)
    optimize.add_argument("--precision", choices=PRECISIONS, default="fp16")
    optimize.add_argument(
        "--keep-f32",
        nargs="*",
        default=list(FP16_KEEP_F32_DEFAULT),
        help="graph names (without .onnx) left in float32 for --precision fp16",
    )

    package = sub.add_parser("package", help="write the LOCAL tokenizer/manifest contract")
    package.add_argument("--index-tts-project", required=True, type=Path)
    package.add_argument("--source-model-dir", required=True, type=Path)
    package.add_argument("--output-dir", required=True, type=Path)
    package.add_argument("--precision", choices=PRECISIONS, default="fp16")
    package.add_argument("--dakeqq-repo", type=Path, help="recorded in provenance when given")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "export":
        run_export(args)
    elif args.command == "optimize":
        run_optimize(args)
    else:
        run_package(args)
    return 0


def _dakeqq_paths(repo: Path) -> tuple[Path, Path]:
    repo = repo.expanduser().resolve()
    v2 = repo / "Index_TTS" / "v2"
    if not (v2 / "Export_IndexTTS2.py").is_file():
        raise SystemExit(f"{repo} is not a Text-to-Speech-TTS-ONNX checkout (missing Index_TTS/v2)")
    for path in (repo, repo / "Index_TTS"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo, v2


def run_export(args: argparse.Namespace) -> None:
    _, v2 = _dakeqq_paths(args.dakeqq_repo)
    project = args.index_tts_project.expanduser().resolve()
    source = args.source_model_dir.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()
    if not (source / "config.yaml").is_file() or not (source / TIKTOKEN_FILE).is_file():
        raise SystemExit(f"{source} is not an IndexTTS-2.5 snapshot")
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))

    from Index_TTS import _indextts2_export_common as shared
    from Index_TTS.v2.STFT_Process import STFT_Process

    _accept_dtype_kwarg()

    profile = shared.ExportProfile(
        model_version=MODEL_VERSION,
        script_dir=v2,
        project_path=project,
        models_path=source,
        # Path.__truediv__ keeps an absolute right-hand side, so the package
        # lands in raw_dir rather than inside the upstream checkout.
        output_folder_name=str(raw_dir),
        text_tokenizer_file=TIKTOKEN_FILE,
        stft_process=STFT_Process,
    )
    started = time.time()
    shared.run_export(profile, None)
    print(f"raw export ready in {time.time() - started:.0f}s: {raw_dir}")


def _accept_dtype_kwarg() -> None:
    """Let the upstream exporter's ``from_pretrained(dtype=...)`` run on the
    transformers 4.52 pinned by the official uv.lock, which still names it
    ``torch_dtype``. Upgrading transformers instead would change the GPT-2
    classes the official checkpoint is loaded into."""
    import transformers

    original = transformers.AutoModelForCausalLM.from_pretrained.__func__

    def from_pretrained(cls, *args: Any, **kwargs: Any):
        if "dtype" in kwargs and "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return original(cls, *args, **kwargs)

    transformers.AutoModelForCausalLM.from_pretrained = classmethod(from_pretrained)


def run_optimize(args: argparse.Namespace) -> None:
    """Build a runtime-only package from the raw export.

    ``fp32`` bundles the raw graphs unchanged. ``fp16`` converts each graph with
    ONNX Runtime's ``convert_float_to_float16`` (float32 graph I/O kept, casts
    inserted around blocked ops). Upstream's F16 optimizer plan is not used: its
    fusion passes leave mixed float16/float32 operands in the Conditioning,
    sampling and CFM graphs. Only the runtime graphs are bundled, which drops
    the Qwen emotion-text graphs and unused decode strategies.
    """
    _dakeqq_paths(args.dakeqq_repo)
    import onnx
    from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16
    from Index_TTS.v2.Shared_Weights import bundle_shared_initializers

    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metadata = read_package_metadata(raw_dir)
    if metadata.get("model_version") != MODEL_VERSION:
        raise SystemExit(f"{raw_dir} is not a raw IndexTTS-2.5 package")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    keep_f32 = set(args.keep_f32)
    started = time.time()
    staged = []
    for key in RUNTIME_GRAPH_KEYS:
        name = metadata[key]
        graph = name.removesuffix(".onnx")
        convert = args.precision == "fp16" and graph not in keep_f32 and key != "model_file_name_metadata"
        graph_started = time.time()
        # Shape inference must run on the file: fp32 GPT graphs exceed the 2 GB
        # in-memory protobuf limit. The inferred copy lives beside the raw graph
        # so its external-data references still resolve.
        inferred = raw_dir / f".{graph}.inferred.onnx"
        try:
            onnx.shape_inference.infer_shapes_path(str(raw_dir / name), str(inferred))
            model = onnx.load(str(inferred), load_external_data=True)
        finally:
            inferred.unlink(missing_ok=True)
        fused = fuse_mha_attention(model)
        if fused:
            print(f"  fused {fused} attention block(s) into MultiHeadAttention", flush=True)
        if convert:
            neutralize_integer_sums(model)
            partial = FP16_PARTIAL.get(graph)
            model = convert_float_to_float16(
                model,
                keep_io_types=True,
                disable_shape_infer=True,
                op_block_list=sorted(set(DEFAULT_OP_BLOCK_LIST) | FP16_OP_BLOCK_EXTRA),
                node_block_list=(
                    [node.name for node in model.graph.node if not node.name.startswith(partial)]
                    if partial
                    else None
                ),
            )
            drop_duplicate_nodes(model)
            removed = remove_cast_round_trips(model)
            if removed:
                print(f"  removed {removed} float16 round-trip cast(s) between float32 nodes", flush=True)
        target = output_dir / name
        onnx.save(
            model,
            str(target),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{name}.data",
            size_threshold=1024,
        )
        del model
        staged.append(target)
        print(f"{'fp16' if convert else 'fp32'} {graph} in {time.time() - graph_started:.0f}s", flush=True)
    stats = bundle_shared_initializers(output_dir, model_paths=staged, metadata=metadata)
    print(
        f"{args.precision} package ready in {time.time() - started:.0f}s: "
        f"{stats['unique_initializers']} tensors, {stats['unique_bytes'] / 2**30:.2f} GiB -> {output_dir}"
    )


def fuse_mha_attention(model: Any) -> int:
    """Replace the DiT's explicit attention with ``com.microsoft.MultiHeadAttention``.

    Upstream exports each CFM layer as ``Softmax((q*s)(k^T*s)) @ v`` over
    ``[B, N, T, H]`` tensors (no mask), materializing a float32 ``[2, 8, T, T]``
    score tensor per layer; at T ~ 4000 mel frames that is ~1 GB, and the
    scores also overflow float16. The fused op runs flash / memory-efficient
    kernels on CUDA with float32 accumulation and O(T) memory. Returns the
    number of rewritten blocks.
    """
    from onnx import helper

    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}
    consumers: dict[str, list[Any]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def only(name: str, op: str) -> Any | None:
        users = consumers.get(name, [])
        return users[0] if len(users) == 1 and users[0].op_type == op else None

    new_nodes = []
    fused = 0
    for softmax in [node for node in graph.node if node.op_type == "Softmax"]:
        scores = producers.get(softmax.input[0])
        if scores is None or scores.op_type != "MatMul":
            continue
        q_scale, k_scale = (producers.get(name) for name in scores.input)
        context = only(softmax.output[0], "MatMul")
        if not (q_scale and k_scale and context) or q_scale.op_type != "Mul" or k_scale.op_type != "Mul":
            continue
        k_transpose = producers.get(k_scale.input[0])
        merge = only(context.output[0], "Transpose")
        flatten = only(merge.output[0], "Reshape") if merge else None
        if k_transpose is None or k_transpose.op_type != "Transpose" or flatten is None:
            continue
        prefix = softmax.name.rsplit("/", 1)[0]
        query_bnsh, key_bnsh, value_bnsh = q_scale.input[0], k_transpose.input[0], context.input[1]
        # Reuse the block's own [B, T, D] output shape for the flattened inputs.
        hidden_shape = flatten.input[1]
        new_inputs = []
        for label, bnsh in (("query", query_bnsh), ("key", key_bnsh), ("value", value_bnsh)):
            bsnh = f"{prefix}/mha_{label}_bsnh"
            bsd = f"{prefix}/mha_{label}_bsd"
            new_nodes.append((flatten, helper.make_node("Transpose", [bnsh], [bsnh], name=f"{prefix}/mha_{label}_transpose", perm=[0, 2, 1, 3])))
            new_nodes.append((flatten, helper.make_node("Reshape", [bsnh, hidden_shape], [bsd], name=f"{prefix}/mha_{label}_reshape")))
            new_inputs.append(bsd)
        new_nodes.append((flatten, helper.make_node(
            "MultiHeadAttention",
            new_inputs,
            [flatten.output[0]],
            name=f"{prefix}/mha",
            domain="com.microsoft",
            num_heads=DIT_ATTENTION_HEADS,
            # q and k are each pre-scaled by sqrt(1/sqrt(head_dim)) upstream.
            scale=(DIT_ATTENTION_HEADS / DIT_HIDDEN_SIZE) ** 0.5,
        )))
        flatten.output[0] = f"{flatten.output[0]}_unfused"
        fused += 1
    if not fused:
        return 0
    # Insert each replacement right before the output reshape it supersedes
    # (all of its inputs exist by then), then drop what is no longer reachable.
    ordered = []
    pending = {}
    for anchor, node in new_nodes:
        pending.setdefault(id(anchor), []).append(node)
    for node in graph.node:
        ordered.extend(pending.get(id(node), []))
        ordered.append(node)
    del graph.node[:]
    graph.node.extend(ordered)
    prune_dead_nodes(model)
    if not any(opset.domain == "com.microsoft" for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return fused


def prune_dead_nodes(model: Any) -> None:
    graph = model.graph
    needed = {output.name for output in graph.output}
    kept = []
    for node in reversed(list(graph.node)):
        if any(output in needed for output in node.output):
            kept.append(node)
            needed.update(name for name in node.input if name)
    kept.reverse()
    del graph.node[:]
    graph.node.extend(kept)


def neutralize_integer_sums(model: Any) -> None:
    """Zero integer-id sums that only feed a ``* 0`` anchor.

    Upstream's 2.5 latent bypass keeps unused Synthesis inputs alive with
    ``(sum(speaker_latent) + sum(emotion_vector) + sum(float(text_ids))) * 0``.
    The text-id sum (~5e5) overflows float16 to inf and ``inf * 0`` turns the
    whole (all-zero) ``gpt_latent`` into NaN. Multiplying the cast ids by 0
    before the ReduceSum keeps the result exactly 0 in either precision.
    """
    from onnx import TensorProto, helper

    graph = model.graph
    integer_inputs = {
        value.name
        for value in graph.input
        if value.type.tensor_type.elem_type in (TensorProto.INT32, TensorProto.INT64)
    }
    producers = {output: node for node in graph.node for output in node.output}
    patched = 0
    for index, node in enumerate(list(graph.node)):
        if node.op_type != "ReduceSum" or not node.name.startswith("latent/"):
            continue
        source = producers.get(node.input[0])
        if source is None or source.op_type != "Cast" or source.input[0] not in integer_inputs:
            continue
        zero = f"{node.name}_zero"
        scaled = f"{node.input[0]}_zeroed"
        graph.initializer.append(helper.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        position = list(graph.node).index(node)
        graph.node.insert(position, helper.make_node("Mul", [node.input[0], zero], [scaled], name=f"{node.name}_zero_ids"))
        # The float16 converter runs without shape inference and types every
        # tensor from value_info, so the new tensor needs one.
        for info in list(graph.value_info):
            if info.name == node.input[0]:
                copy = graph.value_info.add()
                copy.CopyFrom(info)
                copy.name = scaled
                break
        node.input[0] = scaled
        patched += 1
    if patched:
        print(f"  zeroed {patched} integer-id anchor sum(s) before float16 conversion", flush=True)


def remove_cast_round_trips(model: Any) -> int:
    """Bypass ``Cast(float16) -> Cast(float32)`` pairs.

    Without shape inference ``convert_float_to_float16`` wraps every blocked
    node in casts, so two adjacent float32 nodes still exchange their tensor
    through float16, which defeats the block list exactly where values exceed
    the half range. Consumers of the float32 re-cast read the original tensor.
    """
    from onnx import TensorProto

    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}
    graph_outputs = {output.name for output in graph.output}
    # Only float32 sources qualify: the converter also wraps blocked ops that
    # run on integers (shape arithmetic), and those casts must stay.
    elem_types = {info.name: info.type.tensor_type.elem_type for info in [*graph.value_info, *graph.input]}
    elem_types.update({tensor.name: tensor.data_type for tensor in graph.initializer})

    def cast_to(node: Any) -> int | None:
        if node.op_type != "Cast":
            return None
        return next((attr.i for attr in node.attribute if attr.name == "to"), None)

    replacement: dict[str, str] = {}
    removed = set()
    for node in graph.node:
        if cast_to(node) != TensorProto.FLOAT or node.output[0] in graph_outputs:
            continue
        down = producers.get(node.input[0])
        if down is None or cast_to(down) != TensorProto.FLOAT16:
            continue
        if elem_types.get(down.input[0]) != TensorProto.FLOAT:
            continue
        replacement[node.output[0]] = down.input[0]
        removed.add(id(node))
    if not replacement:
        return 0
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name in replacement:
                node.input[index] = replacement[name]
    kept = [node for node in graph.node if id(node) not in removed]
    # Drop float16 casts left without consumers.
    consumed = {name for node in kept for name in node.input} | graph_outputs
    kept = [
        node for node in kept
        if not (cast_to(node) == TensorProto.FLOAT16 and node.output[0] not in consumed)
    ]
    count = len(graph.node) - len(kept)
    del graph.node[:]
    graph.node.extend(kept)
    return count


def drop_duplicate_nodes(model: Any) -> None:
    """Remove byte-identical nodes. ``convert_float_to_float16`` inserts one
    boundary Cast per consumer *input*, so a node reading the same tensor twice
    (e.g. ``Concat(x, x)``) gets two identical Casts with one name and output."""
    seen: set[bytes] = set()
    kept = []
    for node in model.graph.node:
        key = node.SerializeToString()
        if key in seen:
            continue
        seen.add(key)
        kept.append(node)
    if len(kept) != len(model.graph.node):
        del model.graph.node[:]
        model.graph.node.extend(kept)


def run_package(args: argparse.Namespace) -> None:
    project = args.index_tts_project.expanduser().resolve()
    source = args.source_model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metadata = read_package_metadata(output_dir)
    if metadata.get("model_version") != MODEL_VERSION:
        raise SystemExit(f"{output_dir} declares model_version={metadata.get('model_version')!r}")
    shutil.copy2(source / TIKTOKEN_FILE, output_dir / TIKTOKEN_FILE)
    write_json(output_dir / "tokenizer.json", tokenizer_manifest(project, source))
    for name in ("LICENSE",):
        if (source / name).is_file():
            shutil.copy2(source / name, output_dir / name)

    files = sorted({metadata[key] for key in RUNTIME_GRAPH_KEYS})
    files += [metadata["shared_initializer_model_file"], metadata["shared_initializer_data_file"]]
    files += [TIKTOKEN_FILE, "tokenizer.json"]
    missing = [name for name in files if not (output_dir / name).is_file()]
    if missing:
        raise SystemExit(f"package is missing runtime files: {missing}")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "adapter": ADAPTER,
        "model_version": MODEL_VERSION,
        "precision": args.precision,
        "sample_rate": int(metadata["out_sample_rate"]),
        "graphs": {key.removeprefix("model_file_name_"): metadata[key] for key in RUNTIME_GRAPH_KEYS},
        "runtime": {key: int(metadata[key]) for key in RUNTIME_INT_KEYS},
        "device_shared_initializers": device_shared_initializers(
            output_dir,
            [metadata["model_file_name_target_prefill_sampling"], metadata["model_file_name_decode_step_sampling"]],
        ),
        # Session config the fp16-KV graphs need (mirrors upstream inference).
        "disabled_optimizers": (
            ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
            if metadata.get("use_f16_kv") == "1" and metadata.get("compute_in_f32") == "0"
            else []
        ),
        "artifacts": [
            {"path": name, "size_bytes": (output_dir / name).stat().st_size, "sha256": sha256_file(output_dir / name)}
            for name in files
        ],
        "export_provenance": {
            "script": "scripts/local/indextts2_export.py",
            "created_unix": int(time.time()),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "source_model": {"repository": "IndexTeam/IndexTTS-2.5", "revision": hf_revision(source)},
            "index_tts_code": git_revision(project),
            "dakeqq_exporter": git_revision(args.dakeqq_repo) if args.dakeqq_repo else None,
            "license": "bilibili Model Use License Agreement (see LICENSE)",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"packaged {len(files)} runtime files in {output_dir}")


def device_shared_initializers(folder: Path, graphs: list[str]) -> dict[str, Any]:
    """Blob ranges of the initializers every graph in ``graphs`` references.

    The prefill and decode GPT graphs carry the same ~1.2 GiB of weights. The
    runtime uploads these ranges to the device once and hands the same values
    to both sessions (``AddInitializer``) instead of letting each session copy
    its own.
    """
    import onnx

    per_graph = []
    for name in graphs:
        model = onnx.load(str(folder / name), load_external_data=False)
        per_graph.append({tensor.name: tensor for tensor in model.graph.initializer})
    common = set(per_graph[0]).intersection(*per_graph[1:])
    tensors = []
    locations = set()
    for name in sorted(common):
        tensor = per_graph[0][name]
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        external = {entry.key: entry.value for entry in tensor.external_data}
        locations.add(external["location"])
        tensors.append({
            "name": name,
            "element": ONNX_ELEMENTS[tensor.data_type],
            "shape": [int(dim) for dim in tensor.dims],
            "offset": int(external.get("offset", "0")),
            "length": int(external["length"]),
        })
    if len(locations) > 1:
        raise SystemExit(f"shared GPT initializers span several data files: {sorted(locations)}")
    return {"graphs": graphs, "data_file": locations.pop() if locations else None, "tensors": tensors}


def read_package_metadata(folder: Path) -> dict[str, str]:
    import onnx

    model = onnx.load(str(folder / "IndexTTS2_Metadata.onnx"), load_external_data=False)
    return {prop.key: prop.value for prop in model.metadata_props}


def tokenizer_manifest(project: Path, source: Path) -> dict[str, Any]:
    """Serialize the official tokenizer tables so Rust cannot drift from them."""
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from indextts.utils import tokenizer as upstream

    encoding = upstream.get_encoding(
        name=TIKTOKEN_FILE.removesuffix(".tiktoken"),
        num_languages=99,
        model_dir=str(source),
    )
    specials = sorted(encoding._special_tokens.items(), key=lambda item: item[1])
    return {
        "schema": TOKENIZER_SCHEMA,
        "tiktoken_file": TIKTOKEN_FILE,
        "pattern": encoding._pat_str,
        "special_tokens": [[name, int(token_id)] for name, token_id in specials],
        "languages": {code: int(index) for code, index in upstream.LANGUAGE_DICT.items()},
        "fallback_language": "common",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def hf_revision(source: Path) -> str | None:
    marker = source / ".cache" / "huggingface" / "download"
    for meta in sorted(marker.glob("*.metadata")) if marker.is_dir() else []:
        first = meta.read_text(encoding="utf-8").splitlines()[:1]
        if first:
            return first[0].strip()
    return None


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
