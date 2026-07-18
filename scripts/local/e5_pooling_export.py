"""Append masked mean pooling and L2 normalization to E5 ONNX graphs.

The upstream multilingual-e5-small exports ``last_hidden_state``.  Returning
that tensor forces the CUDA execution provider to copy ``[batch, seq, 384]``
back to the host so Rust can pool it.  This tool produces an equivalent graph
whose only output is the normalized ``[batch, 384]`` sentence embedding.

Run with an isolated dependency environment, for example::

    uv run --with onnx --python 3.12 \
      python -m scripts.local.e5_pooling_export --model-dir workdir/models
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import onnx
from onnx import TensorProto, helper


DEFAULT_GRAPHS = (
    "model_O4.onnx",
    "model_qint8_avx512_vnni.onnx",
)


def pooled_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_pooled.onnx")


def append_pooling(source: Path, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"[e5-pooling] exists: {destination}")
        return

    model = onnx.load(str(source), load_external_data=True)
    original_checker_error: onnx.checker.ValidationError | None = None
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as error:
        # The upstream O4 graph is an ORT-optimized graph containing fused
        # operators that the generic ONNX checker does not recognize at its
        # declared opset. ORT is the authority for that graph; still require
        # clean upstream graphs (such as qint8) to remain checker-clean.
        original_checker_error = error
    graph = model.graph
    output_names = {output.name for output in graph.output}
    if "last_hidden_state" not in output_names:
        raise RuntimeError(
            f"{source} does not expose last_hidden_state; outputs={sorted(output_names)}"
        )
    input_names = {value.name for value in graph.input}
    if "attention_mask" not in input_names:
        raise RuntimeError(f"{source} does not expose attention_mask")

    prefix = "local_e5_pool"
    epsilon_name = f"{prefix}_epsilon"
    graph.initializer.append(
        helper.make_tensor(epsilon_name, TensorProto.FLOAT, [], [1.0e-12])
    )
    graph.node.extend(
        [
            helper.make_node(
                "Cast",
                ["attention_mask"],
                [f"{prefix}_mask_f32"],
                name=f"{prefix}_cast_mask",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Unsqueeze",
                [f"{prefix}_mask_f32"],
                [f"{prefix}_mask_3d"],
                name=f"{prefix}_unsqueeze_mask",
                axes=[2],
            ),
            helper.make_node(
                "Mul",
                ["last_hidden_state", f"{prefix}_mask_3d"],
                [f"{prefix}_masked_hidden"],
                name=f"{prefix}_mask_hidden",
            ),
            helper.make_node(
                "ReduceSum",
                [f"{prefix}_masked_hidden"],
                [f"{prefix}_pooled_sum"],
                name=f"{prefix}_sum_hidden",
                axes=[1],
                keepdims=0,
            ),
            helper.make_node(
                "ReduceSum",
                [f"{prefix}_mask_f32"],
                [f"{prefix}_token_count"],
                name=f"{prefix}_sum_mask",
                axes=[1],
                keepdims=1,
            ),
            helper.make_node(
                "Div",
                [f"{prefix}_pooled_sum", f"{prefix}_token_count"],
                [f"{prefix}_pooled"],
                name=f"{prefix}_mean",
            ),
            helper.make_node(
                "Mul",
                [f"{prefix}_pooled", f"{prefix}_pooled"],
                [f"{prefix}_squared"],
                name=f"{prefix}_square",
            ),
            helper.make_node(
                "ReduceSum",
                [f"{prefix}_squared"],
                [f"{prefix}_norm_squared"],
                name=f"{prefix}_sum_squares",
                axes=[1],
                keepdims=1,
            ),
            helper.make_node(
                "Sqrt",
                [f"{prefix}_norm_squared"],
                [f"{prefix}_norm"],
                name=f"{prefix}_sqrt_norm",
            ),
            helper.make_node(
                "Max",
                [f"{prefix}_norm", epsilon_name],
                [f"{prefix}_safe_norm"],
                name=f"{prefix}_clamp_norm",
            ),
            helper.make_node(
                "Div",
                [f"{prefix}_pooled", f"{prefix}_safe_norm"],
                ["sentence_embedding"],
                name=f"{prefix}_normalize",
            ),
        ]
    )

    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info(
            "sentence_embedding", TensorProto.FLOAT, ["batch_size", 384]
        )
    )
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError:
        if original_checker_error is None:
            raise
        print(
            "[e5-pooling] preserving upstream checker warning: "
            f"{str(original_checker_error).splitlines()[0]}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    onnx.save(model, str(temporary), save_as_external_data=False)
    os.replace(temporary, destination)
    print(
        f"[e5-pooling] wrote {destination} "
        f"({destination.stat().st_size / 1024 / 1024:.1f} MiB)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("workdir/models"),
        help="Local model store root (default: workdir/models)",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_dir = args.model_dir / "multilingual-e5-small-onnx" / "onnx"
    for name in DEFAULT_GRAPHS:
        source = graph_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"missing source graph: {source}")
        append_pooling(source, pooled_path(source), args.force)


if __name__ == "__main__":
    main()
