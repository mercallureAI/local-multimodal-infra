"""Validate split-v2 IndexTTS E PyTorch/ONNX numerical parity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from indextts_export import IndexTtsE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-tts-project", required=True, type=Path)
    parser.add_argument("--source-model-dir", required=True, type=Path)
    parser.add_argument("--onnx-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    sys.path.insert(0, str(args.index_tts_project.resolve()))
    from indextts.infer import IndexTTS
    from indextts.utils import front as front_module

    original_load = front_module.TextNormalizer.load
    front_module.TextNormalizer.load = lambda self: None
    try:
        tts = IndexTTS(
            cfg_path=str(args.source_model_dir / "config.yaml"),
            model_dir=str(args.source_model_dir),
            is_fp16=False,
            device="cpu",
            use_cuda_kernel=False,
        )
    finally:
        front_module.TextNormalizer.load = original_load
    gpt = tts.gpt.eval()
    for parameter in gpt.parameters():
        parameter.requires_grad_(False)

    layers = int(gpt.layers)
    prefill_torch = IndexTtsE(gpt, layers, prefill=True).eval()
    decode_torch = IndexTtsE(gpt, layers, prefill=False).eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    prefill_ort = ort.InferenceSession(
        str(args.onnx_dir / "IndexTTS_E_Prefill.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    decode_ort = ort.InferenceSession(
        str(args.onnx_dir / "IndexTTS_E.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    torch.manual_seed(25851)
    hidden = torch.randn(1, 11, 1280, dtype=torch.float32) * 0.01
    mask = torch.ones(1, 11, dtype=torch.int64)
    with torch.no_grad():
        torch_outputs = prefill_torch(hidden, mask)
    ort_outputs = prefill_ort.run(None, {"hidden_state": hidden.numpy(), "attention_mask": mask.numpy()})
    maxima = compare_outputs("prefill", torch_outputs, ort_outputs)
    torch_cache = list(torch_outputs[: layers * 2])
    ort_cache = list(ort_outputs[: layers * 2])

    for step in range(args.steps):
        torch.manual_seed(25852 + step)
        hidden = torch.randn(1, 1, 1280, dtype=torch.float32) * 0.01
        mask = torch.ones(1, 12 + step, dtype=torch.int64)
        with torch.no_grad():
            torch_outputs = decode_torch(*torch_cache, hidden, mask)
        feeds = {}
        for index in range(layers):
            feeds[f"in_key_{index}"] = ort_cache[index * 2]
            feeds[f"in_value_{index}"] = ort_cache[index * 2 + 1]
        feeds["hidden_state"] = hidden.numpy()
        feeds["attention_mask"] = mask.numpy()
        ort_outputs = decode_ort.run(None, feeds)
        step_maxima = compare_outputs(f"decode_{step + 1}", torch_outputs, ort_outputs)
        maxima = tuple(max(left, right) for left, right in zip(maxima, step_maxima))
        torch_cache = list(torch_outputs[: layers * 2])
        ort_cache = list(ort_outputs[: layers * 2])
        expected = 12 + step
        assert ort_cache[0].shape == (1, 20, expected, 64), ort_cache[0].shape

    print(
        f"parity_ok steps={args.steps} max_cache_abs={maxima[0]:.9g} "
        f"max_hidden_abs={maxima[1]:.9g} max_logits_abs={maxima[2]:.9g}"
    )
    return 0


def compare_outputs(label: str, torch_outputs, ort_outputs) -> tuple[float, float, float]:
    arrays = [value.detach().cpu().numpy() for value in torch_outputs]
    assert len(arrays) == len(ort_outputs) == 50
    cache_error = max(float(np.max(np.abs(left - right))) for left, right in zip(arrays[:48], ort_outputs[:48]))
    hidden_error = float(np.max(np.abs(arrays[48] - ort_outputs[48])))
    logits_error = float(np.max(np.abs(arrays[49] - ort_outputs[49])))
    if cache_error > 2e-4 or hidden_error > 2e-4 or logits_error > 2e-3:
        raise AssertionError(
            f"{label} parity exceeded tolerance: cache={cache_error}, hidden={hidden_error}, logits={logits_error}"
        )
    return cache_error, hidden_error, logits_error


if __name__ == "__main__":
    raise SystemExit(main())
