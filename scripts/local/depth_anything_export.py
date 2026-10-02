"""Export Depth Anything V2 Metric Indoor Small to ONNX.

The model (DINOv2 ViT-S + DPT head, fine-tuned on Hypersim; 24.8M params,
Apache-2.0 as depth-anything/Depth-Anything-V2-Metric-Hypersim-Small) is
taken in its transformers form, pinned by revision, and exported for one
input size (``--size HxW``, multiples of 14): tracing bakes the sizes of the
position-embedding and output interpolations into the graph, so a dynamic
export would answer every size at the traced one. The graph takes
``pixel_values`` ``[1, 3, H, W]`` (ImageNet-normalized RGB) and returns
``depth`` ``[1, H, W]`` in metres (0..max_depth, 20 for indoor). 308x546
fits 16:9 game frames at a fraction of the standard 518 short side's cost.
The export is checked against PyTorch on an image, and ``config.json`` plus
``preprocessor_config.json`` are copied next to ``model.onnx``.

Run with an isolated dependency environment, for example::

    uv run --python 3.12 --with torch --with "transformers>=4.45" \\
      --with onnx --with onnxruntime --with pillow --with huggingface_hub \\
      python -m scripts.local.depth_anything_export --size 308x546 \\
        --output-dir workdir/models/depth-anything-v2-metric-indoor-small-onnx
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
REVISION = "8078d68a9c75a972131914f6afd0c1723be0da7f"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def pixel_values(image_path: Path, height: int, width: int):
    """The image as the model input: resized to ``height`` x ``width``
    (bicubic), ImageNet-normalized."""
    import numpy as np
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image.resize((width, height), Image.BICUBIC), dtype=np.float32) / 255.0
    array = (array - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return array.transpose(2, 0, 1)[None].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-model-dir", type=Path,
                        help="a local copy of the HF repo (default: download it)")
    parser.add_argument("--check-image", type=Path,
                        default=Path(__file__).resolve().parents[1] / "assets" / "depth-input.jpg")
    parser.add_argument("--size", default="308x546", help="input HxW, multiples of 14")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    height, width = (int(v) for v in args.size.split("x"))
    if height % 14 or width % 14:
        raise SystemExit("--size must be multiples of 14")

    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForDepthEstimation

    source = args.source_model_dir
    if source is None:
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(REPO_ID, revision=REVISION))
    model = AutoModelForDepthEstimation.from_pretrained(source).eval()

    class Depth(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            return self.inner(pixel_values=pixel_values).predicted_depth

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "model.onnx"
    # Checked before it takes the model's name.
    checked = args.output_dir / "model.onnx.unchecked"
    sample = torch.from_numpy(pixel_values(args.check_image, height, width))
    torch.onnx.export(
        Depth(model), (sample,), str(checked),
        input_names=["pixel_values"], output_names=["depth"],
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
    )
    for name in ("config.json", "preprocessor_config.json"):
        shutil.copy2(Path(source) / name, args.output_dir / name)

    session = ort.InferenceSession(str(checked), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = Depth(model)(sample).numpy()
    actual = session.run(["depth"], {"pixel_values": sample.numpy()})[0]
    diff = float(np.abs(actual - expected).max())
    print(f"[depth-export] {height}x{width}: depth {actual.min():.2f}..{actual.max():.2f} m, "
          f"max |onnx - torch| {diff:.5f}")
    del session  # its file is renamed or removed next
    if actual.shape != (1, height, width) or diff > 1e-2:
        checked.unlink(missing_ok=True)
        raise SystemExit("the export does not match PyTorch")
    checked.replace(out)
    print(f"[depth-export] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
