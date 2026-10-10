"""Export an NSFW image classifier to ONNX for the ``nsfw_vit`` adapter.

The graph takes ``pixels`` float32 ``[N, 3, S, S]`` in 0..1 (RGB, already
resized) and returns ``probs`` ``[N, C]``: the model's normalization and the
softmax are inside, so the runtime only resizes and scales by 1/255.
``nsfw_meta.json`` beside it gives ``size``, ``resize`` (``squash``: stretch
to S x S; ``center``: shorter side to S, centre square), ``labels`` and
``nsfw_labels`` (whose probabilities add up to the NSFW probability).

Models (pinned revisions):

- ``freepik``: Freepik/nsfw_image_detector, EVA02-base 448 (MIT); four
  levels neutral / low / medium / high (``low``: suggestive, e.g. swimwear).
- ``falconsai``: Falconsai/nsfw_image_detection, ViT-base 224 (Apache-2.0);
  normal / nsfw.
- ``marqo``: Marqo/nsfw-image-detection-384, ViT-tiny 384 (Apache-2.0);
  NSFW / SFW, centre crop.

The export is checked against PyTorch on random input (max |diff| < 1e-3)
before it takes the model's name. ``--fp16`` stores the weights and computes
in float16 (input and output stay float32): half the file and GPU memory,
checked within 2e-2 of the float32 probabilities. Run with an isolated environment::

    uv run --python 3.12 --with torch --with timm --with "transformers>=4.48" \\
      --with onnx --with onnxruntime --with safetensors --with huggingface_hub \\
      python -m scripts.local.nsfw_export freepik \\
        --output-dir workdir/models/freepik-nsfw-image-detector-onnx --fp16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = {
    "freepik": {
        "repo": "Freepik/nsfw_image_detector",
        "revision": "15b85477e4fd2000db76ae9aae0f89a72f95e2e3",
        "arch": "timm", "size": 448, "resize": "squash",
        "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711],
        "labels": ["neutral", "low", "medium", "high"], "nsfw_labels": ["low", "medium", "high"],
    },
    "falconsai": {
        "repo": "Falconsai/nsfw_image_detection",
        "revision": "96cb0d0342c7afb80cab76ecc58b265fa44da256",
        "arch": "hf_vit", "size": 224, "resize": "squash",
        "mean": [0.5] * 3, "std": [0.5] * 3,
        "labels": ["normal", "nsfw"], "nsfw_labels": ["nsfw"],
    },
    "marqo": {
        "repo": "Marqo/nsfw-image-detection-384",
        "revision": "0c26ec22111b83f106d72a55f611ec35962bcb65",
        "arch": "timm", "size": 384, "resize": "center",
        "mean": [0.5] * 3, "std": [0.5] * 3,
        "labels": ["NSFW", "SFW"], "nsfw_labels": ["NSFW"],
    },
}


def load(spec: dict, source: Path):
    import torch

    if spec["arch"] == "timm":
        import timm
        from safetensors.torch import load_file

        config = json.loads((source / "config.json").read_text())
        net = timm.create_model(config["architecture"], pretrained=False,
                                num_classes=config["num_classes"])
        state = load_file(source / "model.safetensors")
        net.load_state_dict({k.removeprefix("timm_model."): v.float() for k, v in state.items()})
        hf = False
    else:
        from transformers import AutoModelForImageClassification

        net = AutoModelForImageClassification.from_pretrained(source)
        hf = True
    labels = config_labels(source, hf)
    if labels != spec["labels"]:
        raise SystemExit(f"the checkpoint's labels {labels} are not {spec['labels']}")

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net.eval().float()
            self.register_buffer("mean", torch.tensor(spec["mean"]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(spec["std"]).view(1, 3, 1, 1))

        def forward(self, pixels):
            x = (pixels - self.mean) / self.std
            logits = self.net(pixel_values=x).logits if hf else self.net(x)
            return torch.softmax(logits.float(), -1)

    return Classifier().eval()


def config_labels(source: Path, hf: bool) -> list[str]:
    config = json.loads((source / "config.json").read_text())
    if hf:
        return [config["id2label"][str(i)] for i in range(len(config["id2label"]))]
    return config["label_names"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-model-dir", type=Path,
                        help="a local copy of the HF repo at the pinned revision (default: download)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true", help="float16 weights and compute")
    args = parser.parse_args()
    spec = MODELS[args.model]

    import numpy as np
    import onnxruntime as ort
    import torch

    source = args.source_model_dir
    if source is None:
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(spec["repo"], revision=spec["revision"],
                                        allow_patterns=["config.json", "model.safetensors"]))
    model = load(spec, Path(source))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "model.onnx"
    # Checked before it takes the model's name.
    checked = args.output_dir / "model.onnx.unchecked"
    size = spec["size"]
    sample = torch.rand(2, 3, size, size, generator=torch.Generator().manual_seed(0))
    torch.onnx.export(model, (sample,), str(checked), input_names=["pixels"],
                      output_names=["probs"], dynamic_axes={"pixels": {0: "n"}, "probs": {0: "n"}},
                      opset_version=args.opset, do_constant_folding=True, dynamo=False)
    tolerance = 1e-3
    if args.fp16:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        # Averages stay float32: EVA02's token mean (1024 tokens of values up
        # to ~250) overflows a float16 accumulator, NaN for ~5% of images.
        half = convert_float_to_float16(
            onnx.load(str(checked)), keep_io_types=True, op_block_list=["ReduceMean"])
        onnx.save(half, str(checked))
        tolerance = 2e-2
    session = ort.InferenceSession(str(checked), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = model(sample).numpy()
    actual = session.run(["probs"], {"pixels": sample.numpy()})[0]
    diff = float(np.abs(actual - expected).max())
    print(f"[nsfw-export] {args.model}: max |onnx - torch| {diff:.2e}")
    del session  # its file is renamed or removed next
    if actual.shape != expected.shape or diff > tolerance:
        checked.unlink(missing_ok=True)
        raise SystemExit("the export does not match PyTorch")
    meta = {k: spec[k] for k in ("size", "resize", "labels", "nsfw_labels")}
    meta.update(source=spec["repo"], revision=spec["revision"], precision="fp16" if args.fp16 else "fp32")
    (args.output_dir / "nsfw_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    checked.replace(out)
    print(f"[nsfw-export] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
