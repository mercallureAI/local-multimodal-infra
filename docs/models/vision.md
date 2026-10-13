# Vision: object detection and depth

| Model | Task | Get it | Runs on |
| --- | --- | --- | --- |
| `yolo11n.onnx` | object detection, 80 COCO classes | download (~11 MB) | CPU or CUDA |
| `depth-anything-v2-metric-indoor-small-onnx` | monocular metric depth (meters) | local export (~99 MB) | CPU or CUDA |

Both have a per-frame HTTP endpoint that takes the image (PNG, JPEG, BMP) as
the request body, for calling once per video or game frame.

## `yolo11n.onnx`

YOLO11n from [aaurelions/yolo11n.onnx](https://huggingface.co/aaurelions/yolo11n.onnx)
at `f46d9b72aa9a0f02bc00484446e2310b1a549bce`, with the COCO labels from
[Ultralytics](https://github.com/ultralytics/ultralytics/blob/eba96641b5cea142e21641909d6400fef7134244/ultralytics/cfg/datasets/coco.yaml)
(Ultralytics YOLO is AGPL-3.0;
[spec](../../configs/providers/detect/yolo.yaml)).

```bash
curl --data-binary @frame.jpg -H "authorization: Bearer $TOKEN" \
  'http://127.0.0.1:17890/v1/detect/objects?model=yolo11n.onnx'
# {"objects": [{"label", "confidence", "bbox"}]}
```

Also MCP / JSON-RPC `object_detect` (`image` file ref) and the generic task
`object.detect`.

## `depth-anything-v2-metric-indoor-small-onnx`

[Depth Anything V2 Metric Indoor Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf)
(ViT-S fine-tuned on indoor Hypersim, up to 20 m, Apache-2.0) at
`8078d68a9c75a972131914f6afd0c1723be0da7f`
([spec](../../configs/providers/detect/depth-anything-v2.yaml)).

```bash
python -m scripts.local.depth_anything_export --size 308x546 \
  --output-dir workdir/models/depth-anything-v2-metric-indoor-small-onnx
```

The export has a fixed input size and images are stretched to it: `308x546`
suits 16:9 frames (such as 1280×720). For other aspect ratios export another
size with `--size`, or depth is skewed.

```bash
curl --data-binary @frame.jpg -H "authorization: Bearer $TOKEN" \
  'http://127.0.0.1:17890/v1/depth?cols=64&rows=36'
# {"cols", "rows", "max_depth", "depth": [meters, row by row from the top]}
```

Each cell is the mean depth of its area; `grid_cols` / `grid_rows` in the spec
metadata set the default grid (64×36). Generic task: `depth.estimate`
(`params.cols` / `params.rows`). A 1280×720 frame takes ~30 ms on an RTX 4090,
decoding included.
