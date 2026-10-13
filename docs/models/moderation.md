# Content moderation models

`POST /v1/moderations` (OpenAI-compatible) checks text and images before they
are sent to a hosted LLM API. It combines these models, all local exports
except PP-OCRv5:

| Model | Input | Get it | Default |
| --- | --- | --- | --- |
| `qwen3guard-gen-0.6b-int4-onnx` | text | local export (~340 MB) | yes |
| `qwen3guard-gen-0.6b-onnx` | text, FP16, slightly more accurate (~2.7 GB VRAM) | local export (~1.2 GB) | only when named (`model`) |
| `freepik-nsfw-image-detector-onnx` | image (neutral / low / medium / high) | local export (~180 MB) | yes |
| `falconsai-nsfw-image-detection-onnx` | image (normal / nsfw) | local export | only when named (`nsfw_model`) |
| `ppocrv5-mobile-onnx` | text in images | download, see [OCR](ocr.md) | yes |

Together the defaults stay under 2 GB of GPU memory (1.63 GB peak); one image
takes ~120 ms end to end on an RTX 4090.

## Export

```bash
# Qwen3Guard-Gen-0.6B (Apache-2.0), INT4
hf download Qwen/Qwen3Guard-Gen-0.6B --revision fada3b2f655b89601929198343c94cd2f64d93cc --local-dir <src>
python -m onnxruntime_genai.models.builder -i <src> -o workdir/models/qwen3guard-gen-0.6b-int4-onnx \
  -p int4 -e cuda --extra_options int4_algo_config=rtn_last prune_lm_head=true
# FP16 instead: -o workdir/models/qwen3guard-gen-0.6b-onnx -p fp16 -e cuda --extra_options prune_lm_head=true

# NSFW classifiers (needs torch, timm, transformers, onnx, onnxruntime, safetensors, huggingface_hub)
python -m scripts.local.nsfw_export freepik --fp16 --output-dir workdir/models/freepik-nsfw-image-detector-onnx
python -m scripts.local.nsfw_export falconsai --output-dir workdir/models/falconsai-nsfw-image-detection-onnx
```

| Source | Revision | License |
| --- | --- | --- |
| [Qwen/Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) | `fada3b2f655b89601929198343c94cd2f64d93cc` | Apache-2.0 |
| [Freepik/nsfw_image_detector](https://huggingface.co/Freepik/nsfw_image_detector) | `15b85477e4fd2000db76ae9aae0f89a72f95e2e3` | MIT |
| [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) | `96cb0d0342c7afb80cab76ecc58b265fa44da256` | Apache-2.0 |

## Options

| Spec metadata | Model | Meaning |
| --- | --- | --- |
| `max_context` | Qwen3Guard | KV capacity (INT4: 2048, FP16: 4096) |
| `window_tokens` | Qwen3Guard | longer texts are judged in windows of this many tokens |
| `nsfw_labels` | NSFW | labels that add up to the NSFW probability (default `low + medium + high`; `[medium, high]` stops counting suggestive images) |
| `cuda_arena_same_as_requested`, `cuda_conv_max_workspace`, `cuda_release_memory_after_run` | all | keep CUDA memory small |

Per request: `threshold` (text, default 0.9), `nsfw_threshold` (0.5),
`nsfw_labels`, `categories`, `ocr`.

The API, how the verdict is read, and the measurements (ToxicChat recall 0.81
at 2.8% false positives, no false positives on 600 local technical documents)
are in [docs/moderation.md](../moderation.md).
