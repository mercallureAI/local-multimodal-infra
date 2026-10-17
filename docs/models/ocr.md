# OCR

| Model | For | Get it | Runs on |
| --- | --- | --- | --- |
| `ppocrv5-mobile-onnx` | text lines in photos, screens, game frames | download (~21 MB) | CPU or CUDA |
| `unlimited-ocr-onnx` | whole document pages, with layout and tables | local export | CUDA only (6 GB+ VRAM) |

## `ppocrv5-mobile-onnx`

PaddleOCR's PP-OCRv5 mobile detection and recognition (Chinese, English,
Japanese and more in one model, Apache-2.0) from
[ilaylow/PP_OCRv5_mobile_onnx](https://huggingface.co/ilaylow/PP_OCRv5_mobile_onnx)
at `f97b337b3ac256f9dffcac5fc53955082d919d58`, with the dictionary from
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/a38c087bcb2579f9ccc2068aea02ec893b1c2311/ppocr/utils/dict/ppocrv5_dict.txt)
([spec](../../configs/providers/ocr/ppocrv5-mobile.yaml)). Lines come back
top to bottom, left to right.

```bash
curl --data-binary @frame.png -H "authorization: Bearer $TOKEN" \
  'http://127.0.0.1:17890/v1/ocr/lines?model=ppocrv5-mobile-onnx'
# {"lines": [{"text", "confidence", "bbox"}]}
```

Generic task: `ocr.lines` (upload `image`). It also reads the text in images
sent to [content moderation](moderation.md).

| Metadata | Spec value | Meaning |
| --- | --- | --- |
| `det_limit_side_len` | 1280 | longest side for detection (adapter default 960); 1280 keeps far, small name tags in 720p frames |
| `rec_batch` | 32 | lines per recognition batch (adapter default 8) |
| `rec_batch_columns` | 6400 | padded columns per batch, bounds VRAM on text-dense screenshots |
| `det_thresh`, `box_thresh`, `unclip_ratio` | 0.3, 0.6, 1.5 | DB detection post-processing |
| `min_confidence` | 0.5 | lines below are dropped |

## `unlimited-ocr-onnx`

[baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) (DeepEncoder +
DeepSeek-V2 MoE decoder with R-SWA attention) at
`07dea832e22aefee32ad281d4b80551282e1c168`, exported with int8 experts
([spec](../../configs/providers/ocr/unlimited-ocr.yaml)). One page per request
(padded to 1024×1024). The result is the page's text, each layout block
preceded by `<|det|>category [x1, y1, x2, y2]<|/det|>` (coordinates 0–999),
tables as HTML.

Export (Python 3.11 with the upstream pins `torch==2.10.0` (CPU is enough),
`torchvision==0.25.0`, `transformers==4.57.1`, plus
`onnx onnxruntime-gpu==1.30.0 einops addict easydict safetensors pillow matplotlib`):

```bash
hf download baidu/Unlimited-OCR --revision 07dea832e22aefee32ad281d4b80551282e1c168 --local-dir <src>
python -m scripts.local.unlimited_ocr_export export --source <src> --out workdir/models/unlimited-ocr-onnx
```

`parity` compares a package with the PyTorch model token by token. The int8
experts run only on the CUDA kernels, so the worker must be built with
`--features cuda`; `--expert-precision fp16` exports FP16 experts instead
(~5.6 GB).

Calling: the task `ocr.recognize` with an `image` file (MCP and JSON-RPC), or
JSON-RPC `ocr_recognize` (`image` file ref or `image_path`). `max_new_tokens` (8192)
bounds the decode per page.

On an RTX 4090: ~145 tokens/s, a ~4.9 GB GPU memory peak; the 14-page paper takes
6.8 s per page against 35 s for the official transformers pipeline on the same
GPU, with identical text on 12 of 14 pages. Details:
[implementation notes](../implementation-notes.md#document-ocr-unlimited-ocr).
