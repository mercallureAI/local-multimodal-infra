# Content moderation

`POST /v1/moderations` checks what is about to be sent to a hosted LLM API
for content its usage policy forbids, on local models of 1B parameters or
less, in under 2 GB of GPU memory together (1.36 GB loaded, 1.63 GB peak):

| Input | Model | Task | Files |
| --- | --- | --- | --- |
| Text | `qwen3guard-gen-0.6b-int4-onnx` (Qwen3Guard-Gen-0.6B, INT4) | `text.moderate` | 340 MB |
| Image | `freepik-nsfw-image-detector-onnx` (EVA02-base 448, FP16) | `image.nsfw` | 180 MB |
| Text in an image | `ppocrv5-mobile-onnx`, then the text model | `ocr.lines` + `text.moderate` | 21 MB |

Also configured, used only when a request names them: `qwen3guard-gen-0.6b-onnx`
(the FP16 export, slightly more accurate, ~2.7 GB of VRAM) as `model`, and
`falconsai-nsfw-image-detection-onnx` (ViT-base 224) as `nsfw_model`.

## API

OpenAI's moderation request and response, plus a few fields:

```bash
curl http://127.0.0.1:17890/v1/moderations -H 'content-type: application/json' \
  -H 'authorization: Bearer <inference token>' \
  --data '{"input": [
    {"type": "text", "text": "..."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]}'
```

- `input`: a string (one result), an array of strings (one result each) or an
  array of `text` / `image_url` parts (one result for them all). Images are
  base64 `data:` URLs (PNG, JPEG, BMP, WebP, GIF's first frame); remote URLs
  are refused, never fetched. At most 16 images, 64 MB per request.
- Optional: `model` (the text model), `nsfw_model`, `ocr_model`, `ocr`
  (default `true`), `threshold` (default 0.9), `nsfw_threshold` (default 0.5),
  `nsfw_labels` (the classifier labels that add up to the NSFW probability;
  default the model's `metadata.nsfw_labels`, e.g. `["medium", "high"]` to
  stop counting suggestive images; an unknown label is a 400 naming the
  model's labels), `categories` (the text categories that flag, default all
  nine; a text judged unsafe only for other categories is not flagged, one
  judged unsafe without a category still is; applies to text read in images,
  not to the NSFW classifier).
- A result has `flagged`, `categories`, `category_scores` and
  `category_applied_input_types` over `violent`, `illegal`, `sexual`, `pii`,
  `self-harm`, `unethical`, `political`, `copyright`, `jailbreak`
  (Qwen3Guard's categories; images feed `sexual`), and beyond OpenAI's
  fields `safety` (`safe` / `controversial` / `unsafe` of the least safe
  text) and `images` (`nsfw`, label `scores`, `ocr_chars`, `ocr_skipped`,
  `ocr_safety`, `ocr_categories`). The text read in images is never returned.
- A result of an array of parts also has `parts`: each part's own verdict in
  input order, `{"type": "text" | "image_url", "flagged", "categories"}`, so
  a caller can drop the flagged parts and keep the rest (AstrBot does this
  with tool results). An image part is flagged by its NSFW probability
  (`sexual`) or by the text read in it.
- A text is flagged when `1 - p(safe) > threshold`; an image when its NSFW
  probability (Freepik: `low + medium + high`) is above `nsfw_threshold`, or
  when its text is flagged. An image flagged as NSFW is not read
  (`ocr_skipped`). A category is flagged when a flagged text names it; its
  score is the highest `1 - p(safe)` among texts naming it (or the NSFW
  probability, for `sexual`).
- A model output that is not a finite number (a float16 overflow, say) fails
  the request rather than passing as safe.

The generic tasks work alone too: `text.moderate` (`params.input`: a string
or strings) and `image.nsfw` (upload `image`).

## How the text model is run

The `qwen3_guard` adapter renders Qwen3Guard's own moderation template with
the text as the user turn and appends the answer's start, `Safety:`. One
prefill gives the next token's logits; the softmax over ` Safe`, ` Unsafe`
and ` Cont` (the first token of ` Controversial`) is the verdict, so no
tokens are generated for a safe text. A text that is not judged safe decodes
its `Categories:` line greedily (a few tokens). The model runs on the
`qwen3_chat` adapter's decoder (its shared KV cache keeps the template's
~290-token preamble between requests, so a short text prefills a few dozen
tokens). The logit verdict agrees with the model's generated first line on
50 of 50 checked prompts. This is the "decision model" pattern of TypeSafe's
Jev and its open copies (a distribution over fixed answers from one forward
pass, no text generated), on a model trained for safety.

A text longer than `window_tokens` (1536 for INT4, 2048 for FP16) is judged
in windows of that many tokens, 64 shared between neighbours, and reported by
its least safe window.

Only prompt moderation (the user turn) is used. Qwen3Guard's categories are
fixed by its training; its `PII` category is about asking for or exposing
someone's personal information, not a PII detector.

## Keeping it under 2 GB

| Step | Worker VRAM: loaded (after the benchmark) |
| --- | --- |
| FP16 guard (4096-token cache), FP32 NSFW, PP-OCR as shipped | ~4.1 GB (5.1 GB) |
| INT4 guard (2048-token cache), FP16 NSFW, OCR batches capped at 6400 columns | ~2.3 GB (2.6 GB) |
| + `cuda_arena_same_as_requested`, `cuda_conv_max_workspace: false` | ~1.7 GB (2.4 GB) |
| + `cuda_release_memory_after_run` | **1.36 GB (1.38 GB; 1.63 GB peak, polled every 50 ms)** |

Per model, with the last row's settings: the guard 1.04 GB (CUDA context,
cuBLAS and 230 MB of KV cache included), the NSFW classifier 0.26 GB, PP-OCR
0.05 GB. The memory settings are model spec metadata read by `backend-ort`
(`CudaMemoryOptions::from_metadata`, `release_memory_after_run`): the arena
grows by what is asked instead of doubling, cuDNN does not take its largest
workspace, and each run's activations go back to the driver afterwards (for
the text model only after prefills of over 512 tokens). PP-OCR's
`rec_batch_columns` caps lines x padded width per recognition batch; name
tags (a few hundred pixels each) still go in one batch.

The FP16 NSFW export keeps its average pooling in FP32: EVA02 averages 1024
tokens with values up to ~250, which overflows a float16 accumulator and gave
NaN for ~5% of images (each of which would have passed). With that one node
in FP32 it agrees with the FP32 export on every one of 1500 test images
(largest difference 0.004) and runs in 8 ms instead of 11.

## Measurements

RTX 4090, ONNX Runtime 1.30 CUDA. Research scripts and raw scores:
`D:\AI_WorkSpace\moderation-research` (outside the repository).

### Text (Qwen3Guard-Gen-0.6B)

`1 - p(safe) > t` on three sets: XSTest (250 safe prompts that look unsafe,
200 unsafe), ToxicChat 0124 test (5083 real user prompts, 362 toxic), and
600 chunks of local technical documents and code (all benign; the traffic
these checks actually see). FP16:

| Threshold | ToxicChat P / R / F1 / FPR | XSTest F1 / FPR | Technical docs FPR |
| --- | --- | --- | --- |
| 0.5 | 0.43 / 0.97 / 0.59 / 10.0% | 0.86 / 22% | 0 / 600 |
| 0.9 (default) | 0.70 / 0.84 / 0.76 / 2.8% | 0.87 / 9.6% | 0 / 600 |
| 0.95 | 0.76 / 0.77 / 0.77 / 1.9% | 0.87 / 6.4% | 0 / 600 |
| `p(unsafe) > 0.5` (loose) | 0.82 / 0.71 / 0.76 / 1.2% | 0.85 / 5.2% | 0 / 600 |

ROC AUC: 0.985 (ToxicChat), 0.959 (XSTest). The FP16 export equals PyTorch
BF16 within noise.

Quantizations (onnxruntime-genai builder, `prune_lm_head=true`), at the
default threshold:

| Export | Files | ToxicChat R / FPR / F1 | XSTest F1 / FPR | Technical docs FPR |
| --- | --- | --- | --- | --- |
| FP16 | 1156 MB | 0.84 / 2.8% / 0.763 | 0.869 / 9.6% | 0 |
| INT8 (embedding left FP16) | 920 MB | 0.83 / 2.7% / 0.764 | 0.863 / 9.6% | 0 |
| INT4 `k_quant_mixed`, block 32 | 475 MB | 0.82 / 3.0% / 0.745 | 0.882 / 7.2% | 0 |
| INT4 `k_quant_last`, block 32 | 424 MB | 0.86 / 3.9% / 0.726 | 0.867 / 10.4% | 0 |
| INT4 `rtn_last`, block 32 | 413 MB | 0.83 / 3.6% / 0.724 | 0.870 / 8.8% | 0 |
| **INT4 `rtn_last` (default)** | **339 MB** | 0.81 / 2.8% / 0.746 | 0.895 / 8.4% | 0 |

The default INT4 export costs ~0.02 F1 on ToxicChat (recall 0.81 against
0.84 at the same false-positive rate) and is no worse on XSTest.

### Images

NSFW classifiers on everyday images (flagged at 0.5). The flagged images
were not reviewed (COCO has beach and swimwear photos), so these counts are
upper bounds of false positives:

| Model | COCO val2017 (5000 photos) | ScreenSpot (1272 UI screenshots) | GPU time |
| --- | --- | --- | --- |
| Freepik EVA02-base 448 (default) | 26 (0.52%) | 0 | ~11 ms (FP32), ~8 ms (FP16) |
| Falconsai ViT-base 224 | 6 (0.12%) | 0 | ~4 ms |
| Marqo ViT-tiny 384 (not configured) | 24 (0.48%) | 0 | ~4 ms |

Recall on `deepghs/nsfw_detect` (revision
`ac763cb1e1557225168be3b6b6b1ee864c17bc36`; 300 random images per
category, flagged at 0.5; its `neutral` and `drawings` are scraped and
noisily labelled, so their flags are upper bounds of false positives):

| Category | Freepik (`low+medium+high`, default) | Freepik (`medium+high`) | Falconsai | Marqo |
| --- | --- | --- | --- | --- |
| porn | 99.3% | 99.3% | 97.3% | 93.0% |
| hentai | 90.0% | 90.0% | 93.3% | 99.3% |
| sexy (suggestive, e.g. lingerie, swimwear) | 99.7% | 43.7% | 59.0% | 69.3% |
| neutral (should pass) | 6.3% | 4.0% | 1.0% | 6.0% |
| drawings (should pass) | 4.0% | 4.0% | 0.3% | 48.3% |

Freepik's model is the default: it catches explicit photos best and is the
only one that separates suggestive images (`low`) from explicit ones; it
misses 10% of explicit anime (judged `neutral`). Whether suggestive images
count is a policy choice: `metadata.nsfw_labels: [medium, high]` in the
model spec stops counting `low` for every request, `"nsfw_labels":
["medium", "high"]` for one. Marqo's model flags half of all drawings
and is not configured. One real image that drew a provider warning: NSFW 1.0
on all three.

Text in images: the 450 XSTest prompts rendered as screenshots and sent as
images only (OCR + the INT4 guard) reach the text path's verdict 440 times
(97.8%; 445 with the FP16 guard).

### Resources and latency

RTX 4090, release build, warm models, the default (2 GB) configuration;
client-side p50 of 20 requests to `/v1/moderations` (in brackets: the
worker's execution time). `moderation-research/bench_service.py` measures
it, polling GPU memory every 50 ms for the peak.

| Model | VRAM | First request (load) |
| --- | --- | --- |
| Qwen3Guard-Gen-0.6B INT4, 2048-token KV cache (+ CUDA context) | 1.04 GB | ~1.1 s |
| Freepik NSFW FP16 | 0.26 GB | 0.6 s |
| PP-OCRv5 mobile | 0.05 GB | 0.5 s |
| All three, peak over every request below | 1.63 GB | |

| Request | p50 | p95 |
| --- | --- | --- |
| Text, short safe | 5 ms (4) | 19 ms |
| Text, short unsafe (categories decoded) | 19 ms (17) | 33 ms |
| Text, ~400 tokens | 6 ms (4) | 16 ms |
| Text, ~2k tokens (more than one window) | 40 ms (38) | 55 ms |
| Text, ~8k tokens (several windows) | 157 ms (144) | 184 ms |
| 16 short texts in one request | 107 ms (104) | 133 ms |
| Photo 1280x720, NSFW only | 45 ms (29) | 66 ms |
| Photo 1280x720, NSFW + OCR | 121 ms (NSFW 30, OCR 52, guard 28) | 144 ms |
| Text-dense screenshot 1920x1080 | 583 ms (OCR 527) | 609 ms |
| 4 such screenshots in one request | 2.28 s | 2.47 s |

NSFW runs before OCR (so a flagged image is not read); the NSFW time is
mostly decoding and resizing on the CPU. OCR dominates text-heavy images:
recognition of long lines, one image at a time (`max_concurrency: 1`).
Moderation tasks skip the job records the generic task flow writes (SQLite,
~20 ms per request on Windows), as the direct frame endpoints do.

### RTX 3060 12 GB (Linux, Docker)

The same build and models on the deployment server, measured over the LAN
from another machine (so each request includes ~15 ms of network and HTTP):
the worker holds 1.14 GB with all three models loaded (1.21 GB the highest
seen while running every request below). Text verdicts equal the RTX 4090's
(6132 of 6133 flags; probabilities within 0.011), NSFW recall per category is
the same to within one image, the 450 rendered screenshots agree with the
text path 440 times, and the real image that drew a provider warning is
flagged (`sexual`, OCR skipped).

| Request | p50 |
| --- | --- |
| Text, short | 31 ms |
| Text, ~2k tokens | 170 ms |
| Text, ~8k tokens | 606 ms |
| 16 short texts | 234 ms |
| Photo 1280x720, NSFW only | 61 ms |
| Photo 1280x720, NSFW + OCR | 155 ms |
| Text-dense screenshot 1920x1080 | 1.03 s |
| 4 such screenshots | 3.90 s |

About 50 texts per second in batches of 32.

## Alternatives considered

- **Jev-style decision models.** TypeSafe's Jev (September 2026) answers
  typed questions with probability distributions instead of text; it is a
  hosted API only (no weights), so it cannot run here. Its open copies are
  general-purpose and not trained for safety: Kev-0.8B (LoRA + pointer head
  on Qwen3.5-0.8B, Apache-2.0) is the weakest of its family against Jev
  (accuracy ~0.65 vs 0.86 on its authors' panel) and larger than
  Qwen3Guard-0.6B; NanoJev (Qwen3-0.6B + choice head, MIT) was trained on
  game-playing decisions. Either would need safety training to replace
  Qwen3Guard, whose verdict is already read the same way (one forward pass).
- **Encoder classifiers** (GLiGuard, 0.3B) are smaller still but publish no
  Chinese results.

## Reproducing

```bash
# Exports (see configs/providers/moderation/*.yaml)
python -m onnxruntime_genai.models.builder -i <Qwen3Guard-Gen-0.6B> -o workdir/models/qwen3guard-gen-0.6b-int4-onnx \
  -p int4 -e cuda --extra_options int4_algo_config=rtn_last prune_lm_head=true
python -m scripts.local.nsfw_export freepik --fp16 --output-dir workdir/models/freepik-nsfw-image-detector-onnx

# Real-model tests (the guard test compares with the FP16 export's Python run)
LOCAL_QWEN3GUARD_MODEL_DIR=workdir/models/qwen3guard-gen-0.6b-onnx LOCAL_TEST_PROVIDER_ORDER=cuda,cpu \
  cargo test --release -p local-adapter-qwen3guard --features cuda real_model -- --nocapture
LOCAL_NSFW_VIT_MODEL_DIR=workdir/models/freepik-nsfw-image-detector-onnx LOCAL_TEST_PROVIDER_ORDER=cuda,cpu \
  cargo test --release -p local-adapter-nsfw-vit --features cuda real_model -- --nocapture

# Service
python -m scripts.local.smoke --tests moderation --workdir ./workdir --model-dir ./workdir/models
```
