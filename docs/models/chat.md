# Chat

| Model | Get it | Runs on |
| --- | --- | --- |
| `qwen3-4b-instruct-2507-int4-onnx` | download, [ModaLeap/qwen3-4b-instruct-2507-onnx](https://huggingface.co/ModaLeap/qwen3-4b-instruct-2507-onnx) (~2.3 GB) | NVIDIA GPU (4 GB+ VRAM) |
| `hy-mt2-1.8b-int4-onnx` | download, [ModaLeap/hy-mt2-1.8b-onnx](https://huggingface.co/ModaLeap/hy-mt2-1.8b-onnx) (~1.1 GB) | NVIDIA GPU (2.5 GB+ VRAM) |

[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
(Apache-2.0) at `cdbee75f17c01a7cc42f958dc650907174af0554`, exported INT4 by
the onnxruntime-genai model builder
([spec](../../configs/providers/chat/qwen3-chat.yaml)). It is also the realtime
voice's chat model. Other Qwen3-family decoder exports with the same inputs
and outputs work the same way.

## Hy-MT2-1.8B

[tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B) (Apache-2.0)
at `9a341cd1b679d3efd23b46e847b01745a71ed792`, Tencent Hunyuan's translation
model: 33 languages, terminology and background information in the prompt
([spec](../../configs/providers/chat/hy-mt2.yaml)). It runs on the same
adapter, so a translator that sends each line after the earlier ones only
prefills the new line. Prompt it with the model card's templates (one user
turn, no system prompt), e.g. `将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n<text>`.
`auto_select: false`: used where a request names it. About 2.1 GB of GPU
memory with a 4,096-token cache (~64 KB per token).

## Packages

Download either with `download_model`. Both were built by the
onnxruntime-genai model builder:

```bash
python -m onnxruntime_genai.models.builder -i <src> -o <dir> \
  -p int4 -e cuda --extra_options int4_algo_config=rtn_last prune_lm_head=true
```

(Hy-MT2: the builder reads `rope_theta`, which Hy-MT2 keeps under
`rope_parameters`; set it on the config before building.) The published
packages name the graph `model_int4.onnx` (`genai_config.json` says so) and
hold `genai_config.json`, `tokenizer.json` and `chat_template.jinja`. Keep
`rtn_last` (int8 LM head shared with the
embedding): with `k_quant*` the model tends to end tool calls without
`</tool_call>`. The decoder graphs need ONNX Runtime 1.30 or newer on CUDA
(older builds produce wrong tokens).

## Calling it

`POST /v1/chat/completions`, OpenAI style:

```bash
curl http://127.0.0.1:17890/v1/chat/completions -H 'content-type: application/json' \
  -H "authorization: Bearer $TOKEN" \
  --data '{"model":"qwen3-4b-instruct-2507-int4-onnx","stream":true,
           "messages":[{"role":"user","content":"Hello"}]}'
```

- `stream: true` answers with server-sent events; `stream_options.include_usage`
  adds a usage chunk, and the last chunk carries `timings` (`prefill_ms`,
  `first_token_ms`, `decode_ms`). A client that disconnects stops generation.
- `tools` / `tool_calls` / `tool` messages as in OpenAI; the model's
  `<tool_call>` blocks become `tool_calls`.
- Sampling: `temperature` (0 = greedy), `top_p`, `top_k`, `presence_penalty`,
  `frequency_penalty`, `seed`, `stop`, `max_tokens`, `logit_bias`, plus
  `tool_call_bias` (offset on `<tool_call>`) and `tool_bias` (tool name →
  offset on the first token of that name).
- The KV cache is kept between requests: a request prefills only what follows
  its common prefix with the previous one (system prompt, tools, earlier
  turns); `usage.prompt_tokens_details.cached_tokens` says how much.

Generic task: `chat.complete` with `params.messages`, `params.tools`,
`params.options`.

## Options

| Metadata | Default | Meaning |
| --- | --- | --- |
| `max_context` | 8192 | KV cache capacity in tokens, allocated at load (~144 KB/token) |

How the prompt, cache and decoding work: [implementation notes](../implementation-notes.md#chat-completion-qwen3).
