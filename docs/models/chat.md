# Chat

| Model | Get it | Runs on |
| --- | --- | --- |
| `qwen3-4b-instruct-2507-int4-onnx` | local export (~2.5 GB) | NVIDIA GPU (4 GB+ VRAM) |

[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
(Apache-2.0) at `cdbee75f17c01a7cc42f958dc650907174af0554`, exported INT4 by
the onnxruntime-genai model builder
([spec](../../configs/providers/chat/qwen3-chat.yaml)). It is also the realtime
voice's chat model. Other Qwen3-family decoder exports with the same inputs
and outputs work the same way.

## Export

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir <src>
python -m onnxruntime_genai.models.builder -i <src> -o workdir/models/qwen3-4b-instruct-2507-int4-onnx \
  -p int4 -e cuda --extra_options int4_algo_config=rtn_last prune_lm_head=true
```

The directory holds `model.onnx(.data)`, `genai_config.json`, `tokenizer.json`
and `chat_template.jinja`. Keep `rtn_last` (int8 LM head shared with the
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
