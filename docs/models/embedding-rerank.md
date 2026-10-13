# Text embedding and reranking

| Model | Task | Get it | Runs on |
| --- | --- | --- | --- |
| `multilingual-e5-small-onnx` | embedding, 384 dimensions | download | CPU or CUDA |
| `mmarco-minilm-l12-onnx` | reranking (cross-encoder) | download | CPU or CUDA |

Both are multilingual (Chinese and English included) and take up to 512
tokens per text.

## `multilingual-e5-small-onnx`

[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
(MIT) at `614241f622f53c4eeff9890bdc4f31cfecc418b3`: E5 `query:` / `passage:`
prefixes, mean pooling and L2 normalization
([spec](../../configs/providers/embedding/multilingual-e5-small.yaml)). On CPU
it runs the official qint8 graph, on CUDA the O4 graph.

Optional, faster: append the pooling to both graphs once after downloading
(the adapter picks up the `_pooled.onnx` files, else pools on the host):

```bash
uv run --with onnx --python 3.12 python -m scripts.local.e5_pooling_export --model-dir workdir/models
```

`POST /v1/embeddings` takes OpenAI input (a string or a list), plus
`input_type`: `passage` (default) or `query`; `encoding_format: base64` is
supported. MCP: task `text.embed`; JSON-RPC: `text_embed`. Benchmark:
`python -m scripts.local.benchmark_text_embeddings`.

## `mmarco-minilm-l12-onnx`

[cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
(Apache-2.0) at `1427fd652930e4ba29e8149678df786c240d8825`: scores each
query/document pair (sigmoid of the logit) and sorts them
([spec](../../configs/providers/rerank/mmarco-minilm-reranker.yaml)). CPU uses
the official AVX2 uint8 graph, CUDA the O4 graph.

`POST /rerank`, `/v1/rerank` or `/v2/rerank` (vLLM / Jina / Cohere style):

```json
{"model": "mmarco-minilm-l12-onnx", "query": "...", "documents": ["...", "..."], "top_n": 3}
```

returns `results[{index, document: {text}, relevance_score}]`. MCP: task
`text.rerank`; JSON-RPC: `text_rerank`.
