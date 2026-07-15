# IndexTTS latency benchmark

This is a **real-model target benchmark**, not a synthetic unit benchmark. Run a
warmed worker/controller on the target Arch Linux host, enable INFO logs, and
submit the same reference WAV with a short-long-short-short text sequence.
Keep requests continuous (or concurrent when measuring admission queueing) and
retain the canonical request ID returned/used by the client.

Start the controller in terminal 1 using the repository config:

```bash
RUST_LOG=local_controller=info cargo run -p local-cli --bin controller
```

For each setting, start the worker in terminal 2. This is a foreground process;
stop it with Ctrl-C before changing `intra`, then repeat for 1, 2, 4, 6, and 8:

```bash
intra=4
LOCAL_INDEXTTS_ORT_INTRA_THREADS=$intra \
LOCAL_INDEXTTS_ORT_INTER_THREADS=1 \
RUST_LOG=local_runtime=info,local_worker=info,local_adapter_index_tts=info \
cargo run -p local-cli --bin worker 2>&1 | tee "indextts-intra-${intra}.log"
```

Tracing currently uses the standard human-readable formatter, not JSON. The
provided parser handles its `field=value` output.

Set an absolute worker-visible reference path and use the checked-in OpenAI
speech route. This exact shell sequence sends short-long-short-short:

```bash
export REF_WAV=/absolute/path/to/reference-24k-mono.wav
texts=(
  'A short warmed latency sample.'
  'This is the deliberately longer benchmark request. It should contain enough ordinary prose to exercise natural segmentation and sustained decoding while remaining identical in every measured cycle.'
  'A short warmed latency sample.'
  'A short warmed latency sample.'
)
for text in "${texts[@]}"; do
  curl --fail-with-body -sS http://127.0.0.1:17890/v1/audio/speech \
    -H 'content-type: application/json' \
    --data "$(jq -n --arg text "$text" --arg ref "$REF_WAV" \
      '{model:"indextts-1.5-onnx",input:$text,reference_path:$ref}')" >/dev/null
done
```

With the worker log already recording, first send exactly one complete
short-long-short-short cycle to lazy-load/warm the model. Do not truncate or
rotate the active log. Then send at
least 20 measured cycles (80 requests), waiting for each response before the
next request when measuring model execution (wrap the loop above in
`for cycle in $(seq 1 20)`). Repeat with concurrent requests
separately if admission queueing is the subject. IndexTTS remains serial
(`max_concurrency: 1`); do not overlap its mutable sessions. The structured
events expose:

* runtime `queue_wait_ms`, model `acquire_ms`/`load_ms`, `execution_ms`, total;
* adapter frontend and reference read/A timing;
* per chunk B/C-initial/D/decode-loop/F timing, decode budget/steps/STOP and samples;
* WAV encode/write and worker handler total.

After stopping the worker, compute p50/p95 directly:

```bash
python3 scripts/local/summarize_indextts_latency.py \
  --discard-first 4 --expect-success 80 indextts-intra-4.log
```

`--discard-first 4` removes the first four terminal runtime requests—the warmup
sequence—from the primary sample. `--expect-success 80` fails unless exactly 80
complete successful synthesis requests remain. Failed and incomplete requests
are reported separately and never contribute to a percentile. The script joins
events by canonical request ID and reports p50/p95 for `queue_wait_ms`,
`execution_ms`, aggregate `decode_steps`, and real-time factor:
`execution_ms / (audio_samples / 24000 * 1000)`. Also inspect failures and split
kind/chunk count. A budget exhaustion is intentionally an error: never count it
as successful audio, crop output, or substitute silence.

Compare intra-thread values 1, 2, 4, 6, and 8 with inter-thread fixed at 1.
Choose only from target measurements; this change does not assert an unproven
new thread default. A near-zero queue wait plus high E time/steps indicates a
decode tail; high queue wait indicates same-model admission backlog.

## CUDA-resident E KV cache experiment

On eligible CUDA builds, IndexTTS E/E-prefill uses I/O Binding by default so the
48 FP32 KV cache tensors remain on the selected CUDA device between decode
steps. Set `LOCAL_INDEXTTS_RESIDENT_KV=0` for a host-cache A/B run or emergency
rollback. Accepted explicit false values are `0`, `false`, `no`, and `off`;
accepted true values are `1`, `true`, `yes`, and `on` (case-insensitive and
whitespace-trimmed). Unset, empty, and unknown values remain enabled; a
non-empty unknown value emits a warning so typos are visible.

The adapter enables the resident path only when E and E-prefill are
CUDA-selected without a whole-session CPU retry and use the same device ID.
CPU and other ineligible provider selections quietly use host caches. Actual
bound KV placement, type, shape, progression, and device ID are validated on
every generation. ORT may still intentionally assign ordinary shape/control
nodes to CPU; this is not the whole-session fallback reported by the provider
status. A setup/prefill failure before sampling transparently reruns the
existing host path, while a failure after sampling has started fails that
request and disables the resident path for later requests on the loaded
adapter. Keep `max_concurrency: 1`.

The current performance and device-wide 200 ms VRAM evidence comes from an
RTX4090 and does not validate the production RTX3060 12 GB margin. Monitor VRAM
and request failures after rollout; use `LOCAL_INDEXTTS_RESIDENT_KV=0` as the
immediate rollback while validating 12 GB deployment behavior.
