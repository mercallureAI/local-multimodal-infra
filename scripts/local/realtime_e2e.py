"""End-to-end check of /v1/realtime on a local-multimodal-infra build.

Starts the release controller + worker (harness helpers), opens a realtime
session, plays spoken test utterances at real-time pace and reports what the
bot did and how fast it started answering. The bot's audio is saved. Needs
numpy, librosa and websockets >= 14 (``additional_headers``).

    python realtime_e2e.py --infra D:/AI_WorkSpace/local-multimodal-infra-chat \
        --model-dir D:/AI_WorkSpace/local-multimodal-infra/workdir/models [--group] [--mode audio]

``--mode audio`` checks the audio-only protocol: the script plays the client's
part, answering each utterance that wants a reply with a canned text streamed
in small deltas (as a cloud model would).
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import wave
from pathlib import Path

import librosa
import numpy as np
import websockets

# The spoken test utterances (16 kHz mono WAV), set by --audio-dir.
ROUTER_AUDIO = Path(".")
# (wav, what the bot should do)
PRIVATE_STEPS = [
    ("r15.wav", "reply"),  # 小乐，一百二十三加四百五十六等于多少？
    ("r07.wav", "backend_task"),  # 小乐，现在几点了？
    ("r13.wav", "reply"),  # 小乐，给大家讲个笑话吧。
]
GROUP_STEPS = [
    ("r00.wav", "silence"),  # 老王，晚上一起去吃火锅吗？
    ("r15.wav", "reply"),
    ("r04.wav", "silence"),  # 小李你帮我看一下左边有没有人。
    ("r07.wav", "backend_task"),
]
CHUNK = 320  # 20 ms at 16 kHz
REF_AUDIO = Path("scripts/assets/tts-input-mon3tr.wav")
# What REF_AUDIO says (Qwen3-TTS in-context cloning), set by --ref-text.
REF_TEXT = ""
# Audio mode: what the client answers (a long one to talk over).
SHORT_REPLY = "好的，你刚才说的是：{text}"
LONG_REPLY = (
    "从前有一座山，山里有一座庙，庙里住着一个老和尚和一个小和尚。"
    "有一天老和尚对小和尚说，我给你讲个故事吧，从前有一座山，山里有一座庙。"
    "庙里住着一个老和尚和一个小和尚，老和尚又开始讲故事了，讲的还是同一个故事。"
)


def pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


class Client:
    def __init__(self, ws, audio_mode: bool = False) -> None:
        self.ws = ws
        self.audio_mode = audio_mode
        self.long_reply = False
        self.replies = 0
        # Audio mode: the reply task and response id of each utterance id.
        self.reply_tasks: dict = {}
        self.events: list[tuple[float, dict]] = []
        self.audio = bytearray()
        self.audio_times: list[float] = []
        self.first_audio_after: float | None = None
        self.mark = 0.0

    async def reader(self) -> None:
        async for message in self.ws:
            now = time.perf_counter()
            if isinstance(message, bytes):
                if self.mark and self.first_audio_after is None:
                    self.first_audio_after = now - self.mark
                self.audio += message
                self.audio_times.append(now)
                continue
            event = json.loads(message)
            self.events.append((now, event))
            kind = event["type"]
            if kind == "tool.call":
                print(f"  [{kind}] {event['name']} {event['arguments']} heard={event['heard']!r}")
                # The backend answers after a moment.
                asyncio.get_running_loop().call_later(
                    1.0,
                    lambda call_id=event["call_id"]: asyncio.ensure_future(
                        self.ws.send(
                            json.dumps({"type": "tool.result", "call_id": call_id, "output": "现在是下午三点四十分。"})
                        )
                    ),
                )
            elif kind in ("input.transcript", "response.text", "response.cut", "error"):
                print(f"  [{kind}] {event.get('speaker', '')} {event.get('text', event.get('message', ''))}")
                if self.audio_mode and kind == "input.transcript" and event.get("respond"):
                    print(f"    id={event.get('id')} replaces={event.get('replaces')}")
                    # An utterance that continues an earlier one replaces it:
                    # the earlier reply goes.
                    old = self.reply_tasks.pop(event.get("replaces"), None)
                    if old is not None:
                        old[0].cancel()
                        await self.ws.send(json.dumps({"type": "response.cancel", "response_id": old[1]}))
                    self.replies += 1
                    response_id = f"reply-{self.replies}"
                    task = asyncio.ensure_future(self.reply(event["text"], response_id))
                    self.reply_tasks[event.get("id")] = (task, response_id)
            elif kind in ("response.done", "state"):
                print(f"  [{kind}] {json.dumps({k: v for k, v in event.items() if k != 'type'}, ensure_ascii=False)}")

    async def reply(self, heard: str, response_id: str) -> None:
        """Audio mode: streams an answer, a few characters every 30 ms."""
        text = LONG_REPLY if self.long_reply else SHORT_REPLY.format(text=heard)
        await asyncio.sleep(0.3)  # the model's first token
        for i in range(0, len(text), 4):
            await self.ws.send(
                json.dumps({"type": "response.delta", "response_id": response_id, "text": text[i : i + 4]})
            )
            await asyncio.sleep(0.03)
        await self.ws.send(json.dumps({"type": "response.end", "response_id": response_id}))

    async def speak(self, samples: np.ndarray, tail_seconds: float) -> None:
        rng = np.random.default_rng(0)
        audio = np.concatenate([samples, np.zeros(int(16000 * tail_seconds), np.float32)])
        audio = audio + rng.normal(0, 10 ** (-60 / 20), len(audio)).astype(np.float32)
        start = time.perf_counter()
        for i in range(0, len(audio), CHUNK):
            if i >= len(samples) and not self.mark:
                self.mark = time.perf_counter()
                self.first_audio_after = None
            await self.ws.send(pcm16(audio[i : i + CHUNK]))
            wait = start + (i + CHUNK) / 16000 - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)


async def session(url: str, token: str, group: bool, out_wav: Path, audio_mode: bool = False) -> None:
    ref = REF_AUDIO.read_bytes()
    async with websockets.connect(
        url, additional_headers={"Authorization": f"Bearer {token}"}, max_size=None, ping_interval=None
    ) as ws:
        client = Client(ws, audio_mode)
        reader = asyncio.create_task(client.reader())
        t = time.perf_counter()
        await ws.send(
            json.dumps(
                {
                    "type": "session.start",
                    "config": {
                        "name": "小乐",
                        "aliases": ["小月"],
                        "group": group,
                        "speaker": "测试员",
                        "ref_audio": base64.b64encode(ref).decode(),
                        **({"ref_text": REF_TEXT} if REF_TEXT else {}),
                        "tool_filler": "好的，我查一下。",
                        "mode": "audio" if audio_mode else "cascade",
                    },
                }
            )
        )
        while not any(e["type"] in ("session.started", "error") for _, e in client.events):
            await asyncio.sleep(0.1)
        print(f"session started in {time.perf_counter() - t:.1f}s: {client.events[-1][1]}")
        for name, want in GROUP_STEPS if group else PRIVATE_STEPS:
            samples, _ = librosa.load(str(ROUTER_AUDIO / name), sr=16000, mono=True)
            print(f"-> {name} (want {want})")
            client.mark = 0.0
            await client.speak(samples.astype(np.float32), tail_seconds=8.0 if want == "backend_task" else 5.0)
            if client.first_audio_after is not None:
                print(f"  first audio {client.first_audio_after * 1000:.0f} ms after the utterance's audio ended")
            else:
                print("  no audio")
        # Barge-in: talk over a long answer.
        samples, _ = librosa.load(str(ROUTER_AUDIO / "r13.wav"), sr=16000, mono=True)
        long_answer, _ = librosa.load(str(ROUTER_AUDIO / "r18.wav"), sr=16000, mono=True)
        print("-> barge-in: ask for a joke, then talk over it")
        client.mark = 0.0
        client.long_reply = True
        await client.speak(samples.astype(np.float32), tail_seconds=2.5)
        cuts_before = sum(1 for _, e in client.events if e["type"] == "response.cut")
        await client.speak(long_answer.astype(np.float32), tail_seconds=4.0)
        cuts = sum(1 for _, e in client.events if e["type"] == "response.cut") - cuts_before
        print(f"  response.cut events after talking over: {cuts}")
        if audio_mode:
            dones = [e for _, e in client.events if e["type"] == "response.done"]
            print(f"  response.done events: {len(dones)}, cut: {sum(1 for e in dones if e['cut'])}")
        for task, _ in client.reply_tasks.values():
            task.cancel()
        await ws.send(json.dumps({"type": "session.stop"}))
        await asyncio.sleep(0.5)
        reader.cancel()
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(bytes(client.audio))
    print(f"bot audio {len(client.audio) / 48000:.1f}s saved to {out_wav}")


def main() -> None:
    global ROUTER_AUDIO, REF_AUDIO, REF_TEXT
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True, help="the spoken test utterances (r00.wav ...)")
    ap.add_argument("--model-dir", default="./workdir/models")
    ap.add_argument("--ref", default=str(REF_AUDIO), help="the reference voice (WAV)")
    ap.add_argument("--ref-text", default="", help="what --ref says (Qwen3-TTS in-context cloning)")
    ap.add_argument("--group", action="store_true")
    ap.add_argument("--mode", choices=["cascade", "audio"], default="cascade")
    ap.add_argument("--out", default="workdir/data/realtime_bot.wav")
    args = ap.parse_args()
    ROUTER_AUDIO, REF_AUDIO, REF_TEXT = Path(args.audio_dir), Path(args.ref), args.ref_text
    infra = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(infra))
    from scripts.local import smoke
    from scripts.local.processes import cleanup_processes, start_service, wait_ports_closed

    workdir = infra / "workdir"
    data_dir = workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + "-realtime"
    env = dict(os.environ)
    env.update(
        LOCAL_WORKER_REGISTRATION_TOKEN=smoke.REGISTRATION_TOKEN,
        LOCAL_ADMIN_TOKEN=smoke.ADMIN_TOKEN,
        LOCAL_MCP_INFER_TOKENS=smoke.INFER_TOKEN,
        LOCAL_INDEXTTS_TEXT_FRONTEND="official_like",
    )
    env.setdefault("RUST_LOG", "info")
    procs = []
    try:
        controller = start_service(
            "controller",
            [
                str(infra / "target" / "release" / "controller.exe"), "configs/controller.yaml",
                "--workdir", str(workdir), "--model-dir", args.model_dir,
                "--worker-registration-token", smoke.REGISTRATION_TOKEN,
                "--public-base-url", smoke.CONTROLLER_URL, "--admin-token", smoke.ADMIN_TOKEN,
                "--mcp-bind", "127.0.0.1:17892",
            ],
            infra, data_dir, stamp, env,
        )
        procs.append(controller)
        smoke.wait_health("controller", f"{smoke.CONTROLLER_URL}/health", controller, 60, 15, data_dir)
        smoke.rpc_enable_indextts(30)
        worker = start_service(
            "worker",
            [
                str(infra / "target" / "release" / "worker.exe"), "configs/worker.yaml",
                "--workdir", str(workdir), "--model-dir", args.model_dir,
                "--registration-token", smoke.REGISTRATION_TOKEN,
            ],
            infra, data_dir, stamp, env,
        )
        procs.append(worker)
        smoke.wait_health("worker", f"{smoke.WORKER_URL}/health", worker, 60, 15, data_dir)
        time.sleep(1.0)  # first heartbeat
        asyncio.run(
            session(
                "ws://127.0.0.1:17890/v1/realtime", smoke.INFER_TOKEN, args.group, Path(args.out), args.mode == "audio"
            )
        )
    finally:
        cleanup_processes(procs)
        wait_ports_closed(smoke.PORTS, 10)
        print(f"worker log: {data_dir / f'worker-{stamp}.stdout.log'}")


if __name__ == "__main__":
    main()
