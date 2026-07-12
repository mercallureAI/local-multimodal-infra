#!/usr/bin/env python3
"""Summarize successful IndexTTS requests from human-readable tracing logs."""

import argparse
import math
import re


REQUEST = re.compile(r"\brequest_id=([0-9a-f-]+)")
INTEGER = re.compile(
    r"\b(queue_wait_ms|execution_ms|decode_steps|generated_steps|audio_samples)=(\d+)"
)
SUCCESS = re.compile(r"\bsuccess=(true|false)\b")


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def parse_lines(lines):
    """Join only known event types; do not confuse unrelated success fields."""
    requests = {}
    completion_order = []
    for line in lines:
        request_match = REQUEST.search(line)
        if not request_match:
            continue
        request_id = request_match.group(1)
        record = requests.setdefault(
            request_id,
            {
                "chunk_steps": [],
                "synthesis_complete": False,
                "runtime_complete": None,
            },
        )
        fields = {name: int(value) for name, value in INTEGER.findall(line)}

        if "IndexTTS synthesized chunk" in line and "generated_steps" in fields:
            record["chunk_steps"].append(fields["generated_steps"])
        elif "IndexTTS synthesis stages" in line:
            record["synthesis_complete"] = True
            record["decode_steps"] = fields.get("decode_steps")
            record["audio_samples"] = fields.get("audio_samples")
        elif "runtime inference stages" in line:
            record["execution_ms"] = fields.get("execution_ms")
        elif "runtime inference completed" in line:
            success_match = SUCCESS.search(line)
            record["runtime_complete"] = (
                success_match is not None and success_match.group(1) == "true"
            )
            record["queue_wait_ms"] = fields.get("queue_wait_ms")
            if "completion_index" not in record:
                record["completion_index"] = len(completion_order)
                completion_order.append(request_id)

    # Older logs without request-level decode_steps can still be summarized
    # correctly by summing every chunk, never overwriting the last chunk.
    for record in requests.values():
        if record.get("decode_steps") is None and record["chunk_steps"]:
            record["decode_steps"] = sum(record["chunk_steps"])
    return requests, completion_order


def summarize(requests, completion_order, discard_first=0):
    discarded = set(completion_order[:discard_first])
    considered = {
        request_id: record
        for request_id, record in requests.items()
        if request_id not in discarded
    }
    successful = [
        record
        for record in considered.values()
        if record["runtime_complete"] is True
        and record["synthesis_complete"]
        and record.get("execution_ms") is not None
        and record.get("decode_steps") is not None
        and record.get("audio_samples", 0) > 0
        and record.get("queue_wait_ms") is not None
    ]
    failed = sum(record["runtime_complete"] is False for record in considered.values())
    incomplete = len(considered) - len(successful) - failed
    metrics = {
        "queue_wait_ms": [record["queue_wait_ms"] for record in successful],
        "execution_ms": [record["execution_ms"] for record in successful],
        "decode_steps": [record["decode_steps"] for record in successful],
        "rtf": [
            record["execution_ms"] / (record["audio_samples"] / 24_000 * 1_000)
            for record in successful
        ],
    }
    return {
        "requests_seen": len(requests),
        "discarded": len(discarded),
        "successful": len(successful),
        "failed": failed,
        "incomplete": incomplete,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file")
    parser.add_argument(
        "--discard-first",
        type=int,
        default=0,
        metavar="N",
        help="discard the first N terminal requests (use 4 for one warmup cycle)",
    )
    parser.add_argument(
        "--expect-success",
        type=int,
        metavar="N",
        help="exit nonzero unless exactly N complete successful requests remain",
    )
    args = parser.parse_args()
    if args.discard_first < 0:
        parser.error("--discard-first must be non-negative")

    with open(args.log_file, encoding="utf-8", errors="replace") as log:
        requests, completion_order = parse_lines(log)
    result = summarize(requests, completion_order, args.discard_first)
    print(
        f"requests_seen={result['requests_seen']} discarded={result['discarded']} "
        f"successful={result['successful']} failed={result['failed']} "
        f"incomplete={result['incomplete']}"
    )
    for name, values in result["metrics"].items():
        print(
            f"{name}: n={len(values)} "
            f"p50={percentile(values, 0.50):.3f} p95={percentile(values, 0.95):.3f}"
        )
    if args.expect_success is not None and result["successful"] != args.expect_success:
        raise SystemExit(
            f"expected {args.expect_success} successful measured requests, "
            f"found {result['successful']}"
        )


if __name__ == "__main__":
    main()
