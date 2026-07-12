import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("summarize_indextts_latency.py")
SPEC = importlib.util.spec_from_file_location("latency_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def event(request_id, message, fields=""):
    return f"INFO {message} request_id={request_id} {fields}\n"


class SummaryTests(unittest.TestCase):
    def test_aggregates_multichunk_request_level_decode_steps(self):
        request = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        lines = [
            event(request, "IndexTTS synthesized chunk", "generated_steps=20"),
            event(request, "IndexTTS synthesized chunk", "generated_steps=30"),
            event(
                request,
                "IndexTTS synthesis stages",
                "decode_steps=50 audio_samples=24000",
            ),
            event(request, "runtime inference stages", "execution_ms=1000 success=true"),
            event(
                request,
                "runtime inference completed",
                "queue_wait_ms=7 success=true",
            ),
        ]
        requests, order = SUMMARY.parse_lines(lines)
        result = SUMMARY.summarize(requests, order)
        self.assertEqual(result["successful"], 1)
        self.assertEqual(result["metrics"]["decode_steps"], [50])
        self.assertEqual(result["metrics"]["rtf"], [1.0])

    def test_single_chunk_success_and_warmup_discard(self):
        warmup = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        measured = "cccccccc-cccc-cccc-cccc-cccccccccccc"
        lines = []
        for request in (warmup, measured):
            lines.extend(
                [
                    event(request, "IndexTTS synthesized chunk", "generated_steps=12"),
                    event(
                        request,
                        "IndexTTS synthesis stages",
                        "decode_steps=12 audio_samples=12000",
                    ),
                    event(request, "runtime inference stages", "execution_ms=500 success=true"),
                    event(
                        request,
                        "runtime inference completed",
                        "queue_wait_ms=1 success=true",
                    ),
                ]
            )
        requests, order = SUMMARY.parse_lines(lines)
        result = SUMMARY.summarize(requests, order, discard_first=1)
        self.assertEqual(result["discarded"], 1)
        self.assertEqual(result["successful"], 1)

    def test_partial_multichunk_failure_is_excluded(self):
        request = "dddddddd-dddd-dddd-dddd-dddddddddddd"
        lines = [
            event(request, "IndexTTS synthesized chunk", "generated_steps=20"),
            event(request, "runtime inference stages", "execution_ms=800 success=false"),
            event(
                request,
                "runtime inference completed",
                "queue_wait_ms=3 success=false",
            ),
        ]
        requests, order = SUMMARY.parse_lines(lines)
        result = SUMMARY.summarize(requests, order)
        self.assertEqual(result["successful"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["metrics"]["decode_steps"], [])

    def test_missing_terminal_event_is_incomplete(self):
        request = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
        requests, order = SUMMARY.parse_lines(
            [event(request, "IndexTTS synthesized chunk", "generated_steps=20")]
        )
        result = SUMMARY.summarize(requests, order)
        self.assertEqual(result["incomplete"], 1)


if __name__ == "__main__":
    unittest.main()
