import tempfile
import unittest
import json
import sys
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import indextts_export as export


class IndexTtsExportV2Tests(unittest.TestCase):
    def test_disabled_modes_do_not_create_destination(self):
        for mode in ["package", "local-export"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "not-created"
                args = self._main_args(output)
                args.index_tts_project = root
                args.source_model_dir = root
                args.mode = mode
                with mock.patch.object(export, "parse_args", return_value=args):
                    self.assertEqual(export.main([]), 2)
                self.assertFalse(output.exists())

    def test_local_export_is_disabled_for_v2_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in export.MODEL_FILES:
                (root / name).write_bytes(b"legacy")
            args = SimpleNamespace(
                local_export_entry=["legacy:export"],
                index_tts_project=root,
                output_dir=root,
            )
            notes = []
            self.assertFalse(export.invoke_local_exporter(args, notes))
            self.assertIn("disabled", notes[0])

    def test_layout_requires_prefill_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in export.MODEL_FILES:
                (root / name).write_bytes(b"legacy")
            with self.assertRaises(SystemExit) as error:
                export.validate_layout(root, require_onnx=True)
            self.assertIn("IndexTTS_E_Prefill.onnx", str(error.exception))

    def test_raw_export_completeness_requires_prefill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output_dir=root)
            for name in export.MODEL_FILES:
                (root / name).write_bytes(b"legacy")
            with (
                mock.patch.object(export, "require_raw_export_dependencies"),
                mock.patch.object(export.RawIndexTtsExporter, "export_all"),
            ):
                self.assertFalse(export.raw_export(args, []))

    def test_no_overwrite_preflight_preserves_ready_manifests_byte_for_byte(self):
        for mode in ["raw-export", "auto"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in export.REQUIRED_ONNX_FILES:
                    (root / name).write_bytes(b"graph")
                json_bytes = b'{"status":"ready","sentinel":"json"}\n'
                yaml_bytes = b"status: ready\nsentinel: yaml\n"
                (root / "manifest.json").write_bytes(json_bytes)
                (root / "manifest.yaml").write_bytes(yaml_bytes)
                args = self._main_args(root)
                args.mode = mode
                with mock.patch.object(export, "parse_args", return_value=args):
                    self.assertEqual(export.main([]), 2)
                self.assertEqual((root / "manifest.json").read_bytes(), json_bytes)
                self.assertEqual((root / "manifest.yaml").read_bytes(), yaml_bytes)

    def test_no_overwrite_rejects_every_exporter_owned_destination(self):
        owned_cases = [
            *export.REQUIRED_ONNX_FILES,
            "manifest.json",
            "manifest.yaml",
            "bpe.model",
            "IndexTTS_E.onnx.data",
            "IndexTTS_E.onnx.data.1",
        ]
        for mode in ["auto", "raw-export"]:
            for owned in owned_cases:
                with self.subTest(mode=mode, owned=owned), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    sentinel = root / owned
                    sentinel.write_bytes(b"preserve")
                    args = self._main_args(root)
                    args.mode = mode
                    with (
                        mock.patch.object(export, "parse_args", return_value=args),
                        mock.patch.object(export, "raw_export") as raw_export,
                    ):
                        self.assertEqual(export.main([]), 2)
                    raw_export.assert_not_called()
                    self.assertEqual(sentinel.read_bytes(), b"preserve")
                    self.assertEqual(list(root.iterdir()), [sentinel])

    def test_no_overwrite_rejects_complete_and_subset_graph_sets(self):
        for names in [
            export.REQUIRED_ONNX_FILES,
            export.REQUIRED_ONNX_FILES[:3],
        ]:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in names:
                    (root / name).write_bytes(name.encode())
                before = {path.name: path.read_bytes() for path in root.iterdir()}
                args = self._main_args(root)
                with mock.patch.object(export, "parse_args", return_value=args):
                    self.assertEqual(export.main([]), 2)
                self.assertEqual(
                    {path.name: path.read_bytes() for path in root.iterdir()},
                    before,
                )

    def test_unrelated_file_does_not_trigger_no_overwrite_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "user-notes.txt").write_bytes(b"mine")
            args = self._main_args(root)
            self.assertIsNone(export.preflight_export(args))

    def test_precondition_failure_does_not_create_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "not-created"
            args = self._main_args(output)
            args.index_tts_project = root / "missing-project"
            args.source_model_dir = root
            with mock.patch.object(export, "parse_args", return_value=args):
                with self.assertRaises(SystemExit):
                    export.main([])
            self.assertFalse(output.exists())

    def test_strict_e_metadata_validator_rejects_abi_mismatches(self):
        mutations = {
            "too few cache": lambda prefill, decode: decode.inputs.pop(0),
            "wrong name": lambda prefill, decode: setattr(decode.inputs[0], "name", "wrong"),
            "wrong order": lambda prefill, decode: decode.inputs.__setitem__(
                slice(0, 2), list(reversed(decode.inputs[:2]))
            ),
            "wrong dtype": lambda prefill, decode: setattr(prefill.inputs[0], "type", "tensor(float16)"),
            "wrong heads": lambda prefill, decode: decode.inputs[0].shape.__setitem__(1, 16),
            "wrong head dim": lambda prefill, decode: decode.outputs[0].shape.__setitem__(3, 80),
            "wrong hidden width": lambda prefill, decode: prefill.inputs[0].shape.__setitem__(2, 1024),
            "wrong logits width": lambda prefill, decode: decode.outputs[-1].shape.__setitem__(2, 8193),
            "extra output": lambda prefill, decode: decode.outputs.append(
                self._meta("extra", "tensor(float)", [1])
            ),
            "static sequence": lambda prefill, decode: prefill.inputs[0].shape.__setitem__(1, 4),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), self._validator_fixture() as fixture:
                prefill, decode = self._valid_e_sessions()
                mutate(prefill, decode)
                with self.assertRaises(export.UnsupportedExport):
                    self._run_real_validator(fixture, prefill, decode)

    def test_strict_e_metadata_validator_accepts_exact_contract(self):
        with self._validator_fixture() as fixture:
            prefill, decode = self._valid_e_sessions()
            self._run_real_validator(fixture, prefill, decode)

    def test_validator_rejects_source_config_mismatch(self):
        with self._validator_fixture() as fixture:
            (fixture.source / "config.yaml").write_text(self._config_yaml(layers=23))
            prefill, decode = self._valid_e_sessions()
            with self.assertRaisesRegex(export.UnsupportedExport, "source config"):
                self._run_real_validator(fixture, prefill, decode)

    def test_main_never_marks_ready_when_certification_fails(self):
        failures = [
            "wrong pinned provenance",
            "missing IndexTTS_E_Prefill.onnx",
            "ONNX checker failure",
            "ORT load failure",
            "invalid E ABI",
        ]
        for failure in failures:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = self._main_args(root)
                with (
                    mock.patch.object(export, "parse_args", return_value=args),
                    mock.patch.object(export, "raw_export", return_value=True),
                    mock.patch.object(export, "apply_precision"),
                    mock.patch.object(
                        export,
                        "validate_ready_export",
                        side_effect=export.UnsupportedExport(failure),
                    ) as certify,
                    mock.patch.object(export, "copy_bpe"),
                    mock.patch.object(export, "validate_layout"),
                    mock.patch.object(
                        export,
                        "build_manifest",
                        side_effect=lambda _args, status, notes: {"status": status, "notes": notes},
                    ),
                ):
                    self.assertEqual(export.main([]), 2)
                certify.assert_called_once_with(args)
                manifest = export.json.loads((root / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "unsupported")
                self.assertIn(failure, manifest["notes"])

    def test_main_marks_ready_only_after_successful_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._main_args(root)
            with (
                mock.patch.object(export, "parse_args", return_value=args),
                mock.patch.object(export, "raw_export", return_value=True),
                mock.patch.object(export, "apply_precision"),
                mock.patch.object(export, "validate_ready_export") as certify,
                mock.patch.object(export, "copy_bpe"),
                mock.patch.object(export, "validate_layout"),
                mock.patch.object(
                    export,
                    "build_manifest",
                    side_effect=lambda _args, status, notes: {"status": status, "notes": notes},
                ),
            ):
                self.assertEqual(export.main([]), 0)
            certify.assert_called_once_with(args)
            manifest = export.json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "ready")

    def test_native_checker_exception_through_main_invalidates_stale_ready(self):
        self._assert_native_validation_failure("checker")

    def test_native_ort_exception_through_main_invalidates_stale_ready(self):
        self._assert_native_validation_failure("ort")

    def _assert_native_validation_failure(self, failure_kind):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._main_args(root)
            args.overwrite = True
            for name in export.REQUIRED_ONNX_FILES:
                (root / name).write_bytes(b"graph")
            (root / "config.yaml").write_text(self._config_yaml())
            (root / "manifest.json").write_text('{"status":"ready","split_contract_version":2}')
            (root / "manifest.yaml").write_text("status: ready\nsplit_contract_version: 2\n")
            expected = {
                "model_repository": "IndexTeam/IndexTTS-1.5",
                "model_revision": "25851a6036dfd3095bb70fb3c8f49217104672c3",
                "code_tag": "v1.5.0",
                "code_commit": "9098497272d5803bae46cbaf5154cf2ba48f6866",
                "code_tree": "aa0335ccaba54ac42d6d209dac56bb9a8b2e80a7",
            }
            fake_onnx = SimpleNamespace(
                checker=SimpleNamespace(
                    check_model=mock.Mock(
                        side_effect=RuntimeError("native checker boom")
                        if failure_kind == "checker"
                        else None
                    )
                )
            )
            fake_ort = SimpleNamespace(
                InferenceSession=mock.Mock(side_effect=RuntimeError("native ORT boom"))
            )
            modules = {"onnx": fake_onnx, "onnxruntime": fake_ort}
            with (
                mock.patch.object(export, "parse_args", return_value=args),
                mock.patch.object(export, "raw_export", return_value=True),
                mock.patch.object(export, "apply_precision"),
                mock.patch.object(export, "copy_bpe"),
                mock.patch.object(export, "read_source_provenance", return_value=expected),
                mock.patch.dict(sys.modules, modules),
            ):
                self.assertEqual(export.main([]), 2)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "unsupported")
            self.assertNotIn("split_contract_version", manifest)
            diagnostic = " ".join(manifest["notes"])
            self.assertIn("ONNX checker" if failure_kind == "checker" else "ONNX Runtime", diagnostic)

    @staticmethod
    def _main_args(root):
        return SimpleNamespace(
            output_dir=root,
            index_tts_project=root,
            source_model_dir=root,
            check_deps=False,
            mode="raw-export",
            precision=export.SUPPORTED_PRECISION,
            device="cpu",
            opset=17,
            audio_length=24_000,
            max_seq_len=16,
            local_export_entry=[],
            overwrite=False,
            external_data_threshold_mb=512,
        )

    @staticmethod
    def _meta(name, element_type, shape):
        return SimpleNamespace(name=name, type=element_type, shape=list(shape))

    @classmethod
    def _valid_e_sessions(cls):
        cache_inputs = []
        cache_outputs = []
        for index in range(24):
            cache_inputs.extend(
                [
                    cls._meta(f"in_key_{index}", "tensor(float)", [1, 20, "past", 64]),
                    cls._meta(f"in_value_{index}", "tensor(float)", [1, 20, "past", 64]),
                ]
            )
            cache_outputs.extend(
                [
                    cls._meta(f"out_key_{index}", "tensor(float)", [1, 20, "next", 64]),
                    cls._meta(f"out_value_{index}", "tensor(float)", [1, 20, "next", 64]),
                ]
            )
        hidden = cls._meta("hidden_state", "tensor(float)", [1, "sequence", 1280])
        mask = cls._meta("attention_mask", "tensor(int64)", [1, "sequence"])
        tails = [
            cls._meta("last_hidden_state", "tensor(float)", [1, 1, 1280]),
            cls._meta("raw_logits", "tensor(float)", [1, 1, 8194]),
        ]

        def session(inputs, outputs):
            return SimpleNamespace(
                inputs=inputs,
                outputs=outputs,
                get_inputs=lambda: inputs,
                get_outputs=lambda: outputs,
            )

        return (
            session([hidden, mask], [*cache_outputs, *tails]),
            session([*cache_inputs, hidden, mask], [*cache_outputs, *tails]),
        )

    @staticmethod
    def _config_yaml(layers=24):
        return f"""gpt:
  layers: {layers}
  heads: 20
  model_dim: 1280
  number_mel_codes: 8194
  start_mel_token: 8192
  stop_mel_token: 8193
"""

    class _validator_fixture:
        def __init__(self):
            self.temp = tempfile.TemporaryDirectory()
            self.root = Path(self.temp.name)
            self.source = self.root / "source"
            self.output = self.root / "output"

        def __enter__(self):
            self.source.mkdir()
            self.output.mkdir()
            (self.source / "config.yaml").write_text(IndexTtsExportV2Tests._config_yaml())
            for name in export.REQUIRED_ONNX_FILES:
                (self.output / name).write_bytes(b"graph")
            return self

        def __exit__(self, *args):
            self.temp.cleanup()

    @staticmethod
    def _run_real_validator(fixture, prefill, decode):
        args = SimpleNamespace(source_model_dir=fixture.source, output_dir=fixture.output)
        sessions = {
            "IndexTTS_E_Prefill.onnx": prefill,
            "IndexTTS_E.onnx": decode,
        }
        fallback = SimpleNamespace(get_inputs=lambda: [], get_outputs=lambda: [])
        fake_onnx = SimpleNamespace(checker=SimpleNamespace(check_model=mock.Mock()))
        fake_ort = SimpleNamespace(
            InferenceSession=lambda path, providers: sessions.get(Path(path).name, fallback)
        )
        expected_provenance = {
            "model_repository": "IndexTeam/IndexTTS-1.5",
            "model_revision": "25851a6036dfd3095bb70fb3c8f49217104672c3",
            "code_tag": "v1.5.0",
            "code_commit": "9098497272d5803bae46cbaf5154cf2ba48f6866",
            "code_tree": "aa0335ccaba54ac42d6d209dac56bb9a8b2e80a7",
        }
        with (
            mock.patch.object(export, "read_source_provenance", return_value=expected_provenance),
            mock.patch.dict(sys.modules, {"onnx": fake_onnx, "onnxruntime": fake_ort}),
        ):
            export.validate_ready_export(args)


if __name__ == "__main__":
    unittest.main()
