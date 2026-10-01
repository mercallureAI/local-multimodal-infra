# Provenance

Vendored from [SpenserCai/wetext-rs](https://github.com/SpenserCai/wetext-rs)
(Apache-2.0, see `LICENSE`) at commit `eaf3390` (crate `wetext-rs` 0.1.2), a
Rust port of [pengzhendong/wetext](https://github.com/pengzhendong/wetext).
It is maintained here as a local dependency; local changes are listed in the
git history of `crates/text/wetext`.

The FST weight files are not part of the repository. They are the WeText
files published with the `wetext` Python package (the same files IndexTTS
uses on Windows/macOS) and are installed as a model artifact.

## Local changes

- `FstTextNormalizer` picks the best path with exact tropical comparisons
  (OpenFst semantics). rustfst's `shortest_path` treats weights within 1/1024 as
  equal, which chose non-preferred verbalizations ("111" -> "one eleven").
  Output labels are read back as bytes, as kaldifst does.
- Rule FSTs are arc-sorted on load, like kaldifst's `TextNormalizer`.
- TN triggers on any Unicode decimal digit (Python `\d`), so full-width digits
  are normalized.
- Upstream integration tests that relied on bundled FSTs are replaced by
  `tests/parity_test.rs`, which checks 271 cases against the Python `wetext`
  0.1.0 output (`scripts/local/wetext_parity.py`).
