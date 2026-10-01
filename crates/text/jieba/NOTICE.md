# Provenance

Vendored from [messense/jieba-rs](https://github.com/messense/jieba-rs) 0.11.0
(MIT, see `LICENSE`) as a local dependency. The keyword extractors (`tfidf`,
`textrank`) are not included. Local changes are listed below and in the git
history of `crates/text/jieba`.

## Local changes

- `Jieba::posseg_cut`: Python `jieba.posseg.cut(sentence, HMM=True)`
  (jieba 0.42.1 `POSTokenizer.__cut_internal` / `__cut_DAG` /
  `__cut_detail`), which PaddleSpeech's Mandarin frontend uses. `Jieba::tag`
  tags after the plain HMM cut, so it segments unknown words differently
  (`过诗琳/通` vs `诗琳通`, `这/一` vs `这一`).
- `posseg::spans`: a literal port of Python `posseg.viterbi` + `__cut`, used
  by `posseg_cut`. The upstream `viterbi_posseg` (still used by `tag`) only
  ends on E/S states, drops `MIN_FLOAT` states and breaks ties by state index;
  Python may end on any state, keeps them, treats a missing transition as
  `-inf` and breaks ties by the `(position, tag)` tuple. `这一` decodes to
  `这/r 一/m` in Python and `这一/r` upstream. `PossegData` records which
  characters have an explicit state list for this.
- Removed `test_cut_weicheng` and `test_cut_with_custom_hmm_model`, which read
  files outside the crate.
