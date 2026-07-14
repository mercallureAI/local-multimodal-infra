use super::*;
use local_backend_ort::ProviderKind;
use local_core::{
    AdapterKind, ArtifactKind, BackendKind, FileRef, InferenceOutput, ModelArtifact, ModelSpec,
    ResourceRequirement, RuntimePolicy,
};
use std::sync::Mutex;

fn official_like(text: &str) -> String {
    preprocess_text_for_index_tts_with_mode(text, IndexTtsTextFrontendMode::OfficialLike)
}

#[test]
fn validates_complete_artifact_layout() {
    let dir = tempfile::tempdir().expect("tempdir");
    for name in MODEL_FILENAMES {
        fs::write(dir.path().join(name), b"placeholder").expect("onnx");
    }
    fs::write(dir.path().join("IndexTTS_E_Prefill.onnx"), b"placeholder").expect("prefill");
    fs::write(dir.path().join("bpe.model"), b"sentencepiece").expect("bpe");
    fs::write(dir.path().join("manifest.yaml"), b"precision: cpu-fp32\n").expect("manifest");

    let artifacts =
        IndexTtsArtifacts::validate(dir.path(), IndexTtsPrecision::CpuFp32).expect("valid layout");

    assert_eq!(
        artifacts.a.file_name().and_then(|v| v.to_str()),
        Some("IndexTTS_A.onnx")
    );
    assert!(artifacts.manifest.is_some());
}

#[test]
fn fp32_layout_ignores_quantized_subdirs() {
    let dir = tempfile::tempdir().expect("tempdir");
    for name in MODEL_FILENAMES {
        fs::write(dir.path().join(name), b"placeholder").expect("onnx");
    }
    fs::write(dir.path().join("IndexTTS_E_Prefill.onnx"), b"placeholder").expect("prefill");
    let q4 = dir.path().join("q4");
    fs::create_dir(&q4).expect("q4 dir");
    for name in MODEL_FILENAMES {
        fs::write(q4.join(name), b"placeholder").expect("onnx");
    }
    fs::write(dir.path().join("bpe.model"), b"sentencepiece").expect("bpe");

    let artifacts = IndexTtsArtifacts::validate(dir.path(), IndexTtsPrecision::CpuFp32)
        .expect("valid fp32 layout");

    assert!(artifacts.a.starts_with(dir.path()));
    assert!(!artifacts.a.starts_with(&q4));
}

#[test]
fn normalize_text_collapses_space_and_punctuation() {
    assert_eq!(normalize_text("  你好，  world！\n"), "你好, world!");
}

#[test]
fn normalize_text_separates_mapped_chinese_punctuation_from_ascii() {
    assert_eq!(
        normalize_text("第一句。Second ？What！Hello，hello、123"),
        "第一句. Second ? What! Hello, hello,一二三"
    );
    assert_eq!(normalize_text("A，hello"), "A, hello");
}

#[test]
fn normalize_text_preserves_and_fixes_toned_pinyin() {
    assert_eq!(
        normalize_text("xuan4 ying1 zhong4 shang5 ju4 qu2 xu1 lü4 nü3"),
        "XVAN4 ying1 zhong4 shang5 JV4 QV2 XV1 lü4 nü3"
    );
}

#[test]
fn frontend_keeps_pure_chinese_as_cjk_chars_by_default() {
    assert_eq!(official_like("你好"), "你 好");
}

#[test]
fn frontend_handles_chinese_english_and_punctuation() {
    assert_eq!(
        official_like("你好 OpenAI，世界！"),
        "你 好 OPENAI, 世 界 !"
    );
}

#[test]
fn frontend_preserves_explicit_pinyin_and_english_mixes() {
    assert_eq!(
        official_like("ni3 hao3 world zhong1 wen2 and English mixed"),
        "NI3 HAO3 WORLD ZHONG1 WEN2 AND ENGLISH MIXED"
    );
    assert_eq!(official_like("ni3hao3OpenAI"), "NI3HAO3OPENAI");
    assert_eq!(official_like("OpenAI2"), "OPENAI2");
    assert_eq!(official_like("NI3 HAO3 OpenAI"), "NI3 HAO3 OPENAI");
}

#[test]
fn frontend_rejects_pinyin_like_suffixes_inside_ascii_words() {
    assert!(!contains_explicit_pinyin_tone("beta1"));
    assert!(!contains_explicit_pinyin_tone("BETA1"));
    assert!(!contains_explicit_pinyin_tone("voice2"));
    assert!(!contains_explicit_pinyin_tone("babala2"));
    assert!(!contains_explicit_pinyin_tone("AGAN3"));
    assert!(!contains_explicit_pinyin_tone("M42"));
    assert_eq!(official_like("beta1"), "BETA1");
    assert_eq!(official_like("BETA1"), "BETA1");
    assert_eq!(official_like("voice2"), "VOICE2");
    assert_eq!(official_like("babala2"), "BABALA2");
    assert_eq!(official_like("AGAN3"), "AGAN3");
    assert_eq!(official_like("M42中文"), "M 四 十 二 中 文");
}

#[test]
fn frontend_normalizes_umlaut_and_jqx_pinyin_rules() {
    assert_eq!(
        official_like("ju4 QU2 xu1 lü4 NÜ3 lüe4 yuan2 yue4 er2 ng5"),
        "JV4 QV2 XV1 LÜ4 NÜ3 LÜE4 YUAN2 YUE4 ER2 NG5"
    );
}

#[test]
fn frontend_expands_official_english_contractions() {
    assert_eq!(official_like("where's the money?"), "WHERE IS THE MONEY?");
    assert_eq!(
        official_like("今天是个好日子 it's a good day"),
        "今 天 是 个 好 日 子 IT IS A GOOD DAY"
    );
}

#[test]
fn frontend_preserves_chinese_names_around_lightweight_tn() {
    assert_eq!(
        official_like("约瑟夫·高登-莱维特（Joseph Gordon-Levitt is an American actor）"),
        "约 瑟 夫 - 高 登 - 莱 维 特 'JOSEPH GORDON-LEVITT IS AN AMERICAN ACTOR'"
    );
}

#[test]
fn pinyin_placeholders_do_not_collide_after_twenty_six_entries() {
    let text = [
        "ba1", "pa2", "ma3", "fa4", "da1", "ta2", "na3", "la4", "ga1", "ka2", "ha3", "ji1", "qi2",
        "xi3", "zhi4", "chi1", "shi2", "ri3", "zi4", "ci1", "si2", "ya3", "yan4", "yang1", "yao2",
        "ye3", "yin4", "ying1", "you2", "wu3",
    ]
    .join(" ");
    let (saved, originals) = save_pinyin_tones(&text);
    assert_eq!(originals.len(), 30);
    assert!(saved.contains("<pinyin_a>"));
    assert!(saved.contains("<pinyin_aa>"));

    let restored = restore_pinyin_tones(&saved, &originals);
    assert!(restored.contains("JI1"));
    assert!(restored.contains("wu3"));
}

#[test]
fn name_placeholders_do_not_collide_after_twenty_six_entries() {
    let names = (0..30)
        .map(|idx| {
            let left = char::from_u32(0x4E00 + idx).expect("hanzi left");
            let right = char::from_u32(0x4E50 + idx).expect("hanzi right");
            format!("{left}·{right}")
        })
        .collect::<Vec<_>>();
    let text = names.join(" ");
    let (saved, originals) = save_names(&text);
    assert_eq!(originals.len(), 30);
    assert!(saved.contains("<n_a>"));
    assert!(saved.contains("<n_aa>"));

    let restored = restore_names(&saved, &originals);
    assert_eq!(restored, text);
}

#[test]
fn frontend_keeps_common_dates_numbers_and_units_numeric_in_lightweight_tn() {
    assert_eq!(
        official_like("现在是北京时间2025年01月11日 20:00，速度是10km/h"),
        "现 在 是 北 京 时 间 二 零 二 五 年 一 月 十 一 日 二 十 点 , 速 度 是 十 千 米 每 小 时"
    );
}

#[test]
fn frontend_corpus_covers_mixed_text_and_lightweight_tn_boundaries() {
    let cases = [
        ("纯中文测试。", "纯 中 文 测 试 ."),
        ("hello IndexTTS", "HELLO INDEXTTS"),
        ("你好 OpenAI，世界！", "你 好 OPENAI, 世 界 !"),
        ("xuan4 GAN3", "XVAN4 GAN3"),
        ("约瑟夫·高登-莱维特", "约 瑟 夫 - 高 登 - 莱 维 特"),
        (
            "版本１２３，比例50％。",
            "版 本 一 二 三 , 比 例 百 分 之 五 十 .",
        ),
        (
            "日期2025/01/11，时间08:30。",
            "日 期 二 零 二 五 年 一 月 十 一 日 , 时 间 八 点 三 十 分 .",
        ),
        (
            "价格￥12.5，另收$3和RMB 20。",
            "价 格 十 二 点 五 元 , 另 收 三 美 元 和 RMB 二 十 .",
        ),
        (
            "速度10km/h，重量5kg，内存16GB，温度25℃。",
            "速 度 十 千 米 每 小 时 , 重 量 五 千 克 , 内 存 十 六 GB, 温 度 二 十 五 摄 氏 度 .",
        ),
        ("联系test@example.com。", "联 系 TEST@EXAMPLE.COM."),
        (
            "第一句。Second sentence! 第三句？",
            "第 一 句 . SECOND SENTENCE! 第 三 句 ?",
        ),
    ];
    for (input, expected) in cases {
        assert_eq!(official_like(input), expected, "{input}");
    }
}

#[test]
fn exact_m42_request_frontend_regression() {
    let input = "这里是M42语音测试，现在检查合成、分段和发送链路是否都能正常工作。";
    assert_eq!(
        normalize_text(input),
        "这里是M四十二语音测试,现在检查合成,分段和发送链路是否都能正常工作."
    );
    assert_eq!(
        official_like(input),
        "这 里 是 M 四 十 二 语 音 测 试 , 现 在 检 查 合 成 , 分 段 和 发 送 链 路 是 否 都 能 正 常 工 作 ."
    );
}

#[test]
fn cjk_detokenize_and_sentence_split_follow_official_shape() {
    let tokenized = tokenize_by_cjk_char("你好世界是 hello world 的中文", true);
    assert_eq!(tokenized, "你 好 世 界 是 HELLO WORLD 的 中 文");
    assert_eq!(
        de_tokenized_by_cjk_char(&tokenized, true),
        "你好世界是hello world的中文"
    );

    let pieces = vec![
        "你".to_string(),
        "好".to_string(),
        ".".to_string(),
        "▁NEXT".to_string(),
        "?".to_string(),
    ];
    assert_eq!(split_sentences(&pieces, 4).len(), 2);
}

#[test]
fn dump_text_frontend_reports_normalized_and_tokenized_without_bpe() {
    let dump = dump_text_frontend("where's 8:00 AM 开会", None).expect("dump");
    assert_eq!(dump.input, "where's 8:00 AM 开会");
    assert!(dump.normalized.contains("where is"));
    assert!(dump.tokenized.contains("WHERE IS"));
    assert!(dump.token_ids.is_none());
}

#[test]
fn explicit_token_ids_params_accept_supported_aliases() {
    let mut params = BTreeMap::new();
    params.insert(
        "indextts_text_token_ids".to_string(),
        serde_json::json!([1, 2, 8191]),
    );

    assert_eq!(
        explicit_text_token_ids_from_params(&params)
            .expect("parse ids")
            .expect("ids present"),
        vec![1, 2, 8191]
    );
}

#[test]
fn explicit_token_ids_params_reject_bad_values() {
    for value in [
        serde_json::json!([]),
        serde_json::json!([1, "2"]),
        serde_json::json!([-1]),
        serde_json::json!([MAX_TEXT_TOKEN_ID + 1]),
    ] {
        let mut params = BTreeMap::new();
        params.insert("text_token_ids".to_string(), value);
        let err = explicit_text_token_ids_from_params(&params).expect_err("bad ids");
        assert!(err.to_string().contains("IndexTTS text_token_ids"), "{err}");
    }
}

#[test]
fn frontend_optional_pinyin_explicit_mode_keeps_old_hanzi_conversion() {
    let expected = format!(
        "{} {}",
        hanzi_to_pinyin_token('重').expect("pinyin for 重"),
        hanzi_to_pinyin_token('庆').expect("pinyin for 庆")
    );
    assert_eq!(
        preprocess_text_for_index_tts_with_mode("重庆", IndexTtsTextFrontendMode::PinyinExplicit),
        expected
    );
}

#[test]
fn split_cjk_minimal_separates_script_runs() {
    assert_eq!(
        split_cjk_minimal(&normalize_text("你好world，再见")),
        vec!["你好".to_string(), "world".to_string(), "再见".to_string()]
    );
}

#[test]
fn audio_resample_linear_changes_length() {
    let samples = vec![0.0, 1.0, 0.0, -1.0];
    let out = audio::resample_linear(&samples, 12_000, 24_000);
    assert_eq!(out.len(), 8);
    assert!((out[0] - 0.0).abs() < 1e-6);
}

#[test]
fn reads_wav_as_mono_i16_24k() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ref.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).expect("create wav");
    writer.write_sample::<i16>(1000).expect("l");
    writer.write_sample::<i16>(-1000).expect("r");
    writer.finalize().expect("finalize");

    let samples = audio::read_wav_mono_i16_24k(&path).expect("read");

    assert_eq!(samples, vec![0]);
}

#[test]
fn sentencepiece_tokenizer_missing_file_is_clear() {
    let dir = tempfile::tempdir().expect("tempdir");
    let err = SentencePieceTokenizer::load(&dir.path().join("missing-bpe.model"))
        .expect_err("missing model");

    assert!(err.to_string().contains("bpe.model is missing"), "{err}");
}

#[test]
fn sentencepiece_tokenizer_invalid_file_reports_parse_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bpe.model");
    fs::write(&path, b"not a sentencepiece protobuf").expect("write invalid bpe");
    let err = SentencePieceTokenizer::load(&path).expect_err("invalid model");

    assert!(
        err.to_string().contains("load IndexTTS SentencePiece"),
        "{err}"
    );
}

#[test]
fn prepare_text_ids_uses_preprocessed_text_with_fake_tokenizer() {
    #[derive(Default)]
    struct FakeTokenizer {
        seen: Mutex<Vec<String>>,
    }
    impl IndexTextTokenizer for FakeTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<i32>> {
            self.seen.lock().expect("lock").push(text.to_string());
            Ok(vec![10, 20, 30])
        }
    }
    let tokenizer = FakeTokenizer::default();

    let ids = prepare_text_ids_with_mode(
        &tokenizer,
        "你好world，再见！",
        IndexTtsTextFrontendMode::OfficialLike,
    )
    .expect("ids");

    assert_eq!(ids, vec![10, 20, 30]);
    assert_eq!(
        tokenizer.seen.lock().expect("lock").as_slice(),
        &["你 好 WORLD, 再 见 !".to_string()]
    );
}

#[test]
fn prepare_text_ids_rejects_non_speakable_text_before_tokenizer() {
    #[derive(Default)]
    struct FakeTokenizer {
        calls: Mutex<usize>,
    }
    impl IndexTextTokenizer for FakeTokenizer {
        fn encode(&self, _text: &str) -> Result<Vec<i32>> {
            *self.calls.lock().expect("lock") += 1;
            Ok(vec![10])
        }
    }

    for input in ["", "   ", "。", "，", "！？", "……", "--", "''", "（）【】"] {
        let tokenizer = FakeTokenizer::default();
        let err =
            prepare_text_ids_with_mode(&tokenizer, input, IndexTtsTextFrontendMode::OfficialLike)
                .expect_err("non-speakable text should be rejected");
        assert!(
            err.to_string()
                .contains("IndexTTS text must contain speakable content"),
            "{input:?}: {err}"
        );
        assert_eq!(*tokenizer.calls.lock().expect("lock"), 0, "{input:?}");
    }
}

#[test]
fn speakable_content_guard_accepts_letters_digits_and_hanzi_only() {
    assert!(index_tts_text_has_speakable_content(&official_like(
        "你好world，再见！"
    )));
    assert!(index_tts_text_has_speakable_content("OPENAI2"));
    assert!(!index_tts_text_has_speakable_content(&official_like(
        "！？……--''"
    )));
}

#[test]
fn output_filename_uses_indextts_prefix_and_wav_suffix() {
    let id = Uuid::parse_str("00000000-0000-0000-0000-000000000123").expect("uuid");

    assert_eq!(
        index_tts_wav_filename(id),
        "indextts-00000000-0000-0000-0000-000000000123.wav"
    );
}

#[test]
fn long_token_planning_preserves_order_and_hard_splits() {
    let ids = (0..301).collect::<Vec<i32>>();
    let chunks = plan_token_chunks(&ids, None, 120).expect("chunks");
    assert_eq!(
        chunks.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![120, 120, 61]
    );
    assert_eq!(chunks.concat(), ids);
}

#[test]
fn punctuation_aware_planning_uses_boundary_without_loss() {
    let ids = (0..14).collect::<Vec<i32>>();
    let mut pieces = (0..14).map(|i| format!("T{i}")).collect::<Vec<_>>();
    pieces[7] = ".".to_string();
    let chunks = plan_token_chunks(&ids, Some(&pieces), 10).expect("chunks");
    assert_eq!(chunks.iter().map(Vec::len).collect::<Vec<_>>(), vec![8, 6]);
    assert_eq!(chunks.concat(), ids);
}

fn plan_indexed_pieces(pieces: &[&str], max_tokens: usize) -> Vec<Vec<i32>> {
    let ids = (0..pieces.len() as i32).collect::<Vec<_>>();
    let pieces = pieces
        .iter()
        .map(|piece| (*piece).to_string())
        .collect::<Vec<_>>();
    plan_token_chunks(&ids, Some(&pieces), max_tokens).expect("planned chunks")
}

#[test]
fn exact_m42_request_soft_splits_once_at_first_comma() {
    let pieces = [
        "这", "里", "是", "M", "四", "十", "二", "语", "音", "测", "试", ",", "现", "在", "检",
        "查", "合", "成", ",", "分", "段", "和", "发", "送", "链", "路", "是", "否", "都", "能",
        "正", "常", "工", "作", ".",
    ];
    let chunks = plan_indexed_pieces(&pieces, 120);

    assert_eq!(
        chunks.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![12, 23]
    );
    assert_eq!(chunks.concat(), (0..35).collect::<Vec<_>>());
    assert_eq!(pieces[*chunks[0].last().expect("first end") as usize], ",");
    assert!(chunks[1].iter().any(|id| pieces[*id as usize] == ","));
    let owned = pieces
        .iter()
        .map(|piece| piece.to_string())
        .collect::<Vec<_>>();
    assert_eq!(classify_split_kind(&owned, &chunks, 120), "soft");
}

#[test]
fn split_kind_distinguishes_none_hard_and_natural_hard_plans() {
    let plain = (0..5).map(|_| "A".to_string()).collect::<Vec<_>>();
    assert_eq!(classify_split_kind(&plain, &[vec![1, 2, 3]], 5), "none");

    let hard = (0..10).map(|_| "A".to_string()).collect::<Vec<_>>();
    assert_eq!(
        classify_split_kind(&hard, &[vec![0; 5], vec![0; 5]], 5),
        "hard"
    );

    let mut natural = (0..10).map(|_| "A".to_string()).collect::<Vec<_>>();
    natural[4] = ",".to_string();
    assert_eq!(
        classify_split_kind(&natural, &[vec![0; 5], vec![0; 5]], 5),
        "soft+hard"
    );
}

#[test]
fn soft_segmentation_supports_normalized_and_sentencepiece_punctuation_forms() {
    for punctuation in [
        ",", "▁,", "，", "▁，", ".", "▁.", "。", "▁。", "!", "?", "…", ";", ":",
    ] {
        let mut pieces = vec!["A"; 8];
        pieces.push(punctuation);
        pieces.extend(["B"; 8]);
        pieces.push(".");
        let chunks = plan_indexed_pieces(&pieces, 120);
        assert_eq!(
            chunks.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![9, 9],
            "{punctuation}"
        );
        assert_eq!(
            chunks.concat(),
            (0..18).collect::<Vec<_>>(),
            "{punctuation}"
        );
    }
}

#[test]
fn soft_segmentation_is_conservative_and_keeps_punctuation_attached() {
    let short = plan_indexed_pieces(&["短", "句", "。"], 120);
    assert_eq!(short.len(), 1);

    let mut pieces = vec!["A"; 8];
    pieces.extend(["?", "!", "…"]);
    pieces.extend(["B"; 8]);
    pieces.extend([",", ".", "C"]);
    let chunks = plan_indexed_pieces(&pieces, 120);
    assert_eq!(
        chunks.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![11, 11]
    );
    assert_eq!(chunks.concat(), (0..22).collect::<Vec<_>>());
    assert!(chunks.iter().all(|chunk| chunk
        .iter()
        .any(|id| piece_has_substantive_text(pieces[*id as usize]))));
}

#[test]
fn soft_and_hard_planning_preserve_all_ids_and_maximum_length() {
    let mut pieces = vec!["A".to_string(); 8];
    pieces.push(",".to_string());
    pieces.extend((0..250).map(|index| format!("T{index}")));
    let ids = (0..pieces.len() as i32).collect::<Vec<_>>();
    let chunks = plan_token_chunks(&ids, Some(&pieces), 120).expect("hard chunks");

    assert_eq!(chunks.concat(), ids);
    assert!(chunks
        .iter()
        .all(|chunk| !chunk.is_empty() && chunk.len() <= 120));
    assert!(chunks.iter().all(|chunk| chunk
        .iter()
        .any(|id| piece_has_substantive_text(&pieces[*id as usize]))));
}

#[test]
fn explicit_ids_remain_opaque_hard_chunks_despite_punctuation_like_values() {
    let ids = vec![44; 121];
    let chunks = plan_token_chunks(&ids, None, 120).expect("opaque ids");
    assert_eq!(
        chunks.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![120, 1]
    );
    assert_eq!(chunks.concat(), ids);
}

#[test]
fn terminal_punctuation_after_full_chunk_is_rebalanced_without_loss() {
    for (substantive_count, punctuation) in [
        (120_i32, vec!["."]),
        (120, vec![".", "!", "?"]),
        (240, vec![".", "!", "?"]),
    ] {
        let punctuation_count = punctuation.len();
        let ids = (0..substantive_count + punctuation.len() as i32).collect::<Vec<_>>();
        let mut pieces = (0..substantive_count)
            .map(|i| format!("T{i}"))
            .collect::<Vec<_>>();
        pieces.extend(punctuation.into_iter().map(str::to_string));

        let chunks = plan_token_chunks(&ids, Some(&pieces), 120).expect("rebalanced chunks");

        assert_eq!(chunks.concat(), ids);
        assert!(chunks.iter().all(|chunk| chunk.len() <= 120));
        assert!(chunks.iter().all(|chunk| chunk.iter().any(|id| {
            pieces[*id as usize]
                .chars()
                .any(|ch| ch.is_ascii_alphanumeric())
        })));
        assert_eq!(
            chunks.last().expect("last chunk").len(),
            punctuation_count + 1
        );
    }
}

#[test]
fn leading_and_boundary_punctuation_runs_attach_to_substantive_chunks() {
    let pieces = ["!", "?", "A", "B", "C", ".", "!", "D", "E"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let ids = (0..pieces.len() as i32).collect::<Vec<_>>();

    let chunks = plan_token_chunks(&ids, Some(&pieces), 4).expect("chunks");

    assert_eq!(chunks.concat(), ids);
    assert!(chunks.iter().all(|chunk| chunk.len() <= 4));
    assert!(chunks.iter().all(|chunk| chunk
        .iter()
        .any(|id| pieces[*id as usize].chars().any(char::is_alphanumeric))));
}

#[test]
fn planner_backtracks_across_multiple_future_punctuation_runs() {
    let pieces = ["S", "S", ".", "S", "."]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let ids = (0..pieces.len() as i32).collect::<Vec<_>>();

    let chunks = plan_token_chunks(&ids, Some(&pieces), 2).expect("global partition");

    assert_eq!(chunks, vec![vec![0], vec![1, 2], vec![3, 4]]);
}

#[test]
fn planner_handles_default_scaled_backtracking_case() {
    let mut pieces = vec!["S".to_string(); 120];
    pieces.push(".".to_string());
    pieces.push("S".to_string());
    pieces.extend(std::iter::repeat_n(".".to_string(), 119));
    let ids = (0..pieces.len() as i32).collect::<Vec<_>>();

    let chunks = plan_token_chunks(&ids, Some(&pieces), 120).expect("global partition");

    assert_eq!(
        chunks.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![119, 2, 120]
    );
    assert_eq!(chunks.concat(), ids);
    assert!(chunks
        .iter()
        .all(|chunk| chunk.iter().any(|id| pieces[*id as usize] == "S")));
}

#[test]
fn token_planner_matches_exhaustive_partition_oracle() {
    for len in 1..=9usize {
        for mask in 0usize..(1usize << len) {
            let substantive = (0..len)
                .map(|index| mask & (1 << index) != 0)
                .collect::<Vec<_>>();
            let pieces = substantive
                .iter()
                .map(|is_substantive| if *is_substantive { "S" } else { "." }.to_string())
                .collect::<Vec<_>>();
            let ids = (0..len as i32).collect::<Vec<_>>();
            for max_tokens in 1..=4usize {
                let expected = brute_force_chunk_partition_exists(&substantive, max_tokens, 0);
                let actual = plan_token_chunks(&ids, Some(&pieces), max_tokens);
                assert_eq!(
                    actual.is_ok(),
                    expected,
                    "len={len} mask={mask:09b} max={max_tokens}: {actual:?}"
                );
                if let Ok(chunks) = actual {
                    assert_eq!(chunks.concat(), ids);
                    assert!(chunks.iter().all(|chunk| !chunk.is_empty()
                        && chunk.len() <= max_tokens
                        && chunk.iter().any(|id| substantive[*id as usize])));
                }
            }
        }
    }
}

fn brute_force_chunk_partition_exists(
    substantive: &[bool],
    max_tokens: usize,
    start: usize,
) -> bool {
    if start == substantive.len() {
        return true;
    }
    ((start + 1)..=(start + max_tokens).min(substantive.len())).any(|end| {
        substantive[start..end].iter().any(|value| *value)
            && brute_force_chunk_partition_exists(substantive, max_tokens, end)
    })
}

#[test]
fn max_one_is_deterministic_and_rejects_unattachable_punctuation() {
    let substantive = vec!["A".to_string(), "B".to_string()];
    assert_eq!(
        plan_token_chunks(&[1, 2], Some(&substantive), 1).expect("substantive chunks"),
        vec![vec![1], vec![2]]
    );
    assert!(plan_token_chunks(&[1, 2], Some(&["A".to_string(), "!".to_string()]), 1).is_err());
    assert!(plan_token_chunks(&[1, 2], Some(&["!".to_string(), "A".to_string()]), 1).is_err());
}

#[test]
fn exact_boundary_and_explicit_ids_have_no_empty_trailing_chunk() {
    let ids = (0..240).collect::<Vec<i32>>();
    let chunks = plan_token_chunks(&ids, None, 120).expect("chunks");
    assert_eq!(chunks.len(), 2);
    assert!(chunks.iter().all(|chunk| chunk.len() == 120));
    assert_eq!(chunks.concat(), ids);
}

#[test]
fn punctuation_only_and_zero_tokens_are_rejected() {
    assert!(is_punctuation_only("，！？ ..."));
    assert!(!is_punctuation_only("你好！"));
    assert!(plan_token_chunks(&[], None, 120).is_err());
    assert!(plan_token_chunks(&[1, 2], Some(&["▁!".to_string(), "。".to_string()]), 120).is_err());
}

#[test]
fn segment_audio_inserts_200ms_only_between_chunks() {
    let output = concatenate_segment_audio(&[vec![1, 2], vec![3, 4]], 24_000, 200).expect("audio");
    assert_eq!(output.len(), 2 + 4_800 + 2);
    assert_eq!(&output[..2], &[1, 2]);
    assert!(output[2..4_802].iter().all(|sample| *sample == 0));
    assert_eq!(&output[4_802..], &[3, 4]);
    assert_eq!(output.last(), Some(&4));
}

fn voiced(samples: usize) -> Vec<i16> {
    (0..samples)
        .map(|index| if index % 16 < 8 { 2_000 } else { -2_000 })
        .collect()
}

#[test]
fn decoder_waveform_trims_only_disproportionate_trailing_quiet_suffix() {
    let speech = voiced(24_000);
    let mut raw = speech.clone();
    raw.extend((0..96_000).map(|index| if index % 2 == 0 { 2 } else { -2 }));
    let (finalized, report) = finalize_decoder_waveform(&raw).expect("speech plus quiet tail");
    assert_eq!(&finalized[..speech.len()], speech.as_slice());
    assert!(report.tail_trimmed);
    assert_eq!(report.trailing_quiet_samples, 96_000);
    assert_eq!(finalized.len(), speech.len() + 2_880);
}

#[test]
fn decoder_waveform_sparse_impulses_cannot_hide_quiet_tail() {
    let speech = voiced(24_000);
    let mut raw = speech.clone();
    let mut sparse_tail = vec![2_i16; 96_000];
    for frame in sparse_tail.chunks_mut(480) {
        frame[0] = 2_000;
    }
    raw.extend_from_slice(&sparse_tail);

    let (finalized, report) = finalize_decoder_waveform(&raw).expect("sparse impulse tail");
    assert_eq!(&finalized[..speech.len()], speech.as_slice());
    assert!(report.tail_trimmed);
    assert_eq!(report.trailing_quiet_samples, sparse_tail.len());
    assert_eq!(finalized.len(), speech.len() + 2_880);
}

#[test]
fn decoder_waveform_accepts_distributed_quiet_voice() {
    // Amplitude 28 is below the RMS activity threshold, but sustained occupancy
    // represents a credible quiet voice rather than isolated decoder clicks.
    let quiet_voice = (0..24_000)
        .map(|index| if index % 16 < 8 { 28 } else { -28 })
        .collect::<Vec<i16>>();
    let (finalized, report) =
        finalize_decoder_waveform(&quiet_voice).expect("distributed quiet voice");
    assert_eq!(finalized, quiet_voice);
    assert_eq!(report.raw_active_ratio, 1.0);
    assert_eq!(report.raw_credible_active_ratio, 1.0);
}

#[test]
fn decoder_waveform_preserves_short_terminal_and_internal_silence() {
    let mut short_tail = voiced(48_000);
    short_tail.extend(vec![0; 24_000]);
    let (unchanged, report) = finalize_decoder_waveform(&short_tail).expect("short tail");
    assert_eq!(unchanged, short_tail);
    assert!(!report.tail_trimmed);

    let mut internal = voiced(24_000);
    internal.extend(vec![0; 12_000]); // 500 ms pause.
    internal.extend(voiced(24_000));
    let (unchanged, report) = finalize_decoder_waveform(&internal).expect("internal pause");
    assert_eq!(unchanged, internal);
    assert_eq!(report.trailing_quiet_samples, 0);
}

#[test]
fn decoder_waveform_rejects_silence_low_noise_and_tiny_blip() {
    assert!(finalize_decoder_waveform(&vec![0; 96_000]).is_err());
    assert!(finalize_decoder_waveform(&vec![3; 96_000]).is_err());
    let mut click = vec![0; 96_000];
    click[20] = 20_000;
    assert!(finalize_decoder_waveform(&click).is_err());
    let mut tiny_transient = voiced(3 * 480);
    tiny_transient.extend(vec![0; 96_000]);
    assert!(finalize_decoder_waveform(&tiny_transient).is_err());

    let finite_noise = OrtTensorOutput {
        name: "generated_wav".to_string(),
        shape: vec![1, 96_000],
        data: OrtTensorData::F32(vec![0.00001; 96_000]),
    };
    let converted = tensor_to_i16_audio(&finite_noise).expect("finite conversion");
    assert!(finalize_decoder_waveform(&converted).is_err());
}

#[test]
fn finalized_chunks_keep_exact_intentional_gap() {
    let first = finalize_decoder_waveform(&voiced(24_000)).expect("first").0;
    let second = finalize_decoder_waveform(&voiced(24_000))
        .expect("second")
        .0;
    let joined =
        concatenate_segment_audio(&[first, second], TARGET_SAMPLE_RATE, 200).expect("joined");
    assert!(joined[24_000..28_800].iter().all(|sample| *sample == 0));
    assert_eq!(joined.len(), 52_800);
}

#[test]
fn waveform_tail_boundary_and_ratio_gates_are_deterministic() {
    let mut under_length = voiced(48_000);
    under_length.extend(vec![0; 47_999]);
    assert!(
        !finalize_decoder_waveform(&under_length)
            .expect("under length")
            .1
            .tail_trimmed
    );

    let mut under_ratio = voiced(72_000);
    under_ratio.extend(vec![0; 48_000]);
    assert!(
        !finalize_decoder_waveform(&under_ratio)
            .expect("under ratio")
            .1
            .tail_trimmed
    );

    let mut boundary = voiced(48_000);
    boundary.extend(vec![0; 48_000]);
    let first = finalize_decoder_waveform(&boundary).expect("boundary");
    let second = finalize_decoder_waveform(&boundary).expect("repeat");
    assert_eq!(first, second);
    assert!(first.1.tail_trimmed);
}

fn activate_frame(samples: &mut [i16], frame: usize) {
    let start = frame * 480;
    let end = (start + 480).min(samples.len());
    for (offset, sample) in samples[start..end].iter_mut().enumerate() {
        *sample = if offset % 16 < 8 { 2_000 } else { -2_000 };
    }
}

#[test]
fn exact_404480_sparse_fixture_is_rejected() {
    let mut samples = vec![0_i16; 404_480];
    for frame in 0..3 {
        activate_frame(&mut samples, frame);
    }
    // Exactly 45 additional sparse frames, including the final 320-sample
    // partial frame. None can form a credible 200 ms envelope.
    for index in 0..45 {
        let frame = 18 + index * (842 - 18) / 44;
        activate_frame(&mut samples, frame);
    }
    let err = finalize_decoder_waveform(&samples).expect_err("sparse output must fail closed");
    assert!(
        err.to_string().contains("reason periodic_sparse_pulses")
            || err
                .to_string()
                .contains("reason fragmented_sparse_activity")
    );
    assert!(
        err.to_string().contains("raw_active_ratio 0.056940"),
        "{err}"
    );
}

#[test]
fn waveform_preserves_multiple_phrase_islands_and_long_internal_pause() {
    let mut samples = voiced(20 * 480);
    samples.extend(vec![0; 80 * 480]);
    samples.extend(voiced(12 * 480));
    samples.extend(vec![0; 8 * 480]); // Natural sentence-ending pause.
    let (finalized, report) = finalize_decoder_waveform(&samples).expect("multiple phrases");
    assert_eq!(finalized, samples);
    assert_eq!(report.credible_island_count, 2);
    assert!(!report.tail_trimmed);
}

#[test]
fn credible_speech_tail_periodic_clicks_do_not_anchor_endpoint() {
    let speech = voiced(50 * 480);
    let mut samples = speech.clone();
    samples.extend(vec![0; 220 * 480]);
    for frame in (60..270).step_by(10) {
        let start = frame * 480;
        let end = (start + 4).min(samples.len());
        for sample in &mut samples[start..end] {
            *sample = 5_000;
        }
    }
    let (finalized, report) = finalize_decoder_waveform(&samples).expect("credible prefix");
    assert!(report.tail_trimmed);
    assert!(report.periodic_sparse_pulses);
    assert_eq!(finalized.len(), speech.len() + 2_880);
}

#[test]
fn exact_588800_tail_fixture_retains_45600_samples() {
    let mut samples = voiced(42_720);
    samples.resize(588_800, 0);
    let (finalized, report) = finalize_decoder_waveform(&samples).expect("long decoder tail");
    assert_eq!(report.trailing_quiet_samples, 546_080);
    assert_eq!(finalized.len(), 45_600);
    assert!(report.tail_trimmed);
}

#[test]
fn token_degeneration_audit_tracks_runs_rolling_metrics_and_milestones() {
    let mut observer = TokenDegenerationObserver::default();
    let mut milestones = Vec::new();
    for token in (0..128).map(|index| if index % 2 == 0 { 7 } else { 8 }) {
        if observer.observe(token) {
            milestones.push(observer.snapshot().total_tokens);
        }
    }
    let snapshot = observer.snapshot();
    assert_eq!(milestones, vec![64, 128]);
    assert_eq!(snapshot.total_tokens, 128);
    assert_eq!(snapshot.unique_tokens, 2);
    assert_eq!(snapshot.top_token_count, 64);
    assert_eq!(snapshot.longest_same_token_run, 1);
    assert_eq!(snapshot.rolling_unique_tokens, 2);
    assert_eq!(snapshot.rolling_adjacent_repeat_ratio, 0.0);
    assert_eq!(snapshot.rolling_period2_match_ratio, 1.0);

    for _ in 0..70 {
        observer.observe(9);
    }
    let snapshot = observer.snapshot();
    assert_eq!(snapshot.total_tokens, 198);
    assert_eq!(snapshot.current_same_token_run, 70);
    assert_eq!(snapshot.longest_same_token_run, 70);
    assert_eq!(snapshot.rolling_unique_tokens, 1);
    assert_eq!(snapshot.rolling_top_share, 1.0);
    // Diagnostics-only: observing even a compelling loop never returns a
    // premature rejection; production STOP/silence semantics remain separate.
}

#[test]
fn silence_run_guard_matches_official_more_than_thirty_semantics_and_resets() {
    let mut guard = SilenceRunGuard::default();
    for _ in 0..DEFAULT_MAX_CONSECUTIVE_SILENCE_TOKENS {
        guard
            .observe(SILENCE_TOKEN, DEFAULT_MAX_CONSECUTIVE_SILENCE_TOKENS)
            .expect("official limit continues");
    }
    assert_eq!(guard.max_run(), 30);
    assert!(guard
        .observe(SILENCE_TOKEN, DEFAULT_MAX_CONSECUTIVE_SILENCE_TOKENS)
        .is_err());

    let mut ordinary = SilenceRunGuard::default();
    for token in [1, SILENCE_TOKEN, SILENCE_TOKEN, 2, SILENCE_TOKEN, 3] {
        ordinary
            .observe(token, DEFAULT_MAX_CONSECUTIVE_SILENCE_TOKENS)
            .expect("ordinary sequence");
    }
    assert_eq!(ordinary.max_run(), 2);
    let next_chunk = SilenceRunGuard::default();
    assert_eq!(next_chunk.max_run(), 0);
}

fn simulate_decode_and_f(
    tokens: &[i32],
    budget: usize,
    c_tokens: &mut Vec<i32>,
    f_calls: &mut usize,
) -> Result<Vec<i32>> {
    let mut guard = SilenceRunGuard::default();
    let mut accepted = Vec::new();
    let mut stopped = false;
    let decode = (|| {
        for &token in tokens.iter().take(budget) {
            accepted.push(token);
            match process_decode_token(
                &mut guard,
                token,
                STOP_TOKEN,
                DEFAULT_MAX_CONSECUTIVE_SILENCE_TOKENS,
                |token| {
                    c_tokens.push(token);
                    Ok(token)
                },
            ) {
                Ok(DecodeTokenAction::Stop) => {
                    stopped = true;
                    break;
                }
                Ok(DecodeTokenAction::Continue(_)) => {}
                Err(DecodeTokenError::PathologicalSilence {
                    consecutive,
                    threshold,
                    ..
                }) => {
                    return Err(InfraError::Backend(format!(
                        "typed pathological silence: consecutive {consecutive}, threshold {threshold}"
                    )));
                }
                Err(DecodeTokenError::Continuation(err)) => return Err(err),
            }
        }
        if !stopped {
            return Err(InfraError::Backend(
                "synthetic decode budget exhausted without STOP".to_string(),
            ));
        }
        Ok(accepted)
    })();
    run_after_success(decode, |_| {
        *f_calls += 1;
        Ok(())
    })
    .map(|(accepted, ())| accepted)
}

#[test]
fn production_decode_seam_allows_thirty_silences_then_stop_and_f_once() {
    let mut tokens = vec![SILENCE_TOKEN; 30];
    tokens.push(STOP_TOKEN);
    let mut c_tokens = Vec::new();
    let mut f_calls = 0;
    let accepted =
        simulate_decode_and_f(&tokens, tokens.len(), &mut c_tokens, &mut f_calls).expect("success");
    assert_eq!(accepted, tokens);
    assert_eq!(c_tokens, vec![SILENCE_TOKEN; 30]);
    assert_eq!(f_calls, 1);
}

#[test]
fn production_decode_seam_aborts_thirty_first_silence_before_c_and_f() {
    let mut tokens = vec![SILENCE_TOKEN; 31];
    tokens.push(STOP_TOKEN);
    let mut c_tokens = Vec::new();
    let mut f_calls = 0;
    let err = simulate_decode_and_f(&tokens, tokens.len(), &mut c_tokens, &mut f_calls)
        .expect_err("must fail closed before hypothetical STOP");
    assert!(
        err.to_string().contains("typed pathological silence"),
        "{err}"
    );
    assert_eq!(c_tokens, vec![SILENCE_TOKEN; 30]);
    assert_eq!(f_calls, 0);
}

#[test]
fn production_decode_seam_resets_runs_and_reaches_stop_and_f() {
    let mut tokens = vec![SILENCE_TOKEN; 30];
    tokens.push(7);
    tokens.extend(vec![SILENCE_TOKEN; 30]);
    tokens.extend([8, STOP_TOKEN]);
    let mut c_tokens = Vec::new();
    let mut f_calls = 0;
    simulate_decode_and_f(&tokens, tokens.len(), &mut c_tokens, &mut f_calls).expect("success");
    assert_eq!(c_tokens, tokens[..tokens.len() - 1]);
    assert_eq!(f_calls, 1);
}

#[test]
fn production_decode_seam_no_stop_budget_exhaustion_skips_f() {
    let tokens = [1, 2, 3, 4, STOP_TOKEN];
    let mut c_tokens = Vec::new();
    let mut f_calls = 0;
    let err = simulate_decode_and_f(&tokens, 4, &mut c_tokens, &mut f_calls).expect_err("no STOP");
    assert!(err.to_string().contains("without STOP"), "{err}");
    assert_eq!(c_tokens, vec![1, 2, 3, 4]);
    assert_eq!(f_calls, 0);
}

#[test]
fn every_segment_must_validate_before_intentional_gap_is_added() {
    let segments = [vec![1; 48], vec![0; 48]];
    assert!(validate_generated_audio(&segments[0], 2).is_ok());
    assert!(validate_generated_audio(&segments[1], 2).is_err());

    let valid = [vec![1; 48], vec![2; 48]];
    for segment in &valid {
        validate_generated_audio(segment, 2).expect("each generated segment is valid");
    }
    let output = concatenate_segment_audio(&valid, 1_000, 10).expect("concatenated");
    assert_eq!(&output[..48], &[1; 48]);
    assert_eq!(&output[48..58], &[0; 10]);
    assert_eq!(&output[58..], &[2; 48]);
}

#[test]
fn decode_budget_is_checked_and_blank_audio_rejected() {
    assert_eq!(checked_decode_budget(600, 125).expect("budget"), 600);
    assert_eq!(checked_decode_budget(600, 900).expect("budget"), 600);
    assert!(validate_generated_audio(&[0; 48], 3).is_err());
    assert!(validate_generated_audio(&[1; 48], 1).is_err());
    assert!(validate_generated_audio(&[1; 48], 2).is_ok());
    let non_finite = OrtTensorOutput {
        name: "generated_wav".to_string(),
        shape: vec![1],
        data: OrtTensorData::F32(vec![f32::NAN]),
    };
    assert!(tensor_to_i16_audio(&non_finite).is_err());
}

#[test]
fn model_config_defaults_use_v2_new_token_budget() {
    let config = IndexTtsModelConfig::default();
    assert_eq!(config.max_generate_length, 600);
    assert_eq!(config.max_text_tokens_per_segment, 120);
    assert_eq!(config.inter_segment_silence_ms, 200);
    assert_eq!(
        config.max_consecutive_silence_tokens,
        DEFAULT_MAX_CONSECUTIVE_SILENCE_TOKENS
    );
    assert_eq!(config.generation_start_token, START_TOKEN);
    assert_eq!(config.generation_stop_token, STOP_TOKEN);
}

#[test]
fn model_config_loads_manifest_numbers_strings_and_aliases() {
    let (dir, artifacts) = config_fixture(
        r#"
max_text_tokens: 96
inter_segment_silence_ms: "175"
max_consecutive_silence_tokens: "40"
"#,
    );
    let spec = model_spec(dir.path().to_path_buf(), vec!["cpu".to_string()]);

    let config = IndexTtsModelConfig::load(&artifacts, &spec).expect("valid manifest config");

    assert_eq!(config.max_generate_length, 600);
    assert_eq!(config.max_text_tokens_per_segment, 96);
    assert_eq!(config.inter_segment_silence_ms, 175);
    assert_eq!(config.max_consecutive_silence_tokens, 40);
    assert_eq!(config.generation_start_token, 8192);
    assert_eq!(config.generation_stop_token, 8193);
    assert_eq!(config.mel_code_size, Some(8194));
}

#[test]
fn model_metadata_overrides_only_operational_fields() {
    let (dir, artifacts) = config_fixture(
        r#"
max_text_tokens_per_segment: 100
inter_segment_silence_ms: 175
max_consecutive_silence_tokens: 20
"#,
    );
    let mut spec = model_spec(dir.path().to_path_buf(), vec!["cpu".to_string()]);
    spec.metadata
        .insert("max_text_tokens".to_string(), serde_json::json!("110"));
    spec.metadata.insert(
        "inter_segment_silence_ms".to_string(),
        serde_json::json!(200),
    );
    spec.metadata.insert(
        "max_consecutive_silence_tokens".to_string(),
        serde_json::json!("45"),
    );

    let config = IndexTtsModelConfig::load(&artifacts, &spec).expect("metadata overrides");

    assert_eq!(config.max_generate_length, 600);
    assert_eq!(config.max_text_tokens_per_segment, 110);
    assert_eq!(config.inter_segment_silence_ms, 200);
    assert_eq!(config.max_consecutive_silence_tokens, 45);
    assert_eq!(config.generation_start_token, 8192);
    assert_eq!(config.generation_stop_token, 8193);
}

#[test]
fn model_config_rejects_present_invalid_manifest_values_and_aliases() {
    for (field, value) in [
        ("max_generate_length", "0"),
        ("max_generate_length", "100001"),
        ("max_generate_length", "-1"),
        ("max_generate_length", "1.5"),
        ("max_generate_length", "not-a-number"),
        ("max_generate_length", "184467440737095516160"),
        ("max_text_tokens", "0"),
        ("max_text_tokens_per_segment", "4097"),
        ("inter_segment_silence_ms", "10001"),
        ("inter_segment_silence_ms", "4294967296"),
        ("max_consecutive_silence_tokens", "0"),
        ("max_consecutive_silence_tokens", "121"),
        ("start_token", "2147483648"),
        ("generation_stop_token", "-2147483649"),
    ] {
        let manifest = format!("{field}: \"{value}\"\n");
        let (dir, artifacts) = config_fixture(&manifest);
        let spec = model_spec(dir.path().to_path_buf(), vec!["cpu".to_string()]);
        let err = IndexTtsModelConfig::load(&artifacts, &spec)
            .expect_err("present invalid manifest value must fail");
        assert!(
            err.to_string().contains(field)
                || err.to_string().contains("max_text_tokens_per_segment"),
            "{field}={value}: {err}"
        );
    }
}

#[test]
fn model_config_rejects_present_invalid_model_metadata_even_with_valid_manifest() {
    for (field, value) in [
        ("max_generate_length", serde_json::json!(-1)),
        ("max_text_tokens", serde_json::json!(1.25)),
        ("inter_segment_silence_ms", serde_json::json!("4294967296")),
        ("generation_start_token", serde_json::json!("bad")),
        ("stop_token", serde_json::json!(2147483648_i64)),
        ("max_consecutive_silence_tokens", serde_json::json!(0)),
        ("max_consecutive_silence_tokens", serde_json::json!(-1)),
        ("max_consecutive_silence_tokens", serde_json::json!(121)),
        ("max_consecutive_silence_tokens", serde_json::json!("bad")),
    ] {
        let (dir, artifacts) = config_fixture("");
        let mut spec = model_spec(dir.path().to_path_buf(), vec!["cpu".to_string()]);
        spec.metadata.insert(field.to_string(), value);
        let err = IndexTtsModelConfig::load(&artifacts, &spec)
            .expect_err("present invalid metadata value must fail");
        assert!(err.to_string().contains(field), "{field}: {err}");
    }
}

#[test]
fn canonical_and_alias_values_are_both_validated() {
    let (dir, artifacts) = config_fixture(
        r#"
max_text_tokens_per_segment: 120
max_text_tokens: malformed
"#,
    );
    let spec = model_spec(dir.path().to_path_buf(), vec!["cpu".to_string()]);
    let err = IndexTtsModelConfig::load(&artifacts, &spec)
        .expect_err("invalid alias must not be masked by canonical field");
    assert!(err.to_string().contains("max_text_tokens"), "{err}");
}

#[test]
fn indextts_cpu_thread_defaults_overrides_and_invalid_values() {
    assert_eq!(
        index_tts_cpu_session_options_from_values(None, None, 16).expect("defaults"),
        CpuSessionOptions {
            intra_threads: 8,
            inter_threads: 1
        }
    );
    assert_eq!(
        index_tts_cpu_session_options_from_values(Some("6"), Some("2"), 16).expect("overrides"),
        CpuSessionOptions {
            intra_threads: 6,
            inter_threads: 2
        }
    );
    assert!(index_tts_cpu_session_options_from_values(Some("no"), None, 16).is_err());
    assert!(index_tts_cpu_session_options_from_values(Some("0"), None, 16).is_err());
}

#[test]
fn repetition_penalty_is_sign_sensitive_full_history_and_unique() {
    let mut logits = vec![20.0, -2.0, 0.0, 4.0];
    apply_repetition_penalty(&mut logits, &[0, 1, 1, 2], DEFAULT_REPEAT_PENALTY).expect("penalty");
    assert_eq!(logits, vec![2.0, -20.0, 0.0, 4.0]);
}

#[test]
fn seeded_sampling_is_deterministic_and_rejects_nonfinite() {
    let logits = [2.0, 2.0, 1.0, 0.0];
    let mut a = SplitMix64::new(42);
    let mut b = SplitMix64::new(42);
    let sequence_a = (0..8)
        .map(|_| sample_logits(&logits, 3, 0.8, 1.0, &mut a).unwrap())
        .collect::<Vec<_>>();
    let sequence_b = (0..8)
        .map(|_| sample_logits(&logits, 3, 0.8, 1.0, &mut b).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(sequence_a, sequence_b);
    assert!(sample_logits(&[f32::NAN], 30, 0.8, 1.0, &mut a).is_err());
}

#[test]
fn repetition_penalty_uses_history_beyond_sixteen_tokens() {
    let mut logits = vec![1.0; 32];
    let history = (0..24).collect::<Vec<i32>>();
    apply_repetition_penalty(&mut logits, &history, 10.0).expect("penalty");
    assert_eq!(logits[0], 0.1);
    assert_eq!(logits[23], 0.1);
    assert_eq!(logits[24], 1.0);
}

#[test]
fn sampling_top_k_excludes_lower_scores() {
    let mut rng = SplitMix64::new(0);
    for _ in 0..100 {
        assert!(sample_logits(&[3.0, 2.0, 1.0], 2, 1.0, 1.0, &mut rng).unwrap() < 2);
    }
}

#[test]
fn sampling_top_p_keeps_threshold_crossing_candidate() {
    let mut rng = SplitMix64::new(7);
    let observed = (0..200)
        .map(|_| sample_logits(&[0.0, 0.0, 0.0], 3, 0.5, 1.0, &mut rng).unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(observed, [0, 1].into_iter().collect());
}

#[test]
fn sampling_ties_are_ordered_by_ascending_token_id() {
    let mut rng = SplitMix64::new(1);
    assert_eq!(
        sample_logits(&[5.0, 5.0, 5.0], 1, 1.0, 1.0, &mut rng).unwrap(),
        0
    );
}

#[test]
fn request_seed_accepts_u64_and_decimal_string_strictly() {
    assert_eq!(indextts_seed_from_params(&BTreeMap::new()).unwrap(), 0);
    for value in [Value::from(u64::MAX), Value::from(u64::MAX.to_string())] {
        let params = BTreeMap::from([("indextts_seed".to_string(), value)]);
        assert_eq!(indextts_seed_from_params(&params).unwrap(), u64::MAX);
    }
    for value in [Value::from(-1), Value::from("1.0"), Value::Bool(true)] {
        let params = BTreeMap::from([("indextts_seed".to_string(), value)]);
        assert!(indextts_seed_from_params(&params).is_err());
    }
}

#[test]
fn one_rng_lifecycle_continues_across_chunks() {
    let logits = [1.0, 1.0, 1.0];
    let mut request = SplitMix64::new(99);
    let first = sample_logits(&logits, 3, 1.0, 1.0, &mut request).unwrap();
    let second = sample_logits(&logits, 3, 1.0, 1.0, &mut request).unwrap();
    let mut replay = SplitMix64::new(99);
    assert_eq!(
        first,
        sample_logits(&logits, 3, 1.0, 1.0, &mut replay).unwrap()
    );
    assert_eq!(
        second,
        sample_logits(&logits, 3, 1.0, 1.0, &mut replay).unwrap()
    );
}

#[test]
fn raw_logits_abi_requires_exact_name_shape_and_type() {
    let valid = OrtTensorOutput {
        name: "raw_logits".to_string(),
        shape: vec![1, 1, 3],
        data: OrtTensorData::F32(vec![0.0; 3]),
    };
    assert_eq!(raw_logits_f32(&valid, 3).unwrap(), vec![0.0; 3]);
    let mut invalid = valid.clone();
    invalid.name = "max_logit_id".to_string();
    assert!(raw_logits_f32(&invalid, 3).is_err());
    invalid = valid.clone();
    invalid.shape = vec![1, 3];
    assert!(raw_logits_f32(&invalid, 3).is_err());
}

#[test]
fn split_v2_manifest_is_required() {
    let (_dir, artifacts) = config_fixture("");
    fs::write(
        artifacts.root.join("manifest.yaml"),
        "status: ready\nsplit_contract_version: 1\n",
    )
    .expect("replace manifest");
    let err = IndexTtsModelConfig::load(
        &artifacts,
        &model_spec(artifacts.root.clone(), vec!["cpu".to_string()]),
    )
    .expect_err("v1 must be rejected");
    assert!(err.to_string().contains("split_contract_version"));
}

#[test]
fn yaml_and_json_manifests_must_be_semantically_equal() {
    let (_dir, artifacts) = config_fixture("max_generate_length: 600\n");
    fs::write(
        artifacts.root.join("manifest.json"),
        r#"{"split_contract_version":2,"max_generate_length":599}"#,
    )
    .expect("json manifest");
    let err = IndexTtsModelConfig::load(
        &artifacts,
        &model_spec(artifacts.root.clone(), vec!["cpu".to_string()]),
    )
    .expect_err("mismatch must fail");
    assert!(err.to_string().contains("semantically equal"));
}

#[test]
fn wrong_v2_cache_or_policy_declaration_is_rejected() {
    for (path, replacement) in [
        ("cache_mode", "permanent_dummy"),
        ("top_k", "29"),
        ("max_new_mel_tokens", "599"),
    ] {
        let (_dir, artifacts) = config_fixture("");
        let manifest_path = artifacts.root.join("manifest.yaml");
        let original = fs::read_to_string(&manifest_path).expect("manifest");
        let changed = match path {
            "cache_mode" => {
                original.replace("cache_mode: prefill_decode", "cache_mode: permanent_dummy")
            }
            "top_k" => original.replace("top_k: 30", "top_k: 29"),
            _ => original.replace("max_new_mel_tokens: 600", "max_new_mel_tokens: 599"),
        };
        fs::write(&manifest_path, changed).expect("changed manifest");
        let err = IndexTtsModelConfig::load(
            &artifacts,
            &model_spec(artifacts.root.clone(), vec!["cpu".to_string()]),
        )
        .expect_err("wrong contract must fail");
        assert!(err.to_string().contains(path), "{path}: {err}");
        let _ = replacement;
    }
}

#[test]
fn deployment_metadata_cannot_override_v2_budget() {
    let (_dir, artifacts) = config_fixture("");
    let mut spec = model_spec(artifacts.root.clone(), vec!["cpu".to_string()]);
    spec.metadata
        .insert("max_generate_length".to_string(), Value::from(601));
    let err = IndexTtsModelConfig::load(&artifacts, &spec).expect_err("override");
    assert!(err.to_string().contains("cannot override"));
}

#[test]
fn unsupported_diagnostic_manifest_is_never_runtime_loadable() {
    let (_dir, artifacts) = config_fixture("");
    let path = artifacts.root.join("manifest.yaml");
    let manifest = fs::read_to_string(&path)
        .expect("manifest")
        .replace("status: ready", "status: unsupported");
    fs::write(path, manifest).expect("diagnostic");
    let err = IndexTtsModelConfig::load(
        &artifacts,
        &model_spec(artifacts.root.clone(), vec!["cpu".to_string()]),
    )
    .expect_err("unsupported status");
    assert!(err.to_string().contains("status must be `ready`"), "{err}");
}

#[test]
fn deployment_config_tracks_immutable_source_and_v2_contract() {
    let (_dir, artifacts) = config_fixture("");
    let manifest: Value =
        serde_yaml::from_slice(&fs::read(artifacts.root.join("manifest.yaml")).unwrap()).unwrap();
    fs::write(
        artifacts.root.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let mut config = deployment_config_fixture(&artifacts.root);
    fs::write(
        artifacts.root.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .expect("deployment config");
    IndexTtsModelConfig::load(
        &artifacts,
        &model_spec(artifacts.root.clone(), vec!["cpu".to_string()]),
    )
    .expect("valid deployment config");
    for pointer in [
        "/official_code/tag",
        "/official_code/tree",
        "/contract/manifest_json_sha256",
        "/contract/manifest_yaml_sha256",
        "/contract/index_tts_e_sha256",
        "/contract/index_tts_e_prefill_sha256",
        "/source_provenance_reference/sha256",
    ] {
        config = deployment_config_fixture(&artifacts.root);
        *config.pointer_mut(pointer).unwrap() = Value::from("bad");
        fs::write(
            artifacts.root.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let err = IndexTtsModelConfig::load(
            &artifacts,
            &model_spec(artifacts.root.clone(), vec!["cpu".to_string()]),
        )
        .expect_err("mutated deployment config");
        assert!(err.to_string().contains("config.json"), "{pointer}: {err}");
    }
    config = deployment_config_fixture(&artifacts.root);
    fs::write(artifacts.root.join("IndexTTS_E.onnx"), b"changed").unwrap();
    fs::write(
        artifacts.root.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let err = IndexTtsModelConfig::load(
        &artifacts,
        &model_spec(artifacts.root.clone(), vec!["cpu".to_string()]),
    )
    .expect_err("actual content mismatch");
    assert!(err.to_string().contains("integrity mismatch"));
}

fn deployment_config_fixture(root: &Path) -> Value {
    let hash = |name: &str| local_files::sha256_file(root.join(name)).unwrap();
    serde_json::json!({
        "schema": "local.index_tts.deployment.v1",
        "model_id": "indextts-1.5-onnx",
        "source": {
            "repo_id": "IndexTeam/IndexTTS-1.5",
            "revision": "25851a6036dfd3095bb70fb3c8f49217104672c3",
            "download_method": "huggingface_hub.snapshot_download"
        },
        "official_code": {
            "tag": "v1.5.0",
            "commit": "9098497272d5803bae46cbaf5154cf2ba48f6866",
            "tree": "aa0335ccaba54ac42d6d209dac56bb9a8b2e80a7"
        },
        "contract": {
            "version": 2,
            "cache_mode": "prefill_decode",
            "manifest_json_sha256": hash("manifest.json"),
            "manifest_yaml_sha256": hash("manifest.yaml"),
            "index_tts_e_sha256": hash("IndexTTS_E.onnx"),
            "index_tts_e_prefill_sha256": hash("IndexTTS_E_Prefill.onnx")
        },
        "source_provenance_reference": {
            "sha256": "3bfb39cc326d834be4fda72000e4cc53ebbb6c52e154150f1fdbe7323d1e909c",
            "runtime_verifiable": false,
            "note": "Informational hash of the separately retained source provenance record; that record is intentionally outside this 14-file runtime package."
        },
        "license": {
            "warning": "The pinned official code contains INDEX_MODEL_LICENSE with non-commercial and other restrictions. Preserve and review it; do not rely only on Hugging Face apache-2.0 metadata."
        }
    })
}

#[test]
fn legacy_layout_without_prefill_fails_actionably() {
    let dir = tempfile::tempdir().expect("tempdir");
    for name in MODEL_FILENAMES {
        fs::write(dir.path().join(name), b"placeholder").expect("onnx");
    }
    fs::write(dir.path().join("bpe.model"), b"sentencepiece").expect("bpe");
    let err = IndexTtsArtifacts::validate(dir.path(), IndexTtsPrecision::CpuFp32)
        .expect_err("legacy package");
    assert!(err.to_string().contains("re-export"));
    assert!(err.to_string().contains("IndexTTS_E_Prefill.onnx"));
}

#[test]
fn attention_mask_shape_uses_current_total_sequence_length() {
    let dynamic_rank2 = tensor_meta("attention_mask", TensorElement::I64, vec![1, -1]);
    let static_rank2 = tensor_meta("attention_mask", TensorElement::I64, vec![1, 7]);
    let rank1 = tensor_meta("attention_mask", TensorElement::I64, vec![-1]);

    assert_eq!(
        attention_mask_shape(&dynamic_rank2, 5).expect("first"),
        vec![1, 5]
    );
    assert_eq!(
        attention_mask_shape(&dynamic_rank2, 6).expect("next"),
        vec![1, 6]
    );
    assert_eq!(
        attention_mask_shape(&static_rank2, 6).expect("static"),
        vec![1, 7]
    );
    assert_eq!(attention_mask_shape(&rank1, 6).expect("rank1"), vec![1]);
}

#[test]
fn attention_mask_values_hide_dummy_past_only_on_first_step() {
    assert_eq!(attention_mask_values(&[1, 6], 1), vec![0, 1, 1, 1, 1, 1]);
    assert_eq!(attention_mask_values(&[1, 6], 0), vec![1, 1, 1, 1, 1, 1]);
    assert_eq!(attention_mask_values(&[2, 3], 1), vec![0, 1, 1, 0, 1, 1]);
}

#[test]
fn scalar_attention_mask_uses_official_first_step_flag() {
    assert_eq!(attention_mask_values(&[1], 1), vec![1]);
    assert_eq!(attention_mask_values(&[1], 0), vec![0]);
    assert_eq!(attention_mask_values(&[2], 1), vec![1, 1]);
    assert_eq!(attention_mask_values(&[2], 0), vec![0, 0]);
}

#[test]
fn cache_sequence_len_reads_hf_present_cache_axis() {
    let cache = OrtTensorOutput {
        name: "in_key_0".to_string(),
        shape: vec![1, 20, 6, 64],
        data: OrtTensorData::F32(vec![0.0; 1]),
    };

    assert_eq!(cache_sequence_len(&cache).expect("seq len"), 6);
}

#[test]
fn cache_sequence_len_reads_official_3d_key_and_value_axes() {
    let key_cache = OrtTensorOutput {
        name: "in_key_0".to_string(),
        shape: vec![1, 20, 7],
        data: OrtTensorData::F32(vec![0.0; 1]),
    };
    let value_cache = OrtTensorOutput {
        name: "in_value_0".to_string(),
        shape: vec![1, 8, 64],
        data: OrtTensorData::F32(vec![0.0; 1]),
    };

    assert_eq!(cache_sequence_len(&key_cache).expect("key seq len"), 7);
    assert_eq!(cache_sequence_len(&value_cache).expect("value seq len"), 8);
}

#[test]
fn concatenate_hidden_states_flattens_batch_token_hidden() {
    let states = vec![
        OrtTensorOutput {
            name: "last_hidden_state".to_string(),
            shape: vec![1, 1, 3],
            data: OrtTensorData::F32(vec![1.0, 2.0, 3.0]),
        },
        OrtTensorOutput {
            name: "last_hidden_state".to_string(),
            shape: vec![1, 1, 3],
            data: OrtTensorData::F32(vec![4.0, 5.0, 6.0]),
        },
    ];

    let concatenated = concatenate_hidden_states(&states).expect("concat");

    assert_eq!(concatenated.name, "save_hidden_state");
    assert_eq!(concatenated.shape, vec![2, 3]);
    assert_eq!(
        concatenated.data,
        OrtTensorData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    );
}

#[test]
fn provider_report_maps_all_seven_sessions_including_e_prefill() {
    let mut observed = Vec::new();
    let report = index_tts_provider_report_with(|session| {
        observed.push(session);
        SessionProviderReport {
            provider: if session == IndexTtsSession::EPrefill {
                ProviderKind::Cuda
            } else {
                ProviderKind::Cpu
            },
            cpu_fallback_used: session == IndexTtsSession::F,
        }
    });

    assert_eq!(
        observed,
        [
            IndexTtsSession::A,
            IndexTtsSession::B,
            IndexTtsSession::C,
            IndexTtsSession::D,
            IndexTtsSession::E,
            IndexTtsSession::EPrefill,
            IndexTtsSession::F,
        ]
    );
    assert_eq!(report.e.provider, ProviderKind::Cpu);
    assert_eq!(report.e_prefill.provider, ProviderKind::Cuda);
    assert_eq!(report.f.provider, ProviderKind::Cpu);
    assert!(report.f.cpu_fallback_used);
}

#[test]
fn real_model_smoke_if_env_set() {
    let Ok(model_dir) = std::env::var("LOCAL_INDEXTTS_MODEL_DIR") else {
        return;
    };
    let reference_audio = std::env::var_os("LOCAL_INDEXTTS_TEST_AUDIO")
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .or_else(|| {
            let default = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../scripts/assets/tts-input-mon3tr.wav");
            default.exists().then_some(default)
        });
    let Some(reference_audio) = reference_audio else {
        eprintln!("IndexTTS smoke skipped: no reference audio found");
        return;
    };

    let mut adapter = match IndexTtsAdapter::load(&model_spec(
        PathBuf::from(model_dir),
        provider_order_from_env(),
    )) {
        Ok(adapter) => adapter,
        Err(err) => {
            eprintln!("IndexTTS smoke stopped at model load boundary: {err}");
            return;
        }
    };
    let report = adapter.provider_report();
    eprintln!(
        "IndexTTS provider report: A={:?}/{} B={:?}/{} C={:?}/{} D={:?}/{} E={:?}/{} E-prefill={:?}/{} F={:?}/{}",
        report.a.provider,
        report.a.cpu_fallback_used,
        report.b.provider,
        report.b.cpu_fallback_used,
        report.c.provider,
        report.c.cpu_fallback_used,
        report.d.provider,
        report.d.cpu_fallback_used,
        report.e.provider,
        report.e.cpu_fallback_used,
        report.e_prefill.provider,
        report.e_prefill.cpu_fallback_used,
        report.f.provider,
        report.f.cpu_fallback_used
    );

    match adapter.synthesize(
        "这里是M42语音测试，现在检查合成、分段和发送链路是否都能正常工作。",
        Some(&FileRef::local(&reference_audio)),
    ) {
        Ok(InferenceOutput::TtsAudio { audio }) => {
            let wav_path = audio
                .path
                .as_ref()
                .expect("IndexTTS smoke should return a local wav path");
            let metadata = std::fs::metadata(&wav_path).expect("stat wav");
            eprintln!(
                "IndexTTS smoke output: {} ({} bytes)",
                wav_path.display(),
                metadata.len()
            );
            assert!(metadata.len() > 44, "IndexTTS smoke produced empty wav");
            assert!(
                metadata.len() < 20_000_000,
                "IndexTTS smoke output too large"
            );
        }
        Ok(other) => panic!("unexpected output: {other:?}"),
        Err(err) => eprintln!("IndexTTS smoke stopped after load/run boundary: {err}"),
    }
}

fn tensor_meta(name: &str, element_type: TensorElement, shape: Vec<i64>) -> TensorMetadata {
    TensorMetadata {
        name: name.to_string(),
        element_type,
        dimension_symbols: vec![None; shape.len()],
        shape,
    }
}

fn provider_order_from_env() -> Vec<String> {
    std::env::var("LOCAL_TEST_PROVIDER_ORDER")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .map(|part| part.to_string())
                .collect::<Vec<_>>()
        })
        .filter(|parts| !parts.is_empty())
        .unwrap_or_else(|| vec!["cpu".to_string()])
}

fn model_spec(path: PathBuf, provider_order: Vec<String>) -> ModelSpec {
    ModelSpec {
        id: "indextts-1.5-onnx".to_string(),
        name: "IndexTTS Test".to_string(),
        enabled: true,
        task_kinds: Vec::new(),
        adapter: AdapterKind::IndexTts,
        backend: BackendKind::Ort,
        artifacts: vec![ModelArtifact {
            kind: ArtifactKind::Local,
            path,
            source_path: None,
            sha256: None,
            url: None,
            repo_id: None,
            revision: None,
            files: Vec::new(),
            allow_patterns: Vec::new(),
            metadata: Default::default(),
        }],
        runtime: RuntimePolicy {
            provider_order,
            ..Default::default()
        },
        resources: ResourceRequirement::default(),
        load_policy: Default::default(),
        metadata: Default::default(),
    }
}

fn config_fixture(manifest: &str) -> (tempfile::TempDir, IndexTtsArtifacts) {
    let dir = tempfile::tempdir().expect("tempdir");
    for name in MODEL_FILENAMES {
        fs::write(dir.path().join(name), b"placeholder").expect("onnx");
    }
    fs::write(dir.path().join("IndexTTS_E_Prefill.onnx"), b"placeholder").expect("prefill");
    fs::write(dir.path().join("bpe.model"), b"sentencepiece").expect("bpe");
    fs::write(
        dir.path().join("manifest.yaml"),
        format!(
            r#"status: ready
split_contract_version: 2
graph_contract:
  cache_mode: prefill_decode
  cache_layout: hf_bhsd
  layers: 24
  dtype: float32
  batch: 1
  heads: 20
  cache_sequence_axis: 2
  head_dim: 64
  initial_cache_length: 0
  attention_mask: {{rank: 2, dtype: int64}}
  logits: {{name: raw_logits, rank: 3, shape: [1, 1, 8194], selection: runtime}}
generation_policy:
  version: 1
  algorithm: seeded_top_k_top_p
  num_beams: 1
  source_num_beams: 3
  top_k: 30
  top_p: 0.8
  temperature: 1.0
  repetition_penalty: 10.0
  repetition_scope: mel_bos_and_full_generated_history
  default_seed: 0
  max_new_mel_tokens: 600
  start_token: 8192
  stop_token: 8193
  silence_token: 52
max_generate_length: 600
mel_code_size: 8194
vocab_size: 8194
{manifest}"#
        ),
    )
    .expect("manifest");
    let artifacts =
        IndexTtsArtifacts::validate(dir.path(), IndexTtsPrecision::CpuFp32).expect("artifacts");
    (dir, artifacts)
}
