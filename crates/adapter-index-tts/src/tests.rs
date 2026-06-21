use super::*;
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
    assert_eq!(official_like("beta1"), "BETA1");
    assert_eq!(official_like("BETA1"), "BETA1");
    assert_eq!(official_like("voice2"), "VOICE2");
    assert_eq!(official_like("babala2"), "BABALA2");
    assert_eq!(official_like("AGAN3"), "AGAN3");
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
            "第 一 句 .SECOND SENTENCE! 第 三 句 ?",
        ),
    ];
    for (input, expected) in cases {
        assert_eq!(official_like(input), expected, "{input}");
    }
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
fn output_filename_uses_indextts_prefix_and_wav_suffix() {
    let id = Uuid::parse_str("00000000-0000-0000-0000-000000000123").expect("uuid");

    assert_eq!(
        index_tts_wav_filename(id),
        "indextts-00000000-0000-0000-0000-000000000123.wav"
    );
}

#[test]
fn repeat_penalty_width_prefers_manifest_then_static_metadata() {
    let meta = tensor_meta("repeat_penality", TensorElement::F32, vec![1, -1]);
    let config = IndexTtsModelConfig {
        mel_code_size: Some(8194),
        vocab_size: None,
    };
    assert_eq!(
        repeat_penalty_width_from_metadata(&meta, &config),
        Some(8194)
    );

    let meta = tensor_meta("repeat_penality", TensorElement::F32, vec![1, 2048]);
    assert_eq!(
        repeat_penalty_width_from_metadata(&meta, &IndexTtsModelConfig::default()),
        Some(2048)
    );

    let meta = tensor_meta("repeat_penality", TensorElement::F32, vec![1, -1]);
    assert_eq!(
        repeat_penalty_width_from_metadata(&meta, &IndexTtsModelConfig::default()),
        None
    );
}

#[test]
fn repeat_penalty_window_resets_evicted_tokens() {
    let mut penalties = vec![1.0; 6];
    let mut history = Vec::new();

    apply_repeat_penalty_token(&mut penalties, &mut history, 2, 2, DEFAULT_REPEAT_PENALTY);
    apply_repeat_penalty_token(&mut penalties, &mut history, 3, 2, DEFAULT_REPEAT_PENALTY);
    apply_repeat_penalty_token(&mut penalties, &mut history, 4, 2, DEFAULT_REPEAT_PENALTY);

    assert_eq!(history, vec![3, 4]);
    assert_eq!(penalties[2], 1.0);
    assert!(DEFAULT_REPEAT_PENALTY < 1.0);
    assert_eq!(penalties[3], DEFAULT_REPEAT_PENALTY);
    assert_eq!(penalties[4], DEFAULT_REPEAT_PENALTY);
}

#[test]
fn e_loop_control_lengths_use_prior_kv_sequence() {
    let concat_len = 5;

    assert_eq!(e_loop_control_lengths(true, concat_len, 0), (0, 5));
    assert_eq!(e_loop_control_lengths(false, concat_len, 5), (5, 1));
    assert_eq!(e_loop_control_lengths(false, concat_len, 6), (6, 1));
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
fn dynamic_cache_shape_uses_dummy_dimension_for_ort() {
    let meta = tensor_meta("in_key_0", TensorElement::F32, vec![1, 20, -1, 64]);

    let shape = concrete_or_empty_shape(&meta);

    assert_eq!(shape, vec![1, 20, 1, 64]);
    assert!(shape.iter().all(|dim| *dim >= 1));
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

fn tensor_meta(name: &str, element_type: TensorElement, shape: Vec<i64>) -> TensorMetadata {
    TensorMetadata {
        name: name.to_string(),
        element_type,
        dimension_symbols: vec![None; shape.len()],
        shape,
    }
}
