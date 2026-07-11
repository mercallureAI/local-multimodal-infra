use super::*;

#[derive(Debug)]
pub struct IndexTtsAdapter {
    model_id: String,
    artifacts: IndexTtsArtifacts,
    a: OrtSession,
    b: OrtSession,
    c: OrtSession,
    d: OrtSession,
    e: OrtSession,
    f: OrtSession,
    tokenizer: SentencePieceTokenizer,
    config: IndexTtsModelConfig,
    output_dir: PathBuf,
}

impl IndexTtsAdapter {
    pub fn load(spec: &ModelSpec) -> Result<Self> {
        let precision = IndexTtsPrecision::from_spec(spec);
        let root = IndexTtsArtifacts::resolve(spec);
        let artifacts = IndexTtsArtifacts::validate(root, precision)?;
        let cpu_options = index_tts_cpu_session_options_from_env()?;
        let backend = OrtBackend::new(ProviderSelection::from_strings(
            &spec.runtime.provider_order,
        ))
        .with_cpu_session_options(cpu_options)?;
        let a = backend.load_session(&artifacts.a)?;
        let b = backend.load_session(&artifacts.b)?;
        let c = backend.load_session(&artifacts.c)?;
        let d = backend.load_session(&artifacts.d)?;
        let e = backend.load_session(&artifacts.e)?;
        let f = backend.load_session(&artifacts.f)?;
        validate_sessions([&a, &b, &c, &d, &e, &f])?;
        let tokenizer = SentencePieceTokenizer::load(&artifacts.bpe_model)?;
        let config = IndexTtsModelConfig::load(&artifacts, spec)?;
        let output_dir = env::var_os("LOCAL_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("workdir/data"));
        Ok(Self {
            model_id: spec.id.clone(),
            artifacts,
            a,
            b,
            c,
            d,
            e,
            f,
            tokenizer,
            config,
            output_dir,
        })
    }

    pub fn synthesize(
        &mut self,
        text: &str,
        reference_audio: Option<&FileRef>,
    ) -> Result<InferenceOutput> {
        self.synthesize_with_params(text, reference_audio, &BTreeMap::new())
    }

    pub fn synthesize_with_params(
        &mut self,
        text: &str,
        reference_audio: Option<&FileRef>,
        params: &BTreeMap<String, Value>,
    ) -> Result<InferenceOutput> {
        let reference_audio = reference_audio.ok_or_else(|| {
            InfraError::BadRequest("IndexTTS synthesis requires reference_audio".to_string())
        })?;
        let reference_path = local_files::local_path(reference_audio)?;
        let reference = audio::read_wav_mono_i16_24k(&reference_path)?;
        let explicit_ids = explicit_text_token_ids_from_params(params)?;
        if is_punctuation_only(text) && explicit_ids.is_none() {
            return Err(InfraError::BadRequest(
                "IndexTTS text must contain something other than whitespace or punctuation"
                    .to_string(),
            ));
        }
        let chunks = match explicit_ids {
            Some(ids) => plan_token_chunks(&ids, None, self.config.max_text_tokens_per_segment)?,
            None => {
                let prepared = preprocess_text_for_index_tts(text);
                let (ids, pieces) = self.tokenizer.encode_ids_and_pieces(&prepared)?;
                plan_token_chunks(&ids, Some(&pieces), self.config.max_text_tokens_per_segment)?
            }
        };
        let reference_state = self.run_a(&reference)?;
        let mut segment_audio = Vec::with_capacity(chunks.len());
        for (chunk_index, text_ids) in chunks.iter().enumerate() {
            let generated = self.run_bf(&reference_state, text_ids, chunk_index, chunks.len())?;
            segment_audio.push(generated);
        }
        let generated = concatenate_segment_audio(
            &segment_audio,
            TARGET_SAMPLE_RATE,
            self.config.inter_segment_silence_ms,
        )?;
        let output = self.write_wav(&generated)?;
        Ok(InferenceOutput::TtsAudio { audio: output })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn artifacts(&self) -> &IndexTtsArtifacts {
        &self.artifacts
    }

    pub fn provider_report(&self) -> IndexTtsProviderReport {
        IndexTtsProviderReport {
            a: self.a.provider_report(),
            b: self.b.provider_report(),
            c: self.c.provider_report(),
            d: self.d.provider_report(),
            e: self.e.provider_report(),
            f: self.f.provider_report(),
        }
    }

    fn run_bf(
        &mut self,
        reference_state: &ReferenceState,
        text_ids: &[i32],
        chunk_index: usize,
        chunk_count: usize,
    ) -> Result<Vec<i16>> {
        let text_hidden = self.run_b(text_ids)?;
        let c_start = self.run_c_token(self.config.generation_start_token, 0)?;
        let concat = self.run_d(
            &reference_state.conds_latent,
            &text_hidden,
            &c_start.gpt_hidden_state,
        )?;
        let concat_len = concat.concat_len;
        let decode = self.run_e_loop(concat, c_start.next_gen_len)?;
        let wav = self.run_f(reference_state, &decode.save_hidden_state)?;
        validate_generated_audio(&wav, decode.generated_steps)?;
        tracing::info!(
            chunk_index = chunk_index + 1,
            chunk_count,
            token_count = text_ids.len(),
            concat_len,
            generated_steps = decode.generated_steps,
            sample_count = wav.len(),
            stopped = decode.stopped,
            "IndexTTS synthesized chunk"
        );
        Ok(wav)
    }

    fn run_a(&mut self, samples: &[i16]) -> Result<ReferenceState> {
        require_input(&self.a, "audio", "IndexTTS_A")?;
        let outputs = self
            .a
            .run_tensors(&[tensor_i16(
                "audio",
                vec![1, 1, samples.len()],
                samples.to_vec(),
            )])
            .map_err(|err| session_error("IndexTTS_A", &self.a, err))?;
        let conds_latent = find_output(outputs.clone(), "conds_latent", "IndexTTS_A", &self.a)?;
        Ok(ReferenceState {
            outputs,
            conds_latent,
        })
    }

    fn run_b(&mut self, text_ids: &[i32]) -> Result<OrtTensorOutput> {
        require_input(&self.b, "text_ids", "IndexTTS_B")?;
        let outputs = self
            .b
            .run_tensors(&[tensor_i32(
                "text_ids",
                vec![1, text_ids.len()],
                text_ids.to_vec(),
            )])
            .map_err(|err| session_error("IndexTTS_B", &self.b, err))?;
        find_output(outputs, "text_hidden_state", "IndexTTS_B", &self.b)
    }

    fn run_c_token(&mut self, token: i32, gen_len: i64) -> Result<CState> {
        let len_input = c_len_input_name(&self.c)?.to_string();
        require_inputs(&self.c, &["gpt_ids", &len_input], "IndexTTS_C")?;
        let outputs = self
            .c
            .run_tensors(&[
                tensor_i32("gpt_ids", vec![1, 1], vec![token]),
                tensor_i64(&len_input, vec![1], vec![gen_len]),
            ])
            .map_err(|err| session_error("IndexTTS_C", &self.c, err))?;
        let gpt_hidden_state =
            find_output(outputs.clone(), "gpt_hidden_state", "IndexTTS_C", &self.c)?;
        let next_gen_len =
            find_named_output(&outputs, &["next_gen_len", "next_kv_seq_len", "kv_seq_len"])
                .and_then(|output| first_i64_or_i32(output, &output.name).ok())
                .unwrap_or(gen_len + 1);
        Ok(CState {
            gpt_hidden_state,
            next_gen_len,
        })
    }

    fn run_d(
        &mut self,
        conds_latent: &OrtTensorOutput,
        text_hidden: &OrtTensorOutput,
        gpt_hidden: &OrtTensorOutput,
    ) -> Result<ConcatState> {
        require_inputs(&self.d, &["embed_x", "embed_y", "embed_z"], "IndexTTS_D")?;
        let outputs = self
            .d
            .run_tensors(&[
                clone_as_input("embed_x", conds_latent)?,
                clone_as_input("embed_y", text_hidden)?,
                clone_as_input("embed_z", gpt_hidden)?,
            ])
            .map_err(|err| session_error("IndexTTS_D", &self.d, err))?;
        let hidden_state = find_output(
            outputs.clone(),
            "concat_hidden_state",
            "IndexTTS_D",
            &self.d,
        )?;
        let concat_len = find_output(outputs, "concat_len", "IndexTTS_D", &self.d)?;
        let concat_len = first_i64_or_i32(&concat_len, "concat_len")?.max(1) as usize;
        Ok(ConcatState {
            hidden_state,
            concat_len,
        })
    }

    fn run_e_loop(&mut self, concat: ConcatState, mut gen_len: i64) -> Result<DecodeState> {
        require_inputs(
            &self.e,
            &[
                "history_len",
                "repeat_penality",
                "ids_len",
                "hidden_state",
                "attention_mask",
            ],
            "IndexTTS_E",
        )?;
        let layer_count = infer_layer_count(self.e.outputs().len())?;
        let mel_code_size = infer_repeat_penalty_width(&self.e, &self.config)?;
        let mut keys = (0..layer_count)
            .map(|idx| empty_cache(&self.e, &format!("in_key_{idx}")))
            .collect::<Result<Vec<_>>>()?;
        let mut values = (0..layer_count)
            .map(|idx| empty_cache(&self.e, &format!("in_value_{idx}")))
            .collect::<Result<Vec<_>>>()?;
        let mut history = Vec::<i32>::new();
        let mut repeat_penalty = vec![1.0_f32; mel_code_size];
        let mut hidden_state = concat.hidden_state;
        let mut history_len = 0_i64;
        let mut hidden_states = Vec::<OrtTensorOutput>::new();

        let budget = checked_decode_budget(self.config.max_generate_length, concat.concat_len)?;
        let mut stopped = false;
        for step_index in 0..budget {
            let mut inputs = Vec::new();
            for idx in 0..layer_count {
                inputs.push(clone_as_input(&format!("in_key_{idx}"), &keys[idx])?);
                inputs.push(clone_as_input(&format!("in_value_{idx}"), &values[idx])?);
            }
            let first_step = step_index == 0;
            let (history_len_input, ids_len_input) =
                e_loop_control_lengths(first_step, concat.concat_len, history_len);
            // ORT tensor construction in this adapter cannot represent zero-length cache tensors,
            // so `empty_cache` materializes a one-token dummy cache for dynamic cache dimensions.
            // Build the attention mask against the actual cache tensor length that the exported
            // GPT receives as `past_key_values`, while keeping logical history_len/ids_len inputs
            // unchanged for the E graph's own control flow.
            let cache_len = cache_sequence_len(keys.first().ok_or_else(|| {
                InfraError::Backend("IndexTTS_E has no key cache tensors".to_string())
            })?)?;
            let attention_mask_len = cache_len + ids_len_input;
            let masked_prefix_len = if first_step { cache_len } else { 0 };
            inputs.push(tensor_i64("history_len", vec![1], vec![history_len_input]));
            inputs.push(tensor_f32(
                "repeat_penality",
                vec![1, mel_code_size],
                repeat_penalty.clone(),
            ));
            inputs.push(tensor_i64("ids_len", vec![1], vec![ids_len_input]));
            inputs.push(clone_as_input("hidden_state", &hidden_state)?);
            inputs.push(attention_mask_input(
                &self.e,
                attention_mask_len,
                masked_prefix_len,
            )?);

            let outputs = self
                .e
                .run_tensors(&inputs)
                .map_err(|err| session_error("IndexTTS_E", &self.e, err))?;
            let step = parse_e_outputs(outputs, layer_count, &self.e)?;
            keys = step.keys;
            values = step.values;
            history_len = step.kv_seq_len;
            hidden_states.push(step.last_hidden_state.clone());
            let token = step.max_logit_id;
            if token == self.config.generation_stop_token {
                stopped = true;
                break;
            }
            apply_repeat_penalty_token(
                &mut repeat_penalty,
                &mut history,
                token,
                DEFAULT_REPEAT_WINDOW,
                DEFAULT_REPEAT_PENALTY,
            );
            let c_step = self.run_c_token(token, gen_len)?;
            gen_len = c_step.next_gen_len;
            hidden_state = c_step.gpt_hidden_state;
        }
        if !stopped {
            return Err(InfraError::Backend(format!(
                "IndexTTS_E exhausted decode budget {budget} without STOP (concat_len {}, max_generate_length {})",
                concat.concat_len, self.config.max_generate_length
            )));
        }
        if hidden_states.len() < 2 {
            return Err(InfraError::Backend(format!(
                "IndexTTS_E stopped too early after {} decode step(s); refusing incomplete/blank synthesis",
                hidden_states.len()
            )));
        }

        Ok(DecodeState {
            save_hidden_state: concatenate_hidden_states(&hidden_states)?,
            generated_steps: hidden_states.len(),
            stopped,
        })
    }

    fn run_f(
        &mut self,
        reference: &ReferenceState,
        save_hidden_state: &OrtTensorOutput,
    ) -> Result<Vec<i16>> {
        let by_name = reference
            .outputs
            .iter()
            .map(|output| (output.name.as_str(), output))
            .collect::<BTreeMap<_, _>>();
        let mut inputs = Vec::new();
        for input in self.f.inputs() {
            if input.name == "save_hidden_state" {
                inputs.push(clone_as_input("save_hidden_state", save_hidden_state)?);
            } else if let Some(output) = by_name.get(input.name.as_str()) {
                inputs.push(clone_as_input(&input.name, output)?);
            } else {
                return Err(InfraError::Backend(format!(
                    "IndexTTS_F input `{}` was not produced by IndexTTS_A; {}",
                    input.name,
                    format_session_io("IndexTTS_F", self.f.metadata())
                )));
            }
        }
        let outputs = self
            .f
            .run_tensors(&inputs)
            .map_err(|err| session_error("IndexTTS_F", &self.f, err))?;
        let wav = find_output(outputs, "generated_wav", "IndexTTS_F", &self.f)?;
        tensor_to_i16_audio(&wav)
    }

    fn write_wav(&self, samples: &[i16]) -> Result<FileRef> {
        fs::create_dir_all(&self.output_dir)
            .map_err(|e| InfraError::io(Some(self.output_dir.clone()), e))?;
        let path = self.output_dir.join(index_tts_wav_filename(Uuid::new_v4()));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: TARGET_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec)
            .map_err(|e| InfraError::Adapter(format!("create wav {}: {e}", path.display())))?;
        for &sample in samples {
            writer
                .write_sample(sample)
                .map_err(|e| InfraError::Adapter(format!("write wav sample: {e}")))?;
        }
        writer
            .finalize()
            .map_err(|e| InfraError::Adapter(format!("finalize wav {}: {e}", path.display())))?;
        let mut file = FileRef::local(path);
        file.mime = Some("audio/wav".to_string());
        Ok(file)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ReferenceState {
    pub(crate) outputs: Vec<OrtTensorOutput>,
    pub(crate) conds_latent: OrtTensorOutput,
}

#[derive(Debug, Clone)]
pub(crate) struct ConcatState {
    pub(crate) hidden_state: OrtTensorOutput,
    pub(crate) concat_len: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct CState {
    pub(crate) gpt_hidden_state: OrtTensorOutput,
    pub(crate) next_gen_len: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct EStep {
    pub(crate) keys: Vec<OrtTensorOutput>,
    pub(crate) values: Vec<OrtTensorOutput>,
    pub(crate) kv_seq_len: i64,
    pub(crate) last_hidden_state: OrtTensorOutput,
    pub(crate) max_logit_id: i32,
}

#[derive(Debug, Clone)]
pub(crate) struct DecodeState {
    pub(crate) save_hidden_state: OrtTensorOutput,
    pub(crate) generated_steps: usize,
    pub(crate) stopped: bool,
}

pub(crate) fn checked_decode_budget(
    max_generate_length: usize,
    concat_len: usize,
) -> Result<usize> {
    max_generate_length.checked_sub(concat_len).filter(|v| *v > 0).ok_or_else(|| {
        InfraError::Backend(format!(
            "IndexTTS has no decode budget: max_generate_length {max_generate_length}, concat_len {concat_len}"
        ))
    })
}

pub(crate) fn plan_token_chunks(
    ids: &[i32],
    pieces: Option<&[String]>,
    max_tokens: usize,
) -> Result<Vec<Vec<i32>>> {
    if ids.is_empty() {
        return Err(InfraError::BadRequest(
            "IndexTTS text produced zero tokens".to_string(),
        ));
    }
    if max_tokens == 0 {
        return Err(InfraError::BadRequest(
            "IndexTTS max text tokens per segment must be greater than zero".to_string(),
        ));
    }
    if pieces.is_some_and(|pieces| pieces.len() != ids.len()) {
        return Err(InfraError::Backend(
            "IndexTTS token id/piece counts differ".to_string(),
        ));
    }
    let Some(pieces) = pieces else {
        return Ok(ids.chunks(max_tokens).map(|chunk| chunk.to_vec()).collect());
    };
    if !pieces.iter().any(|piece| piece_has_substantive_text(piece)) {
        return Err(InfraError::BadRequest(
            "IndexTTS text tokens contain only punctuation".to_string(),
        ));
    }

    // `feasible[start]` records whether the complete suffix can be partitioned
    // into bounded chunks that each contain substantive content. Computing it
    // backwards makes reconstruction globally correct without exponential
    // backtracking: O(token_count * max_tokens) time and O(token_count) space.
    let mut substantive_prefix = Vec::with_capacity(pieces.len() + 1);
    substantive_prefix.push(0usize);
    for piece in pieces {
        substantive_prefix.push(
            substantive_prefix.last().copied().unwrap_or(0)
                + usize::from(piece_has_substantive_text(piece)),
        );
    }
    let mut feasible = vec![false; pieces.len() + 1];
    feasible[pieces.len()] = true;
    for start in (0..pieces.len()).rev() {
        feasible[start] = ((start + 1)..=start.saturating_add(max_tokens).min(pieces.len()))
            .any(|end| feasible[end] && chunk_has_substantive(&substantive_prefix, start, end));
    }
    if !feasible[0] {
        return Err(InfraError::BadRequest(format!(
            "IndexTTS punctuation cannot be attached to substantive tokens within max_text_tokens_per_segment={max_tokens}"
        )));
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < pieces.len() {
        let candidates = ((start + 1)..=start.saturating_add(max_tokens).min(pieces.len()))
            .filter(|end| feasible[*end] && chunk_has_substantive(&substantive_prefix, start, *end))
            .collect::<Vec<_>>();
        // Correctness is established by `feasible`. Among valid choices,
        // prefer the furthest punctuation boundary, then the furthest boundary
        // overall. This keeps chunks full while retaining sentence punctuation
        // when doing so cannot make the remaining suffix impossible.
        let end = candidates
            .iter()
            .rev()
            .copied()
            .find(|end| is_segment_boundary_piece(&pieces[*end - 1]))
            .or_else(|| candidates.last().copied())
            .expect("feasible suffix has at least one valid next boundary");
        chunks.push(ids[start..end].to_vec());
        start = end;
    }
    Ok(chunks)
}

fn chunk_has_substantive(prefix: &[usize], start: usize, end: usize) -> bool {
    prefix[end] > prefix[start]
}

fn piece_has_substantive_text(piece: &str) -> bool {
    piece.chars().any(|ch| {
        ch != '▁' && !ch.is_whitespace() && !ch.is_ascii_punctuation() && !is_cjk_punctuation(ch)
    })
}

fn is_segment_boundary_piece(piece: &str) -> bool {
    matches!(
        piece.trim_start_matches('▁'),
        "." | "!" | "?" | "…" | "," | ";" | ":"
    )
}

pub(crate) fn is_punctuation_only(text: &str) -> bool {
    !text.trim().is_empty()
        && text
            .chars()
            .all(|ch| ch.is_whitespace() || ch.is_ascii_punctuation() || is_cjk_punctuation(ch))
}

fn is_cjk_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '，' | '。'
            | '！'
            | '？'
            | '：'
            | '；'
            | '、'
            | '…'
            | '—'
            | '～'
            | '“'
            | '”'
            | '‘'
            | '’'
            | '（'
            | '）'
            | '《'
            | '》'
            | '【'
            | '】'
            | '「'
            | '」'
    )
}

pub(crate) fn concatenate_segment_audio(
    segments: &[Vec<i16>],
    sample_rate: u32,
    silence_ms: u32,
) -> Result<Vec<i16>> {
    if segments.is_empty() || segments.iter().any(Vec::is_empty) {
        return Err(InfraError::Backend(
            "IndexTTS cannot concatenate empty segment audio".to_string(),
        ));
    }
    let silence_len_u64 = u64::from(sample_rate)
        .checked_mul(u64::from(silence_ms))
        .ok_or_else(|| InfraError::Backend("IndexTTS silence length overflow".to_string()))?
        / 1000;
    let silence_len = usize::try_from(silence_len_u64)
        .map_err(|_| InfraError::Backend("IndexTTS silence length overflow".to_string()))?;
    let samples_len = segments.iter().try_fold(0usize, |total, segment| {
        total
            .checked_add(segment.len())
            .ok_or_else(|| InfraError::Backend("IndexTTS output length overflow".to_string()))
    })?;
    let silence_total = silence_len
        .checked_mul(segments.len().saturating_sub(1))
        .ok_or_else(|| InfraError::Backend("IndexTTS output length overflow".to_string()))?;
    let total = samples_len
        .checked_add(silence_total)
        .ok_or_else(|| InfraError::Backend("IndexTTS output length overflow".to_string()))?;
    let mut output = Vec::with_capacity(total);
    for (index, segment) in segments.iter().enumerate() {
        if index > 0 {
            let target_len = output.len().checked_add(silence_len).ok_or_else(|| {
                InfraError::Backend("IndexTTS output length overflow".to_string())
            })?;
            output.resize(target_len, 0);
        }
        output.extend_from_slice(segment);
    }
    Ok(output)
}

pub(crate) fn validate_generated_audio(samples: &[i16], generated_steps: usize) -> Result<()> {
    if generated_steps < 2 || samples.len() < 24 || samples.iter().all(|sample| *sample == 0) {
        return Err(InfraError::Backend(format!(
            "IndexTTS generated incomplete/blank audio (generated_steps {generated_steps}, samples {})",
            samples.len()
        )));
    }
    Ok(())
}

pub(crate) fn index_tts_cpu_session_options_from_values(
    intra: Option<&str>,
    inter: Option<&str>,
    logical_cpus: usize,
) -> Result<CpuSessionOptions> {
    fn parse(name: &str, value: Option<&str>, default: usize) -> Result<usize> {
        match value {
            None => Ok(default),
            Some(value) => value
                .parse::<usize>()
                .ok()
                .filter(|v| *v > 0)
                .ok_or_else(|| {
                    InfraError::BadRequest(format!(
                        "{name} must be a positive integer, got `{value}`"
                    ))
                }),
        }
    }
    Ok(CpuSessionOptions {
        intra_threads: parse(
            "LOCAL_INDEXTTS_ORT_INTRA_THREADS",
            intra,
            logical_cpus.clamp(1, 8),
        )?,
        inter_threads: parse("LOCAL_INDEXTTS_ORT_INTER_THREADS", inter, 1)?,
    })
}

fn index_tts_cpu_session_options_from_env() -> Result<CpuSessionOptions> {
    let logical_cpus = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    index_tts_cpu_session_options_from_values(
        env::var("LOCAL_INDEXTTS_ORT_INTRA_THREADS").ok().as_deref(),
        env::var("LOCAL_INDEXTTS_ORT_INTER_THREADS").ok().as_deref(),
        logical_cpus,
    )
}
