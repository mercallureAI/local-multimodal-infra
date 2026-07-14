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
    e_prefill: OrtSession,
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
        let e_prefill = backend.load_session(&artifacts.e_prefill)?;
        let f = backend.load_session(&artifacts.f)?;
        validate_sessions([&a, &b, &c, &d, &e, &f])?;
        validate_e_prefill(&e_prefill)?;
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
            e_prefill,
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
        self.synthesize_with_request_id(Uuid::new_v4(), text, reference_audio, params)
    }

    pub fn synthesize_with_request_id(
        &mut self,
        request_id: Uuid,
        text: &str,
        reference_audio: Option<&FileRef>,
        params: &BTreeMap<String, Value>,
    ) -> Result<InferenceOutput> {
        let total_started = std::time::Instant::now();
        let reference_audio = reference_audio.ok_or_else(|| {
            InfraError::BadRequest("IndexTTS synthesis requires reference_audio".to_string())
        })?;
        let reference_started = std::time::Instant::now();
        let reference_path = local_files::local_path(reference_audio)?;
        let reference = audio::read_wav_mono_i16_24k(&reference_path)?;
        let reference_read_ms = reference_started.elapsed().as_millis() as u64;
        let frontend_started = std::time::Instant::now();
        let explicit_ids = explicit_text_token_ids_from_params(params)?;
        let seed = indextts_seed_from_params(params)?;
        let mut rng = SplitMix64::new(seed);
        let uses_explicit_ids = explicit_ids.is_some();
        if is_punctuation_only(text) && explicit_ids.is_none() {
            return Err(InfraError::BadRequest(
                "IndexTTS text must contain something other than whitespace or punctuation"
                    .to_string(),
            ));
        }
        let (chunks, split_kind) = match explicit_ids {
            Some(ids) => {
                let chunks =
                    plan_token_chunks(&ids, None, self.config.max_text_tokens_per_segment)?;
                let kind = if chunks.len() == 1 { "none" } else { "hard" };
                (chunks, kind)
            }
            None => {
                let prepared = preprocess_text_for_index_tts(text);
                let (ids, pieces) = self.tokenizer.encode_ids_and_pieces(&prepared)?;
                let chunks = plan_token_chunks(
                    &ids,
                    Some(&pieces),
                    self.config.max_text_tokens_per_segment,
                )?;
                let kind =
                    classify_split_kind(&pieces, &chunks, self.config.max_text_tokens_per_segment);
                (chunks, kind)
            }
        };
        let frontend_ms = frontend_started.elapsed().as_millis() as u64;
        tracing::info!(
            request_id = %request_id,
            chunk_count = chunks.len(),
            chunk_token_counts = ?chunks.iter().map(Vec::len).collect::<Vec<_>>(),
            explicit_token_ids = uses_explicit_ids,
            split_kind,
            frontend_ms,
            reference_read_ms,
            "IndexTTS planned text chunks"
        );
        let a_started = std::time::Instant::now();
        let reference_state = self.run_a(&reference)?;
        let reference_a_ms = a_started.elapsed().as_millis() as u64;
        let mut segment_audio = Vec::with_capacity(chunks.len());
        let mut total_decode_steps = 0usize;
        for (chunk_index, text_ids) in chunks.iter().enumerate() {
            let (generated, decode_steps) = self.run_bf(
                request_id,
                &reference_state,
                text_ids,
                chunk_index,
                chunks.len(),
                &mut rng,
            )?;
            total_decode_steps += decode_steps;
            segment_audio.push(generated);
        }
        let generated = concatenate_segment_audio(
            &segment_audio,
            TARGET_SAMPLE_RATE,
            self.config.inter_segment_silence_ms,
        )?;
        let encode_started = std::time::Instant::now();
        let output = self.write_wav(&generated)?;
        tracing::info!(
            request_id = %request_id,
            frontend_ms,
            reference_read_ms,
            reference_a_ms,
            chunk_count = chunks.len(),
            split_kind,
            decode_steps = total_decode_steps,
            audio_samples = generated.len(),
            encode_write_ms = encode_started.elapsed().as_millis() as u64,
            total_ms = total_started.elapsed().as_millis() as u64,
            "IndexTTS synthesis stages"
        );
        Ok(InferenceOutput::TtsAudio { audio: output })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn artifacts(&self) -> &IndexTtsArtifacts {
        &self.artifacts
    }

    pub fn provider_report(&self) -> IndexTtsProviderReport {
        index_tts_provider_report_with(|session| match session {
            IndexTtsSession::A => self.a.provider_report(),
            IndexTtsSession::B => self.b.provider_report(),
            IndexTtsSession::C => self.c.provider_report(),
            IndexTtsSession::D => self.d.provider_report(),
            IndexTtsSession::E => self.e.provider_report(),
            IndexTtsSession::EPrefill => self.e_prefill.provider_report(),
            IndexTtsSession::F => self.f.provider_report(),
        })
    }

    fn run_bf(
        &mut self,
        request_id: Uuid,
        reference_state: &ReferenceState,
        text_ids: &[i32],
        chunk_index: usize,
        chunk_count: usize,
        rng: &mut SplitMix64,
    ) -> Result<(Vec<i16>, usize)> {
        let chunk_started = std::time::Instant::now();
        let b_started = std::time::Instant::now();
        let text_hidden = self.run_b(text_ids)?;
        let b_ms = b_started.elapsed().as_millis() as u64;
        let c_started = std::time::Instant::now();
        let c_start = self.run_c_token(self.config.generation_start_token, 0)?;
        let c_initial_ms = c_started.elapsed().as_millis() as u64;
        let d_started = std::time::Instant::now();
        let concat = self.run_d(
            &reference_state.conds_latent,
            &text_hidden,
            &c_start.gpt_hidden_state,
        )?;
        let d_ms = d_started.elapsed().as_millis() as u64;
        let concat_len = concat.concat_len;
        let budget = checked_decode_budget(self.config.max_generate_length, concat_len)?;
        let e_started = std::time::Instant::now();
        let decode_result = self.run_e_loop(
            request_id,
            chunk_index,
            chunk_count,
            concat,
            c_start.next_gen_len,
            rng,
        );
        // This loop includes each E invocation and continuation C invocation.
        let decode_loop_ms = e_started.elapsed().as_millis() as u64;
        let f_started = std::time::Instant::now();
        let (decode, raw_wav) = run_after_success(decode_result, |decode| {
            self.run_f(reference_state, &decode.save_hidden_state)
        })?;
        let f_ms = f_started.elapsed().as_millis() as u64;
        let (wav, waveform) = finalize_decoder_waveform(&raw_wav).map_err(|err| {
            tracing::warn!(
                request_id = %request_id,
                chunk_index = chunk_index + 1,
                chunk_count,
                generated_steps = decode.generated_steps,
                decode_budget = budget,
                silence_token_count = decode.silence_token_count,
                max_silence_run = decode.max_silence_run,
                f_input_rows = decode.f_input_rows,
                raw_samples = raw_wav.len(),
                quality_decision = "rejected",
                reason = "waveform_quality_rejected",
                error = %err,
                "IndexTTS chunk failed"
            );
            InfraError::Backend(format!(
                "IndexTTS request {request_id} chunk {}/{} waveform quality failure: {err}",
                chunk_index + 1,
                chunk_count
            ))
        })?;
        validate_generated_audio(&wav, decode.generated_steps)?;
        tracing::info!(
            request_id = %request_id,
            chunk_index = chunk_index + 1,
            chunk_count,
            token_count = text_ids.len(),
            concat_len,
            generated_steps = decode.generated_steps,
            decode_budget = budget,
            silence_token_count = decode.silence_token_count,
            max_silence_run = decode.max_silence_run,
            f_input_rows = decode.f_input_rows,
            raw_samples = waveform.raw_samples,
            final_samples = waveform.final_samples,
            trimmed_samples = waveform.trimmed_samples,
            trailing_quiet_samples = waveform.trailing_quiet_samples,
            peak = waveform.peak,
            rms = waveform.rms,
            raw_active_ratio = waveform.raw_active_ratio,
            raw_credible_active_ratio = waveform.raw_credible_active_ratio,
            final_active_ratio = waveform.final_active_ratio,
            final_credible_active_ratio = waveform.final_credible_active_ratio,
            credible_island_count = waveform.credible_island_count,
            longest_credible_run_frames = waveform.longest_credible_run_frames,
            short_glitch_count = waveform.short_glitch_count,
            raw_late_active_ratio = waveform.raw_late_active_ratio,
            credible_late_active_ratio = waveform.credible_late_active_ratio,
            periodic_sparse_pulses = waveform.periodic_sparse_pulses,
            quality_decision = waveform.quality_decision,
            quality_reason = waveform.quality_reason,
            tail_trimmed = waveform.tail_trimmed,
            token_unique = decode.degeneration.unique_tokens,
            token_top = ?decode.degeneration.top_token,
            token_top_count = decode.degeneration.top_token_count,
            token_longest_run = decode.degeneration.longest_same_token_run,
            rolling_unique = decode.degeneration.rolling_unique_tokens,
            rolling_top_share = decode.degeneration.rolling_top_share,
            rolling_adjacent_repeat_ratio = decode.degeneration.rolling_adjacent_repeat_ratio,
            rolling_period2_match_ratio = decode.degeneration.rolling_period2_match_ratio,
            stopped = decode.stopped,
            b_ms,
            c_initial_ms,
            d_ms,
            decode_loop_ms,
            f_ms,
            chunk_total_ms = chunk_started.elapsed().as_millis() as u64,
            "IndexTTS synthesized chunk"
        );
        Ok((wav, decode.generated_steps))
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

    fn run_e_loop(
        &mut self,
        request_id: Uuid,
        chunk_index: usize,
        chunk_count: usize,
        concat: ConcatState,
        mut gen_len: i64,
        rng: &mut SplitMix64,
    ) -> Result<DecodeState> {
        require_inputs(&self.e, &["hidden_state", "attention_mask"], "IndexTTS_E")?;
        let layer_count = infer_layer_count(self.e.outputs().len())?;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        let mut history = vec![self.config.generation_start_token];
        let mut hidden_state = concat.hidden_state;
        let mut hidden_states = Vec::<OrtTensorOutput>::new();
        let mut silence_guard = SilenceRunGuard::default();
        let mut degeneration = TokenDegenerationObserver::default();

        let budget = checked_decode_budget(self.config.max_generate_length, concat.concat_len)?;
        let mut stopped = false;
        for step_index in 0..budget {
            let mut inputs = Vec::new();
            let first_step = step_index == 0;
            if !first_step {
                for idx in 0..layer_count {
                    inputs.push(clone_as_input(&format!("in_key_{idx}"), &keys[idx])?);
                    inputs.push(clone_as_input(&format!("in_value_{idx}"), &values[idx])?);
                }
            }
            let cache_len = if first_step {
                0
            } else {
                cache_sequence_len(&keys[0])?
            };
            let ids_len = if first_step {
                concat.concat_len as i64
            } else {
                1
            };
            inputs.push(clone_as_input("hidden_state", &hidden_state)?);
            inputs.push(attention_mask_input(
                if first_step { &self.e_prefill } else { &self.e },
                cache_len + ids_len,
                0,
            )?);

            let session = if first_step {
                &mut self.e_prefill
            } else {
                &mut self.e
            };
            let outputs = session.run_tensors(&inputs).map_err(|err| {
                InfraError::Backend(format!("IndexTTS_E v2 execution failed: {err}"))
            })?;
            let step = parse_e_outputs(outputs, layer_count, session)?;
            keys = step.keys;
            values = step.values;
            hidden_states.push(step.last_hidden_state.clone());
            let mut logits = raw_logits_f32(&step.raw_logits, DEFAULT_MEL_CODE_SIZE)?;
            apply_repetition_penalty(&mut logits, &history, DEFAULT_REPEAT_PENALTY)?;
            let token = sample_logits(&logits, 30, 0.8, 1.0, rng)?;
            let degeneration_milestone = degeneration.observe(token);
            if degeneration_milestone {
                let snapshot = degeneration.snapshot();
                log_token_degeneration(
                    request_id,
                    chunk_index,
                    chunk_count,
                    "milestone",
                    &snapshot,
                );
            }
            let decision = process_decode_token(
                &mut silence_guard,
                token,
                self.config.generation_stop_token,
                self.config.max_consecutive_silence_tokens,
                |token| self.run_c_token(token, gen_len),
            );
            let decision = match decision {
                Ok(decision) => decision,
                Err(DecodeTokenError::PathologicalSilence {
                    consecutive,
                    threshold,
                    silence_token_count,
                }) => {
                    let snapshot = degeneration.snapshot();
                    log_token_degeneration(
                        request_id,
                        chunk_index,
                        chunk_count,
                        "pathological_silence_reject",
                        &snapshot,
                    );
                    let err = InfraError::Backend(format!(
                        "IndexTTS request {request_id} chunk {}/{} decode rejected: pathological silence loop (silence_token {SILENCE_TOKEN}, consecutive {consecutive}, threshold {threshold}, silence_token_count {silence_token_count})",
                        chunk_index + 1,
                        chunk_count
                    ));
                    tracing::warn!(
                        request_id = %request_id,
                        chunk_index = chunk_index + 1,
                        chunk_count,
                        generated_steps = hidden_states.len(),
                        decode_budget = budget,
                        silence_token_count = silence_guard.silence_token_count(),
                        max_silence_run = silence_guard.max_run(),
                        token_unique = snapshot.unique_tokens,
                        token_longest_run = snapshot.longest_same_token_run,
                        rolling_unique = snapshot.rolling_unique_tokens,
                        rolling_top_share = snapshot.rolling_top_share,
                        rolling_adjacent_repeat_ratio = snapshot.rolling_adjacent_repeat_ratio,
                        quality_decision = "rejected",
                        reason = "pathological_silence_token_run",
                        error = %err,
                        "IndexTTS decode aborted before continuation and vocoder"
                    );
                    return Err(err);
                }
                Err(DecodeTokenError::Continuation(source)) => {
                    let snapshot = degeneration.snapshot();
                    log_token_degeneration(
                        request_id,
                        chunk_index,
                        chunk_count,
                        "continuation_reject",
                        &snapshot,
                    );
                    let err = InfraError::Backend(format!(
                        "IndexTTS request {request_id} chunk {}/{} decode rejected: continuation failed: {source}",
                        chunk_index + 1,
                        chunk_count
                    ));
                    tracing::warn!(
                        request_id = %request_id,
                        chunk_index = chunk_index + 1,
                        chunk_count,
                        generated_steps = hidden_states.len(),
                        decode_budget = budget,
                        quality_decision = "rejected",
                        reason = "decode_continuation_failed",
                        error = %err,
                        "IndexTTS decode failed before vocoder"
                    );
                    return Err(err);
                }
            };
            if step_index < 8
                || step_index.is_power_of_two()
                || token == self.config.generation_stop_token
            {
                tracing::debug!(
                    request_id = %request_id,
                    chunk_index = chunk_index + 1,
                    step = step_index + 1,
                    token_id = token,
                    is_stop = token == self.config.generation_stop_token,
                    is_silence = token == SILENCE_TOKEN,
                    consecutive_silence = silence_guard.consecutive(),
                    "IndexTTS decode token"
                );
            }
            if matches!(decision, DecodeTokenAction::Stop) {
                stopped = true;
                break;
            }
            history.push(token);
            let DecodeTokenAction::Continue(c_step) = decision else {
                unreachable!("STOP handled above")
            };
            gen_len = c_step.next_gen_len;
            hidden_state = c_step.gpt_hidden_state;
        }
        if !stopped {
            let snapshot = degeneration.snapshot();
            log_token_degeneration(
                request_id,
                chunk_index,
                chunk_count,
                "decode_budget_exhausted",
                &snapshot,
            );
            let err = InfraError::Backend(format!(
                "IndexTTS request {request_id} chunk {}/{} decode rejected: exhausted budget {budget} without STOP (concat_len {}, max_generate_length {})",
                chunk_index + 1,
                chunk_count,
                concat.concat_len,
                self.config.max_generate_length
            ));
            tracing::warn!(
                request_id = %request_id,
                chunk_index = chunk_index + 1,
                chunk_count,
                generated_steps = hidden_states.len(),
                decode_budget = budget,
                silence_token_count = silence_guard.silence_token_count(),
                max_silence_run = silence_guard.max_run(),
                quality_decision = "rejected",
                reason = "decode_budget_exhausted_without_stop",
                error = %err,
                "IndexTTS decode failed before vocoder"
            );
            return Err(err);
        }
        if hidden_states.len() < 2 {
            let snapshot = degeneration.snapshot();
            log_token_degeneration(
                request_id,
                chunk_index,
                chunk_count,
                "too_early_stop_reject",
                &snapshot,
            );
            let err = InfraError::Backend(format!(
                "IndexTTS request {request_id} chunk {}/{} decode rejected: STOP after only {} decode step(s); refusing incomplete/blank synthesis",
                chunk_index + 1,
                chunk_count,
                hidden_states.len()
            ));
            tracing::warn!(
                request_id = %request_id,
                chunk_index = chunk_index + 1,
                chunk_count,
                generated_steps = hidden_states.len(),
                decode_budget = budget,
                quality_decision = "rejected",
                reason = "decode_stopped_too_early",
                error = %err,
                "IndexTTS decode failed before vocoder"
            );
            return Err(err);
        }

        let degeneration = degeneration.snapshot();
        log_token_degeneration(request_id, chunk_index, chunk_count, "stop", &degeneration);
        let save_hidden_state = concatenate_hidden_states(&hidden_states)?;
        let f_input_rows = save_hidden_state.shape.first().copied().unwrap_or(0);
        Ok(DecodeState {
            save_hidden_state,
            generated_steps: hidden_states.len(),
            stopped,
            silence_token_count: silence_guard.silence_token_count(),
            max_silence_run: silence_guard.max_run(),
            degeneration,
            f_input_rows,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IndexTtsSession {
    A,
    B,
    C,
    D,
    E,
    EPrefill,
    F,
}

pub(crate) fn index_tts_provider_report_with(
    mut report: impl FnMut(IndexTtsSession) -> SessionProviderReport,
) -> IndexTtsProviderReport {
    IndexTtsProviderReport {
        a: report(IndexTtsSession::A),
        b: report(IndexTtsSession::B),
        c: report(IndexTtsSession::C),
        d: report(IndexTtsSession::D),
        e: report(IndexTtsSession::E),
        e_prefill: report(IndexTtsSession::EPrefill),
        f: report(IndexTtsSession::F),
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DecodeTokenAction<T> {
    Stop,
    Continue(T),
}

#[derive(Debug)]
pub(crate) enum DecodeTokenError {
    PathologicalSilence {
        consecutive: usize,
        threshold: usize,
        silence_token_count: usize,
    },
    Continuation(InfraError),
}

/// Production decode decision seam. STOP has precedence and never invokes C;
/// a pathological silence token is rejected before C; only ordinary tokens
/// invoke the continuation callback.
pub(crate) fn process_decode_token<T>(
    guard: &mut SilenceRunGuard,
    token: i32,
    stop_token: i32,
    threshold: usize,
    continue_c: impl FnOnce(i32) -> Result<T>,
) -> std::result::Result<DecodeTokenAction<T>, DecodeTokenError> {
    if token == stop_token {
        return Ok(DecodeTokenAction::Stop);
    }
    guard
        .observe(token, threshold)
        .map_err(|_| DecodeTokenError::PathologicalSilence {
            consecutive: guard.consecutive(),
            threshold,
            silence_token_count: guard.silence_token_count(),
        })?;
    continue_c(token)
        .map(DecodeTokenAction::Continue)
        .map_err(DecodeTokenError::Continuation)
}

/// The production F gate: callbacks are invoked only for a successful,
/// STOP-terminated decode result. Decode errors (including no STOP and the
/// silence guard) pass through without touching F.
pub(crate) fn run_after_success<T, U>(
    decode: Result<T>,
    run_f: impl FnOnce(&T) -> Result<U>,
) -> Result<(T, U)> {
    let decode = decode?;
    let output = run_f(&decode)?;
    Ok((decode, output))
}

pub(crate) fn classify_split_kind(
    pieces: &[String],
    chunks: &[Vec<i32>],
    max_tokens: usize,
) -> &'static str {
    if chunks.len() <= 1 {
        return "none";
    }
    if pieces.len() <= max_tokens {
        return "soft";
    }
    let mut end = 0usize;
    let has_natural_boundary = chunks.iter().take(chunks.len() - 1).any(|chunk| {
        end += chunk.len();
        end > 0 && is_segment_boundary_piece(&pieces[end - 1])
    });
    if has_natural_boundary {
        "soft+hard"
    } else {
        "hard"
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
    pub(crate) last_hidden_state: OrtTensorOutput,
    pub(crate) raw_logits: OrtTensorOutput,
}

#[derive(Debug, Clone)]
pub(crate) struct DecodeState {
    pub(crate) save_hidden_state: OrtTensorOutput,
    pub(crate) generated_steps: usize,
    pub(crate) stopped: bool,
    pub(crate) silence_token_count: usize,
    pub(crate) max_silence_run: usize,
    pub(crate) degeneration: TokenDegenerationSnapshot,
    pub(crate) f_input_rows: usize,
}

const TOKEN_AUDIT_WINDOW: usize = 64;

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct TokenDegenerationSnapshot {
    pub(crate) total_tokens: usize,
    pub(crate) unique_tokens: usize,
    pub(crate) top_token: Option<i32>,
    pub(crate) top_token_count: usize,
    pub(crate) current_same_token_run: usize,
    pub(crate) longest_same_token_run: usize,
    pub(crate) rolling_unique_tokens: usize,
    pub(crate) rolling_top_share: f64,
    pub(crate) rolling_adjacent_repeat_ratio: f64,
    pub(crate) rolling_period2_match_ratio: f64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct TokenDegenerationObserver {
    histogram: std::collections::HashMap<i32, usize>,
    rolling: std::collections::VecDeque<i32>,
    last_token: Option<i32>,
    current_run: usize,
    longest_run: usize,
    total: usize,
}

impl TokenDegenerationObserver {
    /// Performs only incremental bookkeeping in the decode hot loop. The
    /// bounded/cumulative scans needed for a report happen in `snapshot`, which
    /// callers invoke only at 64-step milestones and terminal/error paths.
    pub(crate) fn observe(&mut self, token: i32) -> bool {
        self.total += 1;
        *self.histogram.entry(token).or_default() += 1;
        if self.last_token == Some(token) {
            self.current_run += 1;
        } else {
            self.current_run = 1;
            self.last_token = Some(token);
        }
        self.longest_run = self.longest_run.max(self.current_run);
        self.rolling.push_back(token);
        if self.rolling.len() > TOKEN_AUDIT_WINDOW {
            self.rolling.pop_front();
        }
        self.total % TOKEN_AUDIT_WINDOW == 0
    }

    pub(crate) fn snapshot(&self) -> TokenDegenerationSnapshot {
        let (top_token, top_token_count) = self
            .histogram
            .iter()
            .max_by_key(|(token, count)| (**count, std::cmp::Reverse(**token)))
            .map(|(token, count)| (Some(*token), *count))
            .unwrap_or((None, 0));
        let mut rolling_counts = BTreeMap::<i32, usize>::new();
        for token in &self.rolling {
            *rolling_counts.entry(*token).or_default() += 1;
        }
        let rolling_len = self.rolling.len();
        let rolling_top = rolling_counts.values().copied().max().unwrap_or(0);
        let adjacent = self
            .rolling
            .iter()
            .zip(self.rolling.iter().skip(1))
            .filter(|(left, right)| left == right)
            .count();
        let period2 = self
            .rolling
            .iter()
            .zip(self.rolling.iter().skip(2))
            .filter(|(left, right)| left == right)
            .count();
        TokenDegenerationSnapshot {
            total_tokens: self.total,
            unique_tokens: self.histogram.len(),
            top_token,
            top_token_count,
            current_same_token_run: self.current_run,
            longest_same_token_run: self.longest_run,
            rolling_unique_tokens: rolling_counts.len(),
            rolling_top_share: ratio_metric(rolling_top, rolling_len),
            rolling_adjacent_repeat_ratio: ratio_metric(adjacent, rolling_len.saturating_sub(1)),
            rolling_period2_match_ratio: ratio_metric(period2, rolling_len.saturating_sub(2)),
        }
    }
}

fn ratio_metric(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 / total as f64
    }
}

fn log_token_degeneration(
    request_id: Uuid,
    chunk_index: usize,
    chunk_count: usize,
    audit_stage: &'static str,
    audit: &TokenDegenerationSnapshot,
) {
    tracing::info!(
        request_id = %request_id,
        chunk_index = chunk_index + 1,
        chunk_count,
        audit_stage,
        total_tokens = audit.total_tokens,
        unique_tokens = audit.unique_tokens,
        top_token = ?audit.top_token,
        top_token_count = audit.top_token_count,
        current_same_token_run = audit.current_same_token_run,
        longest_same_token_run = audit.longest_same_token_run,
        rolling_window = TOKEN_AUDIT_WINDOW.min(audit.total_tokens),
        rolling_unique_tokens = audit.rolling_unique_tokens,
        rolling_top_share = audit.rolling_top_share,
        rolling_adjacent_repeat_ratio = audit.rolling_adjacent_repeat_ratio,
        rolling_period2_match_ratio = audit.rolling_period2_match_ratio,
        enforcement = false,
        "IndexTTS token degeneration audit"
    );
}

pub(crate) fn checked_decode_budget(
    max_generate_length: usize,
    _concat_len: usize,
) -> Result<usize> {
    (max_generate_length > 0)
        .then_some(max_generate_length)
        .ok_or_else(|| {
            InfraError::Backend(format!("IndexTTS max-new-token budget must be positive"))
        })
}

pub(crate) fn indextts_seed_from_params(params: &BTreeMap<String, Value>) -> Result<u64> {
    let Some(value) = params.get("indextts_seed") else {
        return Ok(0);
    };
    value
        .as_u64()
        .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
        .ok_or_else(|| {
            InfraError::BadRequest(format!(
                "indextts_seed must be an unsigned 64-bit integer or decimal string, got {value}"
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

    // A short request can still be too long semantically for one reliable model
    // decode. Make one conservative soft split at the first natural punctuation
    // run that leaves substantial speech on both sides. Deliberately do not
    // recurse: later punctuation remains in the second segment, avoiding the
    // one-clause-per-chunk fragmentation of ordinary prose. Opaque explicit IDs
    // cannot be inspected and retain the compatibility `chunks(max)` behavior.
    if pieces.len() <= max_tokens {
        if let Some(end) = soft_punctuation_boundary(pieces) {
            return Ok(vec![ids[..end].to_vec(), ids[end..].to_vec()]);
        }
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

const MIN_SOFT_SEGMENT_SUBSTANTIVE_PIECES: usize = 8;

fn soft_punctuation_boundary(pieces: &[String]) -> Option<usize> {
    let total_substantive = pieces
        .iter()
        .filter(|piece| piece_has_substantive_text(piece))
        .count();
    let mut substantive_before = 0usize;
    let mut index = 0usize;
    while index < pieces.len() {
        if piece_has_substantive_text(&pieces[index]) {
            substantive_before += 1;
            index += 1;
            continue;
        }
        if !is_segment_boundary_piece(&pieces[index]) {
            index += 1;
            continue;
        }

        // Keep a continuous punctuation run attached to the preceding speech.
        // This also prevents punctuation-only chunks for forms such as "?!…".
        let mut end = index + 1;
        while end < pieces.len()
            && !piece_has_substantive_text(&pieces[end])
            && is_segment_boundary_piece(&pieces[end])
        {
            end += 1;
        }
        let substantive_after = total_substantive.saturating_sub(substantive_before);
        if substantive_before >= MIN_SOFT_SEGMENT_SUBSTANTIVE_PIECES
            && substantive_after >= MIN_SOFT_SEGMENT_SUBSTANTIVE_PIECES
        {
            return Some(end);
        }
        index = end;
    }
    None
}

fn chunk_has_substantive(prefix: &[usize], start: usize, end: usize) -> bool {
    prefix[end] > prefix[start]
}

pub(crate) fn piece_has_substantive_text(piece: &str) -> bool {
    piece.chars().any(|ch| {
        ch != '▁' && !ch.is_whitespace() && !ch.is_ascii_punctuation() && !is_cjk_punctuation(ch)
    })
}

fn is_segment_boundary_piece(piece: &str) -> bool {
    matches!(
        piece.trim_start_matches('▁'),
        "." | "!" | "?" | "…" | "," | ";" | ":" | "。" | "！" | "？" | "，" | "；" | "："
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
