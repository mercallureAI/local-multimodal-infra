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
        let backend = OrtBackend::new(ProviderSelection::from_strings(
            &spec.runtime.provider_order,
        ));
        let a = backend.load_session(&artifacts.a)?;
        let b = backend.load_session(&artifacts.b)?;
        let c = backend.load_session(&artifacts.c)?;
        let d = backend.load_session(&artifacts.d)?;
        let e = backend.load_session(&artifacts.e)?;
        let f = backend.load_session(&artifacts.f)?;
        validate_sessions([&a, &b, &c, &d, &e, &f])?;
        let tokenizer = SentencePieceTokenizer::load(&artifacts.bpe_model)?;
        let config = IndexTtsModelConfig::load(&artifacts, spec)?;
        let output_dir = env::var_os("LCOAL_DATA_DIR")
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
        let reference_path = lcoal_files::local_path(reference_audio)?;
        let reference = audio::read_wav_mono_i16_24k(&reference_path)?;
        let text_ids = match explicit_text_token_ids_from_params(params)? {
            Some(ids) => ids,
            None => prepare_text_ids(&self.tokenizer, text)?,
        };
        let generated = self.run_af(&reference, &text_ids)?;
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

    fn run_af(&mut self, reference_audio: &[i16], text_ids: &[i32]) -> Result<Vec<i16>> {
        let reference_state = self.run_a(reference_audio)?;
        let text_hidden = self.run_b(text_ids)?;
        let c_start = self.run_c_token(START_TOKEN, 0)?;
        let concat = self.run_d(
            &reference_state.conds_latent,
            &text_hidden,
            &c_start.gpt_hidden_state,
        )?;
        let decode = self.run_e_loop(concat, c_start.next_gen_len)?;
        self.run_f(&reference_state, &decode.save_hidden_state)
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

        for step_index in 0..MAX_GENERATE_LENGTH {
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
            if token == STOP_TOKEN {
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

        Ok(DecodeState {
            save_hidden_state: concatenate_hidden_states(&hidden_states)?,
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
}
