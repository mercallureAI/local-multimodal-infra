use super::*;

pub(crate) fn tensor_f32(name: &str, shape: Vec<usize>, data: Vec<f32>) -> OrtTensorInput {
    OrtTensorInput {
        name: name.to_string(),
        shape,
        data: OrtTensorData::F32(data),
    }
}

pub(crate) fn tensor_i16(name: &str, shape: Vec<usize>, data: Vec<i16>) -> OrtTensorInput {
    OrtTensorInput {
        name: name.to_string(),
        shape,
        data: OrtTensorData::I16(data),
    }
}

pub(crate) fn tensor_i32(name: &str, shape: Vec<usize>, data: Vec<i32>) -> OrtTensorInput {
    OrtTensorInput {
        name: name.to_string(),
        shape,
        data: OrtTensorData::I32(data),
    }
}

pub(crate) fn tensor_i64(name: &str, shape: Vec<usize>, data: Vec<i64>) -> OrtTensorInput {
    OrtTensorInput {
        name: name.to_string(),
        shape,
        data: OrtTensorData::I64(data),
    }
}

pub(crate) fn clone_as_input(name: &str, output: &OrtTensorOutput) -> Result<OrtTensorInput> {
    Ok(OrtTensorInput {
        name: name.to_string(),
        shape: output.shape.clone(),
        data: output.data.clone(),
    })
}

pub(crate) fn validate_sessions(sessions: [&OrtSession; 6]) -> Result<()> {
    let expected = [
        ("IndexTTS_A", &["audio"][..]),
        ("IndexTTS_B", &["text_ids"][..]),
        ("IndexTTS_C", &["gpt_ids"][..]),
        ("IndexTTS_D", &["embed_x", "embed_y", "embed_z"][..]),
        (
            "IndexTTS_E",
            &[
                "history_len",
                "repeat_penality",
                "ids_len",
                "hidden_state",
                "attention_mask",
            ][..],
        ),
        ("IndexTTS_F", &["save_hidden_state"][..]),
    ];
    for ((label, required), session) in expected.into_iter().zip(sessions) {
        for name in required {
            if !has_input(session, name) {
                return Err(InfraError::Backend(format!(
                    "{label} is missing expected input `{name}`; {}",
                    format_session_io(label, session.metadata())
                )));
            }
        }
    }
    c_len_input_name(sessions[2])?;
    Ok(())
}

pub(crate) fn require_input(session: &OrtSession, name: &str, label: &str) -> Result<()> {
    require_inputs(session, &[name], label)
}

pub(crate) fn require_inputs(session: &OrtSession, names: &[&str], label: &str) -> Result<()> {
    for name in names {
        if !has_input(session, name) {
            return Err(InfraError::Backend(format!(
                "{label} is missing expected input `{name}`; {}",
                format_session_io(label, session.metadata())
            )));
        }
    }
    Ok(())
}

pub(crate) fn has_input(session: &OrtSession, name: &str) -> bool {
    session.inputs().iter().any(|input| input.name == name)
}

pub(crate) fn c_len_input_name(session: &OrtSession) -> Result<&str> {
    if has_input(session, "gen_len") {
        Ok("gen_len")
    } else if has_input(session, "kv_seq_len") {
        Ok("kv_seq_len")
    } else {
        Err(InfraError::Backend(format!(
            "IndexTTS_C is missing expected generation length input `gen_len` or `kv_seq_len`; {}",
            format_session_io("IndexTTS_C", session.metadata())
        )))
    }
}

pub(crate) fn find_output(
    outputs: Vec<OrtTensorOutput>,
    preferred: &str,
    label: &str,
    session: &OrtSession,
) -> Result<OrtTensorOutput> {
    outputs
        .iter()
        .find(|output| output.name == preferred)
        .cloned()
        .or_else(|| outputs.first().cloned())
        .ok_or_else(|| {
            InfraError::Backend(format!(
                "{label} returned no outputs while looking for `{preferred}`; {}",
                format_session_io(label, session.metadata())
            ))
        })
}

pub(crate) fn infer_layer_count(output_count: usize) -> Result<usize> {
    output_count
        .checked_sub(3)
        .filter(|count| count % 2 == 0)
        .map(|count| count / 2)
        .filter(|count| *count > 0)
        .ok_or_else(|| {
            InfraError::Backend(format!(
                "IndexTTS_E output count {output_count} cannot infer KV layer count"
            ))
        })
}

pub(crate) fn infer_repeat_penalty_width(
    session: &OrtSession,
    config: &IndexTtsModelConfig,
) -> Result<usize> {
    let meta = session
        .inputs()
        .iter()
        .find(|input| input.name == "repeat_penality")
        .ok_or_else(|| {
            InfraError::Backend(format!(
                "IndexTTS_E missing repeat_penality input; {}",
                format_session_io("IndexTTS_E", session.metadata())
            ))
        })?;
    repeat_penalty_width_from_metadata(meta, config).ok_or_else(|| {
        InfraError::NeedImplementation(format!(
            "IndexTTS_E repeat_penality width is dynamic/unknown; add mel_code_size or vocab_size to manifest.yaml/json or model metadata. Documented upstream default is {DEFAULT_MEL_CODE_SIZE}, but this adapter refuses to guess for real artifacts. {}",
            format_session_io("IndexTTS_E", session.metadata())
        ))
    })
}

pub(crate) fn repeat_penalty_width_from_metadata(
    meta: &TensorMetadata,
    config: &IndexTtsModelConfig,
) -> Option<usize> {
    if let Some(width) = config.configured_mel_code_size() {
        return Some(width);
    }
    if let Some(width) = static_last_dim(meta) {
        return Some(width);
    }
    None
}

pub(crate) fn static_last_dim(meta: &TensorMetadata) -> Option<usize> {
    meta.shape
        .last()
        .copied()
        .filter(|dim| *dim > 0)
        .and_then(|dim| usize::try_from(dim).ok())
}

pub(crate) fn empty_cache(session: &OrtSession, name: &str) -> Result<OrtTensorOutput> {
    let meta = session
        .inputs()
        .iter()
        .find(|input| input.name == name)
        .ok_or_else(|| InfraError::Backend(format!("IndexTTS_E missing cache input `{name}`")))?;
    let shape = concrete_or_empty_shape(meta);
    let len = shape.iter().product::<usize>();
    let data = match meta.element_type {
        TensorElement::I8 => OrtTensorData::I8(vec![0; len]),
        TensorElement::I16 => OrtTensorData::I16(vec![0; len]),
        TensorElement::I32 => OrtTensorData::I32(vec![0; len]),
        TensorElement::I64 => OrtTensorData::I64(vec![0; len]),
        _ => OrtTensorData::F32(vec![0.0; len]),
    };
    Ok(OrtTensorOutput {
        name: name.to_string(),
        shape,
        data,
    })
}

pub(crate) fn cache_sequence_len(cache: &OrtTensorOutput) -> Result<i64> {
    match cache.shape.as_slice() {
        // HuggingFace GPT present-key/value cache layout exported by IndexTTS_E:
        // [batch, heads, past_sequence, head_dim].
        [_batch, _heads, seq, _head_dim] => i64::try_from(*seq).map_err(|_| {
            InfraError::Backend(format!(
                "IndexTTS_E cache `{}` sequence length {} does not fit i64; shape {:?}",
                cache.name, seq, cache.shape
            ))
        }),
        // Official/DakeQQ split ONNX cache layout can squeeze the head dimension:
        // key [batch, heads, past_sequence], value [batch, past_sequence, dim].
        // The decode loop sizes the mask from a key cache, but keep value support
        // here so diagnostics/tests cover both official 3-D metadata variants.
        [_batch, second, third] => {
            let seq = if cache.name.contains("value") || cache.name.contains("val") {
                *second
            } else {
                *third
            };
            i64::try_from(seq).map_err(|_| {
                InfraError::Backend(format!(
                    "IndexTTS_E cache `{}` sequence length {} does not fit i64; shape {:?}",
                    cache.name, seq, cache.shape
                ))
            })
        }
        shape => Err(InfraError::NeedImplementation(format!(
            "IndexTTS_E cache `{}` has unsupported shape {shape:?}; expected [batch, heads, seq, head_dim], [batch, heads, seq], or [batch, seq, dim] to build attention_mask",
            cache.name
        ))),
    }
}

pub(crate) fn concrete_or_empty_shape(meta: &TensorMetadata) -> Vec<usize> {
    meta.shape
        .iter()
        .map(|dim| {
            if *dim > 0 {
                *dim as usize
            } else {
                // ORT tensor constructors reject zero-sized dimensions. For dynamic cache inputs
                // such as [1, heads, -1, head_dim], use a one-token dummy cache so the tensor can
                // be constructed; the decode loop separately sizes attention_mask from this real
                // cache tensor shape while preserving logical history_len/ids_len inputs.
                1
            }
        })
        .collect()
}

pub(crate) fn parse_e_outputs(
    outputs: Vec<OrtTensorOutput>,
    layer_count: usize,
    session: &OrtSession,
) -> Result<EStep> {
    let mut keys = Vec::with_capacity(layer_count);
    let mut values = Vec::with_capacity(layer_count);
    for idx in 0..layer_count {
        keys.push(
            find_named_output(
                &outputs,
                &[
                    &format!("out_key_{idx}"),
                    &format!("key_{idx}"),
                    &format!("present_key_{idx}"),
                    &format!("present.{idx}.key"),
                ],
            )
            .cloned()
            .or_else(|| outputs.get(idx * 2).cloned())
            .ok_or_else(|| e_output_error("updated key", idx, session))?,
        );
        values.push(
            find_named_output(
                &outputs,
                &[
                    &format!("out_value_{idx}"),
                    &format!("value_{idx}"),
                    &format!("present_value_{idx}"),
                    &format!("present.{idx}.value"),
                ],
            )
            .cloned()
            .or_else(|| outputs.get(idx * 2 + 1).cloned())
            .ok_or_else(|| e_output_error("updated value", idx, session))?,
        );
    }
    let tail = layer_count * 2;
    let kv_seq_len_output = find_named_output(&outputs, &["kv_seq_len", "next_kv_seq_len"])
        .cloned()
        .or_else(|| outputs.get(tail).cloned())
        .ok_or_else(|| e_tail_output_error("kv_seq_len", session))?;
    let kv_seq_len = first_i64_or_i32(&kv_seq_len_output, &kv_seq_len_output.name)?;
    let last_hidden_state =
        find_named_output(&outputs, &["last_hidden_state", "save_hidden_state"])
            .cloned()
            .or_else(|| outputs.get(tail + 1).cloned())
            .ok_or_else(|| e_tail_output_error("last_hidden_state", session))?;
    let max_logit_output =
        find_named_output(&outputs, &["max_logit_id", "max_logits_id", "token_id"])
            .cloned()
            .or_else(|| outputs.get(tail + 2).cloned())
            .ok_or_else(|| e_tail_output_error("max_logit_id", session))?;
    let max_logit_id = first_i64_or_i32(&max_logit_output, &max_logit_output.name)? as i32;
    Ok(EStep {
        keys,
        values,
        kv_seq_len,
        last_hidden_state,
        max_logit_id,
    })
}

pub(crate) fn find_named_output<'a>(
    outputs: &'a [OrtTensorOutput],
    names: &[&str],
) -> Option<&'a OrtTensorOutput> {
    outputs
        .iter()
        .find(|output| names.iter().any(|name| output.name == *name))
}

pub(crate) fn e_output_error(kind: &str, index: usize, session: &OrtSession) -> InfraError {
    InfraError::Backend(format!(
        "IndexTTS_E missing {kind} output for layer {index}; {}",
        format_session_io("IndexTTS_E", session.metadata())
    ))
}

pub(crate) fn e_tail_output_error(kind: &str, session: &OrtSession) -> InfraError {
    InfraError::Backend(format!(
        "IndexTTS_E missing {kind} tail output; {}",
        format_session_io("IndexTTS_E", session.metadata())
    ))
}

pub(crate) fn attention_mask_input(
    session: &OrtSession,
    total_seq_len: i64,
    masked_prefix_len: i64,
) -> Result<OrtTensorInput> {
    let meta = session
        .inputs()
        .iter()
        .find(|input| input.name == "attention_mask")
        .ok_or_else(|| {
            InfraError::Backend(format!(
                "IndexTTS_E missing attention_mask input; {}",
                format_session_io("IndexTTS_E", session.metadata())
            ))
        })?;
    let shape = attention_mask_shape(meta, total_seq_len)?;
    let data = attention_mask_values(&shape, masked_prefix_len);
    Ok(match meta.element_type {
        TensorElement::F32 => tensor_f32(
            "attention_mask",
            shape,
            data.into_iter().map(|value| value as f32).collect(),
        ),
        TensorElement::I32 => tensor_i32("attention_mask", shape, data),
        TensorElement::I64 | TensorElement::Other => tensor_i64(
            "attention_mask",
            shape,
            data.into_iter().map(i64::from).collect(),
        ),
        TensorElement::I8 => OrtTensorInput {
            name: "attention_mask".to_string(),
            shape,
            data: OrtTensorData::I8(data.into_iter().map(|value| value as i8).collect()),
        },
        TensorElement::I16 => OrtTensorInput {
            name: "attention_mask".to_string(),
            shape,
            data: OrtTensorData::I16(data.into_iter().map(|value| value as i16).collect()),
        },
    })
}

pub(crate) fn attention_mask_values(shape: &[usize], masked_prefix_len: i64) -> Vec<i32> {
    if shape.len() == 1 {
        // Official/DakeQQ ONNX uses rank-1 attention_mask as a scalar control:
        // 1 on the first step while a dummy cache is supplied, 0 on subsequent
        // steps. It is not a token mask, so do not apply prefix hiding here.
        return vec![if masked_prefix_len > 0 { 1 } else { 0 }; shape[0].max(1)];
    }
    let len = shape.iter().product::<usize>().max(1);
    let seq_len = shape.last().copied().unwrap_or(len).max(1);
    let masked_prefix_len = usize::try_from(masked_prefix_len.max(0)).unwrap_or(usize::MAX);
    (0..len)
        .map(|idx| {
            if idx % seq_len < masked_prefix_len {
                0
            } else {
                1
            }
        })
        .collect()
}

pub(crate) fn attention_mask_shape(
    meta: &TensorMetadata,
    total_seq_len: i64,
) -> Result<Vec<usize>> {
    if meta.shape.is_empty() {
        return Ok(vec![1]);
    }
    let seq_len = usize::try_from(total_seq_len.max(1)).map_err(|_| {
        InfraError::Backend(format!(
            "IndexTTS_E total sequence length {total_seq_len} is too large for attention_mask"
        ))
    })?;
    Ok(match meta.shape.len() {
        1 => vec![if meta.shape[0] > 0 { meta.shape[0] as usize } else { 1 }],
        2 => vec![
            if meta.shape[0] > 0 { meta.shape[0] as usize } else { 1 },
            if meta.shape[1] > 0 { meta.shape[1] as usize } else { seq_len },
        ],
        _ => {
            return Err(InfraError::NeedImplementation(format!(
                "IndexTTS_E attention_mask rank {} is unsupported; expected rank 1 or 2. Metadata: {}:{:?}{:?}",
                meta.shape.len(),
                meta.name,
                meta.element_type,
                meta.shape
            )))
        }
    })
}

pub fn apply_repeat_penalty_token(
    penalties: &mut [f32],
    history: &mut Vec<i32>,
    token: i32,
    window: usize,
    penalty: f32,
) {
    history.push(token);
    if let Ok(index) = usize::try_from(token) {
        if let Some(slot) = penalties.get_mut(index) {
            *slot = penalty;
        }
    }
    while history.len() > window {
        let removed = history.remove(0);
        if !history.contains(&removed) {
            if let Ok(index) = usize::try_from(removed) {
                if let Some(slot) = penalties.get_mut(index) {
                    *slot = 1.0;
                }
            }
        }
    }
}

pub fn e_loop_control_lengths(
    first_step: bool,
    concat_len: usize,
    prior_kv_seq_len: i64,
) -> (i64, i64) {
    if first_step {
        (0, concat_len as i64)
    } else {
        (prior_kv_seq_len, 1)
    }
}

pub fn concatenate_hidden_states(hidden_states: &[OrtTensorOutput]) -> Result<OrtTensorOutput> {
    if hidden_states.is_empty() {
        return Err(InfraError::Backend(
            "IndexTTS_E did not produce any last_hidden_state tensors before stopping".to_string(),
        ));
    }
    let mut width = None;
    let mut total_rows = 0usize;
    let mut data = Vec::new();
    for state in hidden_states {
        let state_data = match &state.data {
            OrtTensorData::F32(data) => data,
            other => {
                return Err(InfraError::NeedImplementation(format!(
                    "IndexTTS_E last_hidden_state `{}` has {:?}; only f32 hidden states are supported for F save_hidden_state",
                    state.name,
                    other.element_type()
                )))
            }
        };
        let (rows, hidden) = hidden_rows_and_width(&state.shape, state_data.len(), &state.name)?;
        if let Some(expected) = width {
            if expected != hidden {
                return Err(InfraError::Backend(format!(
                    "IndexTTS_E hidden width changed from {expected} to {hidden} for `{}` shape {:?}",
                    state.name, state.shape
                )));
            }
        } else {
            width = Some(hidden);
        }
        total_rows += rows;
        data.extend_from_slice(state_data);
    }
    let width = width.expect("non-empty hidden states set width");
    Ok(OrtTensorOutput {
        name: "save_hidden_state".to_string(),
        shape: vec![total_rows, width],
        data: OrtTensorData::F32(data),
    })
}

pub(crate) fn hidden_rows_and_width(
    shape: &[usize],
    data_len: usize,
    name: &str,
) -> Result<(usize, usize)> {
    match shape {
        [hidden] => Ok((1, *hidden)),
        [rows, hidden] => Ok((*rows, *hidden)),
        [1, rows, hidden] => Ok((*rows, *hidden)),
        [batch, _rows, _hidden] => Err(InfraError::NeedImplementation(format!(
            "IndexTTS_E hidden output `{name}` has batch {batch}; only batch=1 is supported for concatenating save_hidden_state"
        ))),
        _ if !shape.is_empty() => {
            let hidden = *shape.last().unwrap_or(&0);
            if hidden > 0 && data_len % hidden == 0 {
                Ok((data_len / hidden, hidden))
            } else {
                Err(InfraError::NeedImplementation(format!(
                    "IndexTTS_E hidden output `{name}` shape {shape:?} cannot be flattened to [num_decode, hidden]"
                )))
            }
        }
        _ => Err(InfraError::NeedImplementation(format!(
            "IndexTTS_E hidden output `{name}` has scalar/unknown shape"
        ))),
    }
}

pub(crate) fn first_i64_or_i32(output: &OrtTensorOutput, name: &str) -> Result<i64> {
    match &output.data {
        OrtTensorData::I64(data) => data.first().copied(),
        OrtTensorData::I32(data) => data.first().map(|v| i64::from(*v)),
        _ => None,
    }
    .ok_or_else(|| {
        InfraError::Backend(format!("output `{name}` is not a non-empty i64/i32 tensor"))
    })
}

pub(crate) fn tensor_to_i16_audio(output: &OrtTensorOutput) -> Result<Vec<i16>> {
    let samples = match &output.data {
        OrtTensorData::I16(data) => data.clone(),
        OrtTensorData::F32(data) => data.iter().map(|v| f32_to_i16(*v)).collect(),
        OrtTensorData::I32(data) => data
            .iter()
            .map(|v| (*v).clamp(i16::MIN as i32, i16::MAX as i32) as i16)
            .collect(),
        _ => {
            return Err(InfraError::Backend(format!(
                "generated_wav has unsupported element type {:?}; expected i16/f32/i32",
                output.data.element_type()
            )))
        }
    };
    if samples.is_empty() {
        Err(InfraError::Backend("generated_wav is empty".to_string()))
    } else {
        Ok(samples)
    }
}

pub(crate) fn f32_to_i16(value: f32) -> i16 {
    (value.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
}

pub(crate) fn session_error(label: &str, session: &OrtSession, err: InfraError) -> InfraError {
    InfraError::Backend(format!(
        "{label} failed: {err}; {}",
        format_session_io(label, session.metadata())
    ))
}

pub(crate) fn format_session_io(label: &str, metadata: &SessionMetadata) -> String {
    format!(
        "{label} inputs [{}], outputs [{}]",
        metadata
            .inputs
            .iter()
            .map(format_tensor)
            .collect::<Vec<_>>()
            .join(", "),
        metadata
            .outputs
            .iter()
            .map(format_tensor)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub(crate) fn format_tensor(tensor: &TensorMetadata) -> String {
    format!(
        "{}:{:?}{:?}",
        tensor.name, tensor.element_type, tensor.shape
    )
}
