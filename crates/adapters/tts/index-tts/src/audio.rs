use super::*;

pub fn read_wav_mono_i16_24k(path: &Path) -> Result<Vec<i16>> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| InfraError::Adapter(format!("read wav {}: {e}", path.display())))?;
    let spec = reader.spec();
    if spec.channels == 0 {
        return Err(InfraError::BadRequest(format!(
            "wav {} has zero channels",
            path.display()
        )));
    }
    let channels = spec.channels as usize;
    let mut interleaved = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                interleaved.push(
                    sample.map_err(|e| InfraError::Adapter(format!("decode wav sample: {e}")))?,
                );
            }
        }
        hound::SampleFormat::Int => {
            let denom = (1_i64 << spec.bits_per_sample.saturating_sub(1).max(1)) as f32;
            for sample in reader.samples::<i32>() {
                interleaved.push(
                    sample.map_err(|e| InfraError::Adapter(format!("decode wav sample: {e}")))?
                        as f32
                        / denom,
                );
            }
        }
    }
    let mut mono = Vec::with_capacity(interleaved.len() / channels + 1);
    for frame in interleaved.chunks(channels) {
        mono.push(frame.iter().copied().sum::<f32>() / frame.len() as f32);
    }
    let mono = if spec.sample_rate == TARGET_SAMPLE_RATE {
        mono
    } else {
        resample_linear(&mono, spec.sample_rate, TARGET_SAMPLE_RATE)
    };
    Ok(mono.into_iter().map(super::f32_to_i16).collect())
}

pub fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate == 0 || target_rate == 0 {
        return Vec::new();
    }
    if source_rate == target_rate {
        return samples.to_vec();
    }
    if samples.len() == 1 {
        let target_len = ((target_rate as f64) / (source_rate as f64))
            .round()
            .max(1.0) as usize;
        return vec![samples[0]; target_len];
    }
    let target_len = ((samples.len() as f64) * (target_rate as f64) / (source_rate as f64))
        .round()
        .max(1.0) as usize;
    let scale = source_rate as f64 / target_rate as f64;
    (0..target_len)
        .map(|i| {
            let src = i as f64 * scale;
            let left = src.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let frac = (src - left as f64) as f32;
            samples[left] * (1.0 - frac) + samples[right] * frac
        })
        .collect()
}
