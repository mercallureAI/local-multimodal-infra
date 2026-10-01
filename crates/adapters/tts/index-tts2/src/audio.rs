use local_error::{InfraError, Result};
use std::path::Path;

/// Reads a WAV file as mono f32 in [-1, 1], resampled to `target_rate` and
/// truncated to `max_seconds` (upstream `_load_and_cut_audio(..., 15)`).
pub fn read_wav_mono(path: &Path, target_rate: u32, max_seconds: f32) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| InfraError::BadRequest(format!("read wav {}: {e}", path.display())))?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.sample_rate == 0 {
        return Err(InfraError::BadRequest(format!(
            "wav {} has {} channels at {} Hz",
            path.display(),
            spec.channels,
            spec.sample_rate
        )));
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| InfraError::BadRequest(format!("decode wav sample: {e}")))?,
        hound::SampleFormat::Int => {
            let scale = (1_i64 << spec.bits_per_sample.saturating_sub(1).max(1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| InfraError::BadRequest(format!("decode wav sample: {e}")))?
        }
    };
    let channels = spec.channels as usize;
    let mono: Vec<f32> = samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect();
    let max_source = (max_seconds * spec.sample_rate as f32) as usize;
    let mono = &mono[..mono.len().min(max_source)];
    Ok(resample(mono, spec.sample_rate, target_rate))
}

/// Windowed-sinc (Hann, 16 zero crossings) band-limited resampler. Linear
/// interpolation aliases badly on 44.1/48 kHz references, and the reference
/// frontend feeds a speaker encoder that is sensitive to it.
pub fn resample(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }
    const ZERO_CROSSINGS: f64 = 16.0;
    let ratio = target_rate as f64 / source_rate as f64;
    let cutoff = ratio.min(1.0) * 0.97;
    let half_width = ZERO_CROSSINGS / cutoff;
    let out_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for index in 0..out_len {
        let center = index as f64 / ratio;
        let first = (center - half_width).ceil().max(0.0) as usize;
        let last = ((center + half_width).floor() as usize).min(samples.len() - 1);
        let mut acc = 0.0f64;
        let mut norm = 0.0f64;
        for (offset, sample) in samples[first..=last].iter().enumerate() {
            let t = (first + offset) as f64 - center;
            let x = t * cutoff;
            let sinc = if x.abs() < 1e-9 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            };
            let window = 0.5 + 0.5 * (std::f64::consts::PI * t / half_width).cos();
            let weight = sinc * window;
            acc += *sample as f64 * weight;
            norm += weight;
        }
        out.push(if norm.abs() > 1e-12 { (acc / norm) as f32 } else { 0.0 });
    }
    out
}

pub fn write_wav_i16(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| InfraError::Adapter(format!("create wav {}: {e}", path.display())))?;
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        writer
            .write_sample(value)
            .map_err(|e| InfraError::Adapter(format!("write wav {}: {e}", path.display())))?;
    }
    writer
        .finalize()
        .map_err(|e| InfraError::Adapter(format!("finalize wav {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_preserves_a_low_tone() {
        let source: Vec<f32> = (0..48_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
            .collect();
        let out = resample(&source, 48_000, 22_050);
        assert_eq!(out.len(), 22_050);
        let expected: Vec<f32> = (0..22_050)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22_050.0).sin())
            .collect();
        let max_err = out[200..21_800]
            .iter()
            .zip(&expected[200..21_800])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.02, "max error {max_err}");
    }
}
