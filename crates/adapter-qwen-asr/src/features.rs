use super::*;
use rustfft::{num_complex::Complex32, FftPlanner};

pub const SAMPLE_RATE: u32 = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
const FMIN: f32 = 0.0;
const FMAX: f32 = 8_000.0;

#[derive(Debug, Clone)]
pub struct MelFeatures {
    pub bins: usize,
    pub frames: usize,
    pub data: Vec<f32>,
}

pub fn log_mel_128(samples: &[f32], sample_rate: u32) -> Result<MelFeatures> {
    if sample_rate != SAMPLE_RATE {
        return Err(InfraError::NeedImplementation(format!(
            "Qwen ASR feature extractor expects 16 kHz input after resampling, got {sample_rate}"
        )));
    }
    if samples.is_empty() {
        return Err(InfraError::BadRequest(
            "audio contains no samples".to_string(),
        ));
    }

    let padded = reflect_pad(samples, N_FFT / 2);
    let raw_frames = (padded.len().saturating_sub(N_FFT)) / HOP_LENGTH + 1;
    let kept_frames = raw_frames.saturating_sub(1).max(1);
    let window = hann_window(N_FFT);
    let filters = mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE as f32, FMIN, FMAX);
    let n_freqs = N_FFT / 2 + 1;

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut fft_buffer = vec![Complex32::new(0.0, 0.0); N_FFT];
    let mut mel_spec = vec![0.0f32; N_MELS * raw_frames];

    for frame_idx in 0..raw_frames {
        let start = frame_idx * HOP_LENGTH;
        for i in 0..N_FFT {
            fft_buffer[i] = Complex32::new(padded[start + i] * window[i], 0.0);
        }
        fft.process(&mut fft_buffer);
        let power = fft_buffer[..n_freqs]
            .iter()
            .map(|value| value.norm_sqr())
            .collect::<Vec<_>>();
        for mel_bin in 0..N_MELS {
            let filter_start = mel_bin * n_freqs;
            let energy = (0..n_freqs)
                .map(|freq_bin| filters[filter_start + freq_bin] * power[freq_bin])
                .sum::<f32>();
            mel_spec[mel_bin * raw_frames + frame_idx] = energy;
        }
    }

    let mut log_spec = mel_spec
        .into_iter()
        .map(|value| value.max(1e-10).log10())
        .collect::<Vec<_>>();
    let log_max = log_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let clamp_min = log_max - 8.0;
    for value in &mut log_spec {
        *value = ((*value).max(clamp_min) + 4.0) / 4.0;
    }

    let mut data = vec![0.0f32; N_MELS * kept_frames];
    for mel_bin in 0..N_MELS {
        let src = mel_bin * raw_frames;
        let dst = mel_bin * kept_frames;
        data[dst..dst + kept_frames].copy_from_slice(&log_spec[src..src + kept_frames]);
    }
    Ok(MelFeatures {
        bins: N_MELS,
        frames: kept_frames,
        data,
    })
}

pub fn mel_frame_count(num_samples: usize) -> usize {
    if num_samples == 0 {
        return 0;
    }
    let padded_len = num_samples + N_FFT;
    let raw_frames = (padded_len - N_FFT) / HOP_LENGTH + 1;
    raw_frames.saturating_sub(1).max(1)
}

fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    for i in (0..pad).rev() {
        let src = (i + 1).min(samples.len() - 1);
        padded.push(samples[src]);
    }
    padded.extend_from_slice(samples);
    for i in 0..pad {
        let src = samples.len().saturating_sub(2 + i);
        padded.push(samples[src]);
    }
    padded
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos()))
        .collect()
}

fn mel_filterbank(n_mels: usize, n_fft: usize, sr: f32, fmin: f32, fmax: f32) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_points = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .map(mel_to_hz)
        .collect::<Vec<_>>();
    let fft_freqs = (0..n_freqs)
        .map(|i| i as f32 * sr / n_fft as f32)
        .collect::<Vec<_>>();
    let mut filters = vec![0.0f32; n_mels * n_freqs];
    for mel_bin in 0..n_mels {
        let lower = mel_points[mel_bin];
        let center = mel_points[mel_bin + 1];
        let upper = mel_points[mel_bin + 2];
        for (freq_bin, freq) in fft_freqs.iter().copied().enumerate() {
            let lower_slope = if center > lower {
                (freq - lower) / (center - lower)
            } else {
                0.0
            };
            let upper_slope = if upper > center {
                (upper - freq) / (upper - center)
            } else {
                0.0
            };
            filters[mel_bin * n_freqs + freq_bin] = lower_slope.min(upper_slope).max(0.0);
        }
        let width = upper - lower;
        if width > 0.0 {
            let norm = 2.0 / width;
            for freq_bin in 0..n_freqs {
                filters[mel_bin * n_freqs + freq_bin] *= norm;
            }
        }
    }
    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    if hz < 1_000.0 {
        hz / f_sp
    } else {
        let min_log_hz = 1_000.0;
        let min_log_mel = min_log_hz / f_sp;
        let logstep = (6.4f32).ln() / 27.0;
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1_000.0;
    let min_log_mel = min_log_hz / f_sp;
    if mel < min_log_mel {
        mel * f_sp
    } else {
        let logstep = (6.4f32).ln() / 27.0;
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}
