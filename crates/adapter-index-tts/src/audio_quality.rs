use super::*;

const FRAME_SAMPLES: usize = 480; // 20 ms at 24 kHz.
const QUIET_RMS: f64 = 32.0;
const LOW_SIGNAL_AMPLITUDE: i32 = 24;
const MIN_ENERGETIC_OCCUPANCY: usize = 4;
const MIN_QUIET_VOICE_OCCUPANCY: usize = 24;
const MIN_TRIMMABLE_TAIL_SAMPLES: usize = 48_000; // Ignore ordinary <=2 s endings.
const SAFETY_TAIL_SAMPLES: usize = 2_880; // Retain 120 ms after the last active frame.
const MIN_TAIL_RATIO_PERCENT: usize = 45;
const MIN_SUSTAINED_ACTIVE_FRAMES: usize = 3; // 60 ms rejects isolated clicks.
const MIN_CREDIBLE_ACTIVE_FRAMES: usize = 10; // 200 ms is a conservative minimum utterance.
const MIN_EFFECTIVE_ACTIVE_PERCENT: usize = 1;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WaveformReport {
    pub(crate) raw_samples: usize,
    pub(crate) final_samples: usize,
    pub(crate) trimmed_samples: usize,
    pub(crate) trailing_quiet_samples: usize,
    pub(crate) peak: i32,
    pub(crate) rms: f64,
    pub(crate) effective_active_ratio: f64,
    pub(crate) tail_trimmed: bool,
}

/// Finalize one decoder chunk. A frame needs distributed energy, rather than a
/// peak alone, to be active: RMS above 32 requires at least 4 samples above
/// amplitude 24, while a quiet voice can qualify by occupying at least 24/480
/// samples above 24. Thus a sparse click cannot keep a long decoder tail alive,
/// but low-amplitude sustained speech can. Classification scans backward only
/// and ratio-gates trimming, so internal pauses and normal terminal silence are
/// untouched. This must run before segment gaps are inserted; intentional
/// 200 ms gaps therefore cannot be mistaken for F tails.
pub(crate) fn finalize_decoder_waveform(samples: &[i16]) -> Result<(Vec<i16>, WaveformReport)> {
    if samples.is_empty() {
        return Err(blank_audio_error(0, 0, 0.0));
    }
    let frames = samples
        .chunks(FRAME_SAMPLES)
        .map(frame_stats)
        .collect::<Vec<_>>();
    let active = frames
        .iter()
        .map(|stats| {
            (stats.rms > QUIET_RMS && stats.occupied >= MIN_ENERGETIC_OCCUPANCY)
                || stats.occupied >= MIN_QUIET_VOICE_OCCUPANCY
        })
        .collect::<Vec<_>>();
    let sustained_active = active
        .windows(MIN_SUSTAINED_ACTIVE_FRAMES)
        .any(|window| window.iter().all(|value| *value));
    let active_count = active.iter().filter(|value| **value).count();
    let effective_active_ratio = active_count as f64 / frames.len() as f64;
    let peak = samples
        .iter()
        .map(|sample| i32::from(*sample).abs())
        .max()
        .unwrap_or(0);
    let rms = (samples
        .iter()
        .map(|sample| f64::from(*sample).powi(2))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt();
    if !sustained_active
        || !ratio_at_least(active_count, frames.len(), MIN_EFFECTIVE_ACTIVE_PERCENT)
    {
        return Err(blank_audio_error(
            peak,
            samples.len(),
            effective_active_ratio,
        ));
    }
    if active_count < MIN_CREDIBLE_ACTIVE_FRAMES {
        return Err(blank_audio_error(
            peak,
            samples.len(),
            effective_active_ratio,
        ));
    }

    let trailing_frames = active.iter().rev().take_while(|value| !**value).count();
    let trailing_quiet_samples = if trailing_frames == 0 {
        0
    } else {
        let first_quiet_frame = frames.len() - trailing_frames;
        samples.len() - (first_quiet_frame * FRAME_SAMPLES)
    };
    let disproportionate = trailing_quiet_samples >= MIN_TRIMMABLE_TAIL_SAMPLES
        && ratio_at_least(
            trailing_quiet_samples,
            samples.len(),
            MIN_TAIL_RATIO_PERCENT,
        );
    let final_len = if disproportionate {
        samples
            .len()
            .saturating_sub(trailing_quiet_samples)
            .saturating_add(SAFETY_TAIL_SAMPLES.min(trailing_quiet_samples))
    } else {
        samples.len()
    };
    let report = WaveformReport {
        raw_samples: samples.len(),
        final_samples: final_len,
        trimmed_samples: samples.len() - final_len,
        trailing_quiet_samples,
        peak,
        rms,
        effective_active_ratio,
        tail_trimmed: final_len != samples.len(),
    };
    Ok((samples[..final_len].to_vec(), report))
}

fn ratio_at_least(part: usize, total: usize, percent: usize) -> bool {
    total != 0 && (part as u128) * 100 >= (total as u128) * (percent as u128)
}

struct FrameStats {
    rms: f64,
    occupied: usize,
}

fn frame_stats(frame: &[i16]) -> FrameStats {
    let rms = (frame
        .iter()
        .map(|sample| f64::from(*sample).powi(2))
        .sum::<f64>()
        / frame.len().max(1) as f64)
        .sqrt();
    let occupied = frame
        .iter()
        .filter(|sample| i32::from(**sample).abs() > LOW_SIGNAL_AMPLITUDE)
        .count();
    FrameStats { rms, occupied }
}

fn blank_audio_error(peak: i32, samples: usize, active_ratio: f64) -> InfraError {
    InfraError::Backend(format!(
        "IndexTTS_F produced mostly silent audio (samples {samples}, peak {peak}, effective_active_ratio {active_ratio:.4})"
    ))
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SilenceRunGuard {
    silence_token_count: usize,
    consecutive: usize,
    max_run: usize,
}

impl SilenceRunGuard {
    pub(crate) fn observe(&mut self, token: i32, threshold: usize) -> Result<()> {
        if token == SILENCE_TOKEN {
            self.silence_token_count += 1;
            self.consecutive += 1;
            self.max_run = self.max_run.max(self.consecutive);
            if self.consecutive > threshold {
                return Err(InfraError::Backend(format!(
                    "IndexTTS_E pathological silence loop: silence_token {SILENCE_TOKEN}, consecutive {}, threshold {threshold}, silence_token_count {}",
                    self.consecutive, self.silence_token_count
                )));
            }
        } else {
            self.consecutive = 0;
        }
        Ok(())
    }

    pub(crate) fn silence_token_count(&self) -> usize {
        self.silence_token_count
    }

    pub(crate) fn consecutive(&self) -> usize {
        self.consecutive
    }

    pub(crate) fn max_run(&self) -> usize {
        self.max_run
    }
}
