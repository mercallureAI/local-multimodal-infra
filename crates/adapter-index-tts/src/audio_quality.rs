use super::*;

const FRAME_SAMPLES: usize = 480; // 20 ms at 24 kHz.
const QUIET_RMS: f64 = 32.0;
const LOW_SIGNAL_AMPLITUDE: i32 = 24;
const MIN_ENERGETIC_OCCUPANCY: usize = 4;
const MIN_QUIET_VOICE_OCCUPANCY: usize = 24;
const MIN_ONSET_FRAMES: usize = 3; // 60 ms.
const MAX_INTERNAL_GAP_FRAMES: usize = 5; // 100 ms.
const MIN_CREDIBLE_ISLAND_FRAMES: usize = 10; // 200 ms envelope.
const MIN_CREDIBLE_ACTIVE_FRAMES: usize = 10;
const MIN_TRIMMABLE_TAIL_SAMPLES: usize = 48_000;
const SAFETY_TAIL_SAMPLES: usize = 2_880;
const MIN_TAIL_RATIO_PERCENT: usize = 45;
const PERIODIC_MIN_RUNS: usize = 6;
const PERIODIC_MIN_SPACING_FRAMES: usize = 6; // 120 ms.
const PERIODIC_MAX_SPACING_FRAMES: usize = 60; // 1.2 s.
const PERIODIC_MAX_JITTER_FRAMES: usize = 2; // 40 ms.

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WaveformReport {
    pub(crate) raw_samples: usize,
    pub(crate) final_samples: usize,
    pub(crate) trimmed_samples: usize,
    pub(crate) trailing_quiet_samples: usize,
    pub(crate) peak: i32,
    pub(crate) rms: f64,
    pub(crate) raw_active_ratio: f64,
    pub(crate) raw_credible_active_ratio: f64,
    pub(crate) final_active_ratio: f64,
    pub(crate) final_credible_active_ratio: f64,
    pub(crate) credible_island_count: usize,
    pub(crate) longest_credible_run_frames: usize,
    pub(crate) short_glitch_count: usize,
    pub(crate) raw_late_active_ratio: f64,
    pub(crate) credible_late_active_ratio: f64,
    pub(crate) periodic_sparse_pulses: bool,
    pub(crate) quality_decision: &'static str,
    pub(crate) quality_reason: &'static str,
    pub(crate) tail_trimmed: bool,
}

/// Applies a topology-aware activity mask to one decoder chunk. Short active
/// runs are removed before bounded *internal* gaps are joined. Only envelopes
/// lasting at least 200 ms are credible speech islands, so a click at the end
/// cannot anchor a decoder tail. The existing 2 s / 45% / 120 ms trim policy is
/// deliberately retained.
pub(crate) fn finalize_decoder_waveform(samples: &[i16]) -> Result<(Vec<i16>, WaveformReport)> {
    if samples.is_empty() {
        return Err(quality_error(
            "no_credible_speech",
            0,
            0,
            0.0,
            0.0,
            0,
            0,
            false,
        ));
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
    let runs = true_runs(&active);
    let short_glitch_count = runs
        .iter()
        .filter(|(start, end)| end - start < MIN_ONSET_FRAMES)
        .count();
    let periodic_sparse_pulses = periodic_short_runs(&runs);

    let mut onset = vec![false; active.len()];
    for &(start, end) in &runs {
        if end - start >= MIN_ONSET_FRAMES {
            onset[start..end].fill(true);
        }
    }
    let mut merged = onset.clone();
    let onset_runs = true_runs(&onset);
    for pair in onset_runs.windows(2) {
        let gap_start = pair[0].1;
        let gap_end = pair[1].0;
        if gap_end - gap_start <= MAX_INTERNAL_GAP_FRAMES {
            merged[gap_start..gap_end].fill(true);
        }
    }
    let credible_runs = true_runs(&merged)
        .into_iter()
        .filter(|(start, end)| end - start >= MIN_CREDIBLE_ISLAND_FRAMES)
        .collect::<Vec<_>>();
    let mut credible = vec![false; active.len()];
    for &(start, end) in &credible_runs {
        credible[start..end].fill(true);
    }

    let active_count = count_true(&active);
    let credible_count = count_true(&credible);
    let raw_active_ratio = ratio(active_count, frames.len());
    let raw_credible_active_ratio = ratio(credible_count, frames.len());
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
    let longest = credible_runs
        .iter()
        .map(|(start, end)| end - start)
        .max()
        .unwrap_or(0);
    if credible_runs.is_empty() || credible_count < MIN_CREDIBLE_ACTIVE_FRAMES {
        let reason = if periodic_sparse_pulses {
            "periodic_sparse_pulses"
        } else if runs.len() >= PERIODIC_MIN_RUNS {
            "fragmented_sparse_activity"
        } else if active_count >= MIN_CREDIBLE_ACTIVE_FRAMES {
            "insufficient_credible_activity"
        } else {
            "no_credible_speech"
        };
        return Err(quality_error(
            reason,
            samples.len(),
            peak,
            raw_active_ratio,
            raw_credible_active_ratio,
            credible_runs.len(),
            short_glitch_count,
            periodic_sparse_pulses,
        ));
    }

    let last_credible_end = credible_runs.last().expect("non-empty credible runs").1;
    let trailing_quiet_samples = samples
        .len()
        .saturating_sub((last_credible_end * FRAME_SAMPLES).min(samples.len()));
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
    let final_frames = final_len.div_ceil(FRAME_SAMPLES).min(frames.len());
    let late_start = frames.len() * 3 / 4;
    let final_active_ratio = ratio(count_true(&active[..final_frames]), final_frames);
    let final_credible_active_ratio = ratio(count_true(&credible[..final_frames]), final_frames);
    let low_ratio = raw_credible_active_ratio < 0.10;
    let quality_reason = if disproportionate {
        "credible_speech_with_disproportionate_tail_trimmed"
    } else if low_ratio {
        "low_ratio_but_credible_island_present"
    } else {
        "credible_speech"
    };
    let report = WaveformReport {
        raw_samples: samples.len(),
        final_samples: final_len,
        trimmed_samples: samples.len() - final_len,
        trailing_quiet_samples,
        peak,
        rms,
        raw_active_ratio,
        raw_credible_active_ratio,
        final_active_ratio,
        final_credible_active_ratio,
        credible_island_count: credible_runs.len(),
        longest_credible_run_frames: longest,
        short_glitch_count,
        raw_late_active_ratio: ratio(count_true(&active[late_start..]), frames.len() - late_start),
        credible_late_active_ratio: ratio(
            count_true(&credible[late_start..]),
            frames.len() - late_start,
        ),
        periodic_sparse_pulses,
        quality_decision: if disproportionate {
            "trimmed"
        } else {
            "accepted"
        },
        quality_reason,
        tail_trimmed: disproportionate,
    };
    Ok((samples[..final_len].to_vec(), report))
}

fn true_runs(mask: &[bool]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut index = 0;
    while index < mask.len() {
        if !mask[index] {
            index += 1;
            continue;
        }
        let start = index;
        while index < mask.len() && mask[index] {
            index += 1;
        }
        runs.push((start, index));
    }
    runs
}

fn periodic_short_runs(runs: &[(usize, usize)]) -> bool {
    let starts = runs
        .iter()
        .filter(|(start, end)| end - start < MIN_ONSET_FRAMES)
        .map(|(start, _)| *start)
        .collect::<Vec<_>>();
    if starts.len() < PERIODIC_MIN_RUNS {
        return false;
    }
    let spacings = starts.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>();
    let min = spacings.iter().copied().min().unwrap_or(0);
    let max = spacings.iter().copied().max().unwrap_or(usize::MAX);
    min >= PERIODIC_MIN_SPACING_FRAMES
        && max <= PERIODIC_MAX_SPACING_FRAMES
        && max - min <= PERIODIC_MAX_JITTER_FRAMES
}

fn count_true(mask: &[bool]) -> usize {
    mask.iter().filter(|value| **value).count()
}

fn ratio(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 / total as f64
    }
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

fn quality_error(
    reason: &str,
    samples: usize,
    peak: i32,
    raw_active_ratio: f64,
    credible_ratio: f64,
    islands: usize,
    glitches: usize,
    periodic: bool,
) -> InfraError {
    InfraError::Backend(format!(
        "IndexTTS_F waveform rejected: quality_decision rejected, reason {reason}, samples {samples}, peak {peak}, raw_active_ratio {raw_active_ratio:.6}, raw_credible_active_ratio {credible_ratio:.6}, credible_islands {islands}, short_glitches {glitches}, periodic_sparse_pulses {periodic}"
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
