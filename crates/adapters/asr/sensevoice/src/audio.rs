//! Audio input: any container/codec symphonia knows (WAV, FLAC, MP3, AAC in
//! MP4/M4A or Bilibili's fragmented .m4s, Ogg Vorbis, MKV/WebM, ALAC),
//! decoded packet by packet, mixed to mono and resampled to 16 kHz, so a
//! long recording is never held in memory at its source rate.

use super::*;
use symphonia::core::{
    codecs::audio::{AudioDecoder, AudioDecoderOptions},
    errors::Error as SymphoniaError,
    formats::{probe::Hint, FormatOptions, FormatReader, TrackType},
    io::MediaSourceStream,
    meta::MetadataOptions,
};

pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// A decoded audio file, read in 16 kHz mono stretches.
pub struct AudioStream {
    path: PathBuf,
    format: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track_id: u32,
    source_rate: u32,
    resampler: Option<Resampler>,
    interleaved: Vec<f32>,
    mono: Vec<f32>,
    finished: bool,
}

impl std::fmt::Debug for AudioStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioStream")
            .field("path", &self.path)
            .field("source_rate", &self.source_rate)
            .finish_non_exhaustive()
    }
}

impl AudioStream {
    pub fn open(path: &Path) -> Result<Self> {
        let file = fs::File::open(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
        let mut hint = Hint::new();
        if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
            hint.with_extension(extension);
        }
        let unsupported = |error: SymphoniaError| {
            InfraError::BadRequest(format!(
                "unsupported or unreadable audio {}: {error}",
                path.display()
            ))
        };
        let format = symphonia::default::get_probe()
            .probe(
                &hint,
                MediaSourceStream::new(Box::new(file), Default::default()),
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(unsupported)?;
        let track = format.default_track(TrackType::Audio).ok_or_else(|| {
            InfraError::BadRequest(format!("{} has no audio track", path.display()))
        })?;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|params| params.audio())
            .ok_or_else(|| {
                InfraError::BadRequest(format!("{} has no audio codec parameters", path.display()))
            })?;
        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(params, &AudioDecoderOptions::default())
            .map_err(unsupported)?;
        Ok(Self {
            path: path.to_path_buf(),
            track_id: track.id,
            format,
            decoder,
            source_rate: 0,
            resampler: None,
            interleaved: Vec::new(),
            mono: Vec::new(),
            finished: false,
        })
    }

    /// Appends the next decoded stretch (one or more packets, 16 kHz mono)
    /// to `out`; `false` once the stream is exhausted.
    pub fn read(&mut self, out: &mut Vec<f32>) -> Result<bool> {
        let before = out.len();
        while !self.finished && out.len() == before {
            let packet = match self.format.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => {
                    self.finish(out);
                    break;
                }
                // A truncated file ends where its data ends.
                Err(SymphoniaError::IoError(error))
                    if error.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    self.finish(out);
                    break;
                }
                Err(error) => {
                    return Err(InfraError::BadRequest(format!(
                        "read audio {}: {error}",
                        self.path.display()
                    )))
                }
            };
            if packet.track_id != self.track_id {
                continue;
            }
            let buffer = match self.decoder.decode(&packet) {
                Ok(buffer) => buffer,
                // A corrupt packet is skipped, as players do.
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(error) => {
                    return Err(InfraError::BadRequest(format!(
                        "decode audio {}: {error}",
                        self.path.display()
                    )))
                }
            };
            let rate = buffer.spec().rate();
            let channels = buffer.spec().channels().count().max(1);
            if rate == 0 {
                return Err(InfraError::BadRequest(format!(
                    "audio {} has a zero sample rate",
                    self.path.display()
                )));
            }
            if self.source_rate == 0 {
                self.source_rate = rate;
                self.resampler =
                    (rate != TARGET_SAMPLE_RATE).then(|| Resampler::new(rate, TARGET_SAMPLE_RATE));
            } else if rate != self.source_rate {
                return Err(InfraError::BadRequest(format!(
                    "audio {} changes sample rate from {} to {rate} Hz",
                    self.path.display(),
                    self.source_rate
                )));
            }
            buffer.copy_to_vec_interleaved(&mut self.interleaved);
            self.mono.clear();
            self.mono.extend(
                self.interleaved
                    .chunks(channels)
                    .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32),
            );
            match &mut self.resampler {
                Some(resampler) => resampler.process(&self.mono, out),
                None => out.extend_from_slice(&self.mono),
            }
        }
        Ok(out.len() > before)
    }

    fn finish(&mut self, out: &mut Vec<f32>) {
        self.finished = true;
        if let Some(resampler) = &mut self.resampler {
            resampler.finish(out);
        }
    }
}

/// The whole of `path`, 16 kHz mono.
#[cfg(test)]
pub fn read_audio_mono_16k(path: &Path) -> Result<Vec<f32>> {
    let mut stream = AudioStream::open(path)?;
    let mut samples = Vec::new();
    while stream.read(&mut samples)? {}
    Ok(samples)
}

/// Zero crossings of the windowed sinc on each side of a tap.
const RESAMPLER_ZERO_CROSSINGS: f64 = 16.0;
/// The passband edge as a share of the lower Nyquist frequency.
const RESAMPLER_ROLLOFF: f64 = 0.92;
/// Phases tabulated exactly up to this count; beyond it (sample rates with
/// a tiny common divisor) the fractional position is rounded to it.
const RESAMPLER_MAX_PHASES: u64 = 1024;

/// A streaming band-limited (Blackman-windowed sinc, polyphase) resampler
/// between two fixed rates: the anti-aliasing low-pass linear interpolation
/// lacks when 44.1/48 kHz audio comes down to 16 kHz.
#[derive(Debug, Clone)]
pub struct Resampler {
    /// Output sample n sits at input position n * step / up.
    up: u64,
    step: u64,
    phases: u64,
    half_taps: usize,
    /// `phases` rows of `2 * half_taps` weights, for inputs
    /// `index - half_taps + 1 ..= index + half_taps`.
    table: Vec<f32>,
    /// Inputs from absolute index `pending_start` on (negative: the zeros
    /// before the first sample).
    pending: Vec<f32>,
    pending_start: i64,
    input_len: u64,
    next_output: u64,
}

impl Resampler {
    pub fn new(source_rate: u32, target_rate: u32) -> Self {
        let divisor = gcd(source_rate as u64, target_rate as u64);
        let up = target_rate as u64 / divisor;
        let step = source_rate as u64 / divisor;
        // Cycles per input sample.
        let cutoff = 0.5 * RESAMPLER_ROLLOFF * (target_rate as f64 / source_rate as f64).min(1.0);
        let half_width = RESAMPLER_ZERO_CROSSINGS / (2.0 * cutoff);
        let half_taps = half_width.ceil() as usize;
        let phases = up.min(RESAMPLER_MAX_PHASES);
        let mut table = Vec::with_capacity(phases as usize * 2 * half_taps);
        for phase in 0..phases {
            let fraction = phase as f64 / phases as f64;
            let row_start = table.len();
            let mut sum = 0.0;
            for offset in -(half_taps as i64) + 1..=half_taps as i64 {
                // Distance from the output position to input `index + offset`.
                let distance = fraction - offset as f64;
                let weight = windowed_sinc(distance, cutoff, half_width);
                sum += weight;
                table.push(weight as f32);
            }
            if sum.abs() > 1e-9 {
                for weight in &mut table[row_start..] {
                    *weight /= sum as f32;
                }
            }
        }
        Self {
            up,
            step,
            phases,
            half_taps,
            table,
            pending: vec![0.0; half_taps],
            pending_start: -(half_taps as i64),
            input_len: 0,
            next_output: 0,
        }
    }

    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.pending.extend_from_slice(input);
        self.input_len += input.len() as u64;
        self.emit(out, self.input_len);
    }

    /// Flushes the outputs the last inputs still owe (the zeros past the end
    /// stand in for the missing right-hand taps).
    pub fn finish(&mut self, out: &mut Vec<f32>) {
        let available = self.input_len;
        self.pending
            .extend(std::iter::repeat_n(0.0, self.half_taps + 1));
        let total = (available * self.up).div_ceil(self.step);
        while self.next_output < total {
            self.emit_one(out);
        }
        self.pending.clear();
    }

    /// Emits every output whose taps are all available.
    fn emit(&mut self, out: &mut Vec<f32>, available: u64) {
        loop {
            let (index, _) = self.locate(self.next_output);
            if index + self.half_taps as i64 >= available as i64 {
                break;
            }
            self.emit_one(out);
        }
        // Keep the inputs the next output's leftmost tap still reaches.
        let (index, _) = self.locate(self.next_output);
        let keep_from = index - self.half_taps as i64 + 1;
        let drop = (keep_from - self.pending_start).clamp(0, self.pending.len() as i64) as usize;
        if drop > 0 {
            self.pending.drain(..drop);
            self.pending_start += drop as i64;
        }
    }

    /// The input index left of output `n` and its phase row.
    fn locate(&self, n: u64) -> (i64, usize) {
        let position = n * self.step;
        let index = (position / self.up) as i64;
        let remainder = position % self.up;
        if self.phases == self.up {
            return (index, remainder as usize);
        }
        let phase = (remainder * self.phases + self.up / 2) / self.up;
        if phase == self.phases {
            (index + 1, 0)
        } else {
            (index, phase as usize)
        }
    }

    fn emit_one(&mut self, out: &mut Vec<f32>) {
        let (index, phase) = self.locate(self.next_output);
        let taps = 2 * self.half_taps;
        let row = &self.table[phase * taps..(phase + 1) * taps];
        let first = (index - self.half_taps as i64 + 1 - self.pending_start) as usize;
        let value = self.pending[first..first + taps]
            .iter()
            .zip(row)
            .map(|(sample, weight)| sample * weight)
            .sum();
        out.push(value);
        self.next_output += 1;
    }
}

fn windowed_sinc(distance: f64, cutoff: f64, half_width: f64) -> f64 {
    if distance.abs() >= half_width {
        return 0.0;
    }
    let x = 2.0 * cutoff * distance;
    let sinc = if x.abs() < 1e-12 {
        1.0
    } else {
        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
    };
    // Blackman over [-half_width, half_width].
    let t = (distance / half_width + 1.0) * 0.5;
    let window = 0.42 - 0.5 * (2.0 * std::f64::consts::PI * t).cos()
        + 0.08 * (4.0 * std::f64::consts::PI * t).cos();
    2.0 * cutoff * sinc * window
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

#[cfg(test)]
mod resampler_tests {
    use super::*;

    fn resample_all(input: &[f32], from: u32, to: u32, chunk: usize) -> Vec<f32> {
        let mut resampler = Resampler::new(from, to);
        let mut out = Vec::new();
        for part in input.chunks(chunk) {
            resampler.process(part, &mut out);
        }
        resampler.finish(&mut out);
        out
    }

    fn tone(rate: u32, hz: f64, seconds: f64) -> Vec<f32> {
        (0..(rate as f64 * seconds) as usize)
            .map(|i| (2.0 * std::f64::consts::PI * hz * i as f64 / rate as f64).sin() as f32)
            .collect()
    }

    fn rms(values: &[f32]) -> f64 {
        (values.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
    }

    #[test]
    fn output_length_and_chunking_do_not_change_the_result() {
        let input = tone(44_100, 440.0, 1.3);
        let whole = resample_all(&input, 44_100, 16_000, input.len());
        assert_eq!(
            whole.len(),
            (input.len() as u64 * 160).div_ceil(441) as usize
        );
        for chunk in [1, 7, 1024, 4410] {
            assert_eq!(
                resample_all(&input, 44_100, 16_000, chunk),
                whole,
                "chunk {chunk}"
            );
        }
    }

    #[test]
    fn passband_tone_survives_and_aliasing_tone_is_removed() {
        for from in [44_100, 48_000, 22_050, 8_000] {
            let kept = resample_all(&tone(from, 1_000.0, 1.0), from, 16_000, 4096);
            // A sine's RMS is 1/sqrt(2); skip the filter's edges.
            let middle = &kept[1_000..kept.len() - 1_000];
            assert!(
                (rms(middle) - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.01,
                "{from}: {}",
                rms(middle)
            );
            if from > 16_000 {
                // 12 kHz would fold back to 4 kHz without the low-pass.
                let removed = resample_all(&tone(from, 12_000.0, 1.0), from, 16_000, 4096);
                let middle = &removed[1_000..removed.len() - 1_000];
                assert!(rms(middle) < 0.01, "{from}: {}", rms(middle));
            }
        }
    }

    #[test]
    fn odd_rates_use_rounded_phases() {
        let input = tone(44_099, 500.0, 0.5);
        let out = resample_all(&input, 44_099, 16_000, 999);
        assert_eq!(
            out.len(),
            (input.len() as u64 * 16_000).div_ceil(44_099) as usize
        );
        let middle = &out[500..out.len() - 500];
        assert!((rms(middle) - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.01);
    }
}
