use super::*;
use kaldi_native_fbank::{
    fbank::{FbankComputer, FbankOptions},
    online::{FeatureComputer, OnlineFeature},
};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct SenseVoiceConfigFile {
    pub frontend_conf: FrontendConfig,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct FrontendConfig {
    pub fs: u32,
    pub window: String,
    pub n_mels: usize,
    pub frame_length: u32,
    pub frame_shift: u32,
    pub lfr_m: usize,
    pub lfr_n: usize,
}

#[derive(Debug, Clone)]
pub struct Cmvn {
    means: Vec<f32>,
    vars: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SenseVoiceFeatures {
    pub frames: usize,
    pub dim: usize,
    pub data: Vec<f32>,
}

impl SenseVoiceConfigFile {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
        let config: Self = serde_yaml::from_slice(&bytes)?;
        if config.frontend_conf.fs != audio::TARGET_SAMPLE_RATE
            || config.frontend_conf.n_mels == 0
            || config.frontend_conf.lfr_m == 0
            || config.frontend_conf.lfr_n == 0
        {
            return Err(InfraError::Adapter(format!(
                "unsupported SenseVoice frontend configuration in {}: {:?}",
                path.display(),
                config.frontend_conf
            )));
        }
        Ok(config)
    }
}

impl Cmvn {
    pub fn load(path: &Path, expected_dim: usize) -> Result<Self> {
        let text =
            fs::read_to_string(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
        let lines = text.lines().collect::<Vec<_>>();
        let mut means = None;
        let mut vars = None;
        for (index, line) in lines.iter().enumerate() {
            if line.trim_start().starts_with("<AddShift>") {
                means = lines
                    .get(index + 1)
                    .map(|line| parse_vector(line))
                    .transpose()?;
            } else if line.trim_start().starts_with("<Rescale>") {
                vars = lines
                    .get(index + 1)
                    .map(|line| parse_vector(line))
                    .transpose()?;
            }
        }
        let means = means.ok_or_else(|| {
            InfraError::Adapter(format!("{} has no <AddShift> CMVN vector", path.display()))
        })?;
        let vars = vars.ok_or_else(|| {
            InfraError::Adapter(format!("{} has no <Rescale> CMVN vector", path.display()))
        })?;
        if means.len() != expected_dim || vars.len() != expected_dim {
            return Err(InfraError::Adapter(format!(
                "SenseVoice CMVN dimension mismatch in {}: means={}, vars={}, expected={expected_dim}",
                path.display(), means.len(), vars.len()
            )));
        }
        Ok(Self { means, vars })
    }

    pub fn apply(&self, frame: &mut [f32]) {
        for ((value, mean), var) in frame.iter_mut().zip(&self.means).zip(&self.vars) {
            *value = (*value + *mean) * *var;
        }
    }
}

fn parse_vector(line: &str) -> Result<Vec<f32>> {
    let start = line
        .find('[')
        .ok_or_else(|| InfraError::Adapter(format!("invalid CMVN vector without `[`: {line}")))?;
    let end = line
        .rfind(']')
        .ok_or_else(|| InfraError::Adapter(format!("invalid CMVN vector without `]`: {line}")))?;
    line[start + 1..end]
        .split_whitespace()
        .map(|part| {
            part.parse::<f32>()
                .map_err(|e| InfraError::Adapter(format!("invalid CMVN value `{part}`: {e}")))
        })
        .collect()
}

/// LFR features of `samples`, CMVN applied when given (SenseVoice's
/// `am.mvn`; Fun-ASR-Nano has none).
pub fn extract(
    samples: &[f32],
    frontend: &FrontendConfig,
    cmvn: Option<&Cmvn>,
) -> Result<SenseVoiceFeatures> {
    let raw = fbank_frames(samples, frontend)?;
    if raw.is_empty() {
        return Ok(SenseVoiceFeatures {
            frames: 0,
            dim: frontend.n_mels * frontend.lfr_m,
            data: Vec::new(),
        });
    }
    lfr_cmvn(raw, frontend, cmvn)
}

/// Kaldi FBANK frames of `samples` (16 kHz, [-1, 1]). Edges are snipped, so
/// frame `i` depends on samples `i * shift .. i * shift + length` alone and a
/// stream may be cut into pieces on frame-shift boundaries.
pub fn fbank_frames(samples: &[f32], frontend: &FrontendConfig) -> Result<Vec<Vec<f32>>> {
    let mut options = FbankOptions::default();
    options.frame_opts.dither = 0.0;
    options.frame_opts.samp_freq = frontend.fs as f32;
    options.frame_opts.window_type = frontend.window.clone();
    options.frame_opts.frame_length_ms = frontend.frame_length as f32;
    options.frame_opts.frame_shift_ms = frontend.frame_shift as f32;
    options.frame_opts.snip_edges = true;
    options.mel_opts.num_bins = frontend.n_mels;
    options.energy_floor = 0.0;
    // kaldi-native-fbank C++ defaults to false; the Rust port defaults to true.
    options.use_energy = false;

    let computer = FbankComputer::new(options)
        .map_err(|e| InfraError::Adapter(format!("initialize Kaldi FBANK: {e}")))?;
    let mut fbank = OnlineFeature::new(FeatureComputer::Fbank(computer));
    // Official FunASR feeds normalized PCM multiplied back to the i16 scale.
    let scaled = samples
        .iter()
        .map(|sample| sample * 32_768.0)
        .collect::<Vec<_>>();
    fbank.accept_waveform(frontend.fs as f32, &scaled);
    Ok(std::mem::take(&mut fbank.features))
}

impl FrontendConfig {
    pub fn frame_shift_samples(&self) -> usize {
        (self.fs as usize * self.frame_shift as usize) / 1000
    }

    pub fn frame_length_samples(&self) -> usize {
        (self.fs as usize * self.frame_length as usize) / 1000
    }
}

fn lfr_cmvn(
    mut raw: Vec<Vec<f32>>,
    frontend: &FrontendConfig,
    cmvn: Option<&Cmvn>,
) -> Result<SenseVoiceFeatures> {
    let original_frames = raw.len();
    let output_frames = original_frames.div_ceil(frontend.lfr_n);
    let start_padding = (frontend.lfr_m - 1) / 2;
    let first = raw[0].clone();
    for _ in 0..start_padding {
        raw.insert(0, first.clone());
    }
    let dim = frontend.n_mels * frontend.lfr_m;
    let mut data = Vec::with_capacity(output_frames * dim);
    for output_index in 0..output_frames {
        let start = output_index * frontend.lfr_n;
        let last = raw.last().expect("non-empty FBANK frames");
        let mut frame = Vec::with_capacity(dim);
        for offset in 0..frontend.lfr_m {
            frame.extend_from_slice(raw.get(start + offset).unwrap_or(last));
        }
        if let Some(cmvn) = cmvn {
            cmvn.apply(&mut frame);
        }
        data.extend_from_slice(&frame);
    }
    Ok(SenseVoiceFeatures {
        frames: output_frames,
        dim,
        data,
    })
}

#[cfg(test)]
pub fn lfr_cmvn_for_test(
    raw: Vec<Vec<f32>>,
    frontend: &FrontendConfig,
    cmvn: &Cmvn,
) -> Result<SenseVoiceFeatures> {
    lfr_cmvn(raw, frontend, Some(cmvn))
}
