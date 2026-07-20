use crate::{audio::TARGET_SAMPLE_RATE, vad::SpeechSegment};
use kaldi_native_fbank::{
    fbank::{FbankComputer, FbankOptions},
    online::{FeatureComputer, OnlineFeature},
};
use local_backend_ort::{
    OrtBackend, OrtSession, OrtTensorData, OrtTensorInput, PinnedCudaF32IoBinding, ProviderKind,
    ProviderSelection, SessionProviderReport, TensorElement,
};
use local_error::{InfraError, Result};
use std::{collections::HashMap, path::Path};

const CHUNK_SAMPLES: usize = 24_000;
const CHUNK_SHIFT_SAMPLES: usize = 12_000;
const NEW_SPEAKER_COSINE: f32 = 0.65;
const MERGE_SPEAKER_COSINE: f32 = 0.78;

#[derive(Debug, Clone)]
struct SpeakerChunk {
    start_sample: usize,
    end_sample: usize,
    samples: Vec<f32>,
}

#[derive(Debug)]
pub struct CampPlus {
    session: OrtSession,
    batch_size: usize,
    use_pinned_cuda_io: bool,
    pinned_cuda_bindings: HashMap<(usize, usize), PinnedCudaF32IoBinding>,
}

impl CampPlus {
    pub fn load(
        root: &Path,
        providers: &ProviderSelection,
        batch_size: usize,
        use_pinned_cuda_io: bool,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(InfraError::Adapter(
                "CAM++ batch size must be greater than zero".to_string(),
            ));
        }
        let model = root.join("campplus_cn_en_common_200k.onnx");
        let session = OrtBackend::new(providers.clone()).load_session(&model)?;
        if session.inputs().len() != 1
            || session.inputs()[0].element_type != TensorElement::F32
            || session.inputs()[0].shape.len() != 3
        {
            return Err(InfraError::Adapter(format!(
                "CAM++ expects one rank-3 F32 input, got {:?}",
                session.inputs()
            )));
        }
        Ok(Self {
            session,
            batch_size,
            use_pinned_cuda_io,
            pinned_cuda_bindings: HashMap::new(),
        })
    }

    pub fn label_segments(
        &mut self,
        samples: &[f32],
        vad_segments: &[SpeechSegment],
    ) -> Result<Vec<usize>> {
        let chunks = speaker_chunks(samples, vad_segments);
        if chunks.is_empty() {
            return Ok(vec![0; vad_segments.len()]);
        }
        let mut embeddings = Vec::with_capacity(chunks.len());
        for batch in chunks.chunks(self.batch_size) {
            embeddings.extend(self.embedding_batch(batch)?);
        }
        let labels = cluster_embeddings(&embeddings);
        Ok(vad_segments
            .iter()
            .map(|segment| dominant_label(segment, &chunks, &labels))
            .collect())
    }

    pub fn provider_report(&self) -> SessionProviderReport {
        self.session.provider_report()
    }

    fn embedding_batch(&mut self, chunks: &[SpeakerChunk]) -> Result<Vec<Vec<f32>>> {
        let mut frames = None;
        let mut data = Vec::new();
        for chunk in chunks {
            let (chunk_frames, chunk_data) = fbank(&chunk.samples)?;
            if let Some(frames) = frames {
                if chunk_frames != frames {
                    return Err(InfraError::Adapter(format!(
                        "CAM++ batch contains inconsistent frame counts: {chunk_frames} and {frames}"
                    )));
                }
            } else {
                frames = Some(chunk_frames);
                data.reserve(chunks.len() * chunk_data.len());
            }
            data.extend(chunk_data);
        }
        let frames = frames.unwrap_or(0);
        if frames == 0 {
            return Ok(Vec::new());
        }
        let input = OrtTensorInput {
            name: self.session.inputs()[0].name.clone(),
            shape: vec![chunks.len(), frames, 80],
            data: OrtTensorData::F32(data),
        };
        let outputs = if self.use_pinned_cuda_io && self.session.provider() == ProviderKind::Cuda {
            let key = (chunks.len(), frames);
            if !self.pinned_cuda_bindings.contains_key(&key) {
                let binding = self.session.create_pinned_cuda_f32_binding(
                    self.session.device_id().unwrap_or(0),
                    &input.shape,
                    &[chunks.len(), 192],
                )?;
                self.pinned_cuda_bindings.insert(key, binding);
            }
            self.session.run_pinned_cuda_f32_binding(
                self.pinned_cuda_bindings
                    .get_mut(&key)
                    .expect("binding inserted above"),
                &input,
            )?
        } else {
            self.session.run_tensors(&[input])?
        };
        let output = outputs
            .into_iter()
            .find(|output| output.data.element_type() == TensorElement::F32)
            .ok_or_else(|| InfraError::Adapter("CAM++ returned no F32 embedding".to_string()))?;
        let OrtTensorData::F32(embedding) = output.data else {
            unreachable!()
        };
        decode_embedding_batch(embedding, chunks.len())
    }
}

fn decode_embedding_batch(mut data: Vec<f32>, batch: usize) -> Result<Vec<Vec<f32>>> {
    let expected = batch.checked_mul(192).ok_or_else(|| {
        InfraError::Adapter(format!(
            "CAM++ output batch {batch} overflows embedding size"
        ))
    })?;
    if data.len() != expected {
        return Err(InfraError::Adapter(format!(
            "CAM++ returned {} values instead of {batch} embeddings of 192 values",
            data.len()
        )));
    }
    let mut embeddings = Vec::with_capacity(batch);
    for embedding in data.chunks_exact_mut(192) {
        normalize(embedding);
        embeddings.push(embedding.to_vec());
    }
    if embeddings.len() != batch {
        return Err(InfraError::Adapter(format!(
            "CAM++ returned {} embeddings instead of {batch}",
            embeddings.len()
        )));
    }
    Ok(embeddings)
}

fn speaker_chunks(samples: &[f32], segments: &[SpeechSegment]) -> Vec<SpeakerChunk> {
    let mut chunks = Vec::new();
    for segment in segments {
        let data = &samples[segment.start_sample..segment.end_sample];
        let mut offset = 0usize;
        let mut last_end = 0usize;
        while offset < data.len() {
            let end = (offset + CHUNK_SAMPLES).min(data.len());
            if end <= last_end {
                break;
            }
            last_end = end;
            let start = end.saturating_sub(CHUNK_SAMPLES);
            let mut chunk = data[start..end].to_vec();
            chunk.resize(CHUNK_SAMPLES, 0.0);
            chunks.push(SpeakerChunk {
                start_sample: segment.start_sample + start,
                end_sample: segment.start_sample + end,
                samples: chunk,
            });
            offset += CHUNK_SHIFT_SAMPLES;
        }
    }
    chunks
}

fn fbank(samples: &[f32]) -> Result<(usize, Vec<f32>)> {
    let mut options = FbankOptions::default();
    options.frame_opts.samp_freq = TARGET_SAMPLE_RATE as f32;
    options.frame_opts.dither = 0.0;
    options.mel_opts.num_bins = 80;
    options.energy_floor = 0.0;
    options.use_energy = false;
    let computer = FbankComputer::new(options)
        .map_err(|e| InfraError::Adapter(format!("initialize CAM++ FBANK: {e}")))?;
    let mut fbank = OnlineFeature::new(FeatureComputer::Fbank(computer));
    let scaled = samples
        .iter()
        .map(|sample| sample * 32_768.0)
        .collect::<Vec<_>>();
    fbank.accept_waveform(TARGET_SAMPLE_RATE as f32, &scaled);
    let frames = fbank.num_frames_ready();
    let mut data = Vec::with_capacity(frames * 80);
    for frame in 0..frames {
        data.extend_from_slice(fbank.get_frame(frame).expect("ready CAM++ FBANK frame"));
    }
    for dim in 0..80 {
        let mean = (0..frames).map(|frame| data[frame * 80 + dim]).sum::<f32>() / frames as f32;
        for frame in 0..frames {
            data[frame * 80 + dim] -= mean;
        }
    }
    Ok((frames, data))
}

fn cluster_embeddings(embeddings: &[Vec<f32>]) -> Vec<usize> {
    // This matches FunASR ClusterBackend's short-input guard. Fewer than 20
    // overlapping 1.5 s windows do not contain enough evidence to estimate a
    // stable speaker count, so they are treated as one speaker.
    if embeddings.len() < 20 {
        return vec![0; embeddings.len()];
    }
    let mut centers: Vec<Vec<f32>> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut labels = Vec::with_capacity(embeddings.len());
    for embedding in embeddings {
        let best = centers
            .iter()
            .enumerate()
            .map(|(index, center)| (index, cosine(embedding, center)))
            .max_by(|left, right| left.1.total_cmp(&right.1));
        let label = match best {
            Some((index, score)) if score >= NEW_SPEAKER_COSINE => index,
            _ => {
                centers.push(embedding.clone());
                counts.push(0);
                centers.len() - 1
            }
        };
        counts[label] += 1;
        let count = counts[label] as f32;
        for (center, value) in centers[label].iter_mut().zip(embedding) {
            *center += (*value - *center) / count;
        }
        normalize(&mut centers[label]);
        labels.push(label);
    }
    merge_similar_clusters(&mut labels, embeddings);
    relabel_by_first_appearance(&mut labels);
    labels
}

fn merge_similar_clusters(labels: &mut [usize], embeddings: &[Vec<f32>]) {
    loop {
        let count = labels.iter().copied().max().map_or(0, |value| value + 1);
        if count < 2 {
            return;
        }
        let centers = (0..count)
            .map(|label| centroid(label, labels, embeddings))
            .collect::<Vec<_>>();
        let mut best = None;
        for left in 0..count {
            for right in left + 1..count {
                let score = cosine(&centers[left], &centers[right]);
                if best.is_none_or(|(_, _, current)| score > current) {
                    best = Some((left, right, score));
                }
            }
        }
        let Some((left, right, score)) = best else {
            return;
        };
        if score < MERGE_SPEAKER_COSINE {
            return;
        }
        for label in labels.iter_mut() {
            if *label == right {
                *label = left;
            } else if *label > right {
                *label -= 1;
            }
        }
    }
}

fn centroid(label: usize, labels: &[usize], embeddings: &[Vec<f32>]) -> Vec<f32> {
    let mut center = vec![0.0; embeddings[0].len()];
    let mut count = 0usize;
    for (index, embedding) in embeddings.iter().enumerate() {
        if labels[index] == label {
            count += 1;
            for (target, value) in center.iter_mut().zip(embedding) {
                *target += *value;
            }
        }
    }
    if count > 0 {
        for value in &mut center {
            *value /= count as f32;
        }
    }
    normalize(&mut center);
    center
}

fn dominant_label(segment: &SpeechSegment, chunks: &[SpeakerChunk], labels: &[usize]) -> usize {
    let count = labels.iter().copied().max().map_or(1, |value| value + 1);
    let mut overlap = vec![0usize; count];
    for (chunk, &label) in chunks.iter().zip(labels) {
        let start = segment.start_sample.max(chunk.start_sample);
        let end = segment.end_sample.min(chunk.end_sample);
        overlap[label] += end.saturating_sub(start);
    }
    overlap
        .iter()
        .enumerate()
        .max_by_key(|(_, duration)| *duration)
        .map_or(0, |(label, _)| label)
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    #[test]
    fn decodes_and_normalizes_batched_embeddings() {
        let mut data = vec![0.0; 2 * 192];
        data[0] = 3.0;
        data[1] = 4.0;
        data[192] = 5.0;
        data[193] = 12.0;
        let embeddings = decode_embedding_batch(data, 2).expect("decode embeddings");
        assert_eq!(embeddings.len(), 2);
        assert!((embeddings[0][0] - 0.6).abs() < 1e-6);
        assert!((embeddings[0][1] - 0.8).abs() < 1e-6);
        assert!((embeddings[1][0] - 5.0 / 13.0).abs() < 1e-6);
        assert!((embeddings[1][1] - 12.0 / 13.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_incomplete_batched_embeddings() {
        let error = decode_embedding_batch(vec![0.0; 193], 2).expect_err("invalid batch");
        assert!(error.to_string().contains("2 embeddings of 192"));
    }
}

fn relabel_by_first_appearance(labels: &mut [usize]) {
    let mut mapping = Vec::<usize>::new();
    for label in labels {
        let mapped = mapping
            .iter()
            .position(|value| value == label)
            .unwrap_or_else(|| {
                mapping.push(*label);
                mapping.len() - 1
            });
        *label = mapped;
    }
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn normalize(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for value in values {
            *value /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clustering_is_deterministic_and_separates_directions() {
        let mut embeddings = vec![vec![1.0, 0.0]; 10];
        embeddings.extend(vec![vec![0.0, 1.0]; 10]);
        assert_eq!(
            cluster_embeddings(&embeddings),
            [vec![0; 10], vec![1; 10]].concat()
        );
    }
}
