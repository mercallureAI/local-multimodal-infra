//! Next-token selection from one row of logits.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 0 selects the most likely token.
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

/// SplitMix64: small, seedable and good enough for sampling.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Applies penalties for tokens already generated (OpenAI semantics).
pub fn apply_penalties(logits: &mut [f32], counts: &HashMap<u32, usize>, config: &SamplerConfig) {
    if config.presence_penalty == 0.0 && config.frequency_penalty == 0.0 {
        return;
    }
    for (&token, &count) in counts {
        if let Some(logit) = logits.get_mut(token as usize) {
            *logit -= config.presence_penalty + config.frequency_penalty * count as f32;
        }
    }
}

pub fn sample(logits: &[f32], config: &SamplerConfig, rng: &mut Rng) -> u32 {
    if config.temperature <= 0.0 || config.top_k == 1 {
        return argmax(logits);
    }
    let top_k = if config.top_k == 0 {
        logits.len()
    } else {
        config.top_k.min(logits.len())
    };
    let mut candidates = logits
        .iter()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .map(|(index, logit)| (index as u32, *logit))
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return argmax(logits);
    }
    if top_k < candidates.len() {
        candidates.select_nth_unstable_by(top_k - 1, |a, b| b.1.total_cmp(&a.1));
        candidates.truncate(top_k);
    }
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let max = candidates[0].1;
    let mut probs = candidates
        .iter()
        .map(|(_, logit)| ((logit - max) / config.temperature).exp())
        .collect::<Vec<_>>();
    let total: f32 = probs.iter().sum();
    for prob in &mut probs {
        *prob /= total;
    }
    let mut keep = probs.len();
    if config.top_p > 0.0 && config.top_p < 1.0 {
        let mut cumulative = 0.0;
        for (index, prob) in probs.iter().enumerate() {
            cumulative += prob;
            if cumulative >= config.top_p {
                keep = index + 1;
                break;
            }
        }
    }
    let total: f32 = probs[..keep].iter().sum();
    let mut target = rng.next_f32() * total;
    for (index, prob) in probs[..keep].iter().enumerate() {
        target -= prob;
        if target <= 0.0 {
            return candidates[index].0;
        }
    }
    candidates[keep - 1].0
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, logit) in logits.iter().enumerate() {
        if *logit > logits[best] || logits[best].is_nan() {
            best = index;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(temperature: f32) -> SamplerConfig {
        SamplerConfig {
            temperature,
            top_p: 1.0,
            top_k: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }

    #[test]
    fn greedy_picks_the_largest_logit() {
        let mut rng = Rng::new(1);
        assert_eq!(sample(&[0.1, 3.0, -1.0], &config(0.0), &mut rng), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut rng = Rng::new(1);
        let mut cfg = config(1.0);
        cfg.top_k = 1;
        assert_eq!(sample(&[0.1, 3.0, 2.9], &cfg, &mut rng), 1);
    }

    #[test]
    fn top_p_drops_the_tail() {
        let mut rng = Rng::new(7);
        let mut cfg = config(1.0);
        cfg.top_p = 0.5;
        for _ in 0..200 {
            assert_eq!(sample(&[10.0, 0.0, 0.0, 0.0], &cfg, &mut rng), 0);
        }
    }

    #[test]
    fn sampling_follows_the_distribution() {
        let mut rng = Rng::new(3);
        let cfg = config(1.0);
        let hits = (0..2000)
            .filter(|_| sample(&[0.0, 0.0f32.ln_1p() + 2f32.ln()], &cfg, &mut rng) == 1)
            .count();
        // p(1) = 2/3
        assert!((1200..1466).contains(&hits), "{hits}");
    }

    #[test]
    fn penalties_lower_seen_tokens() {
        let mut logits = vec![1.0, 1.0];
        let counts = HashMap::from([(1u32, 2usize)]);
        let mut cfg = config(1.0);
        cfg.presence_penalty = 0.5;
        cfg.frequency_penalty = 0.25;
        apply_penalties(&mut logits, &counts, &cfg);
        assert_eq!(logits, vec![1.0, 0.0]);
    }
}
