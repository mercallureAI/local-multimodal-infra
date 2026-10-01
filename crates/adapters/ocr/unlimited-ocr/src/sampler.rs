//! Greedy decoding with upstream's `SlidingWindowNoRepeatNgramProcessor`.

use std::collections::HashSet;

/// Tokens that would complete an n-gram already present in the last `window`
/// tokens of `history` (prompt included, no whitelist: EOS can be banned).
pub fn banned_tokens(history: &[u32], size: usize, window: usize) -> HashSet<u32> {
    let mut banned = HashSet::new();
    if size == 0 || history.len() < size {
        return banned;
    }
    let start = history.len().saturating_sub(window);
    let end = history.len() - size + 1;
    let prefix = &history[history.len() - (size - 1)..];
    for i in start..end {
        if &history[i..i + size - 1] == prefix {
            banned.insert(history[i + size - 1]);
        }
    }
    banned
}

/// Arg-max over `logits` skipping `banned`; the first index wins ties.
pub fn greedy(logits: &[f32], banned: &HashSet<u32>) -> u32 {
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    let mut found = false;
    for (i, &score) in logits.iter().enumerate() {
        if banned.contains(&(i as u32)) {
            continue;
        }
        if !found || score > best_score {
            best = i;
            best_score = score;
            found = true;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bans_the_token_that_repeats_an_ngram() {
        // ... 1 2 3 | 1 2 -> 3 would repeat "1 2 3".
        let history = [9, 1, 2, 3, 7, 1, 2];
        let banned = banned_tokens(&history, 3, 128);
        assert_eq!(banned, HashSet::from([3]));
    }

    #[test]
    fn window_limits_the_search() {
        let history = [1, 2, 3, 0, 0, 0, 0, 1, 2];
        assert!(banned_tokens(&history, 3, 4).is_empty());
        assert_eq!(banned_tokens(&history, 3, 9), HashSet::from([3]));
    }

    #[test]
    fn short_history_bans_nothing() {
        assert!(banned_tokens(&[1, 2], 3, 128).is_empty());
    }

    #[test]
    fn greedy_skips_banned() {
        let logits = [0.1, 5.0, 3.0];
        assert_eq!(greedy(&logits, &HashSet::new()), 1);
        assert_eq!(greedy(&logits, &HashSet::from([1])), 2);
    }
}
