//! Keyword-boosted modified beam search over the transducer's output, frame
//! by frame: a port of sherpa-onnx's `TransducerKeywordDecoder`
//! (`sherpa-onnx/csrc/transducer-keyword-decoder.cc`).

use crate::graph::{Graph, ROOT};

/// The decoder's left context (tokens it reads).
pub const CONTEXT: usize = 2;
const BLANK: i64 = 0;

#[derive(Debug, Clone)]
pub struct Hyp {
    /// The tokens so far, after `CONTEXT` padding (`[-1, 0]`): only those of
    /// the keyword being matched (a path back at the root starts over).
    pub ys: Vec<i64>,
    pub log_prob: f32,
    /// Where the path stands in the keyword graph.
    pub state: usize,
    pub trailing_blanks: usize,
    /// Per token: when (the caller's frame position) and its probability.
    pub times: Vec<usize>,
    pub probs: Vec<f32>,
}

impl Hyp {
    pub fn start() -> Self {
        Hyp {
            ys: start_context(),
            log_prob: 0.0,
            state: ROOT,
            trailing_blanks: 0,
            times: Vec::new(),
            probs: Vec::new(),
        }
    }

    /// The decoder's input: the last `CONTEXT` tokens.
    pub fn context(&self) -> &[i64] {
        &self.ys[self.ys.len() - CONTEXT..]
    }
}

fn start_context() -> Vec<i64> {
    let mut ys = vec![-1; CONTEXT];
    ys[CONTEXT - 1] = BLANK;
    ys
}

/// A keyword spotted: its graph node, and when its tokens came.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub node: usize,
    pub first_time: usize,
    pub last_time: usize,
    /// The mean probability of its tokens.
    pub score: f32,
}

/// How the search runs: the joiner's width, the token taken as blank too,
/// how many paths it keeps, the blanks a keyword's end needs.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub vocab: usize,
    pub unk: Option<i64>,
    pub max_paths: usize,
    pub trailing_blanks: usize,
}

/// One frame of the search: `hyps` (one row of `log_probs`, log-softmaxed,
/// per hyp, `vocab` wide) become the best `max_paths` continuations. A
/// match of the best one ends the search: it starts over at the root.
pub fn step(
    graph: &Graph,
    hyps: Vec<Hyp>,
    log_probs: &[f32],
    params: Params,
    time: usize,
) -> (Vec<Hyp>, Option<Match>) {
    let Params {
        vocab,
        unk,
        max_paths,
        trailing_blanks,
    } = params;
    // Total scores, then the best `max_paths` of them all.
    let mut scored: Vec<(f32, usize)> = log_probs
        .iter()
        .enumerate()
        .map(|(k, &lp)| (lp + hyps[k / vocab].log_prob, k))
        .collect();
    let keep = max_paths.min(scored.len());
    scored.select_nth_unstable_by(keep.saturating_sub(1), |a, b| b.0.total_cmp(&a.0));
    scored.truncate(keep);
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut next: Vec<Hyp> = Vec::with_capacity(keep);
    for (total, k) in scored {
        let (index, token) = (k / vocab, (k % vocab) as i64);
        let mut hyp = hyps[index].clone();
        let mut context_score = 0.0;
        // Blank, and unk as blank.
        if token != BLANK && Some(token) != unk {
            hyp.ys.push(token);
            hyp.times.push(time);
            hyp.probs.push(log_probs[k].exp());
            hyp.trailing_blanks = 0;
            let (score, state) = graph.forward(hyp.state, token);
            context_score = score;
            hyp.state = state;
            // Back at the root: the decoder's history goes too.
            if graph.nodes[state].token == -1 {
                hyp.ys = start_context();
                hyp.times.clear();
                hyp.probs.clear();
            }
        } else {
            hyp.trailing_blanks += 1;
        }
        hyp.log_prob = total + context_score;
        // The same tokens by another path: one hyp, probabilities added.
        match next.iter_mut().find(|h| h.ys == hyp.ys) {
            Some(same) => same.log_prob = log_add(same.log_prob, hyp.log_prob),
            None => next.push(hyp),
        }
    }
    let best = next
        .iter()
        .max_by(|a, b| a.log_prob.total_cmp(&b.log_prob))
        .expect("at least one hypothesis");
    if let Some(node) = graph.matched(best.state) {
        let level = graph.nodes[node].level.min(best.probs.len());
        if level > 0 {
            let score = best.probs[..level].iter().sum::<f32>() / level as f32;
            if best.trailing_blanks > trailing_blanks && score >= graph.nodes[node].threshold {
                let found = Match {
                    node,
                    first_time: best.times[best.times.len() - level],
                    last_time: best.times[best.times.len() - 1],
                    score,
                };
                return (vec![Hyp::start()], Some(found));
            }
        }
    }
    (next, None)
}

fn log_add(a: f32, b: f32) -> f32 {
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

/// `logits` (rows of `vocab`) to log probabilities, in place.
pub fn log_softmax(logits: &mut [f32], vocab: usize) {
    for row in logits.chunks_mut(vocab) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
        let log_sum = max + sum.ln();
        for v in row.iter_mut() {
            *v -= log_sum;
        }
    }
}
