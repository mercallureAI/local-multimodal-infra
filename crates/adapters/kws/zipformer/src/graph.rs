//! The keywords as an Aho-Corasick graph of their tokens, scoring the paths
//! that follow one: a port of sherpa-onnx's `ContextGraph`
//! (`sherpa-onnx/csrc/context-graph.cc`), strict mode.

use std::collections::{HashMap, VecDeque};

/// The root node.
pub const ROOT: usize = 0;

#[derive(Debug, Clone)]
pub struct Node {
    /// The token that leads here (-1 at the root).
    pub token: i64,
    /// Score of the arc that leads here.
    pub token_score: f32,
    /// Score of the whole path from the root.
    pub node_score: f32,
    /// Score granted when a keyword ends here (or at its output node).
    pub output_score: f32,
    /// Depth: the number of tokens from the root.
    pub level: usize,
    /// Mean token probability a match must reach (at a keyword's end).
    pub threshold: f32,
    pub is_end: bool,
    /// The keyword that ends here.
    pub phrase: Option<usize>,
    pub next: HashMap<i64, usize>,
    pub fail: usize,
    /// The nearest keyword end along the fail arcs.
    pub output: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
}

/// A keyword to spot: its tokens, the boost of each token's arc, the mean
/// token probability to reach, and which keyword it is (several token
/// sequences may spell one).
#[derive(Debug, Clone, PartialEq)]
pub struct KeywordTokens {
    pub tokens: Vec<i64>,
    pub boost: f32,
    pub threshold: f32,
    pub phrase: usize,
}

impl Graph {
    pub fn new(keywords: &[KeywordTokens]) -> Self {
        let mut graph = Graph {
            nodes: vec![Node {
                token: -1,
                token_score: 0.0,
                node_score: 0.0,
                output_score: 0.0,
                level: 0,
                threshold: 0.0,
                is_end: false,
                phrase: None,
                next: HashMap::new(),
                fail: ROOT,
                output: None,
            }],
        };
        for keyword in keywords {
            let mut node = ROOT;
            let last = keyword.tokens.len().saturating_sub(1);
            for (j, &token) in keyword.tokens.iter().enumerate() {
                let is_last = j == last;
                let parent_score = graph.nodes[node].node_score;
                let child = match graph.nodes[node].next.get(&token) {
                    None => {
                        let index = graph.nodes.len();
                        graph.nodes.push(Node {
                            token,
                            token_score: keyword.boost,
                            node_score: parent_score + keyword.boost,
                            output_score: if is_last {
                                parent_score + keyword.boost
                            } else {
                                0.0
                            },
                            level: j + 1,
                            threshold: if is_last { keyword.threshold } else { 0.0 },
                            is_end: is_last,
                            phrase: is_last.then_some(keyword.phrase),
                            next: HashMap::new(),
                            fail: ROOT,
                            output: None,
                        });
                        graph.nodes[node].next.insert(token, index);
                        index
                    }
                    Some(&index) => {
                        let child = &mut graph.nodes[index];
                        child.token_score = child.token_score.max(keyword.boost);
                        child.node_score = parent_score + child.token_score;
                        child.is_end = is_last || child.is_end;
                        child.output_score = if child.is_end { child.node_score } else { 0.0 };
                        if is_last {
                            child.phrase = Some(keyword.phrase);
                            child.threshold = keyword.threshold;
                        }
                        index
                    }
                };
                node = child;
            }
        }
        graph.fill_fail_output();
        graph
    }

    fn fill_fail_output(&mut self) {
        let mut queue: VecDeque<usize> = VecDeque::new();
        let firsts: Vec<usize> = self.nodes[ROOT].next.values().copied().collect();
        for child in firsts {
            self.nodes[child].fail = ROOT;
            queue.push_back(child);
        }
        while let Some(current) = queue.pop_front() {
            let children: Vec<(i64, usize)> = self.nodes[current]
                .next
                .iter()
                .map(|(&t, &c)| (t, c))
                .collect();
            for (token, child) in children {
                let mut fail = self.nodes[current].fail;
                if let Some(&next) = self.nodes[fail].next.get(&token) {
                    fail = next;
                } else {
                    fail = self.nodes[fail].fail;
                    while !self.nodes[fail].next.contains_key(&token) {
                        fail = self.nodes[fail].fail;
                        if self.nodes[fail].token == -1 {
                            break;
                        }
                    }
                    if let Some(&next) = self.nodes[fail].next.get(&token) {
                        fail = next;
                    }
                }
                self.nodes[child].fail = fail;
                // The nearest keyword end along the fail arcs.
                let mut output = Some(fail);
                while let Some(node) = output {
                    if self.nodes[node].is_end {
                        break;
                    }
                    let next = self.nodes[node].fail;
                    output = (self.nodes[next].token != -1).then_some(next);
                }
                self.nodes[child].output = output;
                if let Some(output) = output {
                    self.nodes[child].output_score += self.nodes[output].output_score;
                }
                queue.push_back(child);
            }
        }
    }

    /// One token from `state`: the score it adds and the state it leads to
    /// (`ForwardOneStep`, strict mode).
    pub fn forward(&self, state: usize, token: i64) -> (f32, usize) {
        let (node, score) = if let Some(&next) = self.nodes[state].next.get(&token) {
            (next, self.nodes[next].token_score)
        } else {
            let mut node = self.nodes[state].fail;
            while !self.nodes[node].next.contains_key(&token) {
                node = self.nodes[node].fail;
                if self.nodes[node].token == -1 {
                    break;
                }
            }
            if let Some(&next) = self.nodes[node].next.get(&token) {
                node = next;
            }
            (
                node,
                self.nodes[node].node_score - self.nodes[state].node_score,
            )
        };
        (score + self.nodes[node].output_score, node)
    }

    /// The keyword end `state` stands at (or reaches through its output arc).
    pub fn matched(&self, state: usize) -> Option<usize> {
        let node = &self.nodes[state];
        if node.is_end {
            Some(state)
        } else {
            node.output
        }
    }
}
