use crate::graph::WordGraph;
use crate::spatial::SpatialGrid;
use crate::walk::{predict_next, Tier};

pub fn ingest_text(graph: &mut WordGraph, text: &str, delta: f32) {
    for line in text.lines() {
        if line.trim().is_empty() { continue; }
        ingest_sentence(graph, line.trim(), delta);
    }
}

pub fn ingest_sentence(graph: &mut WordGraph, sentence: &str, delta: f32) {
    let tokens: Vec<&str> = sentence
        .split_whitespace()
        .filter(|t| !t.is_empty())
        .collect();
    for window in tokens.windows(2) {
        let a = graph.get_or_create(window[0]);
        let b = graph.get_or_create(window[1]);
        graph.reinforce_edge(a, b, delta);
    }
}

pub struct TierStats {
    pub t1: usize,
    pub t2: usize,
    pub t3: usize,
}

impl TierStats {
    pub fn report(&self) {
        let total = (self.t1 + self.t2 + self.t3) as f32;
        if total > 0.0 {
            println!("T1: {:.1}%  T2: {:.1}%  T3: {:.1}%",
                self.t1 as f32 / total * 100.0,
                self.t2 as f32 / total * 100.0,
                self.t3 as f32 / total * 100.0,
            );
        } else {
            println!("No predictions made.");
        }
    }
}

pub fn measure_tier_distribution(
    graph:   &WordGraph,
    spatial: &SpatialGrid,
    _limit:  usize,
) -> TierStats {
    let mut stats = TierStats { t1: 0, t2: 0, t3: 0 };
    for &id in graph.nodes.keys() {
        if let Some(res) = predict_next(graph, spatial, id) {
            match res.tier {
                Tier::One => stats.t1 += 1,
                Tier::Two => stats.t2 += 1,
                Tier::Three => stats.t3 += 1,
            }
        }
    }
    stats
}
