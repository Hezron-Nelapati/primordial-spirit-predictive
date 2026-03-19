use crate::graph::{WordGraph, WordNode, WordEdge};

/// Ingest every non-empty line of `text` as an independent sentence.
pub fn ingest_text(graph: &mut WordGraph, text: &str, base_weight: f32) {
    for line in text.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            ingest_sentence(graph, trimmed, base_weight);
        }
    }
}

/// Tokenise `sentence` by whitespace and insert a directed edge between each
/// consecutive pair into the graph.  If the edge already exists its weight is
/// reinforced by `base_weight`; otherwise a new edge is pushed with that weight.
///
/// Metadata defaults (`statement / neutral / general`) are suitable for raw
/// plain-text corpora.  The Python ingest pipeline assigns richer metadata when
/// processing tagged JSON rows.
pub fn ingest_sentence(graph: &mut WordGraph, sentence: &str, base_weight: f32) {
    let tokens: Vec<&str> = sentence.split_whitespace().filter(|t| !t.is_empty()).collect();
    let mut prev_id: Option<u64> = None;

    for token in tokens {
        let id = WordGraph::generate_id(token);
        graph.by_surface.entry(token.to_string()).or_insert(id);

        let node = graph.nodes.entry(id).or_insert(WordNode {
            id,
            surface: token.to_string(),
            frequency: 0,
            position: [0.0; 3],
            lexical_vector: WordNode::compute_lexical_vector(token),
        });
        node.frequency += 1;

        if let Some(prev) = prev_id {
            // Reinforce an existing directed edge, or push a new one.
            if let Some(edge) = graph.edges.iter_mut().find(|e| e.from == prev && e.to == id) {
                edge.weight += base_weight;
            } else {
                graph.edges.push(WordEdge {
                    from: prev,
                    to: id,
                    weight: base_weight,
                    intent: "statement".to_string(),
                    tone: "neutral".to_string(),
                    domain: "general".to_string(),
                    entity: None,
                    dated: None,
                });
            }
        }
        prev_id = Some(id);
    }
}

// ---------------------------------------------------------------------------
// Graph quality metrics (replaces V1 TierStats)
// ---------------------------------------------------------------------------

pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    /// Mean number of outgoing edges per node.
    pub avg_out_degree: f32,
}

impl GraphStats {
    pub fn compute(graph: &WordGraph) -> Self {
        let node_count = graph.nodes.len();
        let edge_count = graph.edges.len();
        let avg_out_degree = if node_count > 0 {
            edge_count as f32 / node_count as f32
        } else {
            0.0
        };
        Self { node_count, edge_count, avg_out_degree }
    }

    pub fn report(&self) {
        println!(
            "Graph: {} nodes | {} edges | {:.2} avg out-degree",
            self.node_count, self.edge_count, self.avg_out_degree
        );
    }
}
