use crate::db::GraphDb;
use crate::graph::{GraphAccess, WordGraph, WordNode, WordEdge};
use serde::Deserialize;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Corpus JSON schema
// ---------------------------------------------------------------------------

/// Deserialisation target for a single row in the corpus JSON produced by the
/// Python ingest pipeline (`data/corpus.json` or `data/corpus_reinforced.json`).
#[derive(Deserialize, Debug)]
pub struct CorpusRow {
    pub text:     String,
    pub tokens:   Vec<String>,
    pub intent:   String,
    pub tone:     String,
    pub domain:   String,
    pub entities: Vec<String>,
    pub dated:    Option<u16>,
}

// ---------------------------------------------------------------------------
// Corpus ingestion
// ---------------------------------------------------------------------------

/// Ingest a slice of deserialised JSON rows into an existing graph.
///
/// Two edges are considered contextually identical when they share the same
/// `(from, to, intent, domain, dated)` tuple.  Identical edges have their
/// weight reinforced by 1.0 rather than being duplicated — this ensures that
/// repeated corpus facts accumulate strength on a single edge rather than
/// producing a cluster of low-weight duplicates that dilute scoring.
///
/// Different intent/domain combinations for the same token pair are preserved
/// as distinct edges so that multi-signal routing can select among them.
pub fn ingest_rows(graph: &mut WordGraph, rows: Vec<CorpusRow>) {
    for row in rows {
        let mut prev_id: Option<u64> = None;
        for token in row.tokens {
            let id = WordGraph::generate_id(&token);
            graph.by_surface.insert(token.clone(), id);

            let node = graph.nodes.entry(id).or_insert_with(|| {
                let lv  = WordNode::compute_lexical_vector(&token);
                let len = lv[0];
                let pos = [
                    len,
                    if len > 0.0 { lv[3] / len } else { 0.0 }, // vowel density
                    if len > 0.0 { lv[4] / len } else { 0.0 }, // uniqueness density
                ];
                WordNode {
                    id,
                    surface: token.clone(),
                    frequency: 0,
                    position: pos,
                    lexical_vector: lv,
                }
            });
            node.frequency += 1;

            if let Some(prev) = prev_id {
                let existing = graph.edges.iter_mut().find(|e| {
                    e.from == prev
                        && e.to == id
                        && e.intent == row.intent
                        && e.domain == row.domain
                        && e.dated  == row.dated
                });
                if let Some(edge) = existing {
                    edge.weight += 1.0;
                } else {
                    graph.push_edge(WordEdge {
                        from: prev,
                        to: id,
                        weight: 1.0,
                        intent: row.intent.clone(),
                        tone:   row.tone.clone(),
                        domain: row.domain.clone(),
                        entity: row.entities.first().cloned(),
                        dated:  row.dated,
                    });
                }
            }
            prev_id = Some(id);
        }
    }
}

/// Ingest corpus rows into a SQLite `GraphDb` rather than an in-memory
/// `WordGraph`.  Identical `(from, to, intent, domain, dated)` edges have
/// their weight reinforced by 1.0 via an `ON CONFLICT DO UPDATE` upsert,
/// exactly mirroring the deduplication logic of `ingest_rows`.
///
/// Wraps the entire batch in a single transaction for throughput — avoids
/// one fsync per row which would be orders of magnitude slower on large corpora.
pub fn ingest_to_db(db: &GraphDb, rows: Vec<CorpusRow>) {
    // Fix #9: propagate begin/commit errors rather than silently discarding.
    // If BEGIN fails (e.g. database locked), abort before writing any rows.
    if let Err(e) = db.begin() {
        eprintln!("[INGEST] WARN: Could not begin transaction: {}. Aborting ingest.", e);
        return;
    }
    // Cache lexical vectors keyed by surface string — compute_lexical_vector()
    // is pure and deterministic, so every repeated token (e.g. "the", "is")
    // reuses the cached result instead of rehashing and re-sorting its chars.
    let mut lv_cache: HashMap<String, ([f32; 5], [f32; 3])> = HashMap::new();

    for row in rows {
        let mut prev_id: Option<u64> = None;
        for token in &row.tokens {
            let id = WordGraph::generate_id(token);
            let (lv, pos) = lv_cache.entry(token.clone()).or_insert_with(|| {
                let lv  = WordNode::compute_lexical_vector(token);
                let len = lv[0];
                let pos = [
                    len,
                    if len > 0.0 { lv[3] / len } else { 0.0 },
                    if len > 0.0 { lv[4] / len } else { 0.0 },
                ];
                (lv, pos)
            });
            let node = WordNode { id, surface: token.clone(), frequency: 1,
                                  position: *pos, lexical_vector: *lv };
            let _ = db.upsert_node(&node);

            if let Some(prev) = prev_id {
                let _ = db.upsert_edge(
                    prev, id, 1.0,
                    &row.intent, &row.tone, &row.domain,
                    row.entities.first().map(String::as_str),
                    row.dated,
                );
            }
            prev_id = Some(id);
        }
    }
    if let Err(e) = db.commit() {
        eprintln!("[INGEST] WARN: Could not commit transaction: {}. Changes may be lost.", e);
    }
}

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
                graph.push_edge(WordEdge {
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
    pub fn compute(graph: &dyn GraphAccess) -> Self {
        let node_count = graph.node_count();
        let edge_count = graph.edge_count();
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
