use crate::graph::{WordGraph, WordNode, WordEdge};
use crate::reasoning::ReasoningModule;
use crate::spatial::SpatialGrid;
use std::collections::{HashSet, VecDeque};

pub struct WalkConfig {
    pub target_year: Option<u16>,
    pub depth_limit: usize,
}

pub fn predict_next<'a>(
    current_word: &str,
    graph: &'a WordGraph,
    spatial: Option<&SpatialGrid>,
    reasoning: &ReasoningModule,
    config: &WalkConfig,
) -> Option<&'a str> {
    let current_id = match graph.by_surface.get(current_word) {
        Some(id) => *id,
        None => {
            // Guardrail 1: OOV Lexical Fallback
            println!("🚨 OOV Panic: '{}' not inside spatial memory.", current_word);
            println!("⚡ Triggering 5-Dimensional Lexical Vector Fallback...");
            let target_vec = WordNode::compute_lexical_vector(current_word);
            let mut best_dist = f32::MAX;
            let mut best_id = 0;
            let mut nearest_str = "";

            for (id, node) in &graph.nodes {
                let dist = euclidean_dist_5(&node.lexical_vector, &target_vec);
                if dist < best_dist {
                    best_dist = dist;
                    best_id = *id;
                    nearest_str = &node.surface;
                }
            }
            println!("✅ Successfully snapped OOV onto geometric structural neighbor: [{}]", nearest_str);
            best_id
        }
    };

    let active_intent = reasoning.session.intent_stack.last().map(String::as_str).unwrap_or("statement");
    let active_domain = reasoning.session.domain_stack.last().map(String::as_str).unwrap_or("general");
    let active_tone   = reasoning.session.tone_stack.last().map(String::as_str).unwrap_or("");
    let active_entity = reasoning.session.entity_stack.last().map(String::as_str).unwrap_or("");

    // Tier 1: Multi-Tag Match Biasing on direct outgoing edges.
    let edges: Vec<&WordEdge> = graph.edges.iter().filter(|e| e.from == current_id).collect();

    if !edges.is_empty() {
        println!("    [TRACE] Validating {} outgoing pathways geometrically...", edges.len());
        return score_edges(&edges, graph, active_intent, active_domain, active_tone, active_entity, config);
    }

    // Tier 2: KD-tree radial search when no direct edges exist.
    if let Some(grid) = spatial {
        let pos = graph.nodes.get(&current_id).map(|n| n.position).unwrap_or([0.0; 3]);
        let neighbours = grid.query_radius(pos, 3.0);
        // Collect outgoing edges from all nearby nodes (excluding the current node itself).
        let tier2_edges: Vec<&WordEdge> = graph.edges.iter()
            .filter(|e| e.from != current_id && neighbours.contains(&e.from))
            .collect();

        if !tier2_edges.is_empty() {
            println!("    [TIER_2] No direct edges — radial KD-tree search found {} candidate edges from {} neighbours.",
                tier2_edges.len(), neighbours.len());
            return score_edges(&tier2_edges, graph, active_intent, active_domain, active_tone, active_entity, config);
        }
    }

    None
}

/// Score a candidate edge slice and return the surface of the highest-weighted target node.
fn score_edges<'a>(
    edges: &[&WordEdge],
    graph: &'a WordGraph,
    active_intent: &str,
    active_domain:  &str,
    active_tone:    &str,
    active_entity:  &str,
    config: &WalkConfig,
) -> Option<&'a str> {
    let mut best_edge   = None;
    let mut highest_weight = 0.0_f32;

    for &edge in edges {
        let mut adj_weight = edge.weight;
        let target_surface = graph.nodes.get(&edge.to).map(|n| n.surface.as_str()).unwrap_or("?");
        println!("      └─ Scanning Edge toward: [{}]", target_surface);

        if edge.intent == active_intent {
            adj_weight *= 2.0;
            println!("         ↳ Intent Match '{}': Weight * 2.0 (New: {:.2})", active_intent, adj_weight);
        }
        if edge.domain == active_domain {
            adj_weight *= 2.0;
            println!("         ↳ Domain Match '{}': Weight * 2.0 (New: {:.2})", active_domain, adj_weight);
        }
        if !active_tone.is_empty() && edge.tone == active_tone {
            adj_weight *= 2.0;
            println!("         ↳ Tone Match '{}': Weight * 2.0 (New: {:.2})", active_tone, adj_weight);
        }
        if !active_entity.is_empty() && edge.entity.as_deref() == Some(active_entity) {
            adj_weight *= 1.5;
            println!("         ↳ Entity Match '{}': Weight * 1.5 (New: {:.2})", active_entity, adj_weight);
        }

        // Guardrail 5: Axiomatic Hallucination (Temporal Tie-Breaking)
        if let (Some(edge_year), Some(target_year)) = (edge.dated, config.target_year) {
            let diff = target_year.abs_diff(edge_year) as f32;
            let mut tm = 2.0 - (diff / 10.0);
            if tm < 1.0 { tm = 1.0; }
            adj_weight *= tm;
            println!("         ↳ Temporal Proximity [Edge:{}, Target:{}]: Weight * {:.2} (New: {:.2})",
                edge_year, target_year, tm, adj_weight);
        }

        if adj_weight > highest_weight {
            highest_weight = adj_weight;
            best_edge = Some(edge);
        }
    }

    best_edge.and_then(|e| graph.nodes.get(&e.to).map(|n| n.surface.as_str()))
}

fn euclidean_dist_5(a: &[f32; 5], b: &[f32; 5]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

/// MVP Guardrail: Dynamic Entry Node Resolution (Bidirectional Walking)
/// Reverse-walks the topology from the entity node to find the sentence anchor.
pub fn resolve_start_node<'a>(
    entity_word: &str,
    graph: &'a WordGraph,
    reasoning: &ReasoningModule,
    config: &WalkConfig,
) -> Option<&'a str> {
    let mut current_id = match graph.by_surface.get(entity_word) {
        Some(id) => *id,
        None => return None,
    };

    println!("  [SYS_ORCHESTRATOR]: Jumping onto Target Entity Node: [{}]", entity_word);

    for _ in 0..20 {
        let incoming: Vec<&WordEdge> = graph.edges.iter().filter(|e| e.to == current_id).collect();
        if incoming.is_empty() { break; }

        let mut best_prev = None;
        let mut highest_weight = 0.0_f32;
        let active_domain = reasoning.session.domain_stack.last().map(String::as_str).unwrap_or("general");

        for edge in incoming {
            let mut adj_weight = edge.weight;
            if edge.domain == active_domain { adj_weight *= 2.0; }
            if let (Some(edge_year), Some(target_year)) = (edge.dated, config.target_year) {
                let diff = target_year.abs_diff(edge_year) as f32;
                let mut tm = 2.0 - (diff / 10.0);
                if tm < 1.0 { tm = 1.0; }
                adj_weight *= tm;
            }
            if adj_weight > highest_weight {
                highest_weight = adj_weight;
                best_prev = Some(edge);
            }
        }

        if let Some(edge) = best_prev {
            let prev_surface = graph.nodes.get(&edge.from).map(|n| n.surface.as_str()).unwrap_or("?");
            if prev_surface == "." || prev_surface == "?" || prev_surface == "!" {
                break;
            }
            current_id = edge.from;
        } else {
            break;
        }
    }

    graph.nodes.get(&current_id).map(|n| n.surface.as_str())
}

/// BFS reachability: returns `true` if `to_word` is reachable from `from_word`
/// by following forward edges within `max_hops` steps.
/// Returns `false` if either word is absent from the graph.
pub fn is_reachable(from_word: &str, to_word: &str, graph: &WordGraph, max_hops: usize) -> bool {
    let start_id = match graph.by_surface.get(from_word) {
        Some(id) => *id,
        None => return false,
    };
    let target_id = match graph.by_surface.get(to_word) {
        Some(id) => *id,
        None => return false,
    };

    let mut visited: HashSet<u64> = HashSet::new();
    let mut queue: VecDeque<(u64, usize)> = VecDeque::new();
    queue.push_back((start_id, 0));
    visited.insert(start_id);

    while let Some((current, hops)) = queue.pop_front() {
        if current == target_id { return true; }
        if hops >= max_hops { continue; }
        for edge in graph.edges.iter().filter(|e| e.from == current) {
            if visited.insert(edge.to) {
                queue.push_back((edge.to, hops + 1));
            }
        }
    }
    false
}

/// Compute a topology-derived sentence depth limit for `entity_word`.
///
/// Counts unique nodes reachable within 2 forward-edge hops from the entity
/// node and maps the count to a bounded depth:
///
/// | Reachable (2 hops) | Depth |
/// |---|---|
/// | 0–4  | 1 — sparse coverage |
/// | 5–14 | 2 — moderate coverage |
/// | 15–29| 3 — good coverage |
/// | 30+  | 4 — dense coverage |
///
/// Returns 1 if the entity is not in the graph.
pub fn compute_depth_limit(entity_word: &str, graph: &WordGraph) -> usize {
    let start_id = match graph.by_surface.get(entity_word) {
        Some(id) => *id,
        None => return 1,
    };

    let mut reachable: HashSet<u64> = HashSet::new();
    let mut frontier: Vec<u64> = vec![start_id];

    for _ in 0..2 {
        let mut next: Vec<u64> = Vec::new();
        for node_id in frontier {
            for edge in graph.edges.iter().filter(|e| e.from == node_id) {
                if reachable.insert(edge.to) {
                    next.push(edge.to);
                }
            }
        }
        frontier = next;
    }

    match reachable.len() {
        0..=4  => 1,
        5..=14 => 2,
        15..=29 => 3,
        _ => 4,
    }
}
