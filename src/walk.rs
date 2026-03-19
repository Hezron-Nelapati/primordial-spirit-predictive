use crate::graph::{GraphAccess, WordEdge};
use crate::reasoning::ReasoningModule;
use crate::spatial::SpatialGrid;
use std::collections::{HashSet, VecDeque};

/// Controls the traversal strategy used by `predict_next`.
///
/// | Mode | Behaviour | Best for |
/// |---|---|---|
/// | `Forward` | Score outgoing edges by intent/domain/tone/entity/temporal signals; pick highest weight | `statement`, `command` |
/// | `Explain` | Among qualifying edges prefer the target with the most onward edges (widest coverage) | `explain` |
/// | `Question` | Prefer edges that lead toward an already-known entity anchor (answer-oriented) | `question` |
#[derive(Debug, Clone, PartialEq)]
pub enum WalkMode {
    Forward,
    Explain,
    Question,
}

impl WalkMode {
    /// Derive the appropriate walk mode from an intent label.
    ///
    /// `"complaint"` maps to `Question` because both intents are answer-seeking:
    /// the walker must bridge toward an entity anchor rather than extend the chain
    /// linearly (implementation_plan.md §4 "Answer-Seeking Intent").
    pub fn from_intent(intent: &str) -> Self {
        match intent {
            "explain" => WalkMode::Explain,
            "question" | "complaint" => WalkMode::Question,
            _ => WalkMode::Forward,
        }
    }
}

pub struct WalkConfig {
    pub target_year: Option<u16>,
    pub depth_limit: usize,
    pub mode: WalkMode,
}

pub fn predict_next(
    current_word: &str,
    graph: &dyn GraphAccess,
    spatial: Option<&SpatialGrid>,
    reasoning: &ReasoningModule<'_>,
    config: &WalkConfig,
    pos_history: &[[f32; 3]],
) -> Option<String> {
    let current_id = match graph.surface_to_id(current_word) {
        Some(id) => id,
        None => {
            // Guardrail 1: OOV Lexical Fallback
            println!("🚨 OOV Panic: '{}' not inside spatial memory.", current_word);
            println!("⚡ Triggering 5-Dimensional Lexical Vector Fallback...");
            let target_vec = crate::graph::WordNode::compute_lexical_vector(current_word);
            let mut best_dist = f32::MAX;
            let mut best_id = 0u64;
            let mut nearest_str = String::new();

            for node in graph.all_nodes() {
                let dist = euclidean_dist_5(&node.lexical_vector, &target_vec);
                if dist < best_dist {
                    best_dist = dist;
                    best_id = node.id;
                    nearest_str = node.surface.clone();
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
    let edges = graph.edges_from(current_id);

    if !edges.is_empty() {
        println!("    [TRACE] Validating {} outgoing pathways geometrically...", edges.len());
        return match config.mode {
            WalkMode::Explain  => score_edges_explain(&edges, graph),
            WalkMode::Question => score_edges_question(&edges, graph, active_entity),
            WalkMode::Forward  => score_edges(&edges, graph, active_intent, active_domain, active_tone, active_entity, config),
        };
    }

    // Tier 2: KD-tree radial search when no direct edges exist.
    // Search origin: geometric centroid of last N traversed positions when history
    // is available (addresses Context Amnesia §2 of architecture_ideas.md); falls
    // back to current node position for an empty history.
    if let Some(grid) = spatial {
        let current_pos = graph.node_by_id(current_id).map(|n| n.position).unwrap_or([0.0; 3]);
        let search_pos = if pos_history.is_empty() {
            current_pos
        } else {
            let all: Vec<[f32; 3]> = pos_history.iter().copied().chain(std::iter::once(current_pos)).collect();
            let n = all.len() as f32;
            [
                all.iter().map(|p| p[0]).sum::<f32>() / n,
                all.iter().map(|p| p[1]).sum::<f32>() / n,
                all.iter().map(|p| p[2]).sum::<f32>() / n,
            ]
        };

        let neighbours = grid.query_radius(search_pos, 3.0);
        let mut tier2_edges: Vec<WordEdge> = Vec::new();
        for &nb_id in &neighbours {
            if nb_id != current_id {
                tier2_edges.extend(graph.edges_from(nb_id));
            }
        }

        if !tier2_edges.is_empty() {
            println!("    [TIER_2] No direct edges — centroid radial search found {} candidate edges from {} neighbours.",
                tier2_edges.len(), neighbours.len());
            return score_edges(&tier2_edges, graph, active_intent, active_domain, active_tone, active_entity, config);
        }
    }

    // Tier 3: A* spatial bridge (Question/Complaint) or backtrack-and-reroute.
    //
    // For answer-seeking intents (Question mode), the architecture specifies a
    // spatial A* jump toward the entity anchor rather than a linear backtrack
    // (implementation_plan.md §4, walkthrough.md §2.4).  We compute the midpoint
    // between the current node and the entity anchor in 3D space and use the
    // KD-tree to find the nearest bridge node at that location.  If the bridge
    // node is valid and not the current node, we return it directly.
    //
    // For all other modes (or when spatial / entity are unavailable), the classic
    // backtrack-reroute runs as before: find ancestors and pick from their
    // alternative outgoing edges.
    if config.mode == WalkMode::Question {
        if let Some(grid) = spatial {
            if let Some(bridge) = spatial_a_star_bridge(current_id, active_entity, graph, grid) {
                println!("    [TIER_3] Question A* bridge → spatial midpoint jump to [{}].", bridge);
                return Some(bridge);
            }
        }
    }

    // Classic backtrack-reroute (all modes, and Question fallback when bridge fails).
    let ancestor_edges = graph.edges_to(current_id);
    let ancestor_ids: HashSet<u64> = ancestor_edges.iter().map(|e| e.from).collect();

    let mut tier3_edges: Vec<WordEdge> = Vec::new();
    for anc_id in &ancestor_ids {
        for e in graph.edges_from(*anc_id) {
            if e.to != current_id {
                tier3_edges.push(e);
            }
        }
    }

    if !tier3_edges.is_empty() {
        println!("    [TIER_3] No Tier-2 candidates — backtrack-reroute found {} alternate path(s) from {} ancestor(s).",
            tier3_edges.len(), ancestor_ids.len());
        return score_edges(&tier3_edges, graph, active_intent, active_domain, active_tone, active_entity, config);
    }

    None
}

/// Explain mode: prefer the target node with the most onward outgoing edges.
/// Wider topological coverage produces richer explanatory sequences.
/// Falls back to the first edge if all targets are leaves.
fn score_edges_explain(edges: &[WordEdge], graph: &dyn GraphAccess) -> Option<String> {
    let best = edges.iter().max_by_key(|e| graph.edges_from(e.to).len());
    best.and_then(|e| graph.node_by_id(e.to).map(|n| {
        let out_degree = graph.edges_from(e.to).len();
        println!("      [EXPLAIN_MODE] Selected node [{}] with {} onward edges (widest coverage).",
            n.surface, out_degree);
        n.surface
    }))
}

/// Question mode: prefer the edge whose target node is closest (in outgoing hops)
/// to the active entity anchor — routes toward a known answer rather than
/// extending the current chain linearly.
/// Falls back to the first edge when no entity is set or no path closes.
fn score_edges_question(
    edges: &[WordEdge],
    graph: &dyn GraphAccess,
    active_entity: &str,
) -> Option<String> {
    let target_id = match graph.surface_to_id(active_entity) {
        Some(id) => id,
        None => {
            // No known entity anchor — fall back to first candidate.
            return edges.first().and_then(|e| graph.node_by_id(e.to).map(|n| n.surface));
        }
    };

    // Pick the edge whose target has the shortest forward-hop distance to the entity.
    // Distance is approximated by a bounded BFS (max 5 hops); unreachable = usize::MAX.
    let best = edges.iter().min_by_key(|e| bfs_distance(e.to, target_id, graph, 5));

    best.and_then(|e| graph.node_by_id(e.to).map(|n| {
        println!("      [QUESTION_MODE] Selected node [{}] as closest to entity anchor [{}].",
            n.surface, active_entity);
        n.surface
    }))
}

/// BFS hop distance from `from_id` to `to_id`, bounded at `max_hops`.
/// Returns `usize::MAX` if unreachable within the bound.
fn bfs_distance(from_id: u64, to_id: u64, graph: &dyn GraphAccess, max_hops: usize) -> usize {
    if from_id == to_id { return 0; }
    let mut visited: HashSet<u64> = HashSet::new();
    let mut queue: VecDeque<(u64, usize)> = VecDeque::new();
    queue.push_back((from_id, 0));
    visited.insert(from_id);
    while let Some((node, hops)) = queue.pop_front() {
        if hops >= max_hops { continue; }
        for edge in graph.edges_from(node) {
            if edge.to == to_id { return hops + 1; }
            if visited.insert(edge.to) {
                queue.push_back((edge.to, hops + 1));
            }
        }
    }
    usize::MAX
}

/// Score a candidate edge slice and return the surface of the highest-weighted target node.
fn score_edges(
    edges: &[WordEdge],
    graph: &dyn GraphAccess,
    active_intent: &str,
    active_domain:  &str,
    active_tone:    &str,
    active_entity:  &str,
    config: &WalkConfig,
) -> Option<String> {
    let mut best_edge   = None;
    let mut highest_weight = 0.0_f32;

    for edge in edges {
        let mut adj_weight = edge.weight;
        let target_surface = graph.node_by_id(edge.to).map(|n| n.surface).unwrap_or_default();
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
            best_edge = Some(edge.clone());
        }
    }

    best_edge.and_then(|e| graph.node_by_id(e.to).map(|n| n.surface))
}

/// Spatial A* bridge for answer-seeking (Question/Complaint) Tier 3.
///
/// Computes the midpoint in 3D space between the current node and the active
/// entity anchor, then queries the KD-tree for the node nearest to that
/// midpoint.  This spatially jumps the walk toward the entity cluster without
/// requiring a graph-edge path between them — bridging conceptual islands the
/// way `implementation_plan.md §4` specifies for question-intent Tier 3.
///
/// Returns `None` when:
/// - The entity anchor is unknown or not in the graph.
/// - The nearest node is the current node itself (no progress).
/// - The bridge node has no outgoing edges (would immediately dead-end again).
fn spatial_a_star_bridge(
    current_id: u64,
    entity_word: &str,
    graph: &dyn GraphAccess,
    grid: &SpatialGrid,
) -> Option<String> {
    let entity_id = graph.surface_to_id(entity_word)?;
    let current_pos = graph.node_by_id(current_id)?.position;
    let entity_pos  = graph.node_by_id(entity_id)?.position;

    // Midpoint in 3D between current and entity anchor.
    let midpoint = [
        (current_pos[0] + entity_pos[0]) / 2.0,
        (current_pos[1] + entity_pos[1]) / 2.0,
        (current_pos[2] + entity_pos[2]) / 2.0,
    ];

    // Large search radius (50.0) guarantees a hit: positions span x∈[1,20+],
    // y/z∈[0,1], so max inter-node distance is < 25.
    let bridge_id = grid.query_nearest(midpoint, 50.0)?;

    if bridge_id == current_id { return None; }

    // Only bridge to nodes that have outgoing edges — avoids instant re-dead-end.
    if !graph.has_edges_from(bridge_id) { return None; }

    graph.node_by_id(bridge_id).map(|n| n.surface)
}

fn euclidean_dist_5(a: &[f32; 5], b: &[f32; 5]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

/// MVP Guardrail: Dynamic Entry Node Resolution (Bidirectional Walking)
/// Reverse-walks the topology from the entity node to find the sentence anchor.
pub fn resolve_start_node(
    entity_word: &str,
    graph: &dyn GraphAccess,
    reasoning: &ReasoningModule<'_>,
    config: &WalkConfig,
) -> Option<String> {
    let mut current_id = match graph.surface_to_id(entity_word) {
        Some(id) => id,
        None => return None,
    };

    println!("  [SYS_ORCHESTRATOR]: Jumping onto Target Entity Node: [{}]", entity_word);

    for _ in 0..20 {
        let incoming = graph.edges_to(current_id);
        if incoming.is_empty() { break; }

        let mut best_prev = None;
        let mut highest_weight = 0.0_f32;
        let active_domain = reasoning.session.domain_stack.last().map(String::as_str).unwrap_or("general");

        for edge in &incoming {
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
                best_prev = Some(edge.clone());
            }
        }

        if let Some(edge) = best_prev {
            let prev_surface = graph.node_by_id(edge.from).map(|n| n.surface).unwrap_or_default();
            if prev_surface == "." || prev_surface == "?" || prev_surface == "!" {
                break;
            }
            current_id = edge.from;
        } else {
            break;
        }
    }

    graph.node_by_id(current_id).map(|n| n.surface)
}

/// Scan `query` for a secondary entity signal: the first whitespace token that
/// exists in the graph and is not the `primary_entity`.  Punctuation is
/// stripped from both ends of each token before lookup.  Both the raw-case
/// and lowercase forms are tried.  Returns the graph surface form on success.
pub fn secondary_signal(query: &str, primary_entity: &str, graph: &dyn GraphAccess) -> Option<String> {
    for raw in query.split_whitespace() {
        let token = raw.trim_matches(|c: char| c.is_ascii_punctuation());
        if token.is_empty() || token.eq_ignore_ascii_case(primary_entity) {
            continue;
        }
        let id = graph.surface_to_id(token)
            .or_else(|| graph.surface_to_id(&token.to_lowercase()));
        if let Some(node_id) = id {
            return graph.node_by_id(node_id).map(|n| n.surface);
        }
    }
    None
}

/// BFS reachability: returns `true` if `to_word` is reachable from `from_word`
/// by following forward edges within `max_hops` steps.
/// Returns `false` if either word is absent from the graph.
pub fn is_reachable(from_word: &str, to_word: &str, graph: &dyn GraphAccess, max_hops: usize) -> bool {
    let start_id = match graph.surface_to_id(from_word) {
        Some(id) => id,
        None => return false,
    };
    let target_id = match graph.surface_to_id(to_word) {
        Some(id) => id,
        None => return false,
    };

    let mut visited: HashSet<u64> = HashSet::new();
    let mut queue: VecDeque<(u64, usize)> = VecDeque::new();
    queue.push_back((start_id, 0));
    visited.insert(start_id);

    while let Some((current, hops)) = queue.pop_front() {
        if current == target_id { return true; }
        if hops >= max_hops { continue; }
        for edge in graph.edges_from(current) {
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
pub fn compute_depth_limit(entity_word: &str, graph: &dyn GraphAccess) -> usize {
    let start_id = match graph.surface_to_id(entity_word) {
        Some(id) => id,
        None => return 1,
    };

    let mut reachable: HashSet<u64> = HashSet::new();
    let mut frontier: Vec<u64> = vec![start_id];

    for _ in 0..2 {
        let mut next: Vec<u64> = Vec::new();
        for node_id in frontier {
            for edge in graph.edges_from(node_id) {
                if reachable.insert(edge.to) {
                    next.push(edge.to);
                }
            }
        }
        frontier = next;
    }

    match reachable.len() {
        0..=4   => 1,
        5..=14  => 2,
        15..=29 => 3,
        _       => 4,
    }
}
