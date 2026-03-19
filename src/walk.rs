use crate::graph::{WordGraph, WordNode, WordEdge};
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
    pub fn from_intent(intent: &str) -> Self {
        match intent {
            "explain" => WalkMode::Explain,
            "question" => WalkMode::Question,
            _ => WalkMode::Forward,
        }
    }
}

pub struct WalkConfig {
    pub target_year: Option<u16>,
    pub depth_limit: usize,
    pub mode: WalkMode,
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
        return match config.mode {
            WalkMode::Explain   => score_edges_explain(&edges, graph),
            WalkMode::Question  => score_edges_question(&edges, graph, active_entity),
            WalkMode::Forward   => score_edges(&edges, graph, active_intent, active_domain, active_tone, active_entity, config),
        };
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

    // Tier 3: Backtrack-and-reroute.
    // Find ancestor nodes (nodes with edges pointing to current_id), then collect
    // their *other* outgoing edges (not back to current_id) and score them.
    // This escapes dead-ends that Tier 1 and Tier 2 could not resolve.
    let ancestor_ids: HashSet<u64> = graph.edges.iter()
        .filter(|e| e.to == current_id)
        .map(|e| e.from)
        .collect();

    let tier3_edges: Vec<&WordEdge> = graph.edges.iter()
        .filter(|e| ancestor_ids.contains(&e.from) && e.to != current_id)
        .collect();

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
fn score_edges_explain<'a>(edges: &[&WordEdge], graph: &'a WordGraph) -> Option<&'a str> {
    let best = edges.iter().max_by_key(|e| {
        graph.edges.iter().filter(|out| out.from == e.to).count()
    });
    best.and_then(|e| graph.nodes.get(&e.to).map(|n| {
        let out_degree = graph.edges.iter().filter(|out| out.from == e.to).count();
        println!("      [EXPLAIN_MODE] Selected node [{}] with {} onward edges (widest coverage).",
            n.surface.as_str(), out_degree);
        n.surface.as_str()
    }))
}

/// Question mode: prefer the edge whose target node is closest (in outgoing hops)
/// to the active entity anchor — routes toward a known answer rather than
/// extending the current chain linearly.
/// Falls back to the first edge when no entity is set or no path closes.
fn score_edges_question<'a>(
    edges: &[&WordEdge],
    graph: &'a WordGraph,
    active_entity: &str,
) -> Option<&'a str> {
    let target_id = match graph.by_surface.get(active_entity) {
        Some(id) => *id,
        None => {
            // No known entity anchor — fall back to first candidate.
            return edges.first().and_then(|e| graph.nodes.get(&e.to).map(|n| n.surface.as_str()));
        }
    };

    // Pick the edge whose target has the shortest forward-hop distance to the entity.
    // Distance is approximated by a bounded BFS (max 5 hops); unreachable = usize::MAX.
    let best = edges.iter().min_by_key(|e| {
        bfs_distance(e.to, target_id, graph, 5)
    });

    best.and_then(|e| graph.nodes.get(&e.to).map(|n| {
        println!("      [QUESTION_MODE] Selected node [{}] as closest to entity anchor [{}].",
            n.surface.as_str(), active_entity);
        n.surface.as_str()
    }))
}

/// BFS hop distance from `from_id` to `to_id`, bounded at `max_hops`.
/// Returns `usize::MAX` if unreachable within the bound.
fn bfs_distance(from_id: u64, to_id: u64, graph: &WordGraph, max_hops: usize) -> usize {
    if from_id == to_id { return 0; }
    let mut visited: HashSet<u64> = HashSet::new();
    let mut queue: VecDeque<(u64, usize)> = VecDeque::new();
    queue.push_back((from_id, 0));
    visited.insert(from_id);
    while let Some((node, hops)) = queue.pop_front() {
        if hops >= max_hops { continue; }
        for edge in graph.edges.iter().filter(|e| e.from == node) {
            if edge.to == to_id { return hops + 1; }
            if visited.insert(edge.to) {
                queue.push_back((edge.to, hops + 1));
            }
        }
    }
    usize::MAX
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

/// Guardrail 6: Arithmetic / logic interception.
///
/// Returns `true` when the query contains **both**:
///   1. At least one standalone numeric token (parses as a finite `f64`), and
///   2. At least one arithmetic signal — an explicit operator token (`+`, `*`,
///      `/`, `=`, `%`, `^`) or a keyword (`plus`, `minus`, `times`, `divided`,
///      `equals`, `sum`, `product`, `percent`, `sqrt`, `squared`, `cubed`).
///
/// The two-condition gate avoids false positives on queries like
/// "The bank closes at 5 pm" (number present but no arithmetic signal) or
/// "What is the sum of all loans?" (arithmetic word present but no number).
pub fn is_arithmetic_query(query: &str) -> bool {
    const ARITH_WORDS: &[&str] = &[
        "plus", "minus", "times", "divided", "equals", "sum",
        "product", "percent", "sqrt", "squared", "cubed",
    ];
    const ARITH_OPS: &[&str] = &["+", "*", "/", "=", "%", "^"];

    let mut has_number   = false;
    let mut has_arith    = false;

    for raw in query.split_whitespace() {
        let token = raw.trim_matches(|c: char| c.is_ascii_punctuation());
        if token.is_empty() { continue; }

        if !has_number && token.parse::<f64>().map(|v| v.is_finite()).unwrap_or(false) {
            has_number = true;
        }
        if !has_arith {
            let lower = token.to_lowercase();
            if ARITH_WORDS.contains(&lower.as_str()) || ARITH_OPS.contains(&token) {
                has_arith = true;
            }
        }
        if has_number && has_arith { return true; }
    }
    false
}

/// Evaluate a simple arithmetic expression from a natural-language query.
///
/// Supports:
/// - Binary ops: `+`, `-`, `*`, `/`, `%` and their word forms
///   (`plus`, `minus`, `times`, `divided`, `percent`).
/// - Unary ops on a single number: `sqrt`, `squared`, `cubed`.
/// - Multi-number `sum` / `product` (folds the full number list).
///
/// Returns a formatted natural-language result string, or `None` if no
/// computable expression is found.
pub fn evaluate_arithmetic(query: &str) -> Option<String> {
    let mut numbers: Vec<f64> = Vec::new();
    let mut binary_op: Option<char> = None;
    let mut unary_op: Option<&str> = None;
    let mut is_sum     = false;
    let mut is_product = false;

    for raw in query.split_whitespace() {
        // Strip surrounding punctuation but keep minus sign on negative numbers.
        let token = raw.trim_matches(|c: char| c.is_ascii_punctuation() && c != '-' && c != '.');
        if token.is_empty() { continue; }

        if let Ok(n) = token.parse::<f64>() {
            if n.is_finite() { numbers.push(n); }
            continue;
        }

        let lower = token.to_lowercase();
        match lower.as_str() {
            "plus"  | "add"      => { if binary_op.is_none() { binary_op = Some('+'); } }
            "minus" | "subtract" => { if binary_op.is_none() { binary_op = Some('-'); } }
            "times" | "multiplied" => { if binary_op.is_none() { binary_op = Some('*'); } }
            "divided" | "over"   => { if binary_op.is_none() { binary_op = Some('/'); } }
            "percent" | "mod"    => { if binary_op.is_none() { binary_op = Some('%'); } }
            "+"  => { if binary_op.is_none() { binary_op = Some('+'); } }
            "-"  => { if binary_op.is_none() { binary_op = Some('-'); } }
            "*"  => { if binary_op.is_none() { binary_op = Some('*'); } }
            "/"  => { if binary_op.is_none() { binary_op = Some('/'); } }
            "%"  => { if binary_op.is_none() { binary_op = Some('%'); } }
            "sqrt"    => { unary_op = Some("sqrt"); }
            "squared" => { unary_op = Some("squared"); }
            "cubed"   => { unary_op = Some("cubed"); }
            "sum"     => { is_sum     = true; }
            "product" => { is_product = true; }
            _ => {}
        }
    }

    if numbers.is_empty() { return None; }

    fn fmt(v: f64) -> String {
        if v.fract() == 0.0 && v.abs() < 1e15 {
            format!("The answer is {}.", v as i64)
        } else {
            format!("The answer is {:.4}.", v)
        }
    }

    // Unary operations — operate on the first number found.
    if let Some(uop) = unary_op {
        let n = numbers[0];
        let result = match uop {
            "sqrt" => {
                if n < 0.0 { return Some("The answer is undefined (square root of negative number).".to_string()); }
                n.sqrt()
            }
            "squared" => n * n,
            "cubed"   => n * n * n,
            _ => return None,
        };
        return Some(fmt(result));
    }

    // Multi-number fold for sum/product.
    if is_sum && numbers.len() >= 2 {
        return Some(fmt(numbers.iter().sum()));
    }
    if is_product && numbers.len() >= 2 {
        return Some(fmt(numbers.iter().product()));
    }

    // Binary operation on first two numbers.
    if numbers.len() >= 2 {
        let a = numbers[0];
        let b = numbers[1];
        let op = binary_op.unwrap_or('+');
        let result = match op {
            '+' => a + b,
            '-' => a - b,
            '*' => a * b,
            '/' | '%' => {
                if b == 0.0 { return Some("The answer is undefined (division by zero).".to_string()); }
                if op == '/' { a / b } else { a % b }
            }
            _ => return None,
        };
        return Some(fmt(result));
    }

    // Single number with no recognized operation — not enough to compute.
    None
}

/// Scan `query` for a secondary entity signal: the first whitespace token that
/// exists in the graph and is not the `primary_entity`.  Punctuation is
/// stripped from both ends of each token before lookup.  Both the raw-case
/// and lowercase forms are tried.  Returns the graph surface form on success.
pub fn secondary_signal<'a>(query: &str, primary_entity: &str, graph: &'a WordGraph) -> Option<&'a str> {
    for raw in query.split_whitespace() {
        let token = raw.trim_matches(|c: char| c.is_ascii_punctuation());
        if token.is_empty() || token.eq_ignore_ascii_case(primary_entity) {
            continue;
        }
        let id = graph.by_surface.get(token)
            .or_else(|| graph.by_surface.get(&token.to_lowercase()));
        if let Some(&node_id) = id {
            return graph.nodes.get(&node_id).map(|n| n.surface.as_str());
        }
    }
    None
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
