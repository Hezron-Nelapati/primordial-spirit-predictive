use crate::graph::{WordGraph, WordNode, WordEdge};
use crate::reasoning::ReasoningModule;

pub struct WalkConfig {
    pub target_year: Option<u16>,
    pub depth_limit: usize,
}

pub fn predict_next<'a>(
    current_word: &str,
    graph: &'a WordGraph,
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
    
    // Tier 1: Multi-Tag Match Biasing
    let edges: Vec<&WordEdge> = graph.edges.iter().filter(|e| e.from == current_id).collect();
    if edges.is_empty() { return None; }
    
    let mut best_edge = None;
    let mut highest_weight = 0.0;
    
    let active_intent = reasoning.session.intent_stack.last().map(String::as_str).unwrap_or("statement");
    let active_domain = reasoning.session.domain_stack.last().map(String::as_str).unwrap_or("general");
    
    println!("    [TRACE] Validating {} outgoing pathways geometrically...", edges.len());
    for edge in edges {
        let mut adj_weight = edge.weight;
        let target_surface = graph.nodes.get(&edge.to).map(|n| n.surface.as_str()).unwrap_or("?");
        println!("      └─ Scanning Edge toward: [{}]", target_surface);
        
        // Multi-tag matching multipliers
        if edge.intent == active_intent { 
            adj_weight *= 2.0; 
            println!("         ↳ Intent Match '{}': Weight * 2.0 (New: {:.2})", active_intent, adj_weight);
        }
        if edge.domain == active_domain { 
            adj_weight *= 2.0; 
            println!("         ↳ Domain Match '{}': Weight * 2.0 (New: {:.2})", active_domain, adj_weight);
        }
        
        // Guardrail 5: Axiomatic Hallucination (Temporal Tie-Breaking)
        if let (Some(edge_year), Some(target_year)) = (edge.dated, config.target_year) {
            let diff = target_year.abs_diff(edge_year) as f32;
            let mut time_multiplier = 2.0 - (diff / 10.0);
            if time_multiplier < 1.0 { time_multiplier = 1.0; }
            adj_weight *= time_multiplier;
            println!("         ↳ Temporal Proximity [Edge:{}, Target:{}]: Weight * {:.2} (New: {:.2})", edge_year, target_year, time_multiplier, adj_weight);
        }
        
        if adj_weight > highest_weight {
            highest_weight = adj_weight;
            best_edge = Some(edge);
        }
    }
    
    if let Some(edge) = best_edge {
        let best_surface = graph.nodes.get(&edge.to).map(|n| n.surface.as_str()).unwrap_or("?");
        Some(best_surface)
    } else {
        None
    }
}

fn euclidean_dist_5(a: &[f32; 5], b: &[f32; 5]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

/// MVP Guardrail: Dynamic Entry Node Resolution (Bidirectional Walking)
/// Instead of hardcoding start words like "In", the Reasoning Module passes the queried Entity (e.g., "server").
/// The Engine physically jumps onto that Entity node in 3D space, reverse-walks the topology to find the Anchor `[.]` 
/// indicating the start of the thought, and returns the mathematically verifiable start word!
pub fn resolve_start_node<'a>(
    entity_word: &str,
    graph: &'a WordGraph,
    reasoning: &ReasoningModule,
    config: &WalkConfig
) -> Option<&'a str> {
    let mut current_id = match graph.by_surface.get(entity_word) {
        Some(id) => *id,
        None => return None,
    };

    println!("  [SYS_ORCHESTRATOR]: Jumping onto Target Entity Node: [{}]", entity_word);
    
    // Reverse Walk: Find the highest probability INCOMING edge matching our Sessional State!
    for _ in 0..20 { // Max backtrack depth
        let incoming: Vec<&WordEdge> = graph.edges.iter().filter(|e| e.to == current_id).collect();
        if incoming.is_empty() { break; }
        
        let mut best_prev = None;
        let mut highest_weight = 0.0;
        
        let active_domain = reasoning.session.domain_stack.last().map(String::as_str).unwrap_or("general");
        
        for edge in incoming {
            let mut adj_weight = edge.weight;
            if edge.domain == active_domain { adj_weight *= 2.0; }
            if let (Some(edge_year), Some(target_year)) = (edge.dated, config.target_year) {
                let diff = target_year.abs_diff(edge_year) as f32;
                let mut time_multiplier = 2.0 - (diff / 10.0);
                if time_multiplier < 1.0 { time_multiplier = 1.0; }
                adj_weight *= time_multiplier;
            }
            
            if adj_weight > highest_weight {
                highest_weight = adj_weight;
                best_prev = Some(edge);
            }
        }
        
        if let Some(edge) = best_prev {
            let prev_surface = graph.nodes.get(&edge.from).map(|n| n.surface.as_str()).unwrap_or("?");
            if prev_surface == "." || prev_surface == "?" || prev_surface == "!" {
                // If the previous node is punctuation, the CURRENT node is the start of the sentence!
                break;
            }
            current_id = edge.from;
        } else {
            break;
        }
    }
    
    // We found the geometric start of the highest-probability path containing our entity!
    graph.nodes.get(&current_id).map(|n| n.surface.as_str())
}
