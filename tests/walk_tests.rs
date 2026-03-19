use spse_predictive::graph::{WordGraph, WordNode, WordEdge};
use spse_predictive::reasoning::ReasoningModule;
use spse_predictive::spatial::SpatialGrid;
use spse_predictive::reasoning::{evaluate_arithmetic, is_arithmetic_query};
use spse_predictive::walk::{compute_depth_limit, is_reachable, predict_next, resolve_start_node, secondary_signal, WalkConfig, WalkMode};

// ---------------------------------------------------------------------------
// Graph construction helpers
// ---------------------------------------------------------------------------

/// Build a linear token chain with uniform metadata.
/// Returns the graph and the ordered list of surface strings that were inserted.
fn build_chain(
    tokens: &[&str],
    intent: &str,
    tone: &str,
    domain: &str,
    entity: Option<&str>,
    dated: Option<u16>,
) -> WordGraph {
    let mut graph = WordGraph::new();
    let mut prev_id: Option<u64> = None;

    for token in tokens {
        let id = WordGraph::generate_id(token);
        graph.by_surface.insert(token.to_string(), id);
        graph.nodes.entry(id).or_insert(WordNode {
            id,
            surface: token.to_string(),
            frequency: 1,
            position: [0.0; 3],
            lexical_vector: WordNode::compute_lexical_vector(token),
        });

        if let Some(prev) = prev_id {
            graph.edges.push(WordEdge {
                from: prev,
                to: id,
                weight: 1.0,
                intent: intent.to_string(),
                tone: tone.to_string(),
                domain: domain.to_string(),
                entity: entity.map(str::to_string),
                dated,
            });
        }
        prev_id = Some(id);
    }
    graph
}

/// Create a reasoning module with a single context frame already pushed.
fn reasoning_with<'a>(
    graph: &'a WordGraph,
    intent: &str,
    tone: &str,
    domain: &str,
) -> ReasoningModule<'a> {
    let mut r = ReasoningModule::new(graph);
    r.update_context(intent, tone, domain, &[]);
    r
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

#[test]
fn test_graph_node_and_edge_insertion() {
    let graph = build_chain(&["hello", "world"], "statement", "neutral", "general", None, None);

    assert!(graph.by_surface.contains_key("hello"));
    assert!(graph.by_surface.contains_key("world"));
    assert_eq!(graph.nodes.len(), 2);
    assert_eq!(graph.edges.len(), 1);

    let hello_id = WordGraph::generate_id("hello");
    let world_id = WordGraph::generate_id("world");
    let edge = &graph.edges[0];
    assert_eq!(edge.from, hello_id);
    assert_eq!(edge.to, world_id);
    assert_eq!(edge.weight, 1.0);
}

#[test]
fn test_lexical_vector_deterministic() {
    let v1 = WordNode::compute_lexical_vector("server");
    let v2 = WordNode::compute_lexical_vector("server");
    assert_eq!(v1, v2);
}

#[test]
fn test_lexical_vector_empty_word() {
    let v = WordNode::compute_lexical_vector("");
    assert_eq!(v, [0.0; 5]);
}

// ---------------------------------------------------------------------------
// predict_next — basic routing
// ---------------------------------------------------------------------------

#[test]
fn test_predict_next_returns_next_token() {
    let graph = build_chain(&["the", "server", "is", "online", "."], "statement", "neutral", "tech", None, None);
    let reasoning = reasoning_with(&graph, "statement", "neutral", "tech");
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Forward };

    let next = predict_next("the", &graph, None, &reasoning, &config, &[]);
    assert_eq!(next.as_deref(), Some("server"));
}

#[test]
fn test_predict_next_returns_none_at_end_of_chain() {
    let graph = build_chain(&["hello", "world"], "statement", "neutral", "general", None, None);
    let reasoning = reasoning_with(&graph, "statement", "neutral", "general");
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Forward };

    // "world" has no outgoing edges
    let next = predict_next("world", &graph, None, &reasoning, &config, &[]);
    assert!(next.is_none());
}

// ---------------------------------------------------------------------------
// predict_next — intent biasing
// ---------------------------------------------------------------------------

/// Two competing edges from the same source node with different intents.
/// The edge whose intent matches the session should win.
#[test]
fn test_predict_next_intent_bias_selects_correct_branch() {
    let mut graph = WordGraph::new();

    let src_id  = WordGraph::generate_id("bank");
    let yes_id  = WordGraph::generate_id("closes");   // intent matches session
    let no_id   = WordGraph::generate_id("opens");    // intent does not match

    for (id, word) in [(src_id, "bank"), (yes_id, "closes"), (no_id, "opens")] {
        graph.by_surface.insert(word.to_string(), id);
        graph.nodes.insert(id, WordNode {
            id,
            surface: word.to_string(),
            frequency: 1,
            position: [0.0; 3],
            lexical_vector: WordNode::compute_lexical_vector(word),
        });
    }

    // Edge to "closes": weight 1.0, intent = "statement" (will be ×2 when session = statement)
    graph.edges.push(WordEdge {
        from: src_id, to: yes_id, weight: 1.0,
        intent: "statement".to_string(), tone: "neutral".to_string(),
        domain: "finance".to_string(), entity: None, dated: None,
    });
    // Edge to "opens": weight 1.0, intent = "question" (no multiplier when session = statement)
    graph.edges.push(WordEdge {
        from: src_id, to: no_id, weight: 1.0,
        intent: "question".to_string(), tone: "neutral".to_string(),
        domain: "finance".to_string(), entity: None, dated: None,
    });

    let reasoning = reasoning_with(&graph, "statement", "neutral", "finance");
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Forward };

    let next = predict_next("bank", &graph, None, &reasoning, &config, &[]);
    assert_eq!(next.as_deref(), Some("closes"), "intent-biased edge should win");
}

// ---------------------------------------------------------------------------
// predict_next — domain biasing
// ---------------------------------------------------------------------------

#[test]
fn test_predict_next_domain_bias_selects_correct_branch() {
    let mut graph = WordGraph::new();

    let src_id   = WordGraph::generate_id("data");
    let tech_id  = WordGraph::generate_id("server");   // domain = tech
    let sci_id   = WordGraph::generate_id("particle");  // domain = science

    for (id, word) in [(src_id, "data"), (tech_id, "server"), (sci_id, "particle")] {
        graph.by_surface.insert(word.to_string(), id);
        graph.nodes.insert(id, WordNode {
            id, surface: word.to_string(), frequency: 1,
            position: [0.0; 3],
            lexical_vector: WordNode::compute_lexical_vector(word),
        });
    }

    graph.edges.push(WordEdge {
        from: src_id, to: tech_id, weight: 1.0,
        intent: "statement".to_string(), tone: "neutral".to_string(),
        domain: "tech".to_string(), entity: None, dated: None,
    });
    graph.edges.push(WordEdge {
        from: src_id, to: sci_id, weight: 1.0,
        intent: "statement".to_string(), tone: "neutral".to_string(),
        domain: "science".to_string(), entity: None, dated: None,
    });

    let reasoning = reasoning_with(&graph, "statement", "neutral", "tech");
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Forward };

    let next = predict_next("data", &graph, None, &reasoning, &config, &[]);
    assert_eq!(next.as_deref(), Some("server"), "domain-biased edge should win");
}

// ---------------------------------------------------------------------------
// predict_next — temporal biasing
// ---------------------------------------------------------------------------

#[test]
fn test_predict_next_temporal_bias_prefers_closer_year() {
    let mut graph = WordGraph::new();

    let src_id    = WordGraph::generate_id("status");
    let recent_id = WordGraph::generate_id("online");   // dated 2025 — closer to target 2026
    let old_id    = WordGraph::generate_id("offline");  // dated 2010 — further from target 2026

    for (id, word) in [(src_id, "status"), (recent_id, "online"), (old_id, "offline")] {
        graph.by_surface.insert(word.to_string(), id);
        graph.nodes.insert(id, WordNode {
            id, surface: word.to_string(), frequency: 1,
            position: [0.0; 3],
            lexical_vector: WordNode::compute_lexical_vector(word),
        });
    }

    graph.edges.push(WordEdge {
        from: src_id, to: recent_id, weight: 1.0,
        intent: "statement".to_string(), tone: "neutral".to_string(),
        domain: "tech".to_string(), entity: None,
        dated: Some(2025),
    });
    graph.edges.push(WordEdge {
        from: src_id, to: old_id, weight: 1.0,
        intent: "statement".to_string(), tone: "neutral".to_string(),
        domain: "tech".to_string(), entity: None,
        dated: Some(2010),
    });

    let reasoning = reasoning_with(&graph, "statement", "neutral", "tech");
    let config = WalkConfig { target_year: Some(2026), depth_limit: 1, mode: WalkMode::Forward };

    let next = predict_next("status", &graph, None, &reasoning, &config, &[]);
    assert_eq!(next.as_deref(), Some("online"), "temporally closer edge should win");
}

// ---------------------------------------------------------------------------
// predict_next — OOV lexical fallback
// ---------------------------------------------------------------------------

#[test]
fn test_predict_next_oov_snaps_to_graph_and_continues() {
    // Graph only contains known word "server"
    let graph = build_chain(&["server", "online"], "statement", "neutral", "tech", None, None);
    let reasoning = reasoning_with(&graph, "statement", "neutral", "tech");
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Forward };

    // "srvr" is not in the graph — OOV fallback should snap to nearest node and return Some
    let result = predict_next("srvr", &graph, None, &reasoning, &config, &[]);
    assert!(result.is_some(), "OOV fallback must return Some, not panic or return None");
}

// ---------------------------------------------------------------------------
// resolve_start_node
// ---------------------------------------------------------------------------

/// Graph: ". -> The -> server -> is -> online"
/// Reverse walk from "server" should reach "The" (stops because predecessor is ".").
#[test]
fn test_resolve_start_node_stops_at_punctuation() {
    let graph = build_chain(
        &[".", "The", "server", "is", "online"],
        "statement", "neutral", "tech", Some("server"), None,
    );
    let reasoning = reasoning_with(&graph, "statement", "neutral", "tech");
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Forward };

    let start = resolve_start_node("server", &graph, &reasoning, &config);
    // "The" is the first word after the sentence boundary "."
    assert_eq!(start.as_deref(), Some("The"));
}

#[test]
fn test_resolve_start_node_returns_none_for_unknown_entity() {
    let graph = build_chain(&["hello", "world"], "statement", "neutral", "general", None, None);
    let reasoning = reasoning_with(&graph, "statement", "neutral", "general");
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Forward };

    let result = resolve_start_node("unknown_entity_xyz", &graph, &reasoning, &config);
    assert!(result.is_none());
}

// ---------------------------------------------------------------------------
// sanitize_queue (Guardrail 3)
// ---------------------------------------------------------------------------

#[test]
fn test_sanitize_queue_passthrough_when_no_command() {
    let input = vec!["question".to_string(), "statement".to_string()];
    let result = ReasoningModule::sanitize_queue(input.clone());
    assert_eq!(result, input);
}

#[test]
fn test_sanitize_queue_drops_intents_before_command() {
    let input = vec![
        "question".to_string(),
        "support".to_string(),
        "command".to_string(),
    ];
    let result = ReasoningModule::sanitize_queue(input);
    // "question" and "support" should be dropped; only "command" survives
    assert_eq!(result, vec!["command"]);
}

#[test]
fn test_sanitize_queue_preserves_statements_before_command() {
    let input = vec![
        "statement".to_string(),
        "question".to_string(),
        "command".to_string(),
    ];
    let result = ReasoningModule::sanitize_queue(input);
    // "statement" is always kept; "question" is dropped; "command" is kept
    assert_eq!(result, vec!["statement", "command"]);
}

#[test]
fn test_sanitize_queue_empty_input() {
    let result = ReasoningModule::sanitize_queue(vec![]);
    assert!(result.is_empty());
}

// ---------------------------------------------------------------------------
// ReasoningModule::update_context
// ---------------------------------------------------------------------------

#[test]
fn test_update_context_pushes_all_stacks() {
    let graph = build_chain(&["a", "b"], "statement", "neutral", "general", None, None);
    let mut r = ReasoningModule::new(&graph);
    r.update_context("question", "polite", "finance", &["bank".to_string()]);

    assert_eq!(r.session.intent_stack.last().unwrap(), "question");
    assert_eq!(r.session.tone_stack.last().unwrap(), "polite");
    assert_eq!(r.session.domain_stack.last().unwrap(), "finance");
    assert_eq!(r.session.entity_stack.last().unwrap(), "bank");
}

#[test]
fn test_update_context_skips_pronoun_entity() {
    let graph = build_chain(&["a", "b"], "statement", "neutral", "general", None, None);
    let mut r = ReasoningModule::new(&graph);
    r.update_context("statement", "neutral", "finance", &["Pronoun".to_string()]);

    // "Pronoun" must not be pushed onto the entity stack
    assert!(r.session.entity_stack.is_empty());
}

#[test]
fn test_entity_stack_used_for_pronoun_resolution() {
    let graph = build_chain(&["a", "b"], "statement", "neutral", "general", None, None);
    let mut r = ReasoningModule::new(&graph);

    // First turn: real entity pushed
    r.update_context("statement", "neutral", "finance", &["bank".to_string()]);
    // Second turn: pronoun — entity stack must still hold "bank"
    r.update_context("question", "neutral", "finance", &["Pronoun".to_string()]);

    assert_eq!(r.session.entity_stack.last().unwrap(), "bank");
}

// ---------------------------------------------------------------------------
// Tier 2: KD-tree radial routing
// ---------------------------------------------------------------------------

/// "orphan" has no outgoing edges (Tier 1 fails).
/// "anchor" is placed at the *same* 3D position as "orphan" (guaranteed within radius 3.0)
/// and has an edge to "result".
/// With a SpatialGrid, predict_next must route via Tier 2 and return Some("result").
#[test]
fn test_predict_next_tier2_routes_via_nearby_node() {
    let mut graph = WordGraph::new();

    let orphan_id = WordGraph::generate_id("orphan");
    let anchor_id = WordGraph::generate_id("anchor");
    let result_id = WordGraph::generate_id("result");

    let shared_pos = [4.0_f32, 0.5, 0.8]; // same position → distance 0

    for (id, word) in [(orphan_id, "orphan"), (anchor_id, "anchor"), (result_id, "result")] {
        graph.by_surface.insert(word.to_string(), id);
        graph.nodes.insert(id, WordNode {
            id,
            surface: word.to_string(),
            frequency: 1,
            position: shared_pos,
            lexical_vector: WordNode::compute_lexical_vector(word),
        });
    }

    // Only anchor → result edge; orphan has no outgoing edges.
    graph.edges.push(WordEdge {
        from: anchor_id, to: result_id, weight: 1.0,
        intent: "statement".to_string(), tone: "neutral".to_string(),
        domain: "general".to_string(), entity: None, dated: None,
    });

    let spatial = SpatialGrid::build(graph.nodes.values().map(|n| (n.id, n.position)));
    let reasoning = reasoning_with(&graph, "statement", "neutral", "general");
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Forward };

    let next = predict_next("orphan", &graph, Some(&spatial), &reasoning, &config, &[]);
    assert_eq!(next.as_deref(), Some("result"), "Tier 2 should route via anchor's edge to result");
}

#[test]
fn test_predict_next_tier2_not_triggered_when_tier1_succeeds() {
    let graph = build_chain(&["hello", "world"], "statement", "neutral", "general", None, None);
    let spatial = SpatialGrid::build(graph.nodes.values().map(|n| (n.id, n.position)));
    let reasoning = reasoning_with(&graph, "statement", "neutral", "general");
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Forward };

    let next = predict_next("hello", &graph, Some(&spatial), &reasoning, &config, &[]);
    assert_eq!(next.as_deref(), Some("world"), "Tier 1 direct edge should win when it exists");
}

// Tier 2 with position history: when the centroid of history + current differs
// from the dead-end alone, Tier 2 should find neighbours via the centroid.
#[test]
fn test_predict_next_tier2_centroid_shifts_search_origin() {
    let mut graph = WordGraph::new();

    let dead_id   = WordGraph::generate_id("dead");
    let hist_id   = WordGraph::generate_id("hist");
    let anchor_id = WordGraph::generate_id("anchor");
    let result_id = WordGraph::generate_id("result");

    // Geometry:
    //   anchor at origin [0,0,0].
    //   dead at [6,0,0]  — distance 6 from anchor, OUTSIDE radius 3.0.
    //   hist at [-4,0,0] — centroid([hist,dead]) = [1,0,0], distance 1 from anchor, INSIDE radius 3.0.
    let dead_pos:   [f32; 3] = [6.0, 0.0, 0.0];
    let hist_pos:   [f32; 3] = [-4.0, 0.0, 0.0];
    let anchor_pos: [f32; 3] = [0.0, 0.0, 0.0];

    for (id, word, pos) in [
        (dead_id,   "dead",   dead_pos),
        (hist_id,   "hist",   hist_pos),
        (anchor_id, "anchor", anchor_pos),
        (result_id, "result", [0.1_f32, 0.0, 0.0]),
    ] {
        graph.by_surface.insert(word.to_string(), id);
        graph.nodes.insert(id, WordNode {
            id, surface: word.to_string(), frequency: 1,
            position: pos,
            lexical_vector: WordNode::compute_lexical_vector(word),
        });
    }
    graph.edges.push(WordEdge {
        from: anchor_id, to: result_id, weight: 1.0,
        intent: "statement".to_string(), tone: "neutral".to_string(),
        domain: "general".to_string(), entity: None, dated: None,
    });

    let spatial = SpatialGrid::build(graph.nodes.values().map(|n| (n.id, n.position)));
    let reasoning = reasoning_with(&graph, "statement", "neutral", "general");
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Forward };

    // Without history: search from dead_pos [6,0,0] — distance 6 from anchor, outside radius 3.0.
    let without_history = predict_next("dead", &graph, Some(&spatial), &reasoning, &config, &[]);
    // With history [hist_pos = [-4,0,0]]: centroid = [(-4+6)/2, 0, 0] = [1,0,0]
    // — distance 1 from anchor, inside radius 3.0 → finds anchor→result.
    let with_history = predict_next("dead", &graph, Some(&spatial), &reasoning, &config, &[hist_pos]);
    assert_eq!(with_history.as_deref(), Some("result"),
        "Tier 2 centroid with history should find anchor→result");
    assert_ne!(without_history.as_deref(), Some("result"),
        "Tier 2 without history should not find anchor from dead_pos (distance 6 > radius 3)");
}

// ---------------------------------------------------------------------------
// is_reachable
// ---------------------------------------------------------------------------

#[test]
fn test_is_reachable_direct_edge() {
    let graph = build_chain(&["a", "b"], "statement", "neutral", "general", None, None);
    assert!(is_reachable("a", "b", &graph, 1));
}

#[test]
fn test_is_reachable_multi_hop() {
    let graph = build_chain(&["a", "b", "c", "d"], "statement", "neutral", "general", None, None);
    assert!(is_reachable("a", "d", &graph, 3));
}

#[test]
fn test_is_reachable_returns_false_when_unreachable() {
    let mut graph = WordGraph::new();
    for (from, to) in [("a", "b"), ("c", "d")] {
        let fid = WordGraph::generate_id(from);
        let tid = WordGraph::generate_id(to);
        for (id, w) in [(fid, from), (tid, to)] {
            graph.by_surface.insert(w.to_string(), id);
            graph.nodes.insert(id, WordNode {
                id, surface: w.to_string(), frequency: 1,
                position: [0.0; 3],
                lexical_vector: WordNode::compute_lexical_vector(w),
            });
        }
        graph.edges.push(WordEdge {
            from: fid, to: tid, weight: 1.0,
            intent: "statement".to_string(), tone: "neutral".to_string(),
            domain: "general".to_string(), entity: None, dated: None,
        });
    }
    assert!(!is_reachable("a", "d", &graph, 10));
}

#[test]
fn test_is_reachable_same_word_returns_true() {
    let graph = build_chain(&["hello", "world"], "statement", "neutral", "general", None, None);
    assert!(is_reachable("hello", "hello", &graph, 0));
}

#[test]
fn test_is_reachable_missing_from_word() {
    let graph = build_chain(&["a", "b"], "statement", "neutral", "general", None, None);
    assert!(!is_reachable("nonexistent", "b", &graph, 5));
}

#[test]
fn test_is_reachable_missing_to_word() {
    let graph = build_chain(&["a", "b"], "statement", "neutral", "general", None, None);
    assert!(!is_reachable("a", "nonexistent", &graph, 5));
}

#[test]
fn test_is_reachable_respects_max_hops() {
    let graph = build_chain(&["a", "b", "c", "d"], "statement", "neutral", "general", None, None);
    assert!(!is_reachable("a", "d", &graph, 2), "should not reach d in only 2 hops");
    assert!(is_reachable("a", "d", &graph, 3), "should reach d in exactly 3 hops");
}

// ---------------------------------------------------------------------------
// compute_depth_limit
// ---------------------------------------------------------------------------

/// A 2-node chain gives 1 reachable node within 2 hops → depth 1 (sparse bucket).
#[test]
fn test_compute_depth_limit_sparse_returns_1() {
    let graph = build_chain(&["start", "end"], "statement", "neutral", "general", None, None);
    assert_eq!(compute_depth_limit("start", &graph), 1);
}

/// An entity unknown to the graph falls back to depth 1.
#[test]
fn test_compute_depth_limit_unknown_entity_returns_1() {
    let graph = build_chain(&["a", "b"], "statement", "neutral", "general", None, None);
    assert_eq!(compute_depth_limit("nonexistent", &graph), 1);
}

/// A hub node with 6 distinct outgoing edges sits in the 5–14 reachable bucket → depth 2.
#[test]
fn test_compute_depth_limit_moderate_hub_returns_2() {
    let mut graph = WordGraph::new();
    let hub_tokens = ["hub", "a1", "a2", "a3", "a4", "a5", "a6"];
    for tok in &hub_tokens {
        let id = WordGraph::generate_id(tok);
        graph.by_surface.insert(tok.to_string(), id);
        graph.nodes.entry(id).or_insert(WordNode {
            id,
            surface: tok.to_string(),
            frequency: 1,
            position: [0.0; 3],
            lexical_vector: WordNode::compute_lexical_vector(tok),
        });
    }
    let hub_id = WordGraph::generate_id("hub");
    for tok in &["a1", "a2", "a3", "a4", "a5", "a6"] {
        let to_id = WordGraph::generate_id(tok);
        graph.edges.push(WordEdge {
            from: hub_id, to: to_id, weight: 1.0,
            intent: "statement".to_string(), tone: "neutral".to_string(),
            domain: "general".to_string(), entity: None, dated: None,
        });
    }
    // 6 reachable nodes in 1 hop (well within 5–14 bucket) → depth 2
    assert_eq!(compute_depth_limit("hub", &graph), 2);
}

// ---------------------------------------------------------------------------
// SpatialGrid
// ---------------------------------------------------------------------------

/// query_radius with radius 2.0 must include a node at distance 1.0 from the
/// query centre and exclude a node at distance 3.0.
#[test]
fn test_spatial_grid_query_radius_includes_close_excludes_distant() {
    let close_id: u64 = 1001;
    let far_id:   u64 = 1002;
    let close_pos: [f32; 3] = [1.0, 0.0, 0.0];
    let far_pos:   [f32; 3] = [4.0, 0.0, 0.0];

    let grid = SpatialGrid::build(
        [(close_id, close_pos), (far_id, far_pos)].into_iter(),
    );

    let origin: [f32; 3] = [0.0, 0.0, 0.0];
    let hits = grid.query_radius(origin, 2.0);
    assert!(hits.contains(&close_id), "close node should be inside radius 2.0");
    assert!(!hits.contains(&far_id),  "far node should be outside radius 2.0");
}

/// query_nearest must return the closest node when both are within the search radius.
#[test]
fn test_spatial_grid_query_nearest_returns_closest() {
    let near_id: u64 = 2001;
    let far_id:  u64 = 2002;
    let near_pos: [f32; 3] = [0.5, 0.0, 0.0];
    let far_pos:  [f32; 3] = [3.0, 0.0, 0.0];

    let grid = SpatialGrid::build(
        [(near_id, near_pos), (far_id, far_pos)].into_iter(),
    );

    let origin: [f32; 3] = [0.0, 0.0, 0.0];
    let nearest = grid.query_nearest(origin, 5.0);
    assert_eq!(nearest, Some(near_id), "nearest within radius must be the closer node");
}

/// query_nearest returns None when no node falls within the given radius.
#[test]
fn test_spatial_grid_query_nearest_returns_none_when_empty_radius() {
    let distant_id: u64 = 3001;
    let grid = SpatialGrid::build(
        [(distant_id, [10.0_f32, 0.0, 0.0])].into_iter(),
    );
    let result = grid.query_nearest([0.0, 0.0, 0.0], 1.0);
    assert_eq!(result, None, "no node within radius 1.0 — must return None");
}

// ---------------------------------------------------------------------------
// resolve_start_node — temporal bias
// ---------------------------------------------------------------------------

/// Two source nodes point to the same entity; one is dated 2010, the other
/// 2026.  With target_year 2026 the resolver must prefer the 2026-dated
/// incoming edge and return "new_start".
#[test]
fn test_resolve_start_node_prefers_temporally_close_source() {
    let mut graph = WordGraph::new();

    let tokens = ["old_start", "new_start", "server"];
    for tok in &tokens {
        let id = WordGraph::generate_id(tok);
        graph.by_surface.insert(tok.to_string(), id);
        graph.nodes.entry(id).or_insert(WordNode {
            id,
            surface: tok.to_string(),
            frequency: 1,
            position: [0.0; 3],
            lexical_vector: WordNode::compute_lexical_vector(tok),
        });
    }

    let old_id    = WordGraph::generate_id("old_start");
    let new_id    = WordGraph::generate_id("new_start");
    let server_id = WordGraph::generate_id("server");

    // old_start → server (dated 2010)
    graph.edges.push(WordEdge {
        from: old_id, to: server_id, weight: 1.0,
        intent: "statement".to_string(), tone: "neutral".to_string(),
        domain: "tech".to_string(), entity: None, dated: Some(2010),
    });
    // new_start → server (dated 2026)
    graph.edges.push(WordEdge {
        from: new_id, to: server_id, weight: 1.0,
        intent: "statement".to_string(), tone: "neutral".to_string(),
        domain: "tech".to_string(), entity: None, dated: Some(2026),
    });

    let mut reasoning = ReasoningModule::new(&graph);
    reasoning.update_context("statement", "neutral", "tech", &["server".to_string()]);
    let config = WalkConfig { target_year: Some(2026), depth_limit: 1, mode: WalkMode::Forward };

    let result = resolve_start_node("server", &graph, &reasoning, &config);
    assert_eq!(
        result.as_deref(),
        Some("new_start"),
        "with target_year=2026 the 2026-dated incoming edge should win"
    );
}

// ---------------------------------------------------------------------------
// secondary_signal
// ---------------------------------------------------------------------------

/// A secondary entity word present in the graph should be detected and returned.
#[test]
fn test_secondary_signal_finds_graph_word() {
    let graph = build_chain(&["bank", "ATM", "open"], "statement", "neutral", "finance", None, None);
    let result = secondary_signal("Is there an ATM there?", "bank", &graph);
    assert_eq!(result.as_deref(), Some("ATM"));
}

/// The primary entity itself must never be returned as the secondary signal,
/// even if it appears in the query.
#[test]
fn test_secondary_signal_ignores_primary_entity() {
    let graph = build_chain(&["server", "status", "ok"], "statement", "neutral", "tech", None, None);
    // "server" is the primary — secondary_signal must skip it and return "status".
    let result = secondary_signal("server status ok", "server", &graph);
    assert_eq!(result.as_deref(), Some("status"));
}

/// Punctuation attached to a token should be stripped before graph lookup.
#[test]
fn test_secondary_signal_strips_trailing_punctuation() {
    let graph = build_chain(&["bank", "ATM"], "statement", "neutral", "finance", None, None);
    let result = secondary_signal("Check ATM!", "bank", &graph);
    assert_eq!(result.as_deref(), Some("ATM"));
}

/// Case-insensitive lookup: an uppercase query word must match a graph node
/// stored in lowercase via the to_lowercase() fallback path.
#[test]
fn test_secondary_signal_case_insensitive_lookup() {
    // Graph stores "atm" (lowercase); query contains "ATM" (uppercase).
    // secondary_signal tries exact match ("ATM" → miss) then lowercase ("atm" → hit).
    let graph = build_chain(&["bank", "atm"], "statement", "neutral", "finance", None, None);
    let result = secondary_signal("Is the ATM open?", "bank", &graph);
    assert!(result.is_some(), "uppercase query token should match lowercase graph surface");
}

/// When no word in the query (besides the primary entity) exists in the graph,
/// secondary_signal must return None.
#[test]
fn test_secondary_signal_returns_none_when_no_match() {
    let graph = build_chain(&["bank", "loan"], "statement", "neutral", "finance", None, None);
    let result = secondary_signal("xyz abc def", "bank", &graph);
    assert_eq!(result, None);
}

// ---------------------------------------------------------------------------
// is_arithmetic_query  (Guardrail 6)
// ---------------------------------------------------------------------------

/// Classic arithmetic expression: number + operator keyword → true.
#[test]
fn test_is_arithmetic_query_detects_number_plus_keyword() {
    assert!(is_arithmetic_query("What is 2 plus 3?"));
}

/// Operator symbol alongside a number → true.
#[test]
fn test_is_arithmetic_query_detects_operator_symbol() {
    assert!(is_arithmetic_query("5 * 7 equals what?"));
}

/// Number present but no arithmetic signal → false (no false positive).
#[test]
fn test_is_arithmetic_query_number_only_returns_false() {
    assert!(!is_arithmetic_query("The bank closes at 5 pm"));
}

/// Arithmetic keyword present but no numeric token → false.
#[test]
fn test_is_arithmetic_query_keyword_only_returns_false() {
    assert!(!is_arithmetic_query("What is the sum of all loans?"));
}

/// Completely non-arithmetic query → false.
#[test]
fn test_is_arithmetic_query_normal_query_returns_false() {
    assert!(!is_arithmetic_query("Are the servers online?"));
}

/// "squared" keyword with a number triggers the guard.
#[test]
fn test_is_arithmetic_query_detects_squared_keyword() {
    assert!(is_arithmetic_query("What is 9 squared?"));
}

/// Division via keyword: "divided" + number → true.
#[test]
fn test_is_arithmetic_query_detects_divided_keyword() {
    assert!(is_arithmetic_query("10 divided by 2 is what?"));
}

// ---------------------------------------------------------------------------
// evaluate_arithmetic — Phase 12
// ---------------------------------------------------------------------------

#[test]
fn test_evaluate_arithmetic_addition_symbol() {
    let result = evaluate_arithmetic("what is 5 + 3?");
    assert_eq!(result, Some("8".to_string()));
}

#[test]
fn test_evaluate_arithmetic_addition_word() {
    let result = evaluate_arithmetic("What is 12 plus 8?");
    assert_eq!(result, Some("20".to_string()));
}

#[test]
fn test_evaluate_arithmetic_subtraction() {
    let result = evaluate_arithmetic("10 minus 4");
    assert_eq!(result, Some("6".to_string()));
}

#[test]
fn test_evaluate_arithmetic_multiplication() {
    let result = evaluate_arithmetic("7 times 6");
    assert_eq!(result, Some("42".to_string()));
}

#[test]
fn test_evaluate_arithmetic_division() {
    let result = evaluate_arithmetic("20 divided by 4");
    assert_eq!(result, Some("5".to_string()));
}

#[test]
fn test_evaluate_arithmetic_fractional_result() {
    let result = evaluate_arithmetic("1 divided by 3");
    assert!(result.is_some());
    assert!(result.unwrap().contains("0.3333"));
}

#[test]
fn test_evaluate_arithmetic_division_by_zero() {
    let result = evaluate_arithmetic("5 divided by 0");
    assert_eq!(result, Some("undefined (division by zero)".to_string()));
}

#[test]
fn test_evaluate_arithmetic_squared() {
    let result = evaluate_arithmetic("9 squared");
    assert_eq!(result, Some("81".to_string()));
}

#[test]
fn test_evaluate_arithmetic_cubed() {
    let result = evaluate_arithmetic("3 cubed");
    assert_eq!(result, Some("27".to_string()));
}

#[test]
fn test_evaluate_arithmetic_sqrt() {
    let result = evaluate_arithmetic("sqrt 16");
    assert_eq!(result, Some("4".to_string()));
}

#[test]
fn test_evaluate_arithmetic_sqrt_negative() {
    let result = evaluate_arithmetic("sqrt -4");
    assert_eq!(result, Some("undefined (square root of negative number)".to_string()));
}

#[test]
fn test_evaluate_arithmetic_sum_keyword() {
    let result = evaluate_arithmetic("sum of 3 and 7");
    assert_eq!(result, Some("10".to_string()));
}

#[test]
fn test_evaluate_arithmetic_product_keyword() {
    let result = evaluate_arithmetic("product of 4 and 5");
    assert_eq!(result, Some("20".to_string()));
}

#[test]
fn test_evaluate_arithmetic_no_number_returns_none() {
    let result = evaluate_arithmetic("what is the capital of France?");
    assert_eq!(result, None);
}

#[test]
fn test_evaluate_arithmetic_single_number_no_op_returns_none() {
    let result = evaluate_arithmetic("I have 5 apples");
    assert_eq!(result, None);
}

// ---------------------------------------------------------------------------
// Tier 3: backtrack-and-reroute
// ---------------------------------------------------------------------------

/// Build a fork graph:  anchor → dead_end  (no further outgoing from dead_end)
///                      anchor → alt
/// Walking from dead_end with no spatial grid must trigger Tier 3, backtrack
/// to anchor, and return "alt" via the alternative forward edge.
#[test]
fn test_predict_next_tier3_reroutes_via_ancestor() {
    let mut graph = WordGraph::new();

    for tok in &["anchor", "dead_end", "alt"] {
        let id = WordGraph::generate_id(tok);
        graph.by_surface.insert(tok.to_string(), id);
        graph.nodes.entry(id).or_insert(WordNode {
            id,
            surface: tok.to_string(),
            frequency: 1,
            position: [0.0; 3],
            lexical_vector: WordNode::compute_lexical_vector(tok),
        });
    }

    let anchor_id   = WordGraph::generate_id("anchor");
    let dead_end_id = WordGraph::generate_id("dead_end");
    let alt_id      = WordGraph::generate_id("alt");

    // anchor → dead_end
    graph.edges.push(WordEdge {
        from: anchor_id, to: dead_end_id, weight: 1.0,
        intent: "statement".to_string(), tone: "neutral".to_string(),
        domain: "general".to_string(), entity: None, dated: None,
    });
    // anchor → alt  (the reroute target)
    graph.edges.push(WordEdge {
        from: anchor_id, to: alt_id, weight: 1.0,
        intent: "statement".to_string(), tone: "neutral".to_string(),
        domain: "general".to_string(), entity: None, dated: None,
    });

    let reasoning = reasoning_with(&graph, "statement", "neutral", "general");
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Forward };

    // dead_end has no outgoing edges — Tier 1 and Tier 2 (None) fail → Tier 3 fires.
    let result = predict_next("dead_end", &graph, None, &reasoning, &config, &[]);
    assert_eq!(result.as_deref(), Some("alt"), "Tier 3 should reroute via ancestor anchor to alt");
}

/// Tier 3 must NOT return the dead-end node itself (no trivial back-and-forth loop).
#[test]
fn test_predict_next_tier3_does_not_loop_back_to_dead_end() {
    let mut graph = WordGraph::new();

    for tok in &["src", "dead_end", "forward"] {
        let id = WordGraph::generate_id(tok);
        graph.by_surface.insert(tok.to_string(), id);
        graph.nodes.entry(id).or_insert(WordNode {
            id,
            surface: tok.to_string(),
            frequency: 1,
            position: [0.0; 3],
            lexical_vector: WordNode::compute_lexical_vector(tok),
        });
    }

    let src_id      = WordGraph::generate_id("src");
    let dead_end_id = WordGraph::generate_id("dead_end");
    let forward_id  = WordGraph::generate_id("forward");

    graph.edges.push(WordEdge {
        from: src_id, to: dead_end_id, weight: 2.0, // heavier — must still be excluded
        intent: "statement".to_string(), tone: "neutral".to_string(),
        domain: "general".to_string(), entity: None, dated: None,
    });
    graph.edges.push(WordEdge {
        from: src_id, to: forward_id, weight: 1.0,
        intent: "statement".to_string(), tone: "neutral".to_string(),
        domain: "general".to_string(), entity: None, dated: None,
    });

    let reasoning = reasoning_with(&graph, "statement", "neutral", "general");
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Forward };

    let result = predict_next("dead_end", &graph, None, &reasoning, &config, &[]);
    assert_ne!(result.as_deref(), Some("dead_end"), "Tier 3 must not loop back to the dead-end node");
    assert_eq!(result.as_deref(), Some("forward"));
}

/// When a dead-end has no ancestors either, Tier 3 returns None (no infinite search).
#[test]
fn test_predict_next_tier3_returns_none_when_no_ancestors() {
    let graph = build_chain(&["orphan"], "statement", "neutral", "general", None, None);
    let reasoning = reasoning_with(&graph, "statement", "neutral", "general");
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Forward };
    let result = predict_next("orphan", &graph, None, &reasoning, &config, &[]);
    assert_eq!(result, None, "orphan node with no ancestors should return None");
}

// ---------------------------------------------------------------------------
// ReasoningModule::reset_session
// ---------------------------------------------------------------------------

/// After reset, all four stacks must be empty.
#[test]
fn test_reset_session_clears_all_stacks() {
    let graph = build_chain(&["a", "b"], "statement", "neutral", "general", None, None);
    let mut r = ReasoningModule::new(&graph);
    r.update_context("question", "polite", "tech", &["server".to_string()]);

    r.reset_session();

    assert!(r.session.intent_stack.is_empty(),  "intent_stack must be empty after reset");
    assert!(r.session.tone_stack.is_empty(),    "tone_stack must be empty after reset");
    assert!(r.session.domain_stack.is_empty(),  "domain_stack must be empty after reset");
    assert!(r.session.entity_stack.is_empty(),  "entity_stack must be empty after reset");
}

/// reset_session must be idempotent — calling it twice on an already-empty
/// module must not panic.
#[test]
fn test_reset_session_is_idempotent() {
    let graph = build_chain(&["a", "b"], "statement", "neutral", "general", None, None);
    let mut r = ReasoningModule::new(&graph);
    r.reset_session();
    r.reset_session(); // must not panic
    assert!(r.session.intent_stack.is_empty());
}

/// After reset, update_context must work normally — only the new frame is
/// present, with no residue from before the reset.
#[test]
fn test_reset_session_then_update_context_fresh_state() {
    let graph = build_chain(&["a", "b"], "statement", "neutral", "general", None, None);
    let mut r = ReasoningModule::new(&graph);
    r.update_context("command", "angry", "finance", &["bank".to_string()]);

    r.reset_session();
    r.update_context("explain", "neutral", "science", &["quantum".to_string()]);

    assert_eq!(r.session.intent_stack.len(), 1);
    assert_eq!(r.session.intent_stack[0], "explain");
    assert_eq!(r.session.domain_stack[0], "science");
    assert_eq!(r.session.entity_stack[0], "quantum");
    // No residue from the pre-reset frame
    assert!(!r.session.intent_stack.contains(&"command".to_string()));
    assert!(!r.session.entity_stack.contains(&"bank".to_string()));
}

// ---------------------------------------------------------------------------
// WalkMode::Explain — breadth-first (high out-degree preference)
// ---------------------------------------------------------------------------

/// Explain mode selects the target with the most onward edges, not the
/// highest-weighted edge.  Build a fork where one branch has more onward edges
/// and verify Explain mode picks it over the heavier-weighted but leaf branch.
#[test]
fn test_predict_next_explain_mode_prefers_high_out_degree() {
    let mut graph = WordGraph::new();

    for tok in &["src", "leaf", "hub", "a", "b", "c"] {
        let id = WordGraph::generate_id(tok);
        graph.by_surface.insert(tok.to_string(), id);
        graph.nodes.entry(id).or_insert(WordNode {
            id, surface: tok.to_string(), frequency: 1,
            position: [0.0; 3], lexical_vector: WordNode::compute_lexical_vector(tok),
        });
    }

    let src_id  = WordGraph::generate_id("src");
    let leaf_id = WordGraph::generate_id("leaf");
    let hub_id  = WordGraph::generate_id("hub");

    // src → leaf  (weight 2.0 — heavier, but leaf is a dead-end)
    graph.edges.push(WordEdge {
        from: src_id, to: leaf_id, weight: 2.0,
        intent: "explain".to_string(), tone: "neutral".to_string(),
        domain: "science".to_string(), entity: None, dated: None,
    });
    // src → hub  (weight 1.0 — lighter, but hub has 3 onward edges)
    graph.edges.push(WordEdge {
        from: src_id, to: hub_id, weight: 1.0,
        intent: "explain".to_string(), tone: "neutral".to_string(),
        domain: "science".to_string(), entity: None, dated: None,
    });
    // hub → a, hub → b, hub → c
    for tok in &["a", "b", "c"] {
        let to_id = WordGraph::generate_id(tok);
        graph.edges.push(WordEdge {
            from: hub_id, to: to_id, weight: 1.0,
            intent: "explain".to_string(), tone: "neutral".to_string(),
            domain: "science".to_string(), entity: None, dated: None,
        });
    }

    let reasoning = reasoning_with(&graph, "explain", "neutral", "science");
    let config = WalkConfig { target_year: None, depth_limit: 2, mode: WalkMode::Explain };

    let result = predict_next("src", &graph, None, &reasoning, &config, &[]);
    assert_eq!(result.as_deref(), Some("hub"), "Explain mode must prefer hub (3 onward edges) over leaf (0 edges)");
}

// ---------------------------------------------------------------------------
// WalkMode::Question — answer-anchor proximity
// ---------------------------------------------------------------------------

/// Question mode selects the edge whose target is topologically closest to the
/// active entity anchor.  Build a fork: one branch leads toward the anchor in
/// 1 hop; the other is a dead-end.  Question mode must pick the closer branch.
#[test]
fn test_predict_next_question_mode_routes_toward_entity_anchor() {
    let mut graph = WordGraph::new();

    for tok in &["src", "near", "far", "answer"] {
        let id = WordGraph::generate_id(tok);
        graph.by_surface.insert(tok.to_string(), id);
        graph.nodes.entry(id).or_insert(WordNode {
            id, surface: tok.to_string(), frequency: 1,
            position: [0.0; 3], lexical_vector: WordNode::compute_lexical_vector(tok),
        });
    }

    let src_id    = WordGraph::generate_id("src");
    let near_id   = WordGraph::generate_id("near");
    let far_id    = WordGraph::generate_id("far");
    let answer_id = WordGraph::generate_id("answer");

    // src → near → answer  (2 hops total; near is 1 hop from answer)
    graph.edges.push(WordEdge {
        from: src_id, to: near_id, weight: 1.0,
        intent: "question".to_string(), tone: "neutral".to_string(),
        domain: "general".to_string(), entity: None, dated: None,
    });
    graph.edges.push(WordEdge {
        from: near_id, to: answer_id, weight: 1.0,
        intent: "question".to_string(), tone: "neutral".to_string(),
        domain: "general".to_string(), entity: None, dated: None,
    });
    // src → far  (dead-end, no path to answer)
    graph.edges.push(WordEdge {
        from: src_id, to: far_id, weight: 2.0, // heavier weight — must still lose
        intent: "question".to_string(), tone: "neutral".to_string(),
        domain: "general".to_string(), entity: None, dated: None,
    });

    let mut reasoning = ReasoningModule::new(&graph);
    reasoning.update_context("question", "neutral", "general", &["answer".to_string()]);
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Question };

    let result = predict_next("src", &graph, None, &reasoning, &config, &[]);
    assert_eq!(result.as_deref(), Some("near"), "Question mode must route toward the entity anchor");
}

/// WalkMode::from_intent must map correctly for all known intent strings.
#[test]
fn test_walk_mode_from_intent_mapping() {
    assert_eq!(WalkMode::from_intent("explain"),    WalkMode::Explain);
    assert_eq!(WalkMode::from_intent("question"),   WalkMode::Question);
    assert_eq!(WalkMode::from_intent("complaint"),  WalkMode::Question);
    assert_eq!(WalkMode::from_intent("statement"),  WalkMode::Forward);
    assert_eq!(WalkMode::from_intent("command"),    WalkMode::Forward);
    assert_eq!(WalkMode::from_intent("unknown"),    WalkMode::Forward);
}

// ---------------------------------------------------------------------------
// Phase 16: complaint intent + Tier 3 Spatial A* Bridge
// ---------------------------------------------------------------------------

/// Build a graph where `orphan` has no outgoing edges and no ancestors, but
/// `entity` sits at a distinct 3D position.  In Question mode with a spatial
/// grid, Tier 3 must fire the A* bridge and land on `entity` (the only other
/// node in the tree that has outgoing edges).
#[test]
fn test_tier3_question_astar_bridge_jumps_to_entity() {
    let mut graph = WordGraph::new();

    // "orphan": position [1.0, 0.0, 0.0]  — length-1 word
    // "entity": position [2.0, 0.5, 0.5]  — different cluster
    // "next"  : reachable from entity
    for (tok, pos) in &[("o", [1.0_f32, 0.0, 0.0]), ("entity", [6.0, 0.5, 0.5]), ("next", [6.0, 0.5, 0.8])] {
        let id = WordGraph::generate_id(tok);
        graph.by_surface.insert(tok.to_string(), id);
        graph.nodes.entry(id).or_insert(WordNode {
            id, surface: tok.to_string(), frequency: 1,
            position: *pos,
            lexical_vector: WordNode::compute_lexical_vector(tok),
        });
    }
    // entity → next (gives entity an outgoing edge so the bridge is accepted)
    graph.edges.push(WordEdge {
        from: WordGraph::generate_id("entity"),
        to:   WordGraph::generate_id("next"),
        weight: 1.0,
        intent: "question".to_string(), tone: "neutral".to_string(),
        domain: "general".to_string(), entity: None, dated: None,
    });

    let spatial = SpatialGrid::build(graph.nodes.values().map(|n| (n.id, n.position)));

    let mut reasoning = ReasoningModule::new(&graph);
    reasoning.update_context("question", "neutral", "general", &["entity".to_string()]);
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Question };

    // "o" has no outgoing edges and no ancestors — Tier 1 and Tier 3 classic fail.
    // Tier 3 A* bridge should jump to "entity" (nearest node with outgoing edges).
    let result = predict_next("o", &graph, Some(&spatial), &reasoning, &config, &[]);
    assert_eq!(result.as_deref(), Some("entity"), "Tier 3 A* bridge must jump to the entity cluster");
}

/// A* bridge must NOT activate in Forward mode — classic backtrack must run instead.
#[test]
fn test_tier3_astar_bridge_inactive_in_forward_mode() {
    let mut graph = WordGraph::new();

    for tok in &["src", "alt", "entity"] {
        let id = WordGraph::generate_id(tok);
        graph.by_surface.insert(tok.to_string(), id);
        graph.nodes.entry(id).or_insert(WordNode {
            id, surface: tok.to_string(), frequency: 1,
            position: [1.0; 3],
            lexical_vector: WordNode::compute_lexical_vector(tok),
        });
    }
    // src ← alt  (alt is ancestor of src, has forward edge to "alt" but not entity)
    graph.edges.push(WordEdge {
        from: WordGraph::generate_id("alt"),
        to:   WordGraph::generate_id("src"),
        weight: 1.0,
        intent: "statement".to_string(), tone: "neutral".to_string(),
        domain: "general".to_string(), entity: None, dated: None,
    });
    // entity → entity_child
    let eid = WordGraph::generate_id("entity");
    let eid2 = WordGraph::generate_id("entity_child");
    graph.by_surface.insert("entity_child".to_string(), eid2);
    graph.nodes.entry(eid2).or_insert(WordNode {
        id: eid2, surface: "entity_child".to_string(), frequency: 1,
        position: [5.0; 3], lexical_vector: WordNode::compute_lexical_vector("entity_child"),
    });
    graph.edges.push(WordEdge {
        from: eid, to: eid2, weight: 1.0,
        intent: "statement".to_string(), tone: "neutral".to_string(),
        domain: "general".to_string(), entity: None, dated: None,
    });

    let spatial = SpatialGrid::build(graph.nodes.values().map(|n| (n.id, n.position)));
    let mut reasoning = ReasoningModule::new(&graph);
    reasoning.update_context("statement", "neutral", "general", &["entity".to_string()]);
    let config = WalkConfig { target_year: None, depth_limit: 1, mode: WalkMode::Forward };

    // "src" is a dead-end. Forward mode must use classic backtrack (ancestor = alt).
    let result = predict_next("src", &graph, Some(&spatial), &reasoning, &config, &[]);
    // The bridge is NOT active in Forward mode; classic backtrack returns alt (only ancestor with alt edge but alt→src is the only edge, alt has no *other* forward edges here).
    // The ancestor of src is alt; alt has edges only to src; tier3_edges will be empty → None.
    // That's fine — the important assertion is that the bridge didn't jump to entity_child.
    assert_ne!(result.as_deref(), Some("entity_child"), "A* bridge must not activate in Forward mode");
}

// ---------------------------------------------------------------------------
// extract_year_from_query — Phase 15
// ---------------------------------------------------------------------------

use spse_predictive::reasoning::extract_year_from_query;

/// Plain 4-digit year embedded in a sentence is extracted.
#[test]
fn test_extract_year_finds_year_in_sentence() {
    assert_eq!(extract_year_from_query("What happened in 2018?"), Some(2018));
}

/// Year at the very start of the query is found.
#[test]
fn test_extract_year_at_start_of_query() {
    assert_eq!(extract_year_from_query("2003 was a turbulent year."), Some(2003));
}

/// Year at the very end of the query (no trailing whitespace) is found.
#[test]
fn test_extract_year_at_end_of_query() {
    assert_eq!(extract_year_from_query("Tell me about events from 1999"), Some(1999));
}

/// Year with surrounding punctuation (comma, parentheses) is still extracted.
#[test]
fn test_extract_year_with_surrounding_punctuation() {
    assert_eq!(extract_year_from_query("The revolution (2011) changed everything."), Some(2011));
}

/// Short numbers are not mistaken for years.
#[test]
fn test_extract_year_ignores_short_numbers() {
    assert_eq!(extract_year_from_query("The bank closes at 5 pm"), None);
}

/// A 5-digit number is not treated as a year.
#[test]
fn test_extract_year_ignores_5_digit_numbers() {
    assert_eq!(extract_year_from_query("Model A10000 is our product"), None);
}

/// Year below the valid range (1899) is rejected.
#[test]
fn test_extract_year_rejects_out_of_range_low() {
    assert_eq!(extract_year_from_query("It happened in 1899."), None);
}

/// Year above the valid range (2100) is rejected.
#[test]
fn test_extract_year_rejects_out_of_range_high() {
    assert_eq!(extract_year_from_query("Plans for 2100 are speculative."), None);
}

/// Query with no numeric token at all returns None.
#[test]
fn test_extract_year_no_numbers_returns_none() {
    assert_eq!(extract_year_from_query("Are the servers online?"), None);
}

/// Boundary: first valid year (1900) is accepted.
#[test]
fn test_extract_year_boundary_low_accepted() {
    assert_eq!(extract_year_from_query("Events from 1900 onward."), Some(1900));
}

/// Boundary: last valid year (2099) is accepted.
#[test]
fn test_extract_year_boundary_high_accepted() {
    assert_eq!(extract_year_from_query("Planning for 2099."), Some(2099));
}

/// When multiple valid years appear, the first one is returned.
#[test]
fn test_extract_year_returns_first_when_multiple() {
    assert_eq!(extract_year_from_query("From 2010 to 2020."), Some(2010));
}
