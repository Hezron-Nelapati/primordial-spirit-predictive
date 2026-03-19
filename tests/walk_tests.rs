use spse_predictive::graph::{WordGraph, WordNode, WordEdge};
use spse_predictive::reasoning::ReasoningModule;
use spse_predictive::spatial::SpatialGrid;
use spse_predictive::walk::{compute_depth_limit, is_reachable, predict_next, resolve_start_node, WalkConfig};

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
    let config = WalkConfig { target_year: None, depth_limit: 1 };

    let next = predict_next("the", &graph, None, &reasoning, &config);
    assert_eq!(next, Some("server"));
}

#[test]
fn test_predict_next_returns_none_at_end_of_chain() {
    let graph = build_chain(&["hello", "world"], "statement", "neutral", "general", None, None);
    let reasoning = reasoning_with(&graph, "statement", "neutral", "general");
    let config = WalkConfig { target_year: None, depth_limit: 1 };

    // "world" has no outgoing edges
    let next = predict_next("world", &graph, None, &reasoning, &config);
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
    let config = WalkConfig { target_year: None, depth_limit: 1 };

    let next = predict_next("bank", &graph, None, &reasoning, &config);
    assert_eq!(next, Some("closes"), "intent-biased edge should win");
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
    let config = WalkConfig { target_year: None, depth_limit: 1 };

    let next = predict_next("data", &graph, None, &reasoning, &config);
    assert_eq!(next, Some("server"), "domain-biased edge should win");
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
    let config = WalkConfig { target_year: Some(2026), depth_limit: 1 };

    let next = predict_next("status", &graph, None, &reasoning, &config);
    assert_eq!(next, Some("online"), "temporally closer edge should win");
}

// ---------------------------------------------------------------------------
// predict_next — OOV lexical fallback
// ---------------------------------------------------------------------------

#[test]
fn test_predict_next_oov_snaps_to_graph_and_continues() {
    // Graph only contains known word "server"
    let graph = build_chain(&["server", "online"], "statement", "neutral", "tech", None, None);
    let reasoning = reasoning_with(&graph, "statement", "neutral", "tech");
    let config = WalkConfig { target_year: None, depth_limit: 1 };

    // "srvr" is not in the graph — OOV fallback should snap to nearest node and return Some
    let result = predict_next("srvr", &graph, None, &reasoning, &config);
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
    let config = WalkConfig { target_year: None, depth_limit: 1 };

    let start = resolve_start_node("server", &graph, &reasoning, &config);
    // "The" is the first word after the sentence boundary "."
    assert_eq!(start, Some("The"));
}

#[test]
fn test_resolve_start_node_returns_none_for_unknown_entity() {
    let graph = build_chain(&["hello", "world"], "statement", "neutral", "general", None, None);
    let reasoning = reasoning_with(&graph, "statement", "neutral", "general");
    let config = WalkConfig { target_year: None, depth_limit: 1 };

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
    let config = WalkConfig { target_year: None, depth_limit: 1 };

    let next = predict_next("orphan", &graph, Some(&spatial), &reasoning, &config);
    assert_eq!(next, Some("result"), "Tier 2 should route via anchor's edge to result");
}

#[test]
fn test_predict_next_tier2_not_triggered_when_tier1_succeeds() {
    let graph = build_chain(&["hello", "world"], "statement", "neutral", "general", None, None);
    let spatial = SpatialGrid::build(graph.nodes.values().map(|n| (n.id, n.position)));
    let reasoning = reasoning_with(&graph, "statement", "neutral", "general");
    let config = WalkConfig { target_year: None, depth_limit: 1 };

    let next = predict_next("hello", &graph, Some(&spatial), &reasoning, &config);
    assert_eq!(next, Some("world"), "Tier 1 direct edge should win when it exists");
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
    let config = WalkConfig { target_year: Some(2026), depth_limit: 1 };

    let result = resolve_start_node("server", &graph, &reasoning, &config);
    assert_eq!(
        result,
        Some("new_start"),
        "with target_year=2026 the 2026-dated incoming edge should win"
    );
}
