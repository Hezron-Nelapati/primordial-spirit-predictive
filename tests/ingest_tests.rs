use spse_predictive::graph::WordGraph;
use spse_predictive::ingest::{ingest_sentence, ingest_text, GraphStats};

// ---------------------------------------------------------------------------
// ingest_sentence
// ---------------------------------------------------------------------------

#[test]
fn test_ingest_sentence_creates_correct_node_count() {
    let mut graph = WordGraph::new();
    ingest_sentence(&mut graph, "hello world how are you", 1.0);
    assert_eq!(graph.nodes.len(), 5);
}

#[test]
fn test_ingest_sentence_creates_correct_edge_count() {
    let mut graph = WordGraph::new();
    ingest_sentence(&mut graph, "hello world how are you", 1.0);
    // 5 tokens → 4 consecutive edges
    assert_eq!(graph.edges.len(), 4);
}

#[test]
fn test_ingest_sentence_registers_surface_lookup() {
    let mut graph = WordGraph::new();
    ingest_sentence(&mut graph, "hello world", 1.0);
    assert!(graph.by_surface.contains_key("hello"));
    assert!(graph.by_surface.contains_key("world"));
}

#[test]
fn test_ingest_sentence_edge_default_metadata() {
    let mut graph = WordGraph::new();
    ingest_sentence(&mut graph, "alpha beta", 0.5);
    let edge = &graph.edges[0];
    assert_eq!(edge.weight, 0.5);
    assert_eq!(edge.intent, "statement");
    assert_eq!(edge.tone, "neutral");
    assert_eq!(edge.domain, "general");
    assert!(edge.entity.is_none());
    assert!(edge.dated.is_none());
}

#[test]
fn test_ingest_sentence_empty_input_creates_nothing() {
    let mut graph = WordGraph::new();
    ingest_sentence(&mut graph, "", 1.0);
    assert!(graph.nodes.is_empty());
    assert!(graph.edges.is_empty());
}

#[test]
fn test_ingest_sentence_single_token_no_edges() {
    let mut graph = WordGraph::new();
    ingest_sentence(&mut graph, "solo", 1.0);
    assert_eq!(graph.nodes.len(), 1);
    assert_eq!(graph.edges.len(), 0);
}

/// Ingesting the same sentence twice must reinforce the existing edge (weight += delta)
/// rather than pushing a duplicate edge.
#[test]
fn test_ingest_sentence_reinforces_existing_edge() {
    let mut graph = WordGraph::new();
    ingest_sentence(&mut graph, "hello world", 1.0);
    ingest_sentence(&mut graph, "hello world", 0.5);

    assert_eq!(graph.nodes.len(), 2, "no duplicate nodes");
    assert_eq!(graph.edges.len(), 1, "no duplicate edges");
    assert!(
        (graph.edges[0].weight - 1.5).abs() < f32::EPSILON,
        "weight should be 1.0 + 0.5 = 1.5, got {}",
        graph.edges[0].weight
    );
}

#[test]
fn test_ingest_sentence_increments_node_frequency() {
    let mut graph = WordGraph::new();
    ingest_sentence(&mut graph, "hello hello hello", 1.0);
    let hello_id = WordGraph::generate_id("hello");
    let freq = graph.nodes[&hello_id].frequency;
    assert_eq!(freq, 3);
}

// ---------------------------------------------------------------------------
// ingest_text
// ---------------------------------------------------------------------------

#[test]
fn test_ingest_text_processes_multiple_lines() {
    let mut graph = WordGraph::new();
    ingest_text(&mut graph, "hello world\nhow are you", 1.0);
    // 5 unique tokens across 2 lines
    assert_eq!(graph.nodes.len(), 5);
    // 1 edge (hello→world) + 2 edges (how→are, are→you) = 3
    assert_eq!(graph.edges.len(), 3);
}

#[test]
fn test_ingest_text_skips_empty_lines() {
    let mut graph = WordGraph::new();
    ingest_text(&mut graph, "hello world\n\n\nhow are you", 1.0);
    assert_eq!(graph.nodes.len(), 5);
}

#[test]
fn test_ingest_text_empty_string_creates_nothing() {
    let mut graph = WordGraph::new();
    ingest_text(&mut graph, "", 1.0);
    assert!(graph.nodes.is_empty());
    assert!(graph.edges.is_empty());
}

// ---------------------------------------------------------------------------
// GraphStats
// ---------------------------------------------------------------------------

#[test]
fn test_graph_stats_correct_values() {
    let mut graph = WordGraph::new();
    ingest_sentence(&mut graph, "a b c d", 1.0);

    let stats = GraphStats::compute(&graph);
    assert_eq!(stats.node_count, 4);
    assert_eq!(stats.edge_count, 3);
    // avg_out_degree = 3 edges / 4 nodes = 0.75
    assert!((stats.avg_out_degree - 0.75).abs() < 1e-5);
}

#[test]
fn test_graph_stats_empty_graph() {
    let graph = WordGraph::new();
    let stats = GraphStats::compute(&graph);
    assert_eq!(stats.node_count, 0);
    assert_eq!(stats.edge_count, 0);
    assert_eq!(stats.avg_out_degree, 0.0);
}
