use spse_predictive::graph::WordGraph;
use spse_predictive::ingest::{ingest_sentence, ingest_text, ingest_rows, GraphStats, CorpusRow};

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

// ---------------------------------------------------------------------------
// ingest_rows
// ---------------------------------------------------------------------------

fn make_row(tokens: &[&str], intent: &str, domain: &str, dated: Option<u16>) -> CorpusRow {
    CorpusRow {
        text:     tokens.join(" "),
        tokens:   tokens.iter().map(|t| t.to_string()).collect(),
        intent:   intent.to_string(),
        tone:     "neutral".to_string(),
        domain:   domain.to_string(),
        entities: vec![],
        dated,
    }
}

/// A single row with two tokens must create one node per token and one edge.
#[test]
fn test_ingest_rows_basic_node_and_edge_creation() {
    let mut graph = WordGraph::new();
    ingest_rows(&mut graph, vec![make_row(&["hello", "world"], "statement", "general", None)]);
    assert_eq!(graph.nodes.len(), 2);
    assert_eq!(graph.edges.len(), 1);
    assert_eq!(graph.edges[0].weight, 1.0);
}

/// Two rows with the same token pair AND the same (intent, domain, dated)
/// must reinforce the existing edge rather than creating a duplicate.
#[test]
fn test_ingest_rows_reinforces_identical_edges() {
    let mut graph = WordGraph::new();
    let rows = vec![
        make_row(&["server", "online"], "statement", "tech", None),
        make_row(&["server", "online"], "statement", "tech", None),
    ];
    ingest_rows(&mut graph, rows);
    assert_eq!(graph.edges.len(), 1, "identical rows must not duplicate the edge");
    assert!(
        (graph.edges[0].weight - 2.0).abs() < f32::EPSILON,
        "weight should be 1.0 + 1.0 = 2.0, got {}",
        graph.edges[0].weight
    );
}

/// Same token pair but different domains must produce two separate edges,
/// preserving the ability for multi-signal routing to choose between them.
#[test]
fn test_ingest_rows_different_domains_create_distinct_edges() {
    let mut graph = WordGraph::new();
    let rows = vec![
        make_row(&["bank", "open"], "statement", "finance", None),
        make_row(&["bank", "open"], "statement", "tech",    None),
    ];
    ingest_rows(&mut graph, rows);
    assert_eq!(graph.edges.len(), 2, "different domains must produce distinct edges");
}

/// Same token pair but different dated values must produce two separate edges
/// so temporal tie-breaking can discriminate between them.
#[test]
fn test_ingest_rows_different_dated_create_distinct_edges() {
    let mut graph = WordGraph::new();
    let rows = vec![
        make_row(&["server", "status"], "statement", "tech", Some(2020)),
        make_row(&["server", "status"], "statement", "tech", Some(2026)),
    ];
    ingest_rows(&mut graph, rows);
    assert_eq!(graph.edges.len(), 2, "different dated values must produce distinct edges");
}

/// Node positions must be non-zero after ingestion (lexical-vector density
/// coordinates are populated, not left as [0.0; 3]).
#[test]
fn test_ingest_rows_populates_node_positions() {
    let mut graph = WordGraph::new();
    ingest_rows(&mut graph, vec![make_row(&["quantum", "physics"], "explain", "science", None)]);
    for node in graph.nodes.values() {
        // length component (lv[0]) must be > 0 for non-empty words
        assert!(node.position[0] > 0.0, "position[0] (length) must be > 0 for '{}'", node.surface);
    }
}
