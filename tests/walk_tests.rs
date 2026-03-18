use spse_predictive::graph::WordGraph;
use spse_predictive::spatial::SpatialGrid;
use spse_predictive::walk::{predict_next, walk, Tier};
use spse_predictive::ingest::{ingest_sentence, ingest_text, measure_tier_distribution};

#[test]
fn test_greeting_hits_tier1_after_training() {
    let mut graph = WordGraph::default();
    for _ in 0..100 {
        ingest_sentence(&mut graph, "hi hello how are you", 0.05);
    }
    let spatial = SpatialGrid::build(
        graph.nodes.values().map(|n| (n.id, n.position))
    );
    let hi_id   = *graph.by_surface.get("hi").unwrap();
    let result  = predict_next(&graph, &spatial, hi_id).unwrap();
    assert_eq!(result.tier, Tier::One);
    assert!(result.confidence >= 0.60);
}

#[test]
fn test_tier3_fallback_never_panics() {
    let mut graph = WordGraph::default();
    ingest_sentence(&mut graph, "rare obscure word", 0.01);
    let spatial = SpatialGrid::build(
        graph.nodes.values().map(|n| (n.id, n.position))
    );
    let id = *graph.by_surface.get("rare").unwrap();
    let result = predict_next(&graph, &spatial, id);
    assert!(result.is_some());
}

#[test]
fn test_tier_distribution_improves_with_training() {
    let mut graph = WordGraph::default();
    let corpus = std::fs::read_to_string("data/corpus.txt").unwrap();

    ingest_text(&mut graph, &corpus, 0.005);
    let spatial = SpatialGrid::build(
        graph.nodes.values().map(|n| (n.id, n.position))
    );
    let stats_before = measure_tier_distribution(&graph, &spatial, 1000);

    ingest_text(&mut graph, &corpus, 0.005);
    let spatial2 = SpatialGrid::build(
        graph.nodes.values().map(|n| (n.id, n.position))
    );
    let stats_after = measure_tier_distribution(&graph, &spatial2, 1000);

    // After more training, edge weights increase, pushing more into Tier 1 threshold.
    assert!(stats_after.t1 >= stats_before.t1);
}

#[test]
fn test_walk_produces_coherent_output() {
    let mut graph = WordGraph::default();
    for _ in 0..200 {
        ingest_sentence(&mut graph, "hello how are you doing today", 0.05);
    }
    let spatial = SpatialGrid::build(
        graph.nodes.values().map(|n| (n.id, n.position))
    );
    let start  = *graph.by_surface.get("hello").unwrap();
    let path   = walk(&graph, &spatial, start, 5);
    
    let output: Vec<&str> = path.iter()
        .map(|id| graph.nodes[id].surface.as_str())
        .collect();
    println!("Output: {}", output.join(" "));
    
    assert!(output.len() >= 3);
}
