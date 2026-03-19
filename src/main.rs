use spse_predictive::graph::{WordGraph, WordNode, WordEdge};
use spse_predictive::reasoning::ReasoningModule;
use spse_predictive::spatial::SpatialGrid;
use spse_predictive::walk::{compute_depth_limit, is_reachable, predict_next, WalkConfig};
use serde::Deserialize;
use std::fs;
use std::process::Command;

/// Classify `query` using the trained centroid model via `python/classify_query.py`.
/// Returns `(intent, tone, domain)` or `None` if Python / the model is unavailable.
fn classify_query(query: &str) -> Option<(String, String, String)> {
    let output = Command::new("python3")
        .args(["python/classify_query.py", query])
        .output()
        .ok()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let v: serde_json::Value = serde_json::from_str(stdout.trim()).ok()?;
        let intent = v["intent"].as_str()?.to_string();
        let tone   = v["tone"].as_str()?.to_string();
        let domain = v["domain"].as_str()?.to_string();
        Some((intent, tone, domain))
    } else {
        None
    }
}

/// Pass the raw graph fact through `python/minillm_wrapper.py` for
/// conversational stylistic reformatting.  Returns `None` on any subprocess
/// or Python failure so callers can fall back to the raw fact gracefully.
fn llm_style(fact: &str, query: &str) -> Option<String> {
    let output = Command::new("python3")
        .args(["python/minillm_wrapper.py", fact, query])
        .output()
        .ok()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // The wrapper emits "🤖 BOT: <response>" — extract just the response.
        for line in stdout.lines() {
            if let Some(rest) = line.strip_prefix("🤖 BOT:") {
                return Some(rest.trim().to_string());
            }
        }
        // Fallback: return full stdout trimmed if the marker line is absent.
        Some(stdout.trim().to_string())
    } else {
        None
    }
}

#[derive(Deserialize, Debug)]
pub struct V2JsonData {
    pub text: String,
    pub tokens: Vec<String>,
    pub intent: String,
    pub tone: String,
    pub domain: String,
    pub entities: Vec<String>,
    pub dated: Option<u16>,
}

use spse_predictive::walk::resolve_start_node;

fn generate_dynamic_answer(
    query: &str,
    entity: &str,
    graph: &WordGraph,
    spatial: Option<&SpatialGrid>,
    reasoning: &mut ReasoningModule,
    config: &WalkConfig,
) -> String {
    println!("\n[USER_QUERY]: \"{}\"", query);

    // Guardrail 3: Pre-execution sanitization pass on the accumulated intent queue
    let sanitized = ReasoningModule::sanitize_queue(reasoning.session.intent_stack.clone());
    let effective_intent = sanitized.last().map(String::as_str).unwrap_or("statement");
    println!("  [GUARDRAIL_3]: Sanitized intent queue -> {:?}  (effective: '{}')", sanitized, effective_intent);

    // Dynamic Entry Node Resolution Pipeline
    let actual_entity = if entity == "Pronoun" {
        reasoning.session.entity_stack.last().map(String::as_str).unwrap_or("?")
    } else {
        entity
    };

    if actual_entity == "?" {
        return "System Fault: Entity Stack is empty.".to_string();
    }

    // Execute Reverse Walk to find absolute topological start of fact
    let raw_start = resolve_start_node(actual_entity, graph, reasoning, config);
    let start_node = raw_start.unwrap_or(actual_entity);

    println!("  [SYS_ORCHESTRATOR]: Geometrically reverse-walked to sentence anchor: [{}]", start_node);

    let mut output = start_node.to_string();
    let mut current = start_node.to_string();
    let mut sentence_count = 0;

    for _ in 0..50 {
        if let Some(next_word) = predict_next(&current, graph, spatial, reasoning, config) {
            if next_word == "." || next_word == "?" || next_word == "!" {
                output.push_str(next_word);
                sentence_count += 1;
                if sentence_count >= config.depth_limit { break; }
            } else if next_word == "," {
                output.push_str(next_word);
            } else {
                output.push(' ');
                output.push_str(next_word);
            }
            current = next_word.to_string();
        } else {
            break;
        }
    }
    output
}

/// Ingest a slice of deserialised JSON rows into an existing graph.
/// Shared surface tokens accumulate edge weight from every corpus they appear in.
fn ingest_rows(graph: &mut WordGraph, rows: Vec<V2JsonData>) {
    for row in rows {
        let mut prev_id = None;
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
                graph.edges.push(WordEdge {
                    from: prev,
                    to: id,
                    weight: 1.0,
                    intent: row.intent.clone(),
                    tone: row.tone.clone(),
                    domain: row.domain.clone(),
                    entity: row.entities.first().cloned(),
                    dated: row.dated,
                });
            }
            prev_id = Some(id);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut graph = WordGraph::new();

    // Always load V2 (required baseline corpus).
    let v2_data = fs::read_to_string("data/v2_graph_edges.json").expect("data/v2_graph_edges.json missing — run python/v2_ingest.py first");
    let v2_rows: Vec<V2JsonData> = serde_json::from_str(&v2_data).expect("v2_graph_edges.json format error");
    let v2_count = v2_rows.len();
    ingest_rows(&mut graph, v2_rows);

    // Merge V3 when available (CLI mode benefits from the wider corpus).
    let v3_path = "data/v3_graph_edges.json";
    let v3_loaded = if fs::metadata(v3_path).is_ok() {
        let result: Result<Vec<V2JsonData>, String> = fs::read_to_string(v3_path)
            .map_err(|e| e.to_string())
            .and_then(|d| serde_json::from_str(&d).map_err(|e| e.to_string()));
        match result {
            Ok(v3_rows) => {
                let count = v3_rows.len();
                ingest_rows(&mut graph, v3_rows);
                Some(count)
            }
            Err(e) => {
                eprintln!("[WARN] Could not load {}: {}", v3_path, e);
                None
            }
        }
    } else {
        None
    };

    // Build 3D spatial index from populated node positions.
    let spatial = SpatialGrid::build(graph.nodes.values().map(|n| (n.id, n.position)));

    println!(
        "[GRAPH] Loaded V2 ({} sequences){} → {} nodes / {} edges | Spatial index built.",
        v2_count,
        v3_loaded.map(|n| format!(" + V3 ({} sequences)", n)).unwrap_or_default(),
        graph.nodes.len(),
        graph.edges.len(),
    );

    // CLI mode: cargo run -- "query" entity domain [year]
    if args.len() >= 4 {
        let query  = &args[1];
        let entity = &args[2];
        let domain = &args[3];
        let year: Option<u16> = args.get(4).and_then(|y| y.parse().ok());

        println!("\n=========== 💬 CLI QUERY MODE 💬 ===========");
        println!("  Query  : {}", query);
        println!("  Entity : {}", entity);
        println!("  Year   : {:?}", year);

        // Attempt live ML classification; fall back to user-supplied domain + safe defaults.
        println!("\n  [CLASSIFIER]: Running centroid-based intent/tone/domain classification...");
        let (intent, tone, effective_domain) = match classify_query(query) {
            Some((i, t, d)) => {
                println!("  [CLASSIFIER]: intent='{}' tone='{}' domain='{}'", i, t, d);
                (i, t, d)
            }
            None => {
                println!("  [CLASSIFIER]: Unavailable — falling back to defaults (domain: '{}').", domain);
                ("question".to_string(), "neutral".to_string(), domain.to_string())
            }
        };

        let depth = compute_depth_limit(entity, &graph);
        println!("  [TOPOLOGY]: Entity '{}' → {} reachable-2-hop nodes → depth_limit={}", entity, depth, depth);

        let mut reasoning = ReasoningModule::new(&graph);
        reasoning.update_context(&intent, &tone, &effective_domain, &[entity.clone()]);
        let config = WalkConfig { target_year: year, depth_limit: depth };
        let graph_fact = generate_dynamic_answer(query, entity, &graph, Some(&spatial), &mut reasoning, &config);

        // Attempt stylistic reformatting via miniLLM wrapper; fall back to raw graph fact.
        println!("\n  [SYS_ORCHESTRATOR]: Piping graph fact to miniLLM stylistic wrapper...");
        match llm_style(&graph_fact, query) {
            Some(styled) => println!("  -> [BOT_OUTPUT]: \"{}\"", styled),
            None => {
                println!("  [SYS_ORCHESTRATOR]: miniLLM unavailable — returning raw graph fact.");
                println!("  -> [BOT_OUTPUT]: \"{}\"", graph_fact);
            }
        }
        println!("\n=============================================\n");
        return;
    }

    println!("\n=========== 💬 DYNAMIC CHATBOT MVP 💬 ===========");
    let mut reasoning = ReasoningModule::new(&graph);
    
    // Test: Axiomatic Domain Tie Break (No hardcodes!)
    reasoning.update_context("question", "neutral", "tech", &["server".to_string()]);
    let conf1 = WalkConfig { target_year: Some(2026), depth_limit: 1 };
    let ans1 = generate_dynamic_answer("Are the servers online?", "server", &graph, None, &mut reasoning, &conf1);
    println!("  -> [BOT_OUTPUT]: \"{}\"", ans1);
    
    reasoning.update_context("question", "neutral", "tech", &["server".to_string()]);
    let conf2 = WalkConfig { target_year: Some(2020), depth_limit: 1 };
    let ans2 = generate_dynamic_answer("Was it offline back in 2020?", "server", &graph, None, &mut reasoning, &conf2);
    println!("  -> [BOT_OUTPUT]: \"{}\"", ans2);
    
    println!("\n[USER_QUERY]: \"When does the bank close?\"");
    // If the user asks a question, the target structural retrieval intent is a STATEMENT (Fact), not another question!
    reasoning.update_context("statement", "neutral", "finance", &["bank".to_string()]);
    let conf3 = WalkConfig { target_year: None, depth_limit: 1 };
    let ans3 = generate_dynamic_answer("When does the bank close?", "bank", &graph, None, &mut reasoning, &conf3);
    println!("  -> [BOT_OUTPUT]: \"{}\"", ans3);
    
    println!("\n[USER_QUERY]: \"Is there an ATM there?\"");
    reasoning.update_context("statement", "neutral", "finance", &["Pronoun".to_string()]);
    let conf4 = WalkConfig { target_year: None, depth_limit: 1 };
    
    // Multi-Signal validation: context entity is "bank" (from stack); secondary signal is "ATM".
    // BFS reachability check: is "ATM" reachable from "bank" within the graph topology?
    println!("  [SYS_ORCHESTRATOR]: Secondary prompt signal detected -> 'ATM'.");
    println!("  [SYS_ORCHESTRATOR]: BFS reachability trace from [bank] to [ATM] (max 10 hops)...");
    if !is_reachable("bank", "ATM", &graph, 10) {
        println!("  -> [BOT_OUTPUT]: System Fault: [ATM] is not topologically reachable from [bank]. Structural Abort to prevent hallucination!");
    } else {
        let ans4 = generate_dynamic_answer("Is there an ATM there?", "Pronoun", &graph, None, &mut reasoning, &conf4);
        println!("  -> [BOT_OUTPUT]: \"{}\"", ans4);
    }

    // Test: Topological Density limits
    reasoning.update_context("explain", "neutral", "science", &["quantum".to_string()]);
    let conf5 = WalkConfig { target_year: None, depth_limit: 3 };
    let ans5 = generate_dynamic_answer("Explain quantum mechanics.", "quantum", &graph, None, &mut reasoning, &conf5);
    println!("  -> [BOT_OUTPUT]: \"{}\"", ans5);
    
    println!("\n=================================================\n");
}
