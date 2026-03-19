use spse_predictive::db::GraphDb;
use spse_predictive::graph::GraphAccess;
use spse_predictive::ingest::{ingest_v2_to_db, V2JsonData};
use spse_predictive::reasoning::ReasoningModule;
use spse_predictive::spatial::SpatialGrid;
use spse_predictive::reasoning::{evaluate_arithmetic, extract_year_from_query, is_arithmetic_query};
use spse_predictive::walk::{compute_depth_limit, is_reachable, predict_next, secondary_signal, WalkConfig, WalkMode};
use std::fs;
use std::process::Command;

/// Classify `query` using the trained centroid model via `python/classify_query.py`.
/// Returns `(intent, tone, domain, entities)` or `None` if Python / the model is unavailable.
fn classify_query(query: &str, session_id: &str) -> Option<(String, String, String, Vec<String>)> {
    let output = Command::new("python3")
        .args(["python/classify_query.py", query, "--session-id", session_id])
        .output()
        .ok()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let v: serde_json::Value = serde_json::from_str(stdout.trim()).ok()?;
        let intent = v["intent"].as_str()?.to_string();
        let tone   = v["tone"].as_str()?.to_string();
        let domain = v["domain"].as_str()?.to_string();
        let entities: Vec<String> = v["entities"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|e| e.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        Some((intent, tone, domain, entities))
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

use spse_predictive::walk::resolve_start_node;

fn generate_dynamic_answer(
    query: &str,
    entity: &str,
    graph: &dyn GraphAccess,
    spatial: Option<&SpatialGrid>,
    reasoning: &mut ReasoningModule<'_>,
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
    let start_node = raw_start.unwrap_or_else(|| actual_entity.to_string());

    println!("  [SYS_ORCHESTRATOR]: Geometrically reverse-walked to sentence anchor: [{}]", start_node);

    let mut output = start_node.clone();
    let mut current = start_node;
    let mut sentence_count = 0;
    // Rolling position window (last 5 visited nodes) for Tier 2 centroid search.
    const POS_WINDOW: usize = 5;
    let mut pos_history: Vec<[f32; 3]> = Vec::with_capacity(POS_WINDOW);

    for _ in 0..50 {
        if let Some(next_word) = predict_next(&current, graph, spatial, reasoning, config, &pos_history) {
            // Record position of current node before advancing.
            if let Some(id) = graph.surface_to_id(&current) {
                if let Some(node) = graph.node_by_id(id) {
                    pos_history.push(node.position);
                    if pos_history.len() > POS_WINDOW { pos_history.remove(0); }
                }
            }
            if next_word == "." || next_word == "?" || next_word == "!" {
                output.push_str(&next_word);
                sentence_count += 1;
                if sentence_count >= config.depth_limit { break; }
            } else if next_word == "," {
                output.push_str(&next_word);
            } else {
                output.push(' ');
                output.push_str(&next_word);
            }
            current = next_word;
        } else {
            break;
        }
    }
    output
}


fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Count only positional args (ignore --flag pairs)
    let positional_count = {
        let mut count = 0usize;
        let mut skip_next = false;
        for a in args.iter().skip(1) {
            if skip_next { skip_next = false; continue; }
            if a.starts_with("--") { skip_next = true; } else { count += 1; }
        }
        count
    };

    if positional_count < 3 {
        eprintln!("Usage: cargo run -- \"<query>\" <entity> <domain> [year] [--session-id ID]");
        eprintln!("Example: cargo run -- \"Is the server online?\" server tech 2026");
        std::process::exit(1);
    }

    // Extract optional --session-id flag (can appear anywhere after positional args)
    let session_id: String = {
        let mut sid = std::process::id().to_string();
        let mut i = 1usize;
        while i < args.len() {
            if args[i] == "--session-id" && i + 1 < args.len() {
                sid = args[i + 1].clone();
                break;
            }
            i += 1;
        }
        sid
    };

    // -----------------------------------------------------------------------
    // Database connection
    // Verify the data/ directory is accessible before attempting to open.
    // GraphDb::open sets PRAGMA busy_timeout=5000, so SQLite will retry for
    // up to 5 s if another process holds a write lock, then return SQLITE_BUSY
    // as an Err — which we surface as a fatal exit rather than a panic.
    // -----------------------------------------------------------------------
    if !std::path::Path::new("data").is_dir() {
        eprintln!("[FATAL] data/ directory not found. Run from the project root.");
        std::process::exit(1);
    }

    println!("[DB] Connecting to data/graph.db ...");
    let db = match GraphDb::open("data/graph.db") {
        Ok(db) => {
            println!("[DB] Connection established.");
            db
        }
        Err(e) => {
            eprintln!("[FATAL] Database connection failed: {}", e);
            eprintln!("[FATAL] The database may be locked (busy_timeout=5 s exceeded) or the");
            eprintln!("[FATAL] data/ directory may not be writable. Exiting.");
            std::process::exit(1);
        }
    };

    // -----------------------------------------------------------------------
    // First-run corpus ingest
    // Skip when the DB already has nodes (a previous run built it).
    // -----------------------------------------------------------------------
    if db.node_count() == 0 {
        let v2_data = match fs::read_to_string("data/v2_corpus.json") {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[FATAL] data/v2_corpus.json missing or unreadable: {}", e);
                eprintln!("[FATAL] Run `cd python && python3 v2_ingest.py` first.");
                std::process::exit(1);
            }
        };
        let v2_rows: Vec<V2JsonData> = match serde_json::from_str(&v2_data) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[FATAL] v2_corpus.json parse error: {}", e);
                std::process::exit(1);
            }
        };
        let v2_count = v2_rows.len();
        ingest_v2_to_db(&db, v2_rows);

        let v3_path = "data/v3_corpus.json";
        let v3_msg = if fs::metadata(v3_path).is_ok() {
            match fs::read_to_string(v3_path)
                .map_err(|e| e.to_string())
                .and_then(|d| serde_json::from_str::<Vec<V2JsonData>>(&d).map_err(|e| e.to_string()))
            {
                Ok(v3_rows) => {
                    let n = v3_rows.len();
                    ingest_v2_to_db(&db, v3_rows);
                    format!(" + V3 ({} sequences)", n)
                }
                Err(e) => {
                    eprintln!("[WARN] Could not load {}: {}", v3_path, e);
                    String::new()
                }
            }
        } else {
            String::new()
        };
        println!("[GRAPH] First-run ingest: V2 ({} sequences){} → DB built.", v2_count, v3_msg);
    }

    // Build 3D spatial index from all node positions in the DB.
    let spatial = SpatialGrid::build(db.all_nodes().into_iter().map(|n| (n.id, n.position)));
    println!("[GRAPH] {} nodes / {} edges | Spatial index ready.", db.node_count(), db.edge_count());

    // -----------------------------------------------------------------------
    // CLI query
    // -----------------------------------------------------------------------
    let query  = &args[1];
    let entity = &args[2];
    let domain = &args[3];
    let year: Option<u16> = args.get(4)
        .filter(|a| !a.starts_with("--"))
        .and_then(|y| y.parse().ok())
        .or_else(|| extract_year_from_query(query));

    println!("\n=========== 💬 CLI QUERY MODE 💬 ===========");
    println!("  Query  : {}", query);
    println!("  Entity : {}", entity);
    println!("  Year   : {:?}", year);

    // Guardrail 6: Arithmetic interception — bypass graph walk entirely.
    if is_arithmetic_query(query) {
        println!("\n  [GUARDRAIL_6]: Arithmetic/logic query detected — computing natively.");
        let raw_answer = evaluate_arithmetic(query)
            .unwrap_or_else(|| "I couldn't parse that arithmetic expression.".to_string());
        println!("\n  [SYS_ORCHESTRATOR]: Piping arithmetic result to miniLLM stylistic wrapper...");
        match llm_style(&raw_answer, query) {
            Some(styled) => println!("  -> [BOT_OUTPUT]: \"{}\"", styled),
            None => {
                println!("  [SYS_ORCHESTRATOR]: miniLLM unavailable — returning raw result.");
                println!("  -> [BOT_OUTPUT]: \"{}\"", raw_answer);
            }
        }
        println!("\n=============================================\n");
        return;
    }

    // Live ML classification; fall back to CLI-supplied domain on failure.
    println!("\n  [CLASSIFIER]: Running centroid-based intent/tone/domain classification...");
    let (intent, tone, effective_domain, ner_entities) = match classify_query(query, &session_id) {
        Some((i, t, d, e)) => {
            println!("  [CLASSIFIER]: intent='{}' tone='{}' domain='{}' entities={:?}", i, t, d, e);
            (i, t, d, e)
        }
        None => {
            println!("  [CLASSIFIER]: Unavailable — falling back to defaults (domain: '{}').", domain);
            ("question".to_string(), "neutral".to_string(), domain.to_string(), vec![])
        }
    };

    let depth = compute_depth_limit(entity, &db);
    println!("  [TOPOLOGY]: Entity '{}' → depth_limit={}", entity, depth);

    let mut all_entities: Vec<String> = vec![entity.clone()];
    all_entities.extend(ner_entities);

    let mut reasoning = ReasoningModule::new(&db);
    reasoning.update_context(&intent, &tone, &effective_domain, &all_entities);
    let config = WalkConfig { target_year: year, depth_limit: depth, mode: WalkMode::from_intent(&intent) };

    // Multi-signal reachability guard — structural hallucination prevention.
    if let Some(secondary) = secondary_signal(query, entity, &db) {
        println!("\n  [SYS_ORCHESTRATOR]: Secondary prompt signal detected -> '{}'.", secondary);
        println!("  [SYS_ORCHESTRATOR]: BFS reachability trace from [{}] to [{}] (max 10 hops)...", entity, &secondary);
        if !is_reachable(entity, &secondary, &db, 10) {
            println!("  -> [BOT_OUTPUT]: System Fault: [{}] is not topologically reachable from [{}]. Structural Abort.", secondary, entity);
            println!("\n=============================================\n");
            return;
        }
    }

    let graph_fact = generate_dynamic_answer(query, entity, &db, Some(&spatial), &mut reasoning, &config);

    println!("\n  [SYS_ORCHESTRATOR]: Piping graph fact to miniLLM stylistic wrapper...");
    match llm_style(&graph_fact, query) {
        Some(styled) => println!("  -> [BOT_OUTPUT]: \"{}\"", styled),
        None => {
            println!("  [SYS_ORCHESTRATOR]: miniLLM unavailable — returning raw graph fact.");
            println!("  -> [BOT_OUTPUT]: \"{}\"", graph_fact);
        }
    }
    println!("\n=============================================\n");
}
