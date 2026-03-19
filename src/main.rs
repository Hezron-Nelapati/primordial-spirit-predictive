use spse_predictive::db::GraphDb;
use spse_predictive::graph::GraphAccess;
use spse_predictive::ingest::{ingest_to_db, CorpusRow};
use spse_predictive::reasoning::{ReasoningModule, SessionalMemory};
use spse_predictive::spatial::SpatialGrid;
use spse_predictive::reasoning::{evaluate_arithmetic, extract_year_from_query, is_arithmetic_query};
use spse_predictive::walk::{compute_depth_limit, is_reachable, predict_next, secondary_signal, WalkConfig, WalkMode};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::{self, BufRead, Write};
use std::process::Command;

/// Classify `query` using the trained centroid model via `python/classify_query.py`.
/// Only called in CLI mode — server mode receives pre-classified data from Flask.
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
/// conversational stylistic reformatting. Only called in CLI mode.
fn llm_style(fact: &str, query: &str) -> Option<String> {
    let output = Command::new("python3")
        .args(["python/minillm_wrapper.py", fact, query])
        .output()
        .ok()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if let Some(rest) = line.strip_prefix("🤖 BOT:") {
                return Some(rest.trim().to_string());
            }
        }
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
    eprintln!("\n[USER_QUERY]: \"{}\"", query);

    // Guardrail 3: Pre-execution sanitization pass on the accumulated intent queue
    let sanitized = ReasoningModule::sanitize_queue(reasoning.session.intent_stack.clone());
    let effective_intent = sanitized.last().map(String::as_str).unwrap_or("statement");
    eprintln!("  [GUARDRAIL_3]: Sanitized intent queue -> {:?}  (effective: '{}')", sanitized, effective_intent);

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

    eprintln!("  [SYS_ORCHESTRATOR]: Geometrically reverse-walked to sentence anchor: [{}]", start_node);

    let mut output = start_node.clone();
    let mut current = start_node;
    let mut sentence_count = 0;
    // Rolling position window (last 5 visited nodes) for Tier 2 centroid search.
    // VecDeque gives O(1) front-removal vs Vec's O(N) shift.
    const POS_WINDOW: usize = 5;
    let mut pos_history: VecDeque<[f32; 3]> = VecDeque::with_capacity(POS_WINDOW + 1);

    for _ in 0..50 {
        // predict_next expects a slice; convert deque to a temporary vec for the call.
        let pos_slice: Vec<[f32; 3]> = pos_history.iter().copied().collect();
        if let Some(next_word) = predict_next(&current, graph, spatial, reasoning, config, &pos_slice) {
            // Record position of current node before advancing.
            if let Some(id) = graph.surface_to_id(&current) {
                if let Some(node) = graph.node_by_id(id) {
                    if pos_history.len() >= POS_WINDOW { pos_history.pop_front(); }
                    pos_history.push_back(node.position);
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

// ── Server mode structs ────────────────────────────────────────────────────────

/// Incoming JSON request from Flask (written to Rust's stdin).
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct ServerRequest {
    query:      String,
    entity:     String,
    domain:     String,
    intent:     String,
    tone:       String,
    entities:   Vec<String>,
    year:       Option<u16>,
    session_id: String,
}

/// Outgoing JSON response to Flask (written to Rust's stdout).
#[derive(serde::Serialize)]
struct ServerResponse {
    answer: String,
    error:  bool,
}

/// Handle a single classified request in server mode.
/// Accepts the caller's session state and returns the updated session so
/// `run_server_mode` can persist it across requests for the same session_id.
fn handle_server_request(
    req: &ServerRequest,
    db: &GraphDb,
    spatial: &SpatialGrid,
    session: SessionalMemory,
) -> (String, SessionalMemory) {
    eprintln!(
        "\n[SERVER_REQUEST]: \"{}\"  entity='{}' domain='{}' intent='{}' tone='{}'",
        req.query, req.entity, req.domain, req.intent, req.tone
    );

    // Arithmetic interception — bypass graph walk; session unchanged.
    if is_arithmetic_query(&req.query) {
        eprintln!("  [GUARDRAIL_6]: Arithmetic query — computing natively.");
        let answer = evaluate_arithmetic(&req.query)
            .unwrap_or_else(|| "I couldn't parse that arithmetic expression.".to_string());
        return (answer, session);
    }

    let depth = compute_depth_limit(&req.entity, db);
    eprintln!("  [TOPOLOGY]: Entity '{}' → depth_limit={}", req.entity, depth);

    let mut all_entities: Vec<String> = vec![req.entity.clone()];
    all_entities.extend(req.entities.clone());

    // Restore the caller's session so multi-turn context accumulates correctly.
    let mut reasoning = ReasoningModule::with_session(db, session);
    reasoning.update_context(&req.intent, &req.tone, &req.domain, &all_entities);
    let config = WalkConfig {
        target_year: req.year,
        depth_limit: depth,
        mode: WalkMode::from_intent(&req.intent),
    };

    // Multi-signal reachability guard — structural hallucination prevention.
    if let Some(secondary) = secondary_signal(&req.query, &req.entity, db) {
        eprintln!(
            "  [SYS_ORCHESTRATOR]: Secondary signal -> '{}'. BFS reachability check...",
            secondary
        );
        if !is_reachable(&req.entity, &secondary, db, 10) {
            let answer = format!(
                "System Fault: [{}] is not topologically reachable from [{}]. Structural Abort.",
                secondary, req.entity
            );
            let updated_session = reasoning.session;
            return (answer, updated_session);
        }
    }

    let answer = generate_dynamic_answer(&req.query, &req.entity, db, Some(spatial), &mut reasoning, &config);
    let updated_session = reasoning.session;
    (answer, updated_session)
}

/// Run the persistent server loop: signal READY, then process one JSON request
/// per stdin line and write one JSON response per line to stdout.
/// Never returns — exits when stdin closes.
fn run_server_mode(db: GraphDb, spatial: SpatialGrid) -> ! {
    // Signal readiness to Flask (stdout only — stderr is for diagnostics).
    {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        writeln!(out, "READY").unwrap();
        out.flush().unwrap();
    }

    // Session map: persists ReasoningModule session state across requests.
    // Key = session_id string; value = the accumulated SessionalMemory.
    // Fix #7: cap at SESSION_CAP entries to prevent unbounded memory growth.
    // Simple eviction: remove the first (arbitrary) key when cap is exceeded.
    const SESSION_CAP: usize = 1024;
    let mut sessions: HashMap<String, SessionalMemory> = HashMap::new();

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let resp = match serde_json::from_str::<ServerRequest>(&line) {
            Ok(req) => {
                let prior_session = sessions.remove(&req.session_id).unwrap_or_else(SessionalMemory::new);
                let (answer, updated_session) = handle_server_request(&req, &db, &spatial, prior_session);
                // Evict one entry when the cap is reached to bound memory use.
                if sessions.len() >= SESSION_CAP {
                    if let Some(oldest_key) = sessions.keys().next().cloned() {
                        sessions.remove(&oldest_key);
                    }
                }
                sessions.insert(req.session_id.clone(), updated_session);
                let error = answer.starts_with("System Fault");
                ServerResponse { answer, error }
            }
            Err(e) => ServerResponse {
                answer: format!("Server parse error: {}", e),
                error:  true,
            },
        };

        let stdout = io::stdout();
        let mut out = stdout.lock();
        writeln!(out, "{}", serde_json::to_string(&resp).unwrap()).unwrap();
        out.flush().unwrap();
    }

    std::process::exit(0);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // --server flag enables persistent IPC mode (used by Flask web UI).
    let server_mode = args.iter().any(|a| a == "--server");

    if !server_mode {
        // Count only positional args.
        // Fix #6: distinguish flags that consume a value argument (e.g. --session-id)
        // from flags that stand alone (e.g. --server). A blind skip_next for every
        // "--" prefix was incorrectly treating --server as a valued flag.
        const VALUED_FLAGS: &[&str] = &["--session-id"];
        let positional_count = {
            let mut count = 0usize;
            let mut skip_next = false;
            for a in args.iter().skip(1) {
                if skip_next { skip_next = false; continue; }
                if a.starts_with("--") {
                    if VALUED_FLAGS.contains(&a.as_str()) { skip_next = true; }
                } else {
                    count += 1;
                }
            }
            count
        };

        if positional_count < 3 {
            eprintln!("Usage: cargo run -- \"<query>\" <entity> <domain> [year] [--session-id ID]");
            eprintln!("       cargo run -- --server   (persistent JSON-IPC mode for Flask)");
            std::process::exit(1);
        }
    }

    // Extract optional --session-id flag (CLI mode only)
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

    // ── Database connection ────────────────────────────────────────────────────
    if !std::path::Path::new("data").is_dir() {
        eprintln!("[FATAL] data/ directory not found. Run from the project root.");
        std::process::exit(1);
    }

    eprintln!("[DB] Connecting to data/graph.db ...");
    let db = match GraphDb::open("data/graph.db") {
        Ok(db) => {
            eprintln!("[DB] Connection established.");
            db
        }
        Err(e) => {
            eprintln!("[FATAL] Database connection failed: {}", e);
            eprintln!("[FATAL] The data/ directory may not be writable or the DB is locked.");
            std::process::exit(1);
        }
    };

    // ── First-run corpus ingest ────────────────────────────────────────────────
    // Skip when the DB already has nodes (a previous run built it).
    if db.node_count() == 0 {
        eprintln!("[GRAPH] Empty DB — running first-run ingest...");
        // Prefer the reinforced corpus (written by train_pipeline.py) which
        // already merges all source corpora and applies edge-weight passes.
        // Fall back to the single-pass corpus.json when training hasn't run yet.
        let corpus_path = if fs::metadata("data/corpus_reinforced.json").is_ok() {
            eprintln!("[GRAPH] Using reinforced corpus: data/corpus_reinforced.json");
            "data/corpus_reinforced.json"
        } else {
            eprintln!("[GRAPH] Using single-pass corpus: data/corpus.json");
            "data/corpus.json"
        };
        let raw_data = match fs::read_to_string(corpus_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[FATAL] {} missing or unreadable: {}", corpus_path, e);
                eprintln!("[FATAL] Run the training pipeline first: python3 python/train_pipeline.py");
                std::process::exit(1);
            }
        };
        let rows: Vec<CorpusRow> = match serde_json::from_str(&raw_data) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[FATAL] {} parse error: {}", corpus_path, e);
                std::process::exit(1);
            }
        };
        let count = rows.len();
        ingest_to_db(&db, rows);
        eprintln!("[GRAPH] First-run ingest: {} sequences from {} → DB built.", count, corpus_path);
    }

    // Build 3D spatial index from all node positions in the DB.
    let spatial = SpatialGrid::build(db.all_nodes().into_iter().map(|n| (n.id, n.position)));
    eprintln!("[GRAPH] {} nodes / {} edges | Spatial index ready.", db.node_count(), db.edge_count());

    // ── Server mode ────────────────────────────────────────────────────────────
    if server_mode {
        run_server_mode(db, spatial);
        // run_server_mode never returns (exits on stdin close)
    }

    // ── CLI query mode ─────────────────────────────────────────────────────────
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
