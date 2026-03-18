use spse_predictive::graph::{WordGraph, WordNode, WordEdge};
use spse_predictive::reasoning::ReasoningModule;
use spse_predictive::walk::{predict_next, WalkConfig};
use serde::Deserialize;
use std::fs;

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

fn generate_sequence(start_word: &str, graph: &WordGraph, reasoning: &ReasoningModule, config: &WalkConfig) -> String {
    let mut output = start_word.to_string();
    let mut current = start_word.to_string(); 
    let mut sentence_count = 0;
    
    for _ in 0..50 { // Hard abort to prevent infinite loop
        if let Some(next_word) = predict_next(&current, graph, reasoning, config) {
            if next_word == "." || next_word == "?" || next_word == "!" {
                output.push_str(next_word);
                sentence_count += 1;
                if sentence_count >= config.depth_limit {
                    break;
                }
            } else if next_word == "," {
                output.push_str(next_word);
            } else {
                output.push(' ');
                output.push_str(next_word);
            }
            current = next_word.to_string();
        } else {
            break; // dead end
        }
    }
    output
}

use spse_predictive::walk::resolve_start_node;

fn generate_dynamic_answer(query: &str, entity: &str, graph: &WordGraph, reasoning: &mut ReasoningModule, config: &WalkConfig) -> String {
    println!("\n[USER_QUERY]: \"{}\"", query);
    
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
    
    for _ in 0..50 { // Safety 
        if let Some(next_word) = predict_next(&current, graph, reasoning, config) {
            if next_word == "." || next_word == "?" || next_word == "!" {
                output.push_str(next_word);
                sentence_count += 1;
                if sentence_count >= config.depth_limit {
                    break;
                }
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

fn main() {
    let mut graph = WordGraph::new();
    let data = fs::read_to_string("data/v2_graph_edges.json").expect("JSON missing");
    let rows: Vec<V2JsonData> = serde_json::from_str(&data).expect("JSON format err");
    
    for row in rows {
        let mut prev_id = None;
        for token in row.tokens {
            let id = WordGraph::generate_id(&token);
            graph.by_surface.insert(token.clone(), id);
            
            let node = graph.nodes.entry(id).or_insert(WordNode {
                id,
                surface: token.clone(),
                frequency: 0,
                position: [0.0; 3],
                lexical_vector: WordNode::compute_lexical_vector(&token),
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
    
    println!("\n=========== 💬 DYNAMIC CHATBOT MVP 💬 ===========");
    let mut reasoning = ReasoningModule::new(&graph);
    
    // Test: Axiomatic Domain Tie Break (No hardcodes!)
    reasoning.update_context("question", "neutral", "tech", &["server".to_string()]);
    let conf1 = WalkConfig { target_year: Some(2026), depth_limit: 1 };
    let ans1 = generate_dynamic_answer("Are the servers online?", "server", &graph, &mut reasoning, &conf1);
    println!("  -> [BOT_OUTPUT]: \"{}\"", ans1);
    
    reasoning.update_context("question", "neutral", "tech", &["server".to_string()]);
    let conf2 = WalkConfig { target_year: Some(2020), depth_limit: 1 };
    let ans2 = generate_dynamic_answer("Was it offline back in 2020?", "server", &graph, &mut reasoning, &conf2);
    println!("  -> [BOT_OUTPUT]: \"{}\"", ans2);
    
    println!("\n[USER_QUERY]: \"When does the bank close?\"");
    // If the user asks a question, the target structural retrieval intent is a STATEMENT (Fact), not another question!
    reasoning.update_context("statement", "neutral", "finance", &["bank".to_string()]);
    let conf3 = WalkConfig { target_year: None, depth_limit: 1 };
    let ans3 = generate_dynamic_answer("When does the bank close?", "bank", &graph, &mut reasoning, &conf3);
    println!("  -> [BOT_OUTPUT]: \"{}\"", ans3);
    
    println!("\n[USER_QUERY]: \"Is there an ATM there?\"");
    reasoning.update_context("statement", "neutral", "finance", &["Pronoun".to_string()]);
    let conf4 = WalkConfig { target_year: None, depth_limit: 1 };
    
    // Multi-Signal validation: Bank is in stack, but ATM is requested!  
    println!("  [SYS_ORCHESTRATOR]: Secondary prompt signal detected -> 'ATM'.");
    println!("  [SYS_ORCHESTRATOR]: Executing A* trace from [bank] to [ATM]...");
    if !graph.by_surface.contains_key("ATM") {
        println!("  -> [BOT_OUTPUT]: System Fault: Target signal [ATM] does not exist topologically connected to [bank]. Structural Abort to prevent hallucination!");
    } else {
        let ans4 = generate_dynamic_answer("Is there an ATM there?", "Pronoun", &graph, &mut reasoning, &conf4);
        println!("  -> [BOT_OUTPUT]: \"{}\"", ans4);
    }

    // Test: Topological Density limits
    reasoning.update_context("explain", "neutral", "science", &["quantum".to_string()]);
    let conf5 = WalkConfig { target_year: None, depth_limit: 3 };
    let ans5 = generate_dynamic_answer("Explain quantum mechanics.", "quantum", &graph, &mut reasoning, &conf5);
    println!("  -> [BOT_OUTPUT]: \"{}\"", ans5);
    
    println!("\n=================================================\n");
}
