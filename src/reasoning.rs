use crate::graph::WordGraph;

pub struct SessionalMemory {
    pub intent_stack: Vec<String>,
    pub tone_stack: Vec<String>,
    pub domain_stack: Vec<String>,
    pub entity_stack: Vec<String>,
}

impl SessionalMemory {
    pub fn new() -> Self {
        Self {
            intent_stack: Vec::new(),
            tone_stack: Vec::new(),
            domain_stack: Vec::new(),
            entity_stack: Vec::new(),
        }
    }
}

pub struct ReasoningModule<'a> {
    pub graph: &'a WordGraph,
    pub session: SessionalMemory,
}

impl<'a> ReasoningModule<'a> {
    pub fn new(graph: &'a WordGraph) -> Self {
        Self {
            graph,
            session: SessionalMemory::new(),
        }
    }

    /// Guardrail 3: Pre-Execution Sanitization Pass
    /// Loops backward. If a terminating "command" intent is found, preceding queries (like "support" or "question") are dropped.
    pub fn sanitize_queue(intents: Vec<String>) -> Vec<String> {
        let mut sanitized = Vec::new();
        let mut has_terminate = false;
        
        for intent in intents.iter().rev() {
            if intent == "command" {
                has_terminate = true;
            }
            
            // "statement" is an Info sentence that sets context, don't drop Info.
            if !has_terminate || intent == "command" || intent == "statement" {
                sanitized.push(intent.clone());
            }
        }
        sanitized.reverse();
        sanitized
    }

    /// Updates the sessional stack based on new ML parses.
    pub fn update_context(&mut self, intent: &str, tone: &str, domain: &str, entities: &[String]) {
        self.session.intent_stack.push(intent.to_string());
        self.session.tone_stack.push(tone.to_string());
        self.session.domain_stack.push(domain.to_string());
        for e in entities {
            if e != "Pronoun" { // Zero-compute prune 
                self.session.entity_stack.push(e.clone());
            }
        }
    }
}
