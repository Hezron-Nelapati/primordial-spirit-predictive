use crate::graph::WordGraph;

/// Guardrail 6: Arithmetic / logic interception.
///
/// Returns `true` when the query contains **both**:
///   1. At least one standalone numeric token (parses as a finite `f64`), and
///   2. At least one arithmetic signal — an explicit operator token (`+`, `*`,
///      `/`, `=`, `%`, `^`) or a keyword (`plus`, `minus`, `times`, `divided`,
///      `equals`, `sum`, `product`, `percent`, `sqrt`, `squared`, `cubed`).
///
/// The two-condition gate avoids false positives on queries like
/// "The bank closes at 5 pm" (number present but no arithmetic signal) or
/// "What is the sum of all loans?" (arithmetic word present but no number).
pub fn is_arithmetic_query(query: &str) -> bool {
    const ARITH_WORDS: &[&str] = &[
        "plus", "minus", "times", "divided", "equals", "sum",
        "product", "percent", "sqrt", "squared", "cubed",
    ];
    const ARITH_OPS: &[&str] = &["+", "*", "/", "=", "%", "^"];

    let mut has_number = false;
    let mut has_arith  = false;

    for raw in query.split_whitespace() {
        let token = raw.trim_matches(|c: char| c.is_ascii_punctuation());
        if token.is_empty() { continue; }

        if !has_number && token.parse::<f64>().map(|v| v.is_finite()).unwrap_or(false) {
            has_number = true;
        }
        if !has_arith {
            let lower = token.to_lowercase();
            if ARITH_WORDS.contains(&lower.as_str()) || ARITH_OPS.contains(&token) {
                has_arith = true;
            }
        }
        if has_number && has_arith { return true; }
    }
    false
}

/// Evaluate a simple arithmetic expression from a natural-language query.
///
/// Supports:
/// - Binary ops: `+`, `-`, `*`, `/`, `%` and their word forms
///   (`plus`, `minus`, `times`, `divided`, `percent`).
/// - Unary ops on a single number: `sqrt`, `squared`, `cubed`.
/// - Multi-number `sum` / `product` (folds the full number list).
///
/// Returns a raw computed result string (e.g. `"42"` or `"3.1416"`), or
/// `None` if no computable expression is found.  The caller is responsible
/// for conversational formatting (e.g. piping through miniLLM).
pub fn evaluate_arithmetic(query: &str) -> Option<String> {
    let mut numbers: Vec<f64> = Vec::new();
    let mut binary_op: Option<char> = None;
    let mut unary_op: Option<&str> = None;
    let mut is_sum     = false;
    let mut is_product = false;

    for raw in query.split_whitespace() {
        // Strip surrounding punctuation but keep minus sign on negative numbers.
        let token = raw.trim_matches(|c: char| c.is_ascii_punctuation() && c != '-' && c != '.');
        if token.is_empty() { continue; }

        if let Ok(n) = token.parse::<f64>() {
            if n.is_finite() { numbers.push(n); }
            continue;
        }

        let lower = token.to_lowercase();
        match lower.as_str() {
            "plus"  | "add"        => { if binary_op.is_none() { binary_op = Some('+'); } }
            "minus" | "subtract"   => { if binary_op.is_none() { binary_op = Some('-'); } }
            "times" | "multiplied" => { if binary_op.is_none() { binary_op = Some('*'); } }
            "divided" | "over"     => { if binary_op.is_none() { binary_op = Some('/'); } }
            "percent" | "mod"      => { if binary_op.is_none() { binary_op = Some('%'); } }
            "+"  => { if binary_op.is_none() { binary_op = Some('+'); } }
            "-"  => { if binary_op.is_none() { binary_op = Some('-'); } }
            "*"  => { if binary_op.is_none() { binary_op = Some('*'); } }
            "/"  => { if binary_op.is_none() { binary_op = Some('/'); } }
            "%"  => { if binary_op.is_none() { binary_op = Some('%'); } }
            "sqrt"    => { unary_op = Some("sqrt"); }
            "squared" => { unary_op = Some("squared"); }
            "cubed"   => { unary_op = Some("cubed"); }
            "sum"     => { is_sum     = true; }
            "product" => { is_product = true; }
            _ => {}
        }
    }

    if numbers.is_empty() { return None; }

    fn fmt(v: f64) -> String {
        if v.fract() == 0.0 && v.abs() < 1e15 {
            format!("{}", v as i64)
        } else {
            format!("{:.4}", v)
        }
    }

    // Unary operations — operate on the first number found.
    if let Some(uop) = unary_op {
        let n = numbers[0];
        let result = match uop {
            "sqrt" => {
                if n < 0.0 { return Some("undefined (square root of negative number)".to_string()); }
                n.sqrt()
            }
            "squared" => n * n,
            "cubed"   => n * n * n,
            _ => return None,
        };
        return Some(fmt(result));
    }

    // Multi-number fold for sum/product.
    if is_sum && numbers.len() >= 2 {
        return Some(fmt(numbers.iter().sum()));
    }
    if is_product && numbers.len() >= 2 {
        return Some(fmt(numbers.iter().product()));
    }

    // Binary operation on first two numbers.
    if numbers.len() >= 2 {
        let a = numbers[0];
        let b = numbers[1];
        let op = binary_op.unwrap_or('+');
        let result = match op {
            '+' => a + b,
            '-' => a - b,
            '*' => a * b,
            '/' | '%' => {
                if b == 0.0 { return Some("undefined (division by zero)".to_string()); }
                if op == '/' { a / b } else { a % b }
            }
            _ => return None,
        };
        return Some(fmt(result));
    }

    // Single number with no recognized operation — not enough to compute.
    None
}

/// Extract a calendar year from a natural-language query string.
///
/// Scans for a standalone 4-digit token that falls within the plausible range
/// `1900–2099`.  Returns the first match found, or `None` if no year is present.
///
/// The range guard prevents false positives from model numbers ("A100"),
/// version strings ("3.2024"), or large numeric values.
///
/// Used in the CLI path to populate `WalkConfig::target_year` automatically
/// when the user does not supply a year as a positional argument.
pub fn extract_year_from_query(query: &str) -> Option<u16> {
    for raw in query.split_whitespace() {
        let token = raw.trim_matches(|c: char| c.is_ascii_punctuation());
        if token.len() == 4 {
            if let Ok(year) = token.parse::<u16>() {
                if (1900..=2099).contains(&year) {
                    return Some(year);
                }
            }
        }
    }
    None
}

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

    /// Reset all four session stacks to empty.
    ///
    /// Use this between independent conversations or demo scenarios to prevent
    /// context from a prior exchange from biasing the next walk.
    pub fn reset_session(&mut self) {
        self.session.intent_stack.clear();
        self.session.tone_stack.clear();
        self.session.domain_stack.clear();
        self.session.entity_stack.clear();
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
