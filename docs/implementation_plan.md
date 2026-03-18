# Predictive Topological Generation Engine (Master Blueprint)

This document dictates the complete architectural constraints for building a non-LLM, ultra-fast 3D Spatial Predictive Text Generator. It relies strictly on Euclidean geometry, Python topological classifiers, and explicit real-world guardrails to generate deterministic, hallucination-free output.

---

## 1. The Python Sensory Pipeline
The Python endpoint serves as the cognitive intake. It must execute **symmetrically** in both Training Mode (Corpus Ingestion) and Usage Mode (Live Prediction).

### A. Sentence Queuing (Input Parsing)
To prevent embedding degradation ("sludge"), input strings are never embedded as massive blocks. The Python pipeline runs an active Sentence Tokenizer (e.g., `nltk.sent_tokenize`). It slices paragraphs into a `Queue` of independent semantic thoughts, ensuring each sentence receives completely isolated classification.

### B. Topological Classification & Metadata Extraction
Every isolated sentence is converted into dense semantic vectors and rich structural metadata explicitly stamping the Graph's physical architecture:
1. **Three-Tier Centroids**: 384-dimensional models calculate dynamic vectors for functional **Intent** (*question, complaint*), emotional **Tone** (*angry, polite*), and structural **Domain** (*medical, finance*).
2. **NER Extraction**: Dynamically tags hard nouns with their geographic category (e.g., *Entity: Location*).
3. **Temporal Mapping**: Extracts the chronological timestamp of the ingested fact and permanently binds it explicitly as a `dated` Edge tag (e.g., `2024`).

---

## 2. The Rust Spatial Graph Core
The core `Graph` is explicitly a pure geometric topology. It holds physical data but possesses zero internal decision-making capabilities.

### A. Data Structures & Punctuation Anchors (`src/graph.rs`)
Nodes act as 3D physical anchors representing vocabulary. Python isolates all punctuation (`.`, `,`, `?`, `!`) into standalone physical anchor nodes, serving as mathematically explicit structural traffic lights inside the graph layout. 

```rust
pub struct WordNode {
    pub id: NodeId, 
    pub surface: String, 
    pub position: [f32; 3], 
    pub frequency: u32,
    pub lexical_vector: [f32; 5], // Deterministic fast-hash for OOV Fallbacks
}

pub struct WordEdge {
    pub from: NodeId, 
    pub to: NodeId, 
    pub weight: f32,       
    pub intent: String,    
    pub tone: String,      
    pub domain: String,    
    pub entity: Option<String>, 
    pub dated: Option<u16>, 
}
```

---

## 3. The Orchestrating Reasoning Module
The `ReasoningModule` acts as the state-machine. It sits insulated above the graph, holding the parsed arrays from the Python pipeline and mathematically commanding the Graph on how to safely navigate the topology.

### A. Sessional Active Memory (Contextual Stacks)
To guarantee the engine never suffers from "Context Amnesia" or Pronominal Ambiguity, the Orchestrator maintains an infinite **Sessional Stack** representing the user's conversational state across all 4 routing tags (`Intent, Tone, Domain, Entity`). 
* If a user query relies heavily on empty pronouns (e.g., `"I need to go there"`), the Orchestrator safely inherits the explicit target (`Entity: Location [bank]`) sitting at the top of its historic Sessional Stack, bypassing ambiguity with mathematically **zero extra compute**.

### B. Pre-Execution Sanitization
The Orchestrator actively separates Parsing from Execution. If a user provides a contradictory jumbled payload (*"Reset my router... nevermind, cancel my account"*), the Orchestrator logically parses the entire Intent array (`[Intent: support, Intent: terminate]`). Before executing the walk, it runs a pre-execution **Sanitization Pass**, structurally dropping the aborted `support` query due to the chronological dominance of the `terminate` intent.

### C. Logic & Arithmetic Interception
If the Python pipeline flags an `Intent: math` or `system_command`, the Orchestrator bypasses the semantic walk natively in Rust (e.g., `eval("2+2")`), dynamically suppressing Transformer arithmetic failure rates.

---

## 4. The Polymorphic Routing Algorithm
To generate text, the Reasoning Module commands the Graph to walk. The active `Intent` strictly dictates how the mathematical Tiering operates.

### Tier 1: Multi-Tag Match Biasing & Temporal Sorting
The engine reviews all outgoing paths. To eliminate the gravitational pull of high-frequency stop words (`the`, `and`), the confidence `weight` of an edge mathematically multiplies exponentially for **every** tag that structurally matches the user's Sessional Stack. 
* **Axiomatic Hallucination Fallback**: To solve conflicting memory lines (`Server is down [2020]` vs `Server is up [2024]`), the `dated` tag acts as a tie-breaker. The Orchestrator defaults to the current clock (`2026`). Edges matching the targeted temporal chronology automatically receive an exponential multiplier, silently phasing out outdated logic.

### Tier 2: Radial Proximity Search
If matching sequences fail, the system triggers a radius query on the KD-Tree index, blindingly bridging conceptual nodes in the 3D space via frequency proximity.

### Tier 3: A* Conceptual Jumps & The Topological Limit
* **Answer-Seeking Intent (`question`, `complaint`)**: The Orchestrator forces a Tier 3 A* pathfinding calculation from the Prompt cluster bridging completely into a **Response Centroid**.
* **Infinite Generation Fallback**: When traversing sequences, the engine does not statically abort on the first `[.]` punctuation anchor. It evaluates the **Topological Edge Density** of the Prompt entity! If the queried entity is heavily connected (100+ edges), the Orchestrator recognizes a "Dense Subject" and dynamically scales the Sentence Limit to `4` periods. If the entity is sparse (5 edges), the Limit is capped at `1` period.
## Section 6: The RAG Topological Integration (The Output Layer)

> [!IMPORTANT]
> The Engine must not be used as a conversational chatbot natively. It is a Knowledge Database. It is mathematically designed to output raw, unadulterated String facts (`"In 2024, the server was online."`).

### The miniLLM Styling Wrapper
* Once the Rust Graph resolves the exact factual String mathematically, it pipes the String to a lightweight, local LLM interface.
* The LLM is strictly disconnected from the logical process. Its absolute instruction is to ingest the Graph Fact and format it conversationally for the user, retaining infinite stylistic flexibility while inheriting flawless topological truth.
