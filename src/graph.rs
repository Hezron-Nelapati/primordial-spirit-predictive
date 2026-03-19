use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use serde::{Deserialize, Serialize};

// FNV-1a extremely fast non-cryptographic hash for 60 FPS lookups
pub struct Fnv1aHasher(u64);
impl Default for Fnv1aHasher {
    #[inline] fn default() -> Self { Self(0xcbf29ce484222325) }
}
impl Hasher for Fnv1aHasher {
    #[inline] fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= byte as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
    #[inline] fn finish(&self) -> u64 { self.0 }
}
type Fnv1aBuildHasher = BuildHasherDefault<Fnv1aHasher>;
pub type NodeId = u64;

#[derive(Clone, Serialize, Deserialize)]
pub struct WordNode {
    pub id: NodeId,
    pub surface: String,
    pub frequency: u32,
    pub position: [f32; 3], // Pseudo 3D topology
    pub lexical_vector: [f32; 5], // Guardrail 1: OOV Panic Fallback
}

impl WordNode {
    pub fn compute_lexical_vector(word: &str) -> [f32; 5] {
        let chars: Vec<char> = word.chars().collect();
        if chars.is_empty() { return [0.0; 5]; }
        
        let length = chars.len() as f32;
        let first = chars[0] as u32 as f32;
        let last = chars[chars.len() - 1] as u32 as f32;
        
        let vowels = chars.iter().filter(|c| "aeiouAEIOU".contains(**c)).count() as f32;
        
        let mut unique_chars = chars.clone();
        unique_chars.sort_unstable();
        unique_chars.dedup();
        let unique = unique_chars.len() as f32;
        
        [length, first, last, vowels, unique]
    }
}

#[derive(Clone, Debug)]
pub struct WordEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub weight: f32,
    pub intent: String,
    pub tone: String,
    pub domain: String,
    pub entity: Option<String>,
    pub dated: Option<u16>, // Guardrail 5: Axiomatic Hallucinations
}

pub struct WordGraph {
    pub nodes: HashMap<NodeId, WordNode, Fnv1aBuildHasher>,
    pub edges: Vec<WordEdge>,
    pub by_surface: HashMap<String, NodeId, Fnv1aBuildHasher>,
    // Fix #10: index maps for O(degree) edge lookups instead of O(E) linear scans.
    from_index: HashMap<NodeId, Vec<usize>, Fnv1aBuildHasher>,
    to_index:   HashMap<NodeId, Vec<usize>, Fnv1aBuildHasher>,
}

impl WordGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::with_capacity_and_hasher(10_000, Fnv1aBuildHasher::default()),
            edges: Vec::with_capacity(50_000),
            by_surface: HashMap::with_capacity_and_hasher(10_000, Fnv1aBuildHasher::default()),
            from_index: HashMap::with_capacity_and_hasher(10_000, Fnv1aBuildHasher::default()),
            to_index:   HashMap::with_capacity_and_hasher(10_000, Fnv1aBuildHasher::default()),
        }
    }

    /// Push an edge and keep both index maps consistent.
    pub fn push_edge(&mut self, edge: WordEdge) {
        let idx = self.edges.len();
        self.from_index.entry(edge.from).or_default().push(idx);
        self.to_index.entry(edge.to).or_default().push(idx);
        self.edges.push(edge);
    }

    pub fn generate_id(word: &str) -> NodeId {
        let mut hasher = Fnv1aHasher::default();
        hasher.write(word.as_bytes());
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// GraphAccess trait — uniform read interface over WordGraph (tests) and
// GraphDb (production).  Walk functions take `&dyn GraphAccess` so the
// same logic runs against either backing store without code duplication.
// ---------------------------------------------------------------------------

pub trait GraphAccess {
    /// Look up the numeric ID for a surface-form string.
    fn surface_to_id(&self, surface: &str) -> Option<NodeId>;
    /// Fetch a node by its numeric ID (returns owned clone).
    fn node_by_id(&self, id: NodeId) -> Option<WordNode>;
    /// All edges whose `from` field equals `from_id`.
    fn edges_from(&self, from_id: NodeId) -> Vec<WordEdge>;
    /// All edges whose `to` field equals `to_id`.
    fn edges_to(&self, to_id: NodeId) -> Vec<WordEdge>;
    /// True if any edge departs from `id` (cheaper than edges_from for boolean check).
    fn has_edges_from(&self, id: NodeId) -> bool;
    /// Number of outgoing edges from `id` — cheaper than edges_from().len() because
    /// it avoids deserialising edge data; used by score_edges_explain for ranking.
    fn out_degree(&self, id: NodeId) -> usize;
    /// All nodes — used for OOV lexical-vector fallback scan.
    fn all_nodes(&self) -> Vec<WordNode>;
    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;
}

impl GraphAccess for WordGraph {
    fn surface_to_id(&self, surface: &str) -> Option<NodeId> {
        self.by_surface.get(surface).copied()
    }
    fn node_by_id(&self, id: NodeId) -> Option<WordNode> {
        self.nodes.get(&id).cloned()
    }
    fn edges_from(&self, from_id: NodeId) -> Vec<WordEdge> {
        // Fix #10: use index map for O(degree) lookup instead of O(E) scan.
        self.from_index.get(&from_id)
            .map(|idxs| idxs.iter().map(|&i| self.edges[i].clone()).collect())
            .unwrap_or_default()
    }
    fn edges_to(&self, to_id: NodeId) -> Vec<WordEdge> {
        // Fix #10: use index map for O(degree) lookup instead of O(E) scan.
        self.to_index.get(&to_id)
            .map(|idxs| idxs.iter().map(|&i| self.edges[i].clone()).collect())
            .unwrap_or_default()
    }
    fn has_edges_from(&self, id: NodeId) -> bool {
        self.from_index.get(&id).map(|v| !v.is_empty()).unwrap_or(false)
    }
    fn out_degree(&self, id: NodeId) -> usize {
        self.from_index.get(&id).map(|v| v.len()).unwrap_or(0)
    }
    fn all_nodes(&self) -> Vec<WordNode> {
        self.nodes.values().cloned().collect()
    }
    fn node_count(&self) -> usize { self.nodes.len() }
    fn edge_count(&self) -> usize { self.edges.len() }
}
