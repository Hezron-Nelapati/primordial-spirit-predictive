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
}

impl WordGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::with_capacity_and_hasher(10_000, Fnv1aBuildHasher::default()),
            edges: Vec::with_capacity(50_000),
            by_surface: HashMap::with_capacity_and_hasher(10_000, Fnv1aBuildHasher::default()),
        }
    }

    pub fn generate_id(word: &str) -> NodeId {
        let mut hasher = Fnv1aHasher::default();
        hasher.write(word.as_bytes());
        hasher.finish()
    }
}
