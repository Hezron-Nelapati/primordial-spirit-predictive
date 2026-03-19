use crate::graph::NodeId;

/// Spatial index over node positions for OOV (out-of-vocabulary) fallback.
///
/// # Why not a KD-tree?
/// The project previously used `kiddo`'s KD-tree, which panics at startup when
/// too many nodes share the same value on a single axis — e.g. all 6-letter words
/// produce pos_x = 6.0, and with 87k nodes that bucket can exceed 12k items,
/// far above kiddo's default BUCKET_SIZE of 32.
///
/// A linear scan is the correct trade-off here:
/// - OOV lookups are *rare* (only fired when a query token isn't in the graph).
/// - 87k × 3 f32 comparisons take < 1 ms — unnoticeable in practice.
/// - Zero panics, zero configuration, zero dependencies beyond `std`.
pub struct SpatialGrid {
    nodes: Vec<(NodeId, [f32; 3])>,
}

#[inline]
fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

impl SpatialGrid {
    pub fn build(nodes: impl Iterator<Item = (NodeId, [f32; 3])>) -> Self {
        Self { nodes: nodes.collect() }
    }

    /// All nodes within `radius` of `pos` (Euclidean distance).
    pub fn query_radius(&self, pos: [f32; 3], radius: f32) -> Vec<NodeId> {
        let r2 = radius * radius;
        self.nodes
            .iter()
            .filter(|(_, p)| dist2(*p, pos) <= r2)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Nearest node within `radius` of `pos`, or `None` if none found.
    pub fn query_nearest(&self, pos: [f32; 3], radius: f32) -> Option<NodeId> {
        if self.nodes.is_empty() {
            return None;
        }
        let r2 = radius * radius;
        self.nodes
            .iter()
            .filter_map(|(id, p)| {
                let d = dist2(*p, pos);
                if d <= r2 { Some((*id, d)) } else { None }
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }
}
