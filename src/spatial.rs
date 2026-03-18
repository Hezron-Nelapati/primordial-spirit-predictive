use kiddo::{KdTree, SquaredEuclidean};
use crate::graph::NodeId;

pub struct SpatialGrid {
    tree:    KdTree<f32, 3>,
    id_map:  Vec<NodeId>,
}

impl SpatialGrid {
    pub fn build(nodes: impl Iterator<Item = (NodeId, [f32; 3])>) -> Self {
        let mut tree   = KdTree::new();
        let mut id_map = Vec::new();
        for (id, pos) in nodes {
            tree.add(&pos, id_map.len() as u64);
            id_map.push(id);
        }
        Self { tree, id_map }
    }

    pub fn query_radius(&self, pos: [f32; 3], radius: f32) -> Vec<NodeId> {
        self.tree
            .within::<SquaredEuclidean>(&pos, radius * radius)
            .iter()
            .map(|n| self.id_map[n.item as usize])
            .collect()
    }

    pub fn query_nearest(&self, pos: [f32; 3], radius: f32) -> Option<NodeId> {
        if self.tree.size() == 0 { return None; }
        let n = self.tree.nearest_one::<SquaredEuclidean>(&pos);
        if n.distance.sqrt() <= radius {
            Some(self.id_map[n.item as usize])
        } else {
            None
        }
    }
}
