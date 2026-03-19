use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct CentroidStore {
    pub intent_labels:         Vec<String>,
    pub intent_full_centroids: Vec<Vec<f32>>,
    pub intent_pos_centroids:  Vec<Vec<f32>>,

    pub tone_labels:           Vec<String>,
    pub tone_full_centroids:   Vec<Vec<f32>>,
    pub tone_pos_centroids:    Vec<Vec<f32>>,

    // Domain centroids — present only after re-running train_centroids.py (Phase 9).
    // Absent keys deserialize to empty Vecs for backward compatibility.
    #[serde(default)]
    pub domain_labels:         Vec<String>,
    #[serde(default)]
    pub domain_full_centroids: Vec<Vec<f32>>,
    #[serde(default)]
    pub domain_pos_centroids:  Vec<Vec<f32>>,
}

pub struct Classifier {
    store: CentroidStore,
}

impl Classifier {
    pub fn load(path: &str) -> Self {
        let json = std::fs::read_to_string(path).expect("centroids file missing");
        Self { store: serde_json::from_str(&json).expect("bad centroids json") }
    }

    pub fn intent(&self, emb_full: &[f32], emb_pos: &[f32]) -> &str {
        nearest_blended(
            emb_full,
            emb_pos,
            &self.store.intent_full_centroids,
            &self.store.intent_pos_centroids,
            &self.store.intent_labels,
        )
    }

    pub fn tone(&self, emb_full: &[f32], emb_pos: &[f32]) -> &str {
        nearest_blended(
            emb_full,
            emb_pos,
            &self.store.tone_full_centroids,
            &self.store.tone_pos_centroids,
            &self.store.tone_labels,
        )
    }

    /// Classify domain using the centroid model.  Returns `"general"` when
    /// domain centroids are absent (old `centroids.json` predating Phase 9).
    pub fn domain(&self, emb_full: &[f32], emb_pos: &[f32]) -> &str {
        if self.store.domain_labels.is_empty() {
            return "general";
        }
        nearest_blended(
            emb_full,
            emb_pos,
            &self.store.domain_full_centroids,
            &self.store.domain_pos_centroids,
            &self.store.domain_labels,
        )
    }
}

fn nearest_blended<'a>(
    emb_full:       &[f32],
    emb_pos:        &[f32],
    full_centroids: &[Vec<f32>],
    pos_centroids:  &[Vec<f32>],
    labels:         &'a [String],
) -> &'a str {
    let mut best_idx = 0;
    let mut min_dist = f32::MAX;

    for i in 0..labels.len() {
        let d_full = euclidean(emb_full, &full_centroids[i]);
        let d_pos  = euclidean(emb_pos,  &pos_centroids[i]);
        let blended = 0.7 * d_full + 0.3 * d_pos;
        if blended < min_dist {
            min_dist = blended;
            best_idx = i;
        }
    }

    labels[best_idx].as_str()
}

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
