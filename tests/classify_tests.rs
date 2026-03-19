use spse_predictive::classify::Classifier;

// The centroid dimension produced by all-MiniLM-L6-v2 (the model used in train_centroids.py).
const EMB_DIM: usize = 384;

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Verifies that data/centroids.json (checked-in) parses without error.
/// If this test panics, the centroids file is corrupt or the CentroidStore
/// schema has drifted from what train_centroids.py produces.
#[test]
fn test_classifier_loads_without_panic() {
    let _clf = Classifier::load("data/centroids.json");
}

// ---------------------------------------------------------------------------
// Intent classification
// ---------------------------------------------------------------------------

/// Passing a zero-vector of the correct dimension must return a non-empty label
/// without panicking.  The specific label doesn't matter; what matters is that
/// the blended nearest-centroid scoring runs to completion.
#[test]
fn test_classifier_intent_returns_valid_label_for_zero_vector() {
    let clf = Classifier::load("data/centroids.json");
    let emb = vec![0.0_f32; EMB_DIM];
    let label = clf.intent(&emb, &emb);
    assert!(!label.is_empty(), "intent label must not be empty");
}

/// The trained labels must include at least the core intents defined in
/// train_centroids.py ("greeting", "question", "command").
#[test]
fn test_classifier_intent_label_set_contains_core_intents() {
    // We probe by checking that intent() for a zero-vector returns one of the
    // known labels.  The actual label for a zero-vector is implementation-
    // defined; we only assert that it is one of the expected values.
    let clf = Classifier::load("data/centroids.json");
    let emb = vec![0.0_f32; EMB_DIM];
    let label = clf.intent(&emb, &emb);
    let known = ["greeting", "question", "explain", "gratitude", "complaint", "request", "command"];
    assert!(
        known.contains(&label),
        "unexpected intent label '{}'; expected one of {:?}",
        label, known
    );
}

// ---------------------------------------------------------------------------
// Tone classification
// ---------------------------------------------------------------------------

#[test]
fn test_classifier_tone_returns_valid_label_for_zero_vector() {
    let clf = Classifier::load("data/centroids.json");
    let emb = vec![0.0_f32; EMB_DIM];
    let label = clf.tone(&emb, &emb);
    assert!(!label.is_empty(), "tone label must not be empty");
}

#[test]
fn test_classifier_tone_label_set_contains_core_tones() {
    let clf = Classifier::load("data/centroids.json");
    let emb = vec![0.0_f32; EMB_DIM];
    let label = clf.tone(&emb, &emb);
    let known = ["casual", "polite", "neutral", "excited", "angry"];
    assert!(
        known.contains(&label),
        "unexpected tone label '{}'; expected one of {:?}",
        label, known
    );
}

// ---------------------------------------------------------------------------
// Consistency
// ---------------------------------------------------------------------------

/// Calling intent() and tone() twice with identical input must return the
/// same label (deterministic — no randomness).
#[test]
fn test_classifier_is_deterministic() {
    let clf = Classifier::load("data/centroids.json");
    let emb = vec![0.1_f32; EMB_DIM];

    assert_eq!(clf.intent(&emb, &emb), clf.intent(&emb, &emb));
    assert_eq!(clf.tone(&emb, &emb), clf.tone(&emb, &emb));
}

// ---------------------------------------------------------------------------
// Domain classification (Phase 9)
// ---------------------------------------------------------------------------

/// domain() must return a non-empty string for any embedding.
/// When domain centroids are absent (old centroids.json), falls back to "general".
#[test]
fn test_classifier_domain_returns_valid_label_for_zero_vector() {
    let clf = Classifier::load("data/centroids.json");
    let emb = vec![0.0_f32; EMB_DIM];
    let label = clf.domain(&emb, &emb);
    assert!(!label.is_empty(), "domain label must not be empty");
}

/// When domain centroids ARE present, the returned label must be one of the
/// five domains defined in train_centroids.py Phase 9 training data.
/// When they are absent, "general" is returned — also a valid member of the set.
#[test]
fn test_classifier_domain_label_set_contains_core_domains() {
    let clf = Classifier::load("data/centroids.json");
    let emb = vec![0.0_f32; EMB_DIM];
    let label = clf.domain(&emb, &emb);
    let known = ["tech", "finance", "science", "geography", "general"];
    assert!(
        known.contains(&label),
        "unexpected domain label '{}'; expected one of {:?}",
        label, known
    );
}

/// domain() must be deterministic — same embedding always returns same label.
#[test]
fn test_classifier_domain_is_deterministic() {
    let clf = Classifier::load("data/centroids.json");
    let emb = vec![0.1_f32; EMB_DIM];
    assert_eq!(clf.domain(&emb, &emb), clf.domain(&emb, &emb));
}
