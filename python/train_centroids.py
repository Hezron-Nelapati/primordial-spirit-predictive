import json
import os
import random
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestCentroid

# ---------------------------------------------------------------------------
# POS tag sets — must mirror classify_query.py exactly
# ---------------------------------------------------------------------------
INTENT_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "NN", "NNS", "NNP", "NNPS"}
TONE_TAGS   = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS", "UH"}
DOMAIN_TAGS = {"NN", "NNS", "NNP", "NNPS"}

# ---------------------------------------------------------------------------
# Bootstrap examples
# Purpose: seed all label classes that mock_classify() never outputs
# (greeting, gratitude, explain, complaint, request).  Without these, those
# classes would be missing from the centroid store entirely.
# The corpus sentences below then pull each centroid toward real vocabulary.
# ---------------------------------------------------------------------------
BOOTSTRAP_INTENT = [
    ("Hi there",                                             "greeting"),
    ("Hello",                                                "greeting"),
    ("Good morning to you!",                                 "greeting"),
    ("How are you doing today?",                             "greeting"),
    ("Thanks so much, this is awesome!",                     "gratitude"),
    ("I appreciate your help.",                              "gratitude"),
    ("Can you explain how the system works?",                "explain"),
    ("Please explain the graph walk algorithm.",             "explain"),
    ("This is terrible, everything is broken!",              "complaint"),
    ("Nothing is working as expected.",                      "complaint"),
    ("Could you fix the bug please?",                        "request"),
    ("Stop doing that right now.",                           "command"),
    ("Please turn off the lights.",                          "command"),
    ("What is the temperature outside?",                     "question"),
    ("The server is currently online.",                      "statement"),
]

BOOTSTRAP_TONE = [
    ("Hi there",                                             "casual"),
    ("Hello",                                                "casual"),
    ("Good morning to you!",                                 "polite"),
    ("I appreciate your help.",                              "polite"),
    ("Could you fix the bug please?",                        "polite"),
    ("Thanks so much, this is awesome!",                     "excited"),
    ("This is terrible, everything is broken!",              "angry"),
    ("Stop doing that right now.",                           "angry"),
    ("What is the temperature outside?",                     "neutral"),
    ("The server is currently online.",                      "neutral"),
]

BOOTSTRAP_DOMAIN = [
    ("The server is offline and the network router is down.",           "tech"),
    ("Our API endpoint returns a CPU usage metric.",                    "tech"),
    ("The bank charges interest on every loan and credit account.",     "finance"),
    ("The ATM dispenses cash and shows your account balance.",          "finance"),
    ("Quantum mechanics describes the behavior of particles and atoms.","science"),
    ("Newton's laws relate force, mass, and acceleration.",             "science"),
    ("Paris is the capital city of France.",                            "geography"),
    ("The Amazon river flows through the rainforest in South America.", "geography"),
    ("What time does the meeting start today?",                         "general"),
    ("Everything seems to be working fine now.",                        "general"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pos_filter(text, tag_set):
    try:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        words  = [w for w, t in tagged if t in tag_set]
        return " ".join(words) if words else text
    except Exception:
        return text


def load_corpus(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    print(f"  Loaded {len(rows)} sentences from {path}")
    return rows


def sample_per_class(texts, labels, max_per_class):
    """Cap each class at max_per_class to keep encoding time manageable."""
    buckets = {}
    for text, label in zip(texts, labels):
        buckets.setdefault(label, []).append(text)
    out_texts, out_labels = [], []
    for label, items in buckets.items():
        sampled = items if len(items) <= max_per_class else random.sample(items, max_per_class)
        out_texts.extend(sampled)
        out_labels.extend([label] * len(sampled))
    return out_texts, out_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(42)

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading sentence-transformer model (device: {_device})...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=_device)

    # -----------------------------------------------------------------------
    # Load corpus sentences
    # train_centroids.py lives in python/, corpora are in data/
    # Prefer reinforced corpus (produced by pipeline step 5); fall back to
    # single-pass corpus.json so standalone runs still work.
    # -----------------------------------------------------------------------
    reinforced = "../data/corpus_reinforced.json"
    fallback   = "../data/corpus.json"
    corpus_rows = load_corpus(reinforced) if os.path.exists(reinforced) else load_corpus(fallback)

    # -----------------------------------------------------------------------
    # Build training sets
    # Start from bootstrap (guarantees all label classes present), then
    # append corpus sentences (pulls centroids toward real vocabulary).
    # Cap corpus contribution at 500 per class to keep encoding manageable.
    # -----------------------------------------------------------------------
    MAX_PER_CLASS = 500

    # Intent
    intent_texts  = [t for t, _ in BOOTSTRAP_INTENT]
    intent_labels = [l for _, l in BOOTSTRAP_INTENT]
    if corpus_rows:
        ct, cl = sample_per_class(
            [r["text"] for r in corpus_rows],
            [r["intent"] for r in corpus_rows],
            MAX_PER_CLASS,
        )
        intent_texts  += ct
        intent_labels += cl

    # Tone
    tone_texts  = [t for t, _ in BOOTSTRAP_TONE]
    tone_labels = [l for _, l in BOOTSTRAP_TONE]
    if corpus_rows:
        ct, cl = sample_per_class(
            [r["text"] for r in corpus_rows],
            [r["tone"] for r in corpus_rows],
            MAX_PER_CLASS,
        )
        tone_texts  += ct
        tone_labels += cl

    # Domain
    domain_texts  = [t for t, _ in BOOTSTRAP_DOMAIN]
    domain_labels = [l for _, l in BOOTSTRAP_DOMAIN]
    if corpus_rows:
        ct, cl = sample_per_class(
            [r["text"] for r in corpus_rows],
            [r["domain"] for r in corpus_rows],
            MAX_PER_CLASS,
        )
        domain_texts  += ct
        domain_labels += cl

    print(f"  Intent training set : {len(intent_texts)} sentences, "
          f"{len(set(intent_labels))} classes → {sorted(set(intent_labels))}")
    print(f"  Tone training set   : {len(tone_texts)} sentences, "
          f"{len(set(tone_labels))} classes → {sorted(set(tone_labels))}")
    print(f"  Domain training set : {len(domain_texts)} sentences, "
          f"{len(set(domain_labels))} classes → {sorted(set(domain_labels))}")

    # -----------------------------------------------------------------------
    # POS-filtered texts for blended centroid
    # -----------------------------------------------------------------------
    print("Extracting POS-filtered texts...")
    intent_pos_texts = [pos_filter(t, INTENT_TAGS) for t in intent_texts]
    tone_pos_texts   = [pos_filter(t, TONE_TAGS)   for t in tone_texts]
    domain_pos_texts = [pos_filter(t, DOMAIN_TAGS) for t in domain_texts]

    # -----------------------------------------------------------------------
    # Embeddings
    # -----------------------------------------------------------------------
    print("Computing embeddings (this may take a moment for large corpora)...")
    emb_intent_full = model.encode(intent_texts,     batch_size=128, show_progress_bar=True)
    emb_intent_pos  = model.encode(intent_pos_texts, batch_size=128, show_progress_bar=False)

    emb_tone_full   = model.encode(tone_texts,       batch_size=128, show_progress_bar=False)
    emb_tone_pos    = model.encode(tone_pos_texts,   batch_size=128, show_progress_bar=False)

    emb_domain_full = model.encode(domain_texts,     batch_size=128, show_progress_bar=False)
    emb_domain_pos  = model.encode(domain_pos_texts, batch_size=128, show_progress_bar=False)

    # -----------------------------------------------------------------------
    # Fit centroids
    # -----------------------------------------------------------------------
    print("Fitting centroids...")
    clf_intent_full = NearestCentroid().fit(emb_intent_full, intent_labels)
    clf_intent_pos  = NearestCentroid().fit(emb_intent_pos,  intent_labels)

    clf_tone_full   = NearestCentroid().fit(emb_tone_full,   tone_labels)
    clf_tone_pos    = NearestCentroid().fit(emb_tone_pos,    tone_labels)

    clf_domain_full = NearestCentroid().fit(emb_domain_full, domain_labels)
    clf_domain_pos  = NearestCentroid().fit(emb_domain_pos,  domain_labels)

    store = {
        "intent_labels":         clf_intent_full.classes_.tolist(),
        "intent_full_centroids": clf_intent_full.centroids_.tolist(),
        "intent_pos_centroids":  clf_intent_pos.centroids_.tolist(),

        "tone_labels":           clf_tone_full.classes_.tolist(),
        "tone_full_centroids":   clf_tone_full.centroids_.tolist(),
        "tone_pos_centroids":    clf_tone_pos.centroids_.tolist(),

        "domain_labels":         clf_domain_full.classes_.tolist(),
        "domain_full_centroids": clf_domain_full.centroids_.tolist(),
        "domain_pos_centroids":  clf_domain_pos.centroids_.tolist(),
    }

    out_path = "../data/centroids.json"
    with open(out_path, "w") as f:
        json.dump(store, f)

    print(f"\nExported to {out_path}")
    print(f"  Intents : {clf_intent_full.classes_.tolist()}")
    print(f"  Tones   : {clf_tone_full.classes_.tolist()}")
    print(f"  Domains : {clf_domain_full.classes_.tolist()}")


if __name__ == "__main__":
    main()
