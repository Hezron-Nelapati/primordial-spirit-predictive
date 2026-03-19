"""
classify_query.py  —  Runtime intent / tone / domain classification.

Usage:
    python3 classify_query.py "query text" [centroids_path]

Outputs a single JSON line to stdout:
    {"intent": "question", "tone": "neutral", "domain": "tech"}

All diagnostic messages go to stderr so the Rust caller can parse stdout cleanly.
"""

import sys
import json
import math
import os

# ── silence noisy HuggingFace / tokeniser warnings before any import ──────────
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import nltk
from sentence_transformers import SentenceTransformer

# ── POS tag sets — must match train_centroids.py exactly ─────────────────────
INTENT_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "NN", "NNS", "NNP", "NNPS"}
TONE_TAGS   = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS", "UH"}

# ── domain keyword map — mirrors v2_ingest.py mock_classify() ─────────────────
DOMAIN_KEYWORDS = {
    "tech":      {"server", "router", "network", "cpu", "online", "offline", "system", "data", "api", "code"},
    "science":   {"quantum", "physics", "atom", "particle", "energy", "wave", "molecule", "force", "mass"},
    "finance":   {"bank", "atm", "finance", "money", "loan", "credit", "payment", "account", "invest"},
    "geography": {"france", "paris", "rome", "italy", "eiffel", "country", "city", "tower", "capital"},
}


def _ensure_nltk():
    for resource, path in [
        ("punkt_tab",                     "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger_eng","taggers/averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"  [classify_query]: downloading NLTK '{resource}'...", file=sys.stderr)
            nltk.download(resource, quiet=True)


def _pos_filter(text: str, tag_set: set) -> str:
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    words  = [w for w, t in tagged if t in tag_set]
    return " ".join(words) if words else text


def _euclidean(a: list, b: list) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _nearest_blended(emb_full, emb_pos, full_centroids, pos_centroids, labels):
    best_idx  = 0
    min_dist  = float("inf")
    for i in range(len(labels)):
        d_full   = _euclidean(emb_full, full_centroids[i])
        d_pos    = _euclidean(emb_pos,  pos_centroids[i])
        blended  = 0.7 * d_full + 0.3 * d_pos
        if blended < min_dist:
            min_dist = blended
            best_idx = i
    return labels[best_idx]


def _keyword_domain(text: str) -> str:
    words = set(text.lower().split())
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if words & keywords:
            return domain
    return "general"


def classify(query: str, centroids_path: str = "data/centroids.json") -> dict:
    _ensure_nltk()

    print("  [classify_query]: Loading sentence-transformer model...", file=sys.stderr)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open(centroids_path) as f:
        store = json.load(f)

    # Full-text embedding
    emb_full = model.encode(query).tolist()

    # POS-filtered embeddings (must match train_centroids.py filtering)
    intent_text = _pos_filter(query, INTENT_TAGS)
    tone_text   = _pos_filter(query, TONE_TAGS)
    emb_intent_pos = model.encode(intent_text).tolist()
    emb_tone_pos   = model.encode(tone_text).tolist()

    intent = _nearest_blended(
        emb_full, emb_intent_pos,
        store["intent_full_centroids"], store["intent_pos_centroids"],
        store["intent_labels"],
    )
    tone = _nearest_blended(
        emb_full, emb_tone_pos,
        store["tone_full_centroids"], store["tone_pos_centroids"],
        store["tone_labels"],
    )
    domain = _keyword_domain(query)

    return {"intent": intent, "tone": tone, "domain": domain}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 classify_query.py 'query text' [centroids_path]", file=sys.stderr)
        sys.exit(1)

    query_text     = sys.argv[1]
    centroids_path = sys.argv[2] if len(sys.argv) >= 3 else "data/centroids.json"

    result = classify(query_text, centroids_path)
    print(json.dumps(result))
