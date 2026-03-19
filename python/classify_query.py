"""
classify_query.py  —  Runtime intent / tone / domain classification.

Usage:
    python3 classify_query.py "query text" [centroids_path]

Outputs a single JSON line to stdout:
    {"intent": "question", "tone": "neutral", "domain": "tech", "entities": ["Paris"]}

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

# ── Absolute default centroids path (works regardless of caller's CWD) ───────
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CENTROIDS = os.path.join(_MODULE_DIR, "..", "data", "centroids.json")

# ── Module-level singletons — loaded once, reused for every classify() call ──
_model: SentenceTransformer | None = None
_store: dict | None                = None
_nlp                               = None   # spaCy model or False (unavailable)

# ── NER entity types to extract (spaCy label set) ────────────────────────────
NER_TYPES = {"PERSON", "ORG", "GPE", "PRODUCT"}

# ── POS tag sets — must match train_centroids.py exactly ─────────────────────
INTENT_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "NN", "NNS", "NNP", "NNPS"}
TONE_TAGS   = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS", "UH"}
DOMAIN_TAGS = {"NN", "NNS", "NNP", "NNPS"}

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


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("  [classify_query]: loading sentence-transformer model (once)…", file=sys.stderr)
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_store(centroids_path: str = _DEFAULT_CENTROIDS) -> dict:
    global _store
    if _store is None:
        with open(centroids_path) as f:
            _store = json.load(f)
    return _store


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            print("  [classify_query]: spaCy en_core_web_sm loaded (once).", file=sys.stderr)
        except (ImportError, OSError) as exc:
            print(f"  [classify_query]: spaCy unavailable ({exc}) — NER disabled.", file=sys.stderr)
            _nlp = False
    return _nlp


def _ner_entities(text: str) -> list:
    """Extract named entity surface strings from text using spaCy en_core_web_sm.
    Returns an empty list if spaCy or its model is unavailable — callers must
    treat [] as a valid (no-entity) response rather than an error.
    """
    nlp = _get_nlp()
    if not nlp:
        return []
    try:
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in NER_TYPES]
        print(f"  [classify_query]: NER extracted {entities}", file=sys.stderr)
        return entities
    except Exception as exc:
        print(f"  [classify_query]: NER failed ({exc})", file=sys.stderr)
        return []


def classify(query: str, centroids_path: str = _DEFAULT_CENTROIDS,
             session_id: str | None = None) -> dict:
    _ensure_nltk()
    model = _get_model()
    store = _get_store(centroids_path)

    # Full-text embedding
    raw_emb = model.encode(query)

    # Sentence Queue: if a session_id is provided, push this embedding into the
    # rolling window and use the blended context vector for classification.
    # Single-turn calls (no session_id) use the raw embedding unchanged.
    if session_id is not None:
        try:
            from sentence_queue import SentenceQueue
            q = SentenceQueue(session_id=session_id)
            q.push(raw_emb)
            emb_full = q.blended()
            q.save()
            print(f"  [classify_query]: Sentence queue depth={len(q._embeddings)} → blended embedding used.", file=sys.stderr)
        except Exception as exc:
            print(f"  [classify_query]: Sentence queue unavailable ({exc}) — using raw embedding.", file=sys.stderr)
            emb_full = raw_emb.tolist()
    else:
        emb_full = raw_emb.tolist()

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

    # Domain: use centroid model when available; fall back to keyword heuristic.
    if "domain_labels" in store:
        domain_text = _pos_filter(query, DOMAIN_TAGS)
        emb_domain_pos = model.encode(domain_text).tolist()
        domain = _nearest_blended(
            emb_full, emb_domain_pos,
            store["domain_full_centroids"], store["domain_pos_centroids"],
            store["domain_labels"],
        )
        print(f"  [classify_query]: domain via centroid model -> '{domain}'", file=sys.stderr)
    else:
        domain = _keyword_domain(query)
        print(f"  [classify_query]: domain via keyword fallback -> '{domain}'", file=sys.stderr)

    entities = _ner_entities(query)

    return {"intent": intent, "tone": tone, "domain": domain, "entities": entities}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 classify_query.py 'query text' [centroids_path] [--session-id ID]", file=sys.stderr)
        sys.exit(1)

    query_text     = sys.argv[1]
    centroids_path = "data/centroids.json"
    session_id     = None

    # Parse optional positional centroids_path and --session-id flag
    remaining = sys.argv[2:]
    positional_done = False
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if arg == "--session-id" and i + 1 < len(remaining):
            session_id = remaining[i + 1]
            i += 2
        elif not positional_done and not arg.startswith("--"):
            centroids_path = arg
            positional_done = True
            i += 1
        else:
            i += 1

    result = classify(query_text, centroids_path, session_id=session_id)
    print(json.dumps(result))
