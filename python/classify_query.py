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
import spacy
from sentence_transformers import SentenceTransformer

# ── Absolute default centroids path (works regardless of caller's CWD) ───────
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CENTROIDS = os.path.join(_MODULE_DIR, "..", "data", "centroids.json")

# ── Module-level singletons — loaded once, reused for every classify() call ──
_model: SentenceTransformer | None = None
_store: dict | None                = None
_nlp                               = None

# ── NER types — must match ingest.py / ingest_wiki.py exactly ────────────────
NER_TYPES = {"PERSON", "ORG", "GPE", "PRODUCT"}

# ── POS tag sets — must match ingest.py / ingest_wiki.py / train_centroids.py ─
INTENT_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "NN", "NNS", "NNP", "NNPS"}
TONE_TAGS   = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS", "UH"}
DOMAIN_TAGS = {"NN", "NNS", "NNP", "NNPS"}

# ── domain keyword map — mirrors ingest.py mock_classify() ───────────────────
DOMAIN_KEYWORDS = {
    "tech":      {"server", "router", "network", "cpu", "online", "offline", "system", "data", "api", "code"},
    "science":   {"quantum", "physics", "atom", "particle", "energy", "wave", "molecule", "force", "mass"},
    "finance":   {"bank", "atm", "finance", "money", "loan", "credit", "payment", "account", "invest"},
    "geography": {"france", "paris", "rome", "italy", "eiffel", "country", "city", "tower", "capital"},
}


def _ensure_nltk():
    needed = [
        ("punkt_tab",                      "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ]
    for resource, path in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"  [classify_query]: downloading NLTK '{resource}'...", file=sys.stderr)
            nltk.download(resource, quiet=True)


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _nlp


def _pos_filter(text: str, tag_set: set) -> str:
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    words  = [w for w, t in tagged if t in tag_set]
    # Fix #12: return empty string instead of full text when no matching tokens
    # exist.  Returning text here caused the POS-filtered embedding to equal the
    # full embedding, making the 0.3 blend weight completely meaningless.
    return " ".join(words)


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
        from gpu_utils import get_device
        print("  [classify_query]: loading sentence-transformer model…", file=sys.stderr)
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=get_device())
    return _model


def _get_store(centroids_path: str = _DEFAULT_CENTROIDS) -> dict:
    global _store
    if _store is None:
        with open(centroids_path) as f:
            _store = json.load(f)
    return _store


def reset_store() -> None:
    """Invalidate the cached centroids so the next classify() call reloads from disk.
    Call this after training completes to pick up the new centroids.json without
    restarting the process."""
    global _store
    _store = None
    print("  [classify_query]: centroids cache invalidated.", file=sys.stderr)


def _ner_entities(text: str) -> list:
    """Extract named entities with spaCy — identical label set to ingest.py / ingest_wiki.py."""
    try:
        nlp      = _get_nlp()
        doc      = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in NER_TYPES]
        print(f"  [classify_query]: NER → {entities}", file=sys.stderr)
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

    # POS-filtered texts (must match train_centroids.py filtering)
    intent_text = _pos_filter(query, INTENT_TAGS)
    tone_text   = _pos_filter(query, TONE_TAGS)
    has_domain  = "domain_labels" in store

    # Batch all POS-filtered texts into a single model.encode() call —
    # one GPU/MPS round-trip instead of 2-3 separate forward passes.
    if has_domain:
        domain_text = _pos_filter(query, DOMAIN_TAGS)
        batch_embs  = model.encode([intent_text, tone_text, domain_text])
        emb_intent_pos, emb_tone_pos, emb_domain_pos = (e.tolist() for e in batch_embs)
    else:
        batch_embs = model.encode([intent_text, tone_text])
        emb_intent_pos, emb_tone_pos = batch_embs[0].tolist(), batch_embs[1].tolist()

    # Signal-derived distance biases for intent classification.
    #
    # The corpus (declarative Wikipedia) produces a skewed centroid space:
    # "greeting" sits near the embedding mean and absorbs most short queries.
    # Rather than hardcoded rules we derive a per-intent bias multiplier from
    # POS-tag signals already present in the tagger output and apply it to the
    # centroid distances before nearest-neighbour selection.  A multiplier < 1.0
    # shrinks a centroid's effective distance (favours that label); > 1.0 grows
    # it (penalises).  The centroid model still makes the final decision.
    _intent_tokens = nltk.word_tokenize(query)
    _intent_tagged = nltk.pos_tag(_intent_tokens)
    _n             = max(len(_intent_tokens), 1)
    _has_wh        = any(t in {"WP", "WRB", "WDT"} for _, t in _intent_tagged)
    _has_uh        = any(t == "UH" for _, t in _intent_tagged)
    _vb_frac       = sum(1 for _, t in _intent_tagged if t.startswith("VB")) / _n
    _nnp_frac      = sum(1 for _, t in _intent_tagged if t in {"NNP", "NNPS"}) / _n

    def _intent_bias(label: str) -> float:
        b = 1.0
        # WH-tags (WP/WRB/WDT) → "what", "who", "when", "where" etc.
        if _has_wh:
            if label == "question": b *= 0.65
            if label == "greeting": b *= 1.4
        # Pure proper-noun phrase with no verbs (e.g. "James Clark Ross",
        # "Donkey Kong") → entity lookup, treat as question.
        if _nnp_frac > 0.5 and _vb_frac == 0:
            if label == "question": b *= 0.70
            if label == "greeting": b *= 1.4
        # Interjection tokens (UH: "hi", "hey", "hello") → greeting.
        if _has_uh:
            if label == "greeting": b *= 0.65
            if label == "question": b *= 1.4
        # Any verb present (non-greeting declarative/command structure).
        # Even one VB tag signals this is not a greeting — push greeting away.
        if _vb_frac > 0 and not _has_uh:
            if label == "greeting": b *= 1.3
        # High verb fraction → action/command, push harder.
        if _vb_frac > 0.4:
            if label == "greeting": b *= 1.2
            if label in ("command", "statement"): b *= 0.85
        return b

    # Biased nearest-centroid for intent.
    _intent_labels = store["intent_labels"]
    _best_intent, _best_dist = None, float("inf")
    for _i, _lbl in enumerate(_intent_labels):
        _d = (
            0.7 * _euclidean(emb_full, store["intent_full_centroids"][_i])
            + 0.3 * _euclidean(emb_intent_pos, store["intent_pos_centroids"][_i])
        ) * _intent_bias(_lbl)
        if _d < _best_dist:
            _best_dist, _best_intent = _d, _lbl
    intent = _best_intent
    print(f"  [classify_query]: intent via signal-biased centroid -> '{intent}' "
          f"(wh={_has_wh} uh={_has_uh} nnp={_nnp_frac:.2f} vb={_vb_frac:.2f})", file=sys.stderr)
    tone = _nearest_blended(
        emb_full, emb_tone_pos,
        store["tone_full_centroids"], store["tone_pos_centroids"],
        store["tone_labels"],
    )

    if has_domain:
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
    # Fix #13: use the absolute _DEFAULT_CENTROIDS path (avoids CWD-dependent
    # relative-path breakage when called from a different working directory).
    centroids_path = _DEFAULT_CENTROIDS
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
