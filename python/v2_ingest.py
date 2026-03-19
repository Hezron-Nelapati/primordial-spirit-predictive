import json
import math
import re
import nltk

# Download required NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('maxent_ne_chunker_tab', quiet=True)
    nltk.download('words', quiet=True)
except Exception:
    pass

# ── POS tag sets — must mirror classify_query.py exactly ─────────────────────
_INTENT_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "NN", "NNS", "NNP", "NNPS"}
_TONE_TAGS   = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS", "UH"}
_DOMAIN_TAGS = {"NN", "NNS", "NNP", "NNPS"}


def _pos_filter(text, tag_set):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    words  = [w for w, t in tagged if t in tag_set]
    return " ".join(words) if words else text


def _euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _nearest_blended(emb_full, emb_pos, full_centroids, pos_centroids, labels):
    best_idx, min_dist = 0, float("inf")
    for i in range(len(labels)):
        d = 0.7 * _euclidean(emb_full, full_centroids[i]) + 0.3 * _euclidean(emb_pos, pos_centroids[i])
        if d < min_dist:
            min_dist, best_idx = d, i
    return labels[best_idx]


def centroid_classify(text, model, store):
    """Classify intent, tone, and domain using the centroid model.
    Mirrors classify_query.py logic exactly for symmetric training/runtime tags.
    """
    emb_full       = model.encode(text).tolist()
    emb_intent_pos = model.encode(_pos_filter(text, _INTENT_TAGS)).tolist()
    emb_tone_pos   = model.encode(_pos_filter(text, _TONE_TAGS)).tolist()

    intent = _nearest_blended(emb_full, emb_intent_pos,
                               store["intent_full_centroids"], store["intent_pos_centroids"],
                               store["intent_labels"])
    tone   = _nearest_blended(emb_full, emb_tone_pos,
                               store["tone_full_centroids"], store["tone_pos_centroids"],
                               store["tone_labels"])

    if "domain_labels" in store:
        emb_domain_pos = model.encode(_pos_filter(text, _DOMAIN_TAGS)).tolist()
        domain = _nearest_blended(emb_full, emb_domain_pos,
                                  store["domain_full_centroids"], store["domain_pos_centroids"],
                                  store["domain_labels"])
    else:
        domain = mock_classify(text)[2]  # keyword fallback for old centroids.json

    return intent, tone, domain

def extract_entities(text):
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    chunks = nltk.ne_chunk(tags)
    entities = []
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            entities.append(' '.join(c[0] for c in chunk))
    return entities

def extract_date(text):
    match = re.search(r'\b(19|20)\d{2}\b', text)
    if match:
        return int(match.group(0))
    return None

def mock_classify(text):
    # For a deterministic fast protoype under tight compute constraints, we emulate the NearestCentroid.
    text_lower = text.lower()
    
    intent = "statement"
    if "?" in text: intent = "question"
    elif "reset" in text_lower or "cancel" in text_lower: intent = "command"
    
    tone = "neutral"
    if "angry" in text_lower or "!" in text: tone = "angry"
    elif "apology" in text_lower: tone = "polite"
    
    domain = "general"
    if "server" in text_lower or "router" in text_lower: domain = "tech"
    elif "quantum" in text_lower or "physics" in text_lower: domain = "science"
    elif "bank" in text_lower: domain = "finance"
    elif "france" in text_lower or "paris" in text_lower or "rome" in text_lower or "italy" in text_lower: domain = "geography"
    
    return intent, tone, domain

def main():
    # Attempt to load centroid model once for symmetric training/runtime classification.
    # Falls back to mock_classify() if sentence-transformers or centroids.json unavailable.
    clf_model = None
    clf_store = None
    try:
        from sentence_transformers import SentenceTransformer
        clf_model = SentenceTransformer("all-MiniLM-L6-v2")
        with open("../data/centroids.json") as f:
            clf_store = json.load(f)
        print("  [V2_INGEST]: Centroid model loaded — using symmetric classification.")
    except Exception as e:
        print(f"  [V2_INGEST]: Centroid model unavailable ({e}) — falling back to mock_classify().")

    with open('../data/corpus_v2.txt', 'r', encoding='utf-8') as f:
        paragraphs = f.read().split('\n')

    processed_data = []

    for para in paragraphs:
        if not para.strip(): continue

        # Guardrail: Sentence Queuing (nltk.sent_tokenize)
        sentences = nltk.sent_tokenize(para)
        for seq, sent in enumerate(sentences):
            # Guardrail: Punctuation Anchors (word_tokenize natively splits punctuation into independent tokens!)
            tokens = nltk.word_tokenize(sent)

            # Guardrail: Zero-Compute Entities & Temporal Ties
            entities = extract_entities(sent)
            dated = extract_date(sent)

            # Semantic Tags — centroid model when available, else keyword heuristic.
            if clf_model is not None and clf_store is not None:
                intent, tone, domain = centroid_classify(sent, clf_model, clf_store)
            else:
                intent, tone, domain = mock_classify(sent)

            processed_data.append({
                "sequence_id": seq,
                "text": sent,
                "tokens": tokens,
                "intent": intent,
                "tone": tone,
                "domain": domain,
                "entities": entities,
                "dated": dated
            })

    with open('../data/v2_graph_edges.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    print("V2 Python Ingestion Complete! Exported to data/v2_graph_edges.json")

if __name__ == "__main__":
    main()
