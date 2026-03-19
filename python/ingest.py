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


def batch_classify(sentences, model, store):
    """Batch-encode all sentences at once, then classify each.
    Batch encoding is orders of magnitude faster than per-sentence calls —
    one matmul across N sentences vs N separate forward passes.
    Returns a list of (intent, tone, domain) tuples.
    """
    embs_full = model.encode(sentences, batch_size=64, show_progress_bar=False)

    intent_texts = [_pos_filter(s, _INTENT_TAGS) for s in sentences]
    tone_texts   = [_pos_filter(s, _TONE_TAGS)   for s in sentences]
    embs_intent  = model.encode(intent_texts, batch_size=64, show_progress_bar=False)
    embs_tone    = model.encode(tone_texts,   batch_size=64, show_progress_bar=False)

    has_domain = "domain_labels" in store
    if has_domain:
        domain_texts = [_pos_filter(s, _DOMAIN_TAGS) for s in sentences]
        embs_domain  = model.encode(domain_texts, batch_size=64, show_progress_bar=False)

    results = []
    for i in range(len(sentences)):
        intent = _nearest_blended(embs_full[i].tolist(), embs_intent[i].tolist(),
                                  store["intent_full_centroids"], store["intent_pos_centroids"],
                                  store["intent_labels"])
        tone   = _nearest_blended(embs_full[i].tolist(), embs_tone[i].tolist(),
                                  store["tone_full_centroids"], store["tone_pos_centroids"],
                                  store["tone_labels"])
        if has_domain:
            domain = _nearest_blended(embs_full[i].tolist(), embs_domain[i].tolist(),
                                      store["domain_full_centroids"], store["domain_pos_centroids"],
                                      store["domain_labels"])
        else:
            domain = mock_classify(sentences[i])[2]
        results.append((intent, tone, domain))
    return results


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
    # Range 1000–2099 — matches ingest_wiki.py and reasoning.rs for consistency.
    match = re.search(r'\b(10|11|12|13|14|15|16|17|18|19|20)\d{2}\b', text)
    if match:
        return int(match.group(0))
    return None


def mock_classify(text):
    # Keyword heuristic fallback when centroid model is unavailable.
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
    # Load centroid model once for the full corpus — batch encoding reuses the
    # same loaded weights across all 98k+ sentences instead of reloading per call.
    clf_model = None
    clf_store = None
    try:
        from sentence_transformers import SentenceTransformer
        from gpu_utils import get_device
        clf_model = SentenceTransformer("all-MiniLM-L6-v2", device=get_device())
        with open("../data/centroids.json") as f:
            clf_store = json.load(f)
        print("  [INGEST]: Centroid model loaded — using symmetric classification.")
    except Exception as e:
        print(f"  [INGEST]: Centroid model unavailable ({e}) — falling back to mock_classify().")

    with open('../data/corpus.txt', 'r', encoding='utf-8') as f:
        paragraphs = f.read().split('\n')

    processed_data = []
    total_paragraphs = len([p for p in paragraphs if p.strip()])
    print(f"  [INGEST]: Processing {total_paragraphs} paragraphs …")

    for i, para in enumerate(paragraphs):
        if not para.strip():
            continue

        sentences = nltk.sent_tokenize(para)
        if not sentences:
            continue

        # Batch-classify all sentences in this paragraph at once.
        # One model.encode() call per paragraph instead of one per sentence.
        if clf_model is not None and clf_store is not None:
            classifications = batch_classify(sentences, clf_model, clf_store)
        else:
            classifications = [mock_classify(s) for s in sentences]

        for seq, (sent, (intent, tone, domain)) in enumerate(zip(sentences, classifications)):
            tokens   = nltk.word_tokenize(sent)
            entities = extract_entities(sent)
            dated    = extract_date(sent)

            processed_data.append({
                "sequence_id": seq,
                "text": sent,
                "tokens": tokens,
                "intent": intent,
                "tone": tone,
                "domain": domain,
                "entities": entities,
                "dated": dated,
            })

        if i % 5000 == 0 and i > 0:
            print(f"  [INGEST]: Parsed {i}/{total_paragraphs} paragraphs …")

    with open('../data/corpus_tmp.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    print(f"Ingestion complete — {len(processed_data)} sentences exported to data/corpus_tmp.json")


if __name__ == "__main__":
    main()
