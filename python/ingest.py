import json
import math
import re
import nltk

# Download required NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
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


def corpus_batch_classify(all_sentences, model, store, batch_size=256):
    """Encode the entire corpus in one pass.

    Collects every sentence up-front then calls model.encode() once per
    text-variant (full / intent-pos / tone-pos / domain-pos) with a large
    batch_size.  This gives the GPU/MPS a single large matrix multiply instead
    of thousands of tiny ones — typically 10-50× faster than per-paragraph
    batching.

    Returns a list of (intent, tone, domain) tuples, one per sentence.
    """
    n = len(all_sentences)
    if n == 0:
        return []

    print(f"  [INGEST]: Encoding {n:,} sentences (full text) …", flush=True)
    embs_full = model.encode(all_sentences, batch_size=batch_size, show_progress_bar=True)

    print(f"  [INGEST]: Encoding intent-POS filtered texts …", flush=True)
    intent_texts = [_pos_filter(s, _INTENT_TAGS) for s in all_sentences]
    embs_intent  = model.encode(intent_texts, batch_size=batch_size, show_progress_bar=True)

    print(f"  [INGEST]: Encoding tone-POS filtered texts …", flush=True)
    tone_texts = [_pos_filter(s, _TONE_TAGS) for s in all_sentences]
    embs_tone  = model.encode(tone_texts, batch_size=batch_size, show_progress_bar=True)

    has_domain = "domain_labels" in store
    if has_domain:
        print(f"  [INGEST]: Encoding domain-POS filtered texts …", flush=True)
        domain_texts = [_pos_filter(s, _DOMAIN_TAGS) for s in all_sentences]
        embs_domain  = model.encode(domain_texts, batch_size=batch_size, show_progress_bar=True)

    results = []
    for i in range(n):
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
            domain = mock_classify(all_sentences[i])[2]
        results.append((intent, tone, domain))

    return results


def extract_entities(text):
    """Extract named-entity phrases from POS tags.

    Groups consecutive NNP/NNPS tokens into entity phrases
    (e.g. ['New', 'York', 'City'] → 'New York City').

    Replaces nltk.ne_chunk which:
      - loads a heavy MaxEnt model per call
      - spawns loky/multiprocessing workers that collide with the
        torch multiprocessing pool left open after batch encoding,
        causing semaphore leaks and non-zero exit on Python 3.14
    """
    try:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        entities, current = [], []
        for word, tag in tagged:
            if tag in ('NNP', 'NNPS'):
                current.append(word)
            elif current:
                entities.append(' '.join(current))
                current = []
        if current:
            entities.append(' '.join(current))
        return entities
    except Exception:
        return []


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
    clf_model = None
    clf_store = None
    try:
        from sentence_transformers import SentenceTransformer
        from gpu_utils import get_device
        clf_model = SentenceTransformer("all-MiniLM-L6-v2", device=get_device())
        with open("../data/centroids.json") as f:
            clf_store = json.load(f)
        print("  [INGEST]: Centroid model loaded — using symmetric classification.", flush=True)
    except Exception as e:
        print(f"  [INGEST]: Centroid model unavailable ({e}) — falling back to mock_classify().", flush=True)

    with open('../data/corpus.txt', 'r', encoding='utf-8') as f:
        paragraphs = f.read().split('\n')

    # ── Pass 1: tokenize entire corpus into sentences (CPU, NLTK) ────────────
    # Collect all sentences first so we can encode the whole corpus in one shot.
    print(f"  [INGEST]: Tokenizing {len(paragraphs):,} lines into sentences …", flush=True)
    para_sentences = []   # list of (para_idx, [sentence, ...])
    all_sentences  = []   # flat list — fed to corpus_batch_classify
    for para in paragraphs:
        if not para.strip():
            continue
        sents = nltk.sent_tokenize(para)
        if sents:
            para_sentences.append(sents)
            all_sentences.extend(sents)

    print(f"  [INGEST]: {len(all_sentences):,} sentences across {len(para_sentences):,} paragraphs", flush=True)

    # ── Pass 2: encode & classify the full corpus in one batch ───────────────
    if clf_model is not None and clf_store is not None:
        classifications = corpus_batch_classify(all_sentences, clf_model, clf_store)
    else:
        classifications = [mock_classify(s) for s in all_sentences]

    # ── Pass 3: NLTK entity/date extraction + assemble output ────────────────
    print(f"  [INGEST]: Extracting entities and dates …", flush=True)
    processed_data = []
    cls_iter = iter(classifications)
    for sents in para_sentences:
        for seq, sent in enumerate(sents):
            intent, tone, domain = next(cls_iter)
            processed_data.append({
                "sequence_id": seq,
                "text": sent,
                "tokens": nltk.word_tokenize(sent),
                "intent": intent,
                "tone": tone,
                "domain": domain,
                "entities": extract_entities(sent),
                "dated": extract_date(sent),
            })

    with open('../data/corpus_tmp.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    print(f"Ingestion complete — {len(processed_data):,} sentences exported to data/corpus_tmp.json", flush=True)


if __name__ == "__main__":
    main()
