import json
import math
import os
import re
import nltk

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except Exception:
    pass

# ── NER types — must match classify_query.py exactly ─────────────────────────
NER_TYPES = {"PERSON", "ORG", "GPE", "PRODUCT"}

# ── POS tag sets — must match classify_query.py exactly ──────────────────────
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
    """Batch-encode all sentences, classify intent/tone/domain.
    Returns a list of (intent, tone, domain) tuples.
    """
    embs_full   = model.encode(sentences,                                        batch_size=64, show_progress_bar=False)
    embs_intent = model.encode([_pos_filter(s, _INTENT_TAGS) for s in sentences], batch_size=64, show_progress_bar=False)
    embs_tone   = model.encode([_pos_filter(s, _TONE_TAGS)   for s in sentences], batch_size=64, show_progress_bar=False)

    has_domain = "domain_labels" in store
    if has_domain:
        embs_domain = model.encode([_pos_filter(s, _DOMAIN_TAGS) for s in sentences], batch_size=64, show_progress_bar=False)

    results = []
    for i in range(len(sentences)):
        intent = _nearest_blended(embs_full[i].tolist(), embs_intent[i].tolist(),
                                  store["intent_full_centroids"], store["intent_pos_centroids"],
                                  store["intent_labels"])
        tone   = _nearest_blended(embs_full[i].tolist(), embs_tone[i].tolist(),
                                  store["tone_full_centroids"], store["tone_pos_centroids"],
                                  store["tone_labels"])
        domain = (_nearest_blended(embs_full[i].tolist(), embs_domain[i].tolist(),
                                   store["domain_full_centroids"], store["domain_pos_centroids"],
                                   store["domain_labels"])
                  if has_domain else "general")
        results.append((intent, tone, domain))
    return results


def extract_date(text):
    match = re.search(r'\b(10|11|12|13|14|15|16|17|18|19|20)\d{2}\b', text)
    if match:
        return int(match.group(0))
    return None


def main():
    # ── Pass 1: tokenize Wikipedia corpus into sentences (CPU / NLTK) ────────
    try:
        with open('../data/corpus_wiki.txt', 'r', encoding='utf-8') as f:
            paragraphs = f.read().split('\n')
    except Exception as e:
        print(f"  [INGEST_WIKI]: Missing Wikipedia corpus: {e}", flush=True)
        return

    print(f"  [INGEST_WIKI]: Tokenizing {len(paragraphs):,} paragraphs …", flush=True)
    para_sentences = []
    all_sentences  = []
    for para in paragraphs:
        if not para.strip():
            continue
        sents = nltk.sent_tokenize(para)
        if sents:
            para_sentences.append(sents)
            all_sentences.extend(sents)

    print(f"  [INGEST_WIKI]: {len(all_sentences):,} sentences across {len(para_sentences):,} paragraphs", flush=True)

    # ── Pass 2: encode & classify intent / tone / domain (GPU / torch) ───────
    clf_model = None
    clf_store = None
    try:
        from sentence_transformers import SentenceTransformer
        from gpu_utils import get_device
        clf_model = SentenceTransformer("all-MiniLM-L6-v2", device=get_device())
        with open("../data/centroids.json") as f:
            clf_store = json.load(f)
        print("  [INGEST_WIKI]: Centroid model loaded — using symmetric classification.", flush=True)
    except Exception as e:
        print(f"  [INGEST_WIKI]: Centroid model unavailable ({e}) — using statement/neutral/general fallback.", flush=True)

    if clf_model is not None and clf_store is not None:
        classifications = []
        for i, sents in enumerate(para_sentences):
            classifications.extend(batch_classify(sents, clf_model, clf_store))
            if i % 500 == 0:
                print(f"  [INGEST_WIKI]: Classified {i:,}/{len(para_sentences):,} paragraphs …", flush=True)
    else:
        classifications = [("statement", "neutral", "general")] * len(all_sentences)

    # Release torch model before spaCy runs (avoids loky semaphore conflict
    # on Python 3.14).
    del clf_model

    # ── Pass 3: spaCy NER (identical logic to classify_query.py) ─────────────
    print(f"  [INGEST_WIKI]: Extracting entities with spaCy (batch) …", flush=True)
    entity_lists = []
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        for doc in nlp.pipe(all_sentences, batch_size=256):
            entity_lists.append([ent.text for ent in doc.ents if ent.label_ in NER_TYPES])
    except (ImportError, OSError) as e:
        print(f"  [INGEST_WIKI]: spaCy unavailable ({e}) — entities will be empty.", flush=True)
        entity_lists = [[] for _ in all_sentences]

    # ── Pass 4: date extraction + assemble output ─────────────────────────────
    processed_data = []
    cls_iter    = iter(classifications)
    entity_iter = iter(entity_lists)
    for sents in para_sentences:
        for seq, sent in enumerate(sents):
            intent, tone, domain = next(cls_iter)
            entities = next(entity_iter)
            processed_data.append({
                "sequence_id": seq,
                "text":        sent,
                "tokens":      nltk.word_tokenize(sent),
                "intent":      intent,
                "tone":        tone,
                "domain":      domain,
                "entities":    entities,
                "dated":       extract_date(sent),
            })

    with open('../data/corpus_wiki_tmp.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    print(f"  [INGEST_WIKI]: Complete — {len(processed_data):,} sentences exported to data/corpus_wiki_tmp.json", flush=True)


if __name__ == "__main__":
    main()
