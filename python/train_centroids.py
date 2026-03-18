import json
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestCentroid

def main():
    print("Loading models...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng')
        nltk.download('punkt_tab')

    # (Text, Intent, Tone)
    training_data = [
        ("Hi there", "greeting", "casual"),
        ("Hello", "greeting", "casual"),
        ("Good morning to you!", "greeting", "polite"),
        ("How are you doing today?", "greeting", "polite"),
        ("What is the temperature outside?", "question", "neutral"),
        ("Can you explain how the system works?", "explain", "neutral"),
        ("Please explain the graph walk algorithm.", "explain", "neutral"),
        ("Thanks so much, this is awesome!", "gratitude", "excited"),
        ("I appreciate your help.", "gratitude", "polite"),
        ("This is terrible, everything is broken!", "complaint", "angry"),
        ("Could you fix the bug please?", "request", "polite"),
        ("Stop doing that right now.", "command", "angry"),
        ("Please turn off the lights.", "command", "polite"),
    ]

    texts = [d[0] for d in training_data]
    intents = [d[1] for d in training_data]
    tones = [d[2] for d in training_data]

    intent_pos_tags = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "NN", "NNS", "NNP", "NNPS"}
    tone_pos_tags = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS", "UH"}

    intent_texts = []
    tone_texts = []

    print("Extracting POS tags...")
    for text in texts:
        tokens = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        
        # Extract intent words
        i_words = [w for w, t in tags if t in intent_pos_tags]
        intent_texts.append(" ".join(i_words) if i_words else text)
        
        # Extract tone words
        t_words = [w for w, t in tags if t in tone_pos_tags]
        tone_texts.append(" ".join(t_words) if t_words else text)

    print("Computing embeddings...")
    emb_full = model.encode(texts)
    emb_intent = model.encode(intent_texts)
    emb_tone = model.encode(tone_texts)

    print("Fitting centroids...")
    clf_intent_full = NearestCentroid().fit(emb_full, intents)
    clf_intent_pos = NearestCentroid().fit(emb_intent, intents)
    
    clf_tone_full = NearestCentroid().fit(emb_full, tones)
    clf_tone_pos = NearestCentroid().fit(emb_tone, tones)

    store = {
        "intent_labels": clf_intent_full.classes_.tolist(),
        "intent_full_centroids": clf_intent_full.centroids_.tolist(),
        "intent_pos_centroids": clf_intent_pos.centroids_.tolist(),
        
        "tone_labels": clf_tone_full.classes_.tolist(),
        "tone_full_centroids": clf_tone_full.centroids_.tolist(),
        "tone_pos_centroids": clf_tone_pos.centroids_.tolist(),
    }

    out_path = "data/centroids.json"
    with open(out_path, "w") as f:
        json.dump(store, f)

    print(f"Exported to {out_path}.")
    print(f"Intents: {clf_intent_full.classes_.tolist()}")
    print(f"Tones: {clf_tone_full.classes_.tolist()}")

if __name__ == "__main__":
    main()
