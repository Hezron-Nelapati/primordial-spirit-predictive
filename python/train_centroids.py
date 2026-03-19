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

    # Domain training data — (Text, Domain)
    # Nouns carry domain signal: domain POS filter uses only {"NN","NNS","NNP","NNPS"}
    domain_training_data = [
        # tech (5)
        ("The server is offline and the network router is down.", "tech"),
        ("Our API endpoint returns a CPU usage metric.", "tech"),
        ("Deploy the code to the cloud system and restart the daemon.", "tech"),
        ("The database query is slow due to missing index on the data table.", "tech"),
        ("Check the bandwidth and latency on the network interface card.", "tech"),
        # finance (5)
        ("The bank charges interest on every loan and credit account.", "finance"),
        ("Transfer money to the investment fund before the payment deadline.", "finance"),
        ("The ATM dispenses cash and shows your account balance.", "finance"),
        ("Stock market volatility affects bond yield and portfolio returns.", "finance"),
        ("The mortgage rate and insurance premium increased this quarter.", "finance"),
        # science (5)
        ("Quantum mechanics describes the behavior of particles and atoms.", "science"),
        ("The photon energy wave collapses into a definite particle state.", "science"),
        ("Newton's laws relate force, mass, and acceleration.", "science"),
        ("The molecule bonds release energy during the chemical reaction.", "science"),
        ("Gravitational waves propagate through spacetime at the speed of light.", "science"),
        # geography (5)
        ("Paris is the capital city of France and home to the Eiffel Tower.", "geography"),
        ("Rome is an ancient city in Italy with a rich history.", "geography"),
        ("The Amazon river flows through the rainforest in South America.", "geography"),
        ("Mount Everest is the highest peak in the Himalayan mountain range.", "geography"),
        ("The Sahara desert spans multiple countries across North Africa.", "geography"),
        # general (5)
        ("What time does the meeting start today?", "general"),
        ("Please remind me to call back later this afternoon.", "general"),
        ("I am not sure what to do next.", "general"),
        ("Could you help me understand this concept?", "general"),
        ("Everything seems to be working fine now.", "general"),
    ]

    domain_texts   = [d[0] for d in domain_training_data]
    domain_labels  = [d[1] for d in domain_training_data]

    domain_pos_tags = {"NN", "NNS", "NNP", "NNPS"}

    intent_texts = []
    tone_texts = []

    print("Extracting intent/tone POS tags...")
    for text in texts:
        tokens = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokens)

        i_words = [w for w, t in tags if t in intent_pos_tags]
        intent_texts.append(" ".join(i_words) if i_words else text)

        t_words = [w for w, t in tags if t in tone_pos_tags]
        tone_texts.append(" ".join(t_words) if t_words else text)

    print("Extracting domain POS tags...")
    domain_pos_texts = []
    for text in domain_texts:
        tokens = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        d_words = [w for w, t in tags if t in domain_pos_tags]
        domain_pos_texts.append(" ".join(d_words) if d_words else text)

    print("Computing embeddings...")
    emb_full   = model.encode(texts)
    emb_intent = model.encode(intent_texts)
    emb_tone   = model.encode(tone_texts)

    emb_domain_full = model.encode(domain_texts)
    emb_domain_pos  = model.encode(domain_pos_texts)

    print("Fitting centroids...")
    clf_intent_full = NearestCentroid().fit(emb_full, intents)
    clf_intent_pos  = NearestCentroid().fit(emb_intent, intents)

    clf_tone_full = NearestCentroid().fit(emb_full, tones)
    clf_tone_pos  = NearestCentroid().fit(emb_tone, tones)

    clf_domain_full = NearestCentroid().fit(emb_domain_full, domain_labels)
    clf_domain_pos  = NearestCentroid().fit(emb_domain_pos,  domain_labels)

    store = {
        "intent_labels":        clf_intent_full.classes_.tolist(),
        "intent_full_centroids": clf_intent_full.centroids_.tolist(),
        "intent_pos_centroids":  clf_intent_pos.centroids_.tolist(),

        "tone_labels":        clf_tone_full.classes_.tolist(),
        "tone_full_centroids": clf_tone_full.centroids_.tolist(),
        "tone_pos_centroids":  clf_tone_pos.centroids_.tolist(),

        "domain_labels":        clf_domain_full.classes_.tolist(),
        "domain_full_centroids": clf_domain_full.centroids_.tolist(),
        "domain_pos_centroids":  clf_domain_pos.centroids_.tolist(),
    }

    out_path = "data/centroids.json"
    with open(out_path, "w") as f:
        json.dump(store, f)

    print(f"Exported to {out_path}.")
    print(f"Intents: {clf_intent_full.classes_.tolist()}")
    print(f"Tones:   {clf_tone_full.classes_.tolist()}")
    print(f"Domains: {clf_domain_full.classes_.tolist()}")

if __name__ == "__main__":
    main()
