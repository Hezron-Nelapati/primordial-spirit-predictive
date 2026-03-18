import json
import re
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('maxent_ne_chunker_tab', quiet=True)
    nltk.download('words', quiet=True)
except Exception:
    pass

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
            
            # Semantic Tags (Intent, Tone, Domain)
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
