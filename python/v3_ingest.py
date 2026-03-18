import json
import re
import nltk

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
    match = re.search(r'\b(10|11|12|13|14|15|16|17|18|19|20)\d{2}\b', text)
    if match:
        return int(match.group(0))
    return None

def main():
    try:
        with open('../data/corpus_v3_massive.txt', 'r', encoding='utf-8') as f:
            paragraphs = f.read().split('\n')
    except Exception as e:
        print("Missing Wikipedia Corpus!")
        return
        
    processed_data = []
    print(f"  [PYTHON_INGEST]: Processing {len(paragraphs)} Massive Wikipedia Articles...")
    
    for i, para in enumerate(paragraphs):
        if not para.strip(): continue
        
        sentences = nltk.sent_tokenize(para)
        for seq, sent in enumerate(sentences):
            tokens = nltk.word_tokenize(sent)
            entities = extract_entities(sent)
            dated = extract_date(sent)
            
            # Massive dataset fallback logic: Everything is a general statement geometry node
            processed_data.append({
                "sequence_id": seq,
                "text": sent,
                "tokens": tokens,
                "intent": "statement",
                "tone": "neutral",
                "domain": "general",
                "entities": entities,
                "dated": dated
            })
            
        if i % 500 == 0:
            print(f"  [PYTHON_INGEST]: Parsed {i} documents into geometric structural queues...")
            
    with open('../data/v3_graph_edges.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    print("✅ V3 Python Ingestion Complete! Exported thousands of nodes to data/v3_graph_edges.json")

if __name__ == "__main__":
    main()
