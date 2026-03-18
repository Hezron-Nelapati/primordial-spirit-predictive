import datasets
import os

def main():
    print("Downloading Massive Wikipedia dataset slice...")
    # 'simple' english wikipedia has shorter crisp sentences (great for RAG testing). 
    # 1% split is about ~2,000 Wikipedia articles.
    try:
        ds = datasets.load_dataset("wikipedia", "20220301.simple", split="train[:1%]", trust_remote_code=True) 
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    total_sentences = 0
    os.makedirs("../data", exist_ok=True)
    
    with open("../data/corpus_v3_massive.txt", "w", encoding="utf-8") as f:
        for doc in ds:
            # Clean up the markdown spacing
            text = doc['text'].replace('\n\n', '\n').strip()
            f.write(text + "\n")
            total_sentences += len(text.split("."))

    print(f"✅ Downloaded Massive V3 Corpus to data/corpus_v3_massive.txt.")
    print(f"✅ Total Topological Sentences Extracted: ~{total_sentences}")

if __name__ == "__main__":
    main()
