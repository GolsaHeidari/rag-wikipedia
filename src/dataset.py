from datasets import load_dataset
import json
import os

def load_and_save_dataset():
    print("📥 Loading dataset...")

    corpus_ds = load_dataset(
        "rag-datasets/rag-mini-wikipedia",
        "text-corpus"
    )

    qa_ds = load_dataset(
        "rag-datasets/rag-mini-wikipedia",
        "question-answer"
    )

    # 🔍 SAFELY GET CORPUS
    if isinstance(corpus_ds, dict):
        corpus_data = list(corpus_ds[list(corpus_ds.keys())[0]])
    else:
        corpus_data = list(corpus_ds)

    # 🔍 SAFELY GET QA
    if isinstance(qa_ds, dict):
        qa_data = list(qa_ds[list(qa_ds.keys())[0]])[:100]
    else:
        qa_data = list(qa_ds)[:100]

    os.makedirs("data", exist_ok=True)

    with open("data/corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus_data, f, ensure_ascii=False)

    with open("data/qa.json", "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False)

    print("✅ Dataset saved successfully!")