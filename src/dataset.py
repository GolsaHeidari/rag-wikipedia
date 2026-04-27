from datasets import load_dataset
import json
import os

def load_and_save_dataset():
    corpus_ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
    qa_ds = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")

    if hasattr(corpus_ds, "keys"):
        corpus_split = list(corpus_ds.keys())[0]
        corpus_data = list(corpus_ds[corpus_split])
    else:
        corpus_data = list(corpus_ds)

    if hasattr(qa_ds, "keys"):
        qa_split = list(qa_ds.keys())[0]
        qa_data = list(qa_ds[qa_split])[:100]
    else:
        qa_data = list(qa_ds)[:100]

    os.makedirs("data", exist_ok=True)    

    with open("data/corpus.json", "w", encoding = "utf8") as f:
        json.dump(corpus_data, f, ensure_ascii=False)  

    with open("data/qa.json", "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False)                  