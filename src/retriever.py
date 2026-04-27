import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, corpus_path):
       with open(corpus_path, "r", encoding="utf-8") as f:
           corpus = json.load(f)

       self.texts = []
       for item in corpus:
           self.texts.append(item["passage"])

       self.model = SentenceTransformer("all-MiniLM-L6-v2")
       self.embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def retrieve(self, query, top_k=5):
        query_embeddings = self.model.encode([query])
        scores = cosine_similarity(query_embeddings, self.embeddings)[0]
        sorted_top_k_index = np.argsort(scores)[::-1][:top_k]

        results = []
        for item in sorted_top_k_index:
            results.append({
                "text": self.texts[item],
                "score": float(scores[item])
            })

        return results    