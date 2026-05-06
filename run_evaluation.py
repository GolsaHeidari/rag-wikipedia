import json
from src.retriever import Retriever
from src.generator import Generator
from src.prompt import prompt_builder

def run_evaluation():
    with open("data/qa.json", "r", encoding = "utf-8") as f:
        qa_data = json.load(f)
        qa_data = qa_data[:3]

    retriever = Retriever("data/corpus.json")
    generator = Generator("Qwen/Qwen3-0.6B")

    results = []
    for i, sample in enumerate(qa_data):
        print(f"Processing {i+1}/{len(qa_data)}")

        query = sample["question"]
        true_answer = sample["answer"]
        
        docs = retriever.retrieve(query, top_k = 5)
        prompt = prompt_builder(query, docs)

        predicted_answer = generator.generate(prompt, max_new_tokens = 20)
                
        results.append({
            "question" : query,
            "true_answer" : true_answer,
            "predicted_answer" : predicted_answer
        })

    with open("data/results.json", "w", encoding = "utf-8") as f:
        json.dump(results, f, ensure_ascii = False, indent = 2) 

    print("--- EVALUATION COMPLETED ---")

  
if __name__ == "__main__":
    run_evaluation()