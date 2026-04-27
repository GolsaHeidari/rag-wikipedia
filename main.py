from src.retriever import Retriever
from src.prompt import prompt_builder

def main():
    top_k = 5
    corpus_path = "data/corpus.json"
    query = "Who was Lincoln?"

    retriever = Retriever(corpus_path)
    results = retriever.retrieve(query, top_k)

    for i, item in enumerate(results):
        print(f"\nRank {i+1}")
        print(item["text"][:200])
        print("Score: ", item["score"])

    prompt = prompt_builder(query, results)
    print("\n-----PROMPT-----\n")
    print(prompt) 

if __name__ == "__main__":
    main()