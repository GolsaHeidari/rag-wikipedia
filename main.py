from src.retriever import Retriever
from src.prompt import prompt_builder
from src.generator import Generator

def main():
    top_k = 5
    corpus_path = "data/corpus.json"
    query = "Who was Lincoln?"
    model_name = "Qwen/Qwen3-0.6B"
    max_new_tokens = 100

    retriever = Retriever(corpus_path)
    results = retriever.retrieve(query, top_k)

    for i, item in enumerate(results):
        print(f"\nRank {i+1}")
        print(item["text"][:200])
        print("Score: ", item["score"])

    prompt = prompt_builder(query, results)
    print("\n-----PROMPT-----\n")
    print(prompt)

    generator = Generator(model_name)
    response = generator.generate(prompt, max_new_tokens)
    print("\n===== FINAL ANSWER =====\n")
    print(response)

if __name__ == "__main__":
    main()