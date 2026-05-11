import json
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from src.retriever import Retriever
from src.generator import Generator
from src.prompt import prompt_builder

def calculate_similarity(similarity_model, true, pred):
    true_embed = similarity_model.encode(true, convert_to_tensor = True)
    pred_embed = similarity_model.encode(pred, convert_to_tensor = True)
    similarity = util.cos_sim(true_embed, pred_embed).item()
    return similarity

def calculate_bleu(reference, prediction):
    reference_tokens = nltk.word_tokenize(reference.lower())
    prediction_tokens = nltk.word_tokenize(prediction.lower())

    smooth = SmoothingFunction().method4

    score = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function = smooth)

    return score


def run_evaluation():
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    with open("data/qa.json", "r", encoding = "utf-8") as f:
        qa_data = json.load(f)
        qa_data = qa_data[:3]

    retriever = Retriever("data/corpus.json")
    generator = Generator("Qwen/Qwen3-0.6B")

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer = True)

    results = []
    for i, sample in enumerate(qa_data):
        print(f"Processing {i+1}/{len(qa_data)}")

        query = sample["question"]
        true_answer = sample["answer"]
        true_answer = true_answer.strip()
        
        docs = retriever.retrieve(query, top_k = 5)
        prompt = prompt_builder(query, docs)

        predicted_answer = generator.generate(prompt, max_new_tokens = 20)
        predicted_answer = predicted_answer.strip()

        similarity = calculate_similarity(
                similarity_model,
                true_answer,
                predicted_answer)
        
        rouge_score = rouge.score(true_answer, predicted_answer)["rougeL"].fmeasure

        bleu_score = calculate_bleu(true_answer, predicted_answer)
                
        results.append({
            "question" : query,
            "true_answer" : true_answer,
            "predicted_answer" : predicted_answer,
            "similarity" : similarity,
            "rougeL" : rouge_score,
            "bleu_score": bleu_score
        })



    with open("data/results.json", "w", encoding = "utf-8") as f:
        json.dump(results, f, ensure_ascii = False, indent = 2) 

    total_similarity = 0
    for r in results:
        total_similarity +=r["similarity"]

    ave_sem_similarity = total_similarity/len(results)
    print("Average semantic similarity is: ", ave_sem_similarity)

    total_rouge = 0
    for r in results:
        total_rouge += r["rougeL"]

    ave_rouge_score = total_rouge/ len(results)
    print("Average Rouge-L is: ", ave_rouge_score)

    total_bleu = 0
    for r in results:
        total_bleu += r["bleu_score"]

    ave_bleu = total_bleu/ len(results)
    print("Average BLEU is: ", ave_bleu)     

    print("--- EVALUATION COMPLETED ---")

  
if __name__ == "__main__":
    run_evaluation()