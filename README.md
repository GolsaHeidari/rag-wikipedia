# 📚 RAG-Wikipedia Project
This project implements a simple **Retrieval-Augmented Generation (RAG)** pipeline using a Wikipedia-based dataset.  
It combines document retrieval with a large language model to answer questions using relevant context.

## 🚀 Overview
The system follows a standard RAG pipeline:
1. Load a question
2. Retrieve relevant Wikipedia passages
3. Build a prompt with retrieved context
4. Generate an answer using an LLM
5. Evaluate results using multiple metrics

## 🤖 Models Used

- **Retriever:** `all-MiniLM-L6-v2` (Sentence Transformers)
- **Generator (LLM):** `Qwen/Qwen3-0.6B`
- **Dataset:** `rag-datasets/rag-mini-wikipedia`

## 📁 Project Structure

```text
rag-wikipedia/
├── src/                  # Core source code (retriever, generator, prompt builder)
├── data/                 # Dataset files (corpus.json, qa.json, results.json)
├── scripts/              # Evaluation and experiments
├── run_dataset.py        # Dataset preparation script
├── run_evaluation.py     # Evaluation pipeline (RAG + metrics)
├── main.py               # Simple inference entry point
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd rag-wikipedia
pip install -r requirements.txt
```

## ▶️ Usage
1. Run RAG inference
python main.py
2. Run dataset preparation
python run_dataset.py
3. Run evaluation pipeline
python run_evaluation.py

## 📊 Evaluation Metrics
The system evaluates generated answers using:
Semantic Similarity
SentenceTransformer embeddings
Measures meaning similarity
ROUGE-L
Longest common subsequence overlap
Measures lexical similarity
BLEU Score
N-gram overlap metric
Includes smoothing for short answers

## 📌 Example Pipeline
Question → Retriever → Context → Prompt → LLM → Answer

## 📈 Results
The evaluation script outputs:
Per-sample scores
Average semantic similarity
Average ROUGE-L score
Average BLEU score

## 📜 License
This project is for educational purposes.

## 🪜 Steps:
- Step 1: Repository setup
    * Defined separate folders for:
      - data/ → storing datasets (corpus and QA files)
      - src/ → source code (dataset processing, retriever, etc.)
    * Added main project files:
      - main.py → entry point of the application
      - run_dataset.py → script for dataset preparation
      - requirements.txt → dependencies
      - run_evaluation.py → Evaluation pipeline
      - README.md → project documentation
        
- Step 2: Dataset download and setup:
    * Used dataset from HuggingFace: rag-datasets/rag-mini-wikipedia
    * The dataset consists of:
      - text-corpus → Wikipedia passages
      - question-answer → QA pairs [:100]
    * Downloaded using the datasets library

- Step 3: Dataset preparation:
  - Implementation (src/dataset.py)
    * Loaded dataset using load_dataset
    * Extracted:
      - Full text corpus
      - First 100 QA samples
    * Converted dataset into Python lists
    * Saved locally as JSON files (corpus.json and qa.json)
    * Push JSON files to the repository
 
- Step 4: Retriever module:
  - Implementation (src/retriever.py)
    * Loaded corpus from corpus.json
    * Extracted text passages
    * Converted passages into embeddings using:
      - SentenceTransformer("all-MiniLM-L6-v2")
    * Encoded user query into embedding
    * Computed similarity using cosine similarity
    * Ranked documents using NumPy sorting
    * Returned top-k results
   
- Step 5: Prompt construction:
  - Implementation (src/prompt.py)
    * A prompt builder function takes:
      - user question
      - Retrieved documents (output from the retriever (Step 4))
    * It then formats them into a structured prompt.
   
- Step 6: LLM Generation
   * Load pretrained LLM model
   * Initialize tokenizer and model
   * Convert prompt to tokens
   * Generate response from the model
   * Decode tokens to text
   * Output the final generated answer
 
- Step 7: Evaluation Pipeline
    - Generate predicted answer:
      * Load question from dataset
      * Retrieve the top-k most relevant documents (from data/corpus.json)
      * Build prompt using retrieved context
      * Generate answer using LLM (Qwen/Qwen3-0.6B)
      * Clean the predicted answer
      * Store results for analysis
   - Evaluation Metrics:
      * Compare the true_answer with predicted_answer
      * Calculate Semantic Similarity and Average Semantic Similarity:
        - Computed using SentenceTransformers embeddings
        - Measures semantic closeness between predicted and true answers
        - Average semantic similarity is reported across all samples
      * Calculate ROUGE-L and Average ROUGE-L:
        - Measures longest common subsequence overlap
        - Evaluates lexical similarity between answers
        - Average ROUGE-L score is reported 
      * Calculate BLEU and Average BLEU:
        - Measures n-gram overlap between generated and reference answers
        - Uses smoothing for short-answer stability
        - Average BLEU score is reported
