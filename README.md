# RAG-Wikipedia project
This project implements a basic Retrieval-Augmented Generator (RAG) pipeline using a Wikipedia database.
- Dataset: rag-mini-wikipedia
- Retriever model: all-MiniLM-L6-v2
- LLM: Qwen3-0.6B

## Steps:
- Step 1: Repository setup
    * Defined separate folders for:
      - data/ → storing datasets (corpus and QA files)
      - src/ → source code (dataset processing, retriever, etc.)
    * Added main project files:
      - main.py → entry point of the application
      - run_dataset.py → script for dataset preparation
      - requirements.txt → dependencies
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
