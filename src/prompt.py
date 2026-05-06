def prompt_builder(query, retrieved_docs):
    context = ""
    for i , doc in enumerate(retrieved_docs):
        context += f"\nDocument {i+1}:\n{doc['text']}\n"

    prompt = f"""
    You are a strict question answering system.

    Rules:
    - Output ONLY one short phrase.
    - No explanations.
    - No extra words.
    
    Context:
    {context}

    Question:
    {query}

    Answer (one short phrase only):
    """
    return prompt 