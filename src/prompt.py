def prompt_builder(query, retrieved_docs):
    context = ""
    for i , doc in enumerate(retrieved_docs):
        context += f"\nDocument {i+1}:\n{doc['text']}\n"

    prompt = f"""
              You are a helpful assistant. Answer the question using the context below.
                {context}
                Question:
                {query}
                Answer:  
              """

    return prompt 