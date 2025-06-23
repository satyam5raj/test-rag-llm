def build_prompt(relevant_docs: list[str], query: str) -> str:
    context = "\n".join(relevant_docs)
    return f"""You are a helpful assistant. Use the provided context about Satyam, but you can also use your general knowledge to answer questions completely.

Context about Satyam:
{context}

Question: {query}
Answer:"""