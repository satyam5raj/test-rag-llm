import chromadb
from rag.embedder import get_doc_embeddings

client = chromadb.Client()
collection = client.get_or_create_collection("docs")

def add_documents(documents: list[str]):
    embeddings = get_doc_embeddings(documents)
    ids = [f"doc{i}" for i in range(len(documents))]
    collection.add(documents=documents, embeddings=embeddings, ids=ids)

def get_relevant_docs(query_embedding, k=3):
    result = collection.query(query_embeddings=[query_embedding], n_results=k)
    return result['documents'][0]
