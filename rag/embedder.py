from sentence_transformers import SentenceTransformer

_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_query_embedding(text: str):
    return _model.encode([text]).tolist()[0]

def get_doc_embeddings(docs: list[str]):
    return _model.encode(docs).tolist()
