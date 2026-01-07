import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_PATH = "vector_store"


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


if not os.path.exists(VECTOR_PATH):
    raise FileNotFoundError(
        "Vector store not found. Create embeddings before running retriever."
    )


vector_db = FAISS.load_local(
    VECTOR_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

def get_relevant_books(query: str):
    docs = vector_db.similarity_search(query, k=3)
    return "\n".join(doc.page_content for doc in docs)
