from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

persist_directory="db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

query = "Recommend fantasy books with rating above 4.5"

retriever=db.as_retriever(search_kwargs={"k":3})
relevant_docs=retriever.invoke(query)

print(f"User Query: {query}")
print("--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")