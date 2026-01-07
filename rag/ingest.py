from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH="/home/parrot/Documents/book_chatbot/datasets/books.csv"
VECTOR_PATH = "vector_store"

loader=CSVLoader(file_path=DATA_PATH)
documents=loader.load()

embeddings= HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db= FAISS.from_documents(documents,embeddings)
vector_db.save_local(VECTOR_PATH)


print("Book data indexed successfully")