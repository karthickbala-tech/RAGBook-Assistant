import os
from langchain_community.document_loaders import CSVLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma 


def load_documents(docs_path="/home/parrot/Documents/book_chatbot/datasets"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist. Please create it and add files."
        )

    loader = CSVLoader(
        file_path="/home/parrot/Documents/book_chatbot/datasets/books.csv",
        encoding="utf-8"
    )

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError("No documents found in CSV file.")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Length: {len(doc.page_content)}")
        print(f"Preview: {doc.page_content[:100]}...")

    return documents


def split_documents(documents, chunk_size=500, chunk_overlap=100):
    print("\nSplitting documents into chunks...")

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Length: {len(chunk.page_content)}")
        print(chunk.page_content)
        print("-" * 40)

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings using FREE HuggingFace model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    print("Vector store saved successfully!")
    return vectorstore


def main():
    docs_path = "/home/parrot/Documents/book_chatbot/datasets"
    persist_directory = "db/chroma_db"

    if os.path.exists(persist_directory):
        print("Vector store already exists. Loading...")

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )

    else:
        print("nothing here")

        # ✅ Ingestion should happen ONLY when DB does not exist
        documents = load_documents(docs_path)
        chunks = split_documents(documents)
        vectorstore = create_vector_store(chunks, persist_directory)

    print("\nIngestion complete! Ready for queries.")
    return vectorstore


if __name__ == "__main__":
    main()
