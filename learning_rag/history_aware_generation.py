from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


persistent_directory = "db/chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

model = Ollama(model="tinyllama")



chat_history = []


def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

   
    if chat_history:
        messages = [
            SystemMessage(
                content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."
            ),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        search_question = model.invoke(messages).strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

   
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        preview = "\n".join(doc.page_content.split("\n")[:2])
        print(f"Doc {i}: {preview}...")

    documents_text = "\n".join(
        [f"- {doc.page_content}" for doc in docs]
    )

  
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

Documents:
{documents_text}

Please provide a clear, helpful answer using only the information from these documents.
If you can't find the answer in the documents, say:
"I don't have enough information to answer that question based on the provided documents."
"""

   
    messages = [
        SystemMessage(
            content="You are a helpful assistant that answers questions based on provided documents and conversation history."
        ),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    answer = model.invoke(messages)


    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"\nAnswer:\n{answer}")
    return answer


def start_chat():
    print("Ask book-related questions! Type 'quit' to exit.")
    while True:
        question = input("\nYour Question: ")
        if question.lower() == "quit":
            print("Goodbye!")
            break
        ask_question(question)


if __name__ == "__main__":
    start_chat()


