from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# Ans genaration
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage


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

# Ans genaration
combined_input=f"""based on the following documents, please answer thsis question: {query}

Documents:{chr(10).join([f"- (doc.page_content)" for doc in relevant_docs])}

please provide a clear, helpful answer using only the information from these documents.
if you can't find the answer in the documents, say
"I don't have enough information to answer that question based on the provided documents."
"""

model =Ollama(model="mistral")

messages=[
    SystemMessage(context="you are a helpful assistant.").
    HumanMessage(content=coimbaed_input)
]

result = model.invoke(messages)

print("\n--- Generated Response ---")
print("Content only:")
print(result)