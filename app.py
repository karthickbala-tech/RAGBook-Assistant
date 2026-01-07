import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from groq import Groq

from rag.retriever import get_relevant_books

load_dotenv()

app = Flask(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

conversation_memory = {}

def classify_intent(message: str) -> str:
    with open("prompts/intent_prompt.txt", "r") as f:
        prompt = f.read()

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    user_message = data.get("message")
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Message required"}), 400

    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    intent = classify_intent(user_message)

    if intent == "non_book":
        return jsonify({
            "reply": "Sorry, I can only help with book-related questions."
        })

    book_context = get_relevant_books(user_message)

    if intent == "book_recommendation":
        with open("prompts/recommendation_prompt.txt", "r") as f:
            system_prompt = f.read()
    else:
        with open("prompts/system_prompt.txt", "r") as f:
            system_prompt = f.read()

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    messages.extend(conversation_memory[session_id][-4:])

    messages.append({
        "role": "user",
        "content": f"Book Data:\n{book_context}\n\nQuestion:\n{user_message}"
    })

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.5
    )

    assistant_reply = response.choices[0].message.content

    conversation_memory[session_id].append(
        {"role": "user", "content": user_message}
    )
    conversation_memory[session_id].append(
        {"role": "assistant", "content": assistant_reply}
    )

    return jsonify({
        "reply": assistant_reply
    })

if __name__ == "__main__":
    app.run(debug=True)
