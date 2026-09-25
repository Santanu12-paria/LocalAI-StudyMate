import requests


def generate_answer(
    question,
    context,
    chat_history=""
):
    """
    Generate an answer using the local Ollama model.

    Parameters:
        question: Current user question
        context: Relevant text retrieved from the PDF
        chat_history: Previous conversation

    Returns:
        AI-generated answer
    """

    prompt = f"""
You are a helpful study assistant.

Answer the user's current question using ONLY
the information provided in the document context.

You may use the previous conversation to understand
what the user is referring to.

If the current question depends on previous conversation,
use that conversation to understand the reference.

If the answer is not available in the document context,
say:

"I could not find the answer in the uploaded document."

Do not make up information.

Previous Conversation:
{chat_history}

Document Context:
{context}

Current Question:
{question}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"Ollama request failed: "
            f"{response.status_code}"
        )

    result = response.json()

    return result["response"].strip()