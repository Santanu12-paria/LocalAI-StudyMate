import requests


def generate_answer(
    question,
    context
):
    """
    Generate an answer using the local Ollama model.

    Parameters:
        question: User's question
        context: Relevant text retrieved from the PDF

    Returns:
        AI-generated answer
    """

    prompt = f"""
You are a helpful study assistant.

Answer the user's question using ONLY the
information provided in the context below.

If the answer is not available in the context,
say:

"I could not find the answer in the uploaded document."

Do not make up information.

Context:
{context}

Question:
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