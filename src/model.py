import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "llama3.2:3b"


def generate_search_query(
    question,
    chat_history=""
):
    """
    Rewrite the user's current question into a
    standalone search query using conversation history.

    The rewritten query is used only for document retrieval.
    """

    # --------------------------------------------------------
    # If there is no previous conversation, the original
    # question is already sufficient for retrieval.
    # --------------------------------------------------------

    if not chat_history.strip():

        return question.strip()


    prompt = f"""
You are a query rewriting assistant for a document
question-answering system.

Your task is to rewrite the user's current question
into a clear, standalone search query.

Use the previous conversation to understand references
such as:
- it
- this
- that
- they
- them
- these
- those

Keep the meaning of the user's question unchanged.

Do NOT answer the question.

Return ONLY the rewritten search query.
Do not add explanations.
Do not add quotation marks.

Previous Conversation:
{chat_history}

Current Question:
{question}

Standalone Search Query:
"""


    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )


    if response.status_code != 200:

        # ----------------------------------------------------
        # If rewriting fails, fall back to the original
        # question instead of breaking the RAG system.
        # ----------------------------------------------------

        return question.strip()


    result = response.json()

    rewritten_query = result["response"].strip()


    if not rewritten_query:

        return question.strip()


    return rewritten_query


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
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
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