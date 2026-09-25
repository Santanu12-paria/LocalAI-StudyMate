import requests


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "llama3.2:3b"


# ============================================================
# QUERY REWRITING
# ============================================================

def generate_search_query(
    question,
    chat_history=""
):
    """
    Rewrite a follow-up question into a standalone
    search query using the local Ollama model.
    """

    if not question or not question.strip():

        return question


    if not chat_history.strip():

        return question.strip()


    prompt = f"""
You are a query rewriting assistant.

Your task is to rewrite the user's latest question
into a standalone search query for document retrieval.

Use the previous conversation only to understand
references such as:
- it
- this
- that
- they
- them
- these
- those

Do not answer the question.

Return ONLY the rewritten search query.

Previous conversation:
{chat_history}

Latest user question:
{question}

Standalone search query:
"""


    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )


        if response.status_code != 200:

            return question.strip()


        result = response.json()

        rewritten_query = result.get(
            "response",
            ""
        ).strip()


        if not rewritten_query:

            return question.strip()


        return rewritten_query


    except requests.RequestException:

        return question.strip()


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    context,
    chat_history=""
):
    """
    Generate an answer using the local Ollama model.

    The answer is based only on the supplied document
    context.
    """

    prompt = f"""
You are LocalAI StudyMate, a helpful document-based
study assistant.

Answer the user's question using ONLY the information
provided in the document context.

Previous conversation may be used only to understand
references in the current question.

If the answer cannot be found in the document context,
say:

"I could not find relevant information in the uploaded document."

Do not invent information.

Previous conversation:
{chat_history}

Document context:
{context}

User question:
{question}

Answer:
"""


    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )


        if response.status_code != 200:

            return (
                "Unable to generate an answer. "
                "Please make sure Ollama is running."
            )


        result = response.json()

        return result.get(
            "response",
            "No answer was generated."
        ).strip()


    except requests.RequestException:

        return (
            "Unable to connect to Ollama. "
            "Please make sure Ollama is running."
        )


# ============================================================
# STUDY MODE — SUMMARY
# ============================================================

def generate_summary(
    context
):
    """
    Generate a concise study summary from document content.
    """

    prompt = f"""
You are a study assistant.

Create a clear and concise summary of the following
document content.

Rules:
- Use ONLY the provided content.
- Include the main concepts.
- Keep the explanation easy to understand.
- Use headings and bullet points where useful.
- Do not invent information.

Document content:
{context}

Summary:
"""


    return _generate_study_response(
        prompt
    )


# ============================================================
# STUDY MODE — KEY POINTS
# ============================================================

def generate_key_points(
    context
):
    """
    Extract important study points from document content.
    """

    prompt = f"""
You are a study assistant.

Extract the most important points from the following
document content.

Rules:
- Use ONLY the provided content.
- Focus on concepts that are important for studying.
- Use clear bullet points.
- Include important terminology.
- Do not invent information.

Document content:
{context}

Important Key Points:
"""


    return _generate_study_response(
        prompt
    )


# ============================================================
# STUDY MODE — QUESTIONS
# ============================================================

def generate_study_questions(
    context
):
    """
    Generate study questions from document content.
    """

    prompt = f"""
You are a study assistant.

Generate useful study questions from the following
document content.

Create:
1. Five short-answer questions.
2. Five conceptual questions.

Rules:
- Questions must be based ONLY on the provided content.
- Do not provide answers.
- Do not invent information.

Document content:
{context}

Study Questions:
"""


    return _generate_study_response(
        prompt
    )


# ============================================================
# STUDY MODE — DEFINITIONS
# ============================================================

def generate_definitions(
    context
):
    """
    Generate important definitions from document content.
    """

    prompt = f"""
You are a study assistant.

Identify important terms and explain their meanings
using the following document content.

Rules:
- Use ONLY information from the document.
- Give concise explanations.
- Use this format:

Term:
Definition:

Do not invent information.

Document content:
{context}

Important Definitions:
"""


    return _generate_study_response(
        prompt
    )


# ============================================================
# COMMON STUDY RESPONSE FUNCTION
# ============================================================

def _generate_study_response(
    prompt
):
    """
    Send a study-mode prompt to the local Ollama model.
    """

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )


        if response.status_code != 200:

            return (
                "Unable to generate the study material. "
                "Please make sure Ollama is running."
            )


        result = response.json()

        answer = result.get(
            "response",
            ""
        ).strip()


        if not answer:

            return (
                "No study material was generated."
            )


        return answer


    except requests.RequestException:

        return (
            "Unable to connect to Ollama. "
            "Please make sure Ollama is running."
        )