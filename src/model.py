import requests
import json
import re


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "llama3.2:3b"


# ============================================================
# BASIC OLLAMA REQUEST
# ============================================================

def _call_ollama(
    prompt,
    temperature=0.2,
    timeout=180
):
    """
    Send a prompt to the local Ollama model.
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=timeout
        )

        response.raise_for_status()

        data = response.json()

        return data.get(
            "response",
            ""
        ).strip()

    except requests.exceptions.RequestException as error:

        raise RuntimeError(
            "Could not connect to Ollama. "
            "Make sure Ollama is running."
        ) from error


# ============================================================
# SEARCH QUERY GENERATION
# ============================================================

def generate_search_query(
    question,
    chat_history=""
):
    """
    Rewrite the user's question into a better
    search query for document retrieval.
    """

    if not question or not question.strip():

        raise ValueError(
            "The question cannot be empty."
        )

    prompt = f"""
You are a search query generator for a document-based
question answering system.

Convert the user's question into a short,
clear search query.

Use the conversation history only to understand
references such as "it", "this", "that", etc.

Do not answer the question.

Conversation history:
{chat_history}

User question:
{question}

Return ONLY the rewritten search query.
"""

    result = _call_ollama(
        prompt,
        temperature=0.1
    )

    if not result:

        return question.strip()

    return result.strip()


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    context,
    chat_history=""
):
    """
    Generate an answer using only the supplied
    document context.
    """

    if not question or not question.strip():

        raise ValueError(
            "The question cannot be empty."
        )

    if not context or not context.strip():

        return (
            "I could not find relevant information "
            "in the uploaded document."
        )

    prompt = f"""
You are an AI study assistant.

Answer the user's question using ONLY the
document context provided below.

Do not use outside knowledge.

If the answer cannot be found in the document,
clearly say that the information is not available
in the uploaded document.

Conversation history can be used only to understand
references from previous messages.

Do not invent information.

Conversation history:
{chat_history}

Document context:
{context}

User question:
{question}

Give a clear and concise answer.
"""

    return _call_ollama(
        prompt,
        temperature=0.2
    )


# ============================================================
# STUDY MODE - SUMMARY
# ============================================================

def generate_summary(
    context
):
    """
    Generate a concise summary from document context.
    """

    if not context or not context.strip():

        return (
            "No document content is available "
            "for summarization."
        )

    prompt = f"""
You are an AI study assistant.

Create a clear and concise summary of the
following document content.

Use ONLY the supplied content.

Do not add outside information.

Document content:
{context}

Write the summary in simple language.
"""

    return _call_ollama(
        prompt,
        temperature=0.2
    )


# ============================================================
# STUDY MODE - KEY POINTS
# ============================================================

def generate_key_points(
    context
):
    """
    Generate important key points from document context.
    """

    if not context or not context.strip():

        return (
            "No document content is available "
            "for generating key points."
        )

    prompt = f"""
You are an AI study assistant.

Extract the most important points from the
following document content.

Use ONLY the supplied content.

Do not add outside information.

Document content:
{context}

Present the result as clear bullet points.
"""

    return _call_ollama(
        prompt,
        temperature=0.2
    )


# ============================================================
# STUDY MODE - STUDY QUESTIONS
# ============================================================

def generate_study_questions(
    context
):
    """
    Generate study questions from document context.
    """

    if not context or not context.strip():

        return (
            "No document content is available "
            "for generating study questions."
        )

    prompt = f"""
You are an AI study assistant.

Create useful study questions based ONLY on
the following document content.

Do not use outside information.

Document content:
{context}

Generate a mixture of conceptual and
understanding-based questions.

Number the questions clearly.
"""

    return _call_ollama(
        prompt,
        temperature=0.3
    )


# ============================================================
# STUDY MODE - DEFINITIONS
# ============================================================

def generate_definitions(
    context
):
    """
    Extract important terms and definitions
    from document context.
    """

    if not context or not context.strip():

        return (
            "No document content is available "
            "for generating definitions."
        )

    prompt = f"""
You are an AI study assistant.

Extract important technical terms and their
definitions from the following document content.

Use ONLY the supplied content.

Do not add outside information.

Document content:
{context}

Format:

Term:
Definition:
"""

    return _call_ollama(
        prompt,
        temperature=0.2
    )


# ============================================================
# CLEAN JSON RESPONSE
# ============================================================

def _clean_json_response(
    response
):
    """
    Clean an Ollama response and extract JSON.
    """

    if not response:

        raise ValueError(
            "The model returned an empty response."
        )

    cleaned = response.strip()

    # Remove markdown code fences
    cleaned = re.sub(
        r"^```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"^```\s*",
        "",
        cleaned
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    cleaned = cleaned.strip()

    # Try direct JSON parsing first
    try:

        return json.loads(
            cleaned
        )

    except json.JSONDecodeError:
        pass

    # Try extracting JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:

        json_text = cleaned[
            start:end + 1
        ]

        try:

            return json.loads(
                json_text
            )

        except json.JSONDecodeError as error:

            raise ValueError(
                "The model returned invalid JSON."
            ) from error

    raise ValueError(
        "Could not find JSON in the model response."
    )


# ============================================================
# NORMALIZE ANSWER TEXT
# ============================================================

def _normalize_answer_text(
    text
):
    """
    Normalize answer text so Python can compare
    the model's correct_answer_text with options.
    """

    if not isinstance(
        text,
        str
    ):

        return ""

    text = text.strip().lower()

    # Remove option prefixes such as:
    # A.
    # B)
    # 1.
    # 2)
    text = re.sub(
        r"^[a-d1-4][\.\)\:\-]\s*",
        "",
        text
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove surrounding quotes
    text = text.strip(
        "\"' "
    )

    return text


# ============================================================
# FIND CORRECT ANSWER INDEX
# ============================================================

def _find_correct_answer_index(
    options,
    correct_answer_text
):
    """
    Find which option matches the model's
    correct_answer_text.

    Python calculates the numeric index.
    """

    normalized_correct = _normalize_answer_text(
        correct_answer_text
    )

    if not normalized_correct:

        return None

    # Exact normalized match
    for index, option in enumerate(options):

        normalized_option = _normalize_answer_text(
            option
        )

        if normalized_option == normalized_correct:

            return index

    # Slightly more tolerant matching
    for index, option in enumerate(options):

        normalized_option = _normalize_answer_text(
            option
        )

        if (
            normalized_correct in normalized_option
            or
            normalized_option in normalized_correct
        ):

            return index

    return None


# ============================================================
# VALIDATE QUIZ
# ============================================================

def _validate_quiz(
    quiz_data,
    expected_questions
):
    """
    Validate the structure of the generated quiz.

    IMPORTANT:
    The model does NOT decide the numeric
    correct_answer index.

    The model provides correct_answer_text.

    Python finds which option matches that text
    and creates the correct_answer index.

    If Ollama forgets the explanation,
    Python creates a basic explanation automatically.
    """

    if not isinstance(
        quiz_data,
        dict
    ):

        raise ValueError(
            "Quiz response is not a JSON object."
        )

    questions = quiz_data.get(
        "questions"
    )

    if not isinstance(
        questions,
        list
    ):

        raise ValueError(
            "Quiz response does not contain "
            "a valid questions list."
        )

    if len(questions) < expected_questions:

        raise ValueError(
            f"Expected {expected_questions} "
            f"questions but received {len(questions)}."
        )

    validated_questions = []

    for number, question_data in enumerate(
        questions[:expected_questions],
        start=1
    ):

        if not isinstance(
            question_data,
            dict
        ):

            raise ValueError(
                f"Question {number} is invalid."
            )

        # ----------------------------------------------------
        # QUESTION
        # ----------------------------------------------------

        question = question_data.get(
            "question"
        )

        if not isinstance(
            question,
            str
        ) or not question.strip():

            raise ValueError(
                f"Question {number} has empty text."
            )

        # ----------------------------------------------------
        # OPTIONS
        # ----------------------------------------------------

        options = question_data.get(
            "options"
        )

        if not isinstance(
            options,
            list
        ):

            raise ValueError(
                f"Question {number} has invalid options."
            )

        if len(options) != 4:

            raise ValueError(
                f"Question {number} must have exactly 4 options."
            )

        cleaned_options = []

        for option in options:

            if not isinstance(
                option,
                str
            ):

                raise ValueError(
                    f"Question {number} contains "
                    f"an invalid option."
                )

            option = option.strip()

            if not option:

                raise ValueError(
                    f"Question {number} contains "
                    f"an empty option."
                )

            cleaned_options.append(
                option
            )

        # ----------------------------------------------------
        # DUPLICATE OPTIONS
        # ----------------------------------------------------

        normalized_options = [
            _normalize_answer_text(option)
            for option in cleaned_options
        ]

        if len(
            set(normalized_options)
        ) != 4:

            raise ValueError(
                f"Question {number} contains "
                f"duplicate options."
            )

        # ----------------------------------------------------
        # CORRECT ANSWER TEXT
        # ----------------------------------------------------

        correct_answer_text = question_data.get(
            "correct_answer_text"
        )

        if not isinstance(
            correct_answer_text,
            str
        ) or not correct_answer_text.strip():

            raise ValueError(
                f"Question {number} does not contain "
                f"a valid correct_answer_text."
            )

        # ----------------------------------------------------
        # PYTHON FINDS CORRECT INDEX
        # ----------------------------------------------------

        correct_index = _find_correct_answer_index(
            cleaned_options,
            correct_answer_text
        )

        if correct_index is None:

            raise ValueError(
                f"Question {number}: the correct answer "
                f"does not match any of the four options."
            )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        explanation = question_data.get(
            "explanation",
            ""
        )

        if not isinstance(
            explanation,
            str
        ):

            explanation = ""

        explanation = explanation.strip()

        # If Ollama forgot the explanation,
        # automatically create one.
        if not explanation:

            explanation = (
                "The correct answer is: "
                + cleaned_options[correct_index]
                + "."
            )

        # ----------------------------------------------------
        # SAVE VALIDATED QUESTION
        # ----------------------------------------------------

        validated_questions.append(
            {
                "question": question.strip(),
                "options": cleaned_options,

                # IMPORTANT:
                # This index is calculated by Python.
                "correct_answer": correct_index,

                "explanation": explanation
            }
        )

    return {
        "questions": validated_questions
    }


# ============================================================
# CHECK QUIZ CONSISTENCY
# ============================================================

def _check_quiz_consistency(
    quiz_data
):
    """
    Perform an additional lightweight consistency check.

    This check asks Ollama whether the question,
    options and correct answer make sense together.

    IMPORTANT:
    Unclear verifier responses are NOT treated as errors.

    This makes the system more reliable with
    smaller local models such as llama3.2:3b.
    """

    questions = quiz_data.get(
        "questions",
        []
    )

    for number, question_data in enumerate(
        questions,
        start=1
    ):

        question = question_data[
            "question"
        ]

        options = question_data[
            "options"
        ]

        correct_index = question_data[
            "correct_answer"
        ]

        correct_option = options[
            correct_index
        ]

        prompt = f"""
You are checking a multiple-choice question.

Determine whether the marked correct answer
actually answers the question.

Question:
{question}

Options:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

Marked correct answer:
{correct_option}

Return ONLY one word:

VALID

or

WRONG
"""

        try:

            result = _call_ollama(
                prompt,
                temperature=0.0,
                timeout=60
            )

            result = result.strip().upper()

            # Only explicit WRONG rejects the quiz.
            # Unclear responses are accepted because
            # small local models may produce extra text.
            if result == "WRONG":

                return False

        except Exception:

            # Do not fail the complete quiz because
            # the verifier failed.
            continue

    return True


# ============================================================
# GENERATE QUIZ
# ============================================================

def generate_quiz(
    context,
    num_questions=5,
    difficulty="Medium"
):
    """
    Generate a multiple-choice quiz from document content.

    The model generates:
        - question
        - four options
        - correct_answer_text
        - explanation

    Python then calculates:
        correct_answer

    This prevents the previous bug where the model
    gave a correct explanation but an incorrect
    numeric answer index.
    """

    if not context or not context.strip():

        raise ValueError(
            "No document content is available "
            "for quiz generation."
        )

    if num_questions not in [
        5,
        10,
        15
    ]:

        raise ValueError(
            "Number of questions must be 5, 10, or 15."
        )

    allowed_difficulties = [
        "Easy",
        "Medium",
        "Hard"
    ]

    if difficulty not in allowed_difficulties:

        raise ValueError(
            "Invalid quiz difficulty."
        )

    prompt = f"""
You are an AI quiz generator.

Create a multiple-choice quiz using ONLY
the supplied document content.

Do NOT use outside knowledge.

Number of questions:
{num_questions}

Difficulty:
{difficulty}

IMPORTANT RULES:

1. Create exactly {num_questions} questions.

2. Every question must be answerable using
   ONLY the supplied document content.

3. Every question must have exactly 4 options.

4. The four options must be different.

5. Exactly ONE option must be correct.

6. The "correct_answer_text" MUST contain the
   EXACT TEXT of the correct option.

7. Do NOT provide a numeric correct answer index.

8. Provide a short explanation that agrees with
   the correct answer.

9. If you cannot provide an explanation, still
   return the question with the correct_answer_text.
   Python will create the explanation.

10. Do not put "A.", "B.", "C." or "D." inside
    the option text.

11. Do not create trick questions.

12. Do not invent facts that are not present
    in the document.

13. Make sure the question, options,
    correct answer and explanation are consistent.

Return ONLY valid JSON.

Required JSON format:

{{
  "questions": [
    {{
      "question": "Question text",
      "options": [
        "Option 1",
        "Option 2",
        "Option 3",
        "Option 4"
      ],
      "correct_answer_text": "Exact text of the correct option",
      "explanation": "Short explanation"
    }}
  ]
}}

Document content:
{context}
"""

    # --------------------------------------------------------
    # TRY GENERATION MULTIPLE TIMES
    # --------------------------------------------------------

    max_attempts = 3

    last_error = None

    for attempt in range(
        max_attempts
    ):

        try:

            response = _call_ollama(
                prompt,
                temperature=0.2
            )

            quiz_data = _clean_json_response(
                response
            )

            validated_quiz = _validate_quiz(
                quiz_data,
                num_questions
            )

            # Additional consistency check
            is_consistent = _check_quiz_consistency(
                validated_quiz
            )

            if is_consistent:

                return validated_quiz

            last_error = ValueError(
                "Generated quiz failed "
                "content consistency check."
            )

        except Exception as error:

            last_error = error

    # --------------------------------------------------------
    # FALLBACK GENERATION
    # --------------------------------------------------------

    fallback_prompt = f"""
Create a reliable multiple-choice quiz from
ONLY the document content below.

Create exactly {num_questions} questions.

Difficulty:
{difficulty}

Each question MUST contain:

- question
- exactly 4 different options
- correct_answer_text
- explanation

IMPORTANT:

The correct_answer_text must be EXACTLY
the same as one of the four options.

Do NOT use a numeric correct answer.

Make sure the explanation agrees with the
correct answer.

If you cannot provide an explanation, still
return the correct_answer_text.
Python will create the explanation.

Do not use outside knowledge.

Return ONLY valid JSON.

Format:

{{
  "questions": [
    {{
      "question": "...",
      "options": [
        "...",
        "...",
        "...",
        "..."
      ],
      "correct_answer_text": "...",
      "explanation": "..."
    }}
  ]
}}

Document content:
{context}
"""

    try:

        response = _call_ollama(
            fallback_prompt,
            temperature=0.1
        )

        quiz_data = _clean_json_response(
            response
        )

        validated_quiz = _validate_quiz(
            quiz_data,
            num_questions
        )

        return validated_quiz

    except Exception as fallback_error:

        if last_error:

            raise RuntimeError(
                "Could not generate a valid quiz "
                "after multiple attempts. "
                f"Last error: {last_error}"
            ) from fallback_error

        raise RuntimeError(
            "Could not generate quiz."
        ) from fallback_error


# ============================================================
# GENERIC STUDY RESPONSE
# ============================================================

def _generate_study_response(
    prompt
):
    """
    Helper for study-related generation.
    """

    if not prompt or not prompt.strip():

        raise ValueError(
            "Study prompt cannot be empty."
        )

    return _call_ollama(
        prompt,
        temperature=0.2
    )