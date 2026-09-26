import streamlit as st
import hashlib
from io import BytesIO

from sentence_transformers import SentenceTransformer

from src.pdf_processor import (
    extract_pages_from_pdf,
    split_pages_into_chunks
)

from src.rag import (
    create_embeddings,
    retrieve_relevant_chunks
)

from src.model import (
    generate_search_query,
    generate_answer,
    generate_summary,
    generate_key_points,
    generate_study_questions,
    generate_definitions,
    generate_quiz
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LocalAI StudyMate",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# SESSION STATE
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = {}


if "active_document" not in st.session_state:
    st.session_state.active_document = None


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "study_results" not in st.session_state:
    st.session_state.study_results = {}


if "study_type" not in st.session_state:
    st.session_state.study_type = "Summary"


# ---------------- QUIZ STATE ----------------

if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []


if "quiz_current" not in st.session_state:
    st.session_state.quiz_current = 0


if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}


if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False


if "quiz_completed" not in st.session_state:
    st.session_state.quiz_completed = False


if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0


if "quiz_document_id" not in st.session_state:
    st.session_state.quiz_document_id = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_document_id(file_bytes):

    return hashlib.md5(
        file_bytes
    ).hexdigest()


def reset_quiz():

    st.session_state.quiz_questions = []

    st.session_state.quiz_current = 0

    st.session_state.quiz_answers = {}

    st.session_state.quiz_submitted = False

    st.session_state.quiz_completed = False

    st.session_state.quiz_score = 0

    st.session_state.quiz_document_id = None


def process_document(
    file_name,
    file_bytes
):

    document_id = get_document_id(
        file_bytes
    )

    # Use existing cached document
    if document_id in st.session_state.documents:

        return document_id

    pages = extract_pages_from_pdf(
        file_bytes
    )

    chunks = split_pages_into_chunks(
        pages,
        chunk_size=1000,
        overlap=200
    )

    chunk_texts = [
        item["chunk"]
        for item in chunks
    ]

    embeddings = create_embeddings(
        chunk_texts,
        embedding_model
    )

    st.session_state.documents[
        document_id
    ] = {

        "name": file_name,

        "pages": pages,

        "chunks": chunks,

        "embeddings": embeddings,

        "file_size": len(file_bytes)

    }

    return document_id


# ============================================================
# CREATE STUDY MATERIAL PDF
# ============================================================

def create_study_material_pdf(
    document_name,
    study_results
):

    from reportlab.lib.pagesizes import A4

    from reportlab.lib.styles import (
        getSampleStyleSheet
    )

    from reportlab.lib.enums import TA_CENTER

    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer
    )

    from xml.sax.saxutils import escape

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]

    title_style.alignment = TA_CENTER

    heading_style = styles["Heading1"]

    body_style = styles["BodyText"]

    story = []

    story.append(
        Paragraph(
            "LocalAI StudyMate",
            title_style
        )
    )

    story.append(
        Spacer(1, 12)
    )

    story.append(
        Paragraph(
            "Generated Study Material",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 12)
    )

    safe_document_name = escape(
        document_name
    )

    story.append(
        Paragraph(
            f"<b>Document:</b> "
            f"{safe_document_name}",
            body_style
        )
    )

    story.append(
        Spacer(1, 20)
    )

    sections = [

        ("Summary", "summary"),

        ("Key Points", "key_points"),

        ("Study Questions", "questions"),

        ("Important Definitions", "definitions")

    ]

    for title, key in sections:

        if key not in study_results:
            continue

        story.append(
            Paragraph(
                title,
                heading_style
            )
        )

        story.append(
            Spacer(1, 8)
        )

        content = study_results[key]

        lines = content.split("\n")

        for line in lines:

            line = line.strip()

            if not line:

                story.append(
                    Spacer(1, 6)
                )

                continue

            if line.startswith("### "):

                line = line[4:]

            elif line.startswith("## "):

                line = line[3:]

            elif line.startswith("# "):

                line = line[2:]

            elif line.startswith("- "):

                line = "• " + line[2:]

            elif line.startswith("* "):

                line = "• " + line[2:]

            clean_line = escape(
                line
            )

            clean_line = clean_line.replace(
                "•",
                "&#8226;"
            )

            story.append(
                Paragraph(
                    clean_line,
                    body_style
                )
            )

            story.append(
                Spacer(1, 4)
            )

        story.append(
            Spacer(1, 15)
        )

    document.build(
        story
    )

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# ============================================================
# SIDEBAR — DOCUMENT LIBRARY
# ============================================================

with st.sidebar:

    st.title(
        "📚 Document Library"
    )

    st.write(
        "Upload multiple PDF documents "
        "and select one for chatting."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            file_bytes = uploaded_file.getvalue()

            try:

                document_id = process_document(
                    uploaded_file.name,
                    file_bytes
                )

                if (
                    st.session_state.active_document
                    is None
                ):

                    st.session_state.active_document = (
                        document_id
                    )

            except Exception as e:

                st.error(
                    f"Could not process "
                    f"{uploaded_file.name}: {e}"
                )

    st.divider()

    st.subheader(
        "📄 Your Documents"
    )

    if not st.session_state.documents:

        st.info(
            "No documents uploaded yet."
        )

    else:

        document_ids = list(
            st.session_state.documents.keys()
        )

        document_names = [

            st.session_state.documents[
                doc_id
            ]["name"]

            for doc_id in document_ids

        ]

        selected_index = 0

        if (
            st.session_state.active_document
            in document_ids
        ):

            selected_index = document_ids.index(
                st.session_state.active_document
            )

        selected_document_name = st.selectbox(
            "Select document",
            document_names,
            index=selected_index
        )

        selected_document_id = document_ids[
            document_names.index(
                selected_document_name
            )
        ]

        if (
            selected_document_id
            != st.session_state.active_document
        ):

            st.session_state.active_document = (
                selected_document_id
            )

            st.session_state.chat_history = []

            st.session_state.study_results = {}

            reset_quiz()

            st.rerun()

        st.success(
            f"Active: {selected_document_name}"
        )

        st.divider()

        st.subheader(
            "🗑️ Remove Document"
        )

        document_to_remove = st.selectbox(
            "Select document to remove",
            document_names,
            key="remove_document"
        )

        if st.button(
            "🗑️ Remove Selected Document",
            use_container_width=True
        ):

            remove_id = document_ids[
                document_names.index(
                    document_to_remove
                )
            ]

            del st.session_state.documents[
                remove_id
            ]

            if (
                st.session_state.active_document
                == remove_id
            ):

                st.session_state.active_document = (
                    None
                )

                st.session_state.chat_history = []

                st.session_state.study_results = {}

                reset_quiz()

            st.success(
                f"Removed: {document_to_remove}"
            )

            st.rerun()


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "🧠 LocalAI StudyMate"
)

st.caption(
    "Local PDF-based Conversational RAG "
    "with Multi-Document Management"
)


# ============================================================
# CHECK ACTIVE DOCUMENT
# ============================================================

if not st.session_state.documents:

    st.info(
        "👈 Upload one or more PDF documents "
        "from the sidebar to get started."
    )

    st.stop()


if (
    st.session_state.active_document
    not in st.session_state.documents
):

    st.info(
        "👈 Select a document from the sidebar."
    )

    st.stop()


# ============================================================
# ACTIVE DOCUMENT DATA
# ============================================================

active_document = st.session_state.documents[
    st.session_state.active_document
]

active_document_name = active_document[
    "name"
]

chunks = active_document[
    "chunks"
]

document_embeddings = active_document[
    "embeddings"
]


# ============================================================
# ACTIVE DOCUMENT INFORMATION
# ============================================================

st.success(
    f"📖 Currently chatting with: "
    f"**{active_document_name}**"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Pages",
        len(
            active_document["pages"]
        )
    )

with col2:

    st.metric(
        "Chunks",
        len(chunks)
    )

with col3:

    st.metric(
        "Documents",
        len(
            st.session_state.documents
        )
    )


st.divider()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.chat_history:

    role = message["role"]

    content = message["content"]

    with st.chat_message(role):

        st.markdown(content)


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the selected PDF..."
)


if question:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a question."
        )

        st.stop()

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    # --------------------------------------------------------
    # RECENT CHAT HISTORY
    # --------------------------------------------------------

    recent_history = ""

    for message in st.session_state.chat_history[-6:]:

        recent_history += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    # --------------------------------------------------------
    # QUERY REWRITING
    # --------------------------------------------------------

    with st.spinner(
        "Understanding your question..."
    ):

        try:

            rewritten_query = generate_search_query(
                question,
                recent_history
            )

        except Exception:

            rewritten_query = question

    with st.expander(
        "🔎 Retrieval Query"
    ):

        st.write(
            rewritten_query
        )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    with st.spinner(
        "Searching the selected document..."
    ):

        results = retrieve_relevant_chunks(

            rewritten_query,

            chunks,

            document_embeddings,

            embedding_model,

            top_k=3,

            similarity_threshold=0.30
        )

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    if not results:

        answer = (
            "I could not find enough relevant "
            "information in the selected document "
            "to answer this question."
        )

        source_results = []

    else:

        context_parts = []

        source_results = []

        for result in results:

            chunk_index = result["index"]

            score = result["score"]

            chunk_data = chunks[
                chunk_index
            ]

            page_number = chunk_data[
                "page"
            ]

            chunk_text = chunk_data[
                "chunk"
            ]

            context_parts.append(

                f"[Page {page_number}]\n"
                f"{chunk_text}"

            )

            source_results.append(
                {
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "score": score
                }
            )

        context = "\n\n".join(
            context_parts
        )

        with st.spinner(
            "Generating AI answer..."
        ):

            try:

                answer = generate_answer(

                    question,

                    context,

                    recent_history

                )

            except Exception as e:

                answer = (
                    f"Error generating answer: {e}"
                )

    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    if source_results:

        with st.expander(
            "📚 Sources Used"
        ):

            for source in source_results:

                st.write(
                    f"**Page "
                    f"{source['page']}** — "
                    f"Chunk "
                    f"{source['chunk_index'] + 1} "
                    f"— Similarity: "
                    f"{source['score']:.4f}"
                )


# ============================================================
# CONVERSATION CONTROLS
# ============================================================

st.divider()

control_col1, control_col2 = st.columns(2)


with control_col1:

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.success(
            "Conversation cleared."
        )

        st.rerun()


with control_col2:

    if st.session_state.chat_history:

        conversation_text = ""

        for message in st.session_state.chat_history:

            role = message["role"].upper()

            content = message["content"]

            conversation_text += (
                f"{role}\n"
                f"{content}\n\n"
            )

        st.download_button(

            label="📥 Export Conversation",

            data=conversation_text,

            file_name=(
                "LocalAI_StudyMate_"
                "Conversation.txt"
            ),

            mime="text/plain",

            use_container_width=True

        )


# ============================================================
# STUDY MODE
# ============================================================

st.divider()

st.header(
    "📖 Study Mode"
)

st.write(
    "Generate study material from the "
    "currently selected document."
)


# ------------------------------------------------------------
# STUDY TYPE SELECTION
# ------------------------------------------------------------

study_type = st.selectbox(

    "Select Study Material Type",

    [
        "Summary",
        "Key Points",
        "Study Questions",
        "Important Definitions",
        "Generate All"
    ],

    key="study_type"
)


if st.button(
    "📝 Generate Study Material",
    use_container_width=True
):

    all_text = "\n\n".join(

        item["chunk"]
        for item in chunks

    )

    with st.spinner(
        "Generating study material..."
    ):

        try:

            # Clear previous results
            st.session_state.study_results = {}

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            if study_type in [
                "Summary",
                "Generate All"
            ]:

                st.session_state.study_results[
                    "summary"
                ] = generate_summary(
                    all_text
                )

            # ------------------------------------------------
            # KEY POINTS
            # ------------------------------------------------

            if study_type in [
                "Key Points",
                "Generate All"
            ]:

                st.session_state.study_results[
                    "key_points"
                ] = generate_key_points(
                    all_text
                )

            # ------------------------------------------------
            # STUDY QUESTIONS
            # ------------------------------------------------

            if study_type in [
                "Study Questions",
                "Generate All"
            ]:

                st.session_state.study_results[
                    "questions"
                ] = generate_study_questions(
                    all_text
                )

            # ------------------------------------------------
            # DEFINITIONS
            # ------------------------------------------------

            if study_type in [
                "Important Definitions",
                "Generate All"
            ]:

                st.session_state.study_results[
                    "definitions"
                ] = generate_definitions(
                    all_text
                )

            st.success(
                f"{study_type} generated successfully!"
            )

        except Exception as e:

            st.error(
                f"Could not generate "
                f"study material: {e}"
            )


# ============================================================
# DISPLAY STUDY RESULTS
# ============================================================

if st.session_state.study_results:

    results = st.session_state.study_results

    if "summary" in results:

        with st.expander(
            "📌 Summary",
            expanded=True
        ):

            st.markdown(
                results["summary"]
            )

    if "key_points" in results:

        with st.expander(
            "🔑 Key Points",
            expanded=True
        ):

            st.markdown(
                results["key_points"]
            )

    if "questions" in results:

        with st.expander(
            "❓ Study Questions",
            expanded=True
        ):

            st.markdown(
                results["questions"]
            )

    if "definitions" in results:

        with st.expander(
            "📖 Important Definitions",
            expanded=True
        ):

            st.markdown(
                results["definitions"]
            )

    # --------------------------------------------------------
    # PDF DOWNLOAD
    # --------------------------------------------------------

    study_pdf = create_study_material_pdf(

        active_document_name,

        results

    )

    st.download_button(

        label=(
            "📥 Download Study Material as PDF"
        ),

        data=study_pdf,

        file_name=(
            "LocalAI_StudyMate_"
            "Study_Material.pdf"
        ),

        mime="application/pdf",

        use_container_width=True

    )


# ============================================================
# QUIZ MODE
# ============================================================

st.divider()

st.header(
    "📝 Quiz Mode"
)

st.write(
    "Test your understanding of the "
    "currently selected document."
)


# ------------------------------------------------------------
# QUIZ SETTINGS
# ------------------------------------------------------------

quiz_col1, quiz_col2 = st.columns(2)

with quiz_col1:

    quiz_num_questions = st.selectbox(
        "Number of Questions",
        [5, 10, 15],
        index=0,
        key="quiz_num_questions"
    )

with quiz_col2:

    quiz_difficulty = st.selectbox(
        "Difficulty",
        [
            "Easy",
            "Medium",
            "Hard"
        ],
        index=1,
        key="quiz_difficulty"
    )


# ------------------------------------------------------------
# GENERATE QUIZ
# ------------------------------------------------------------

if st.button(
    "🚀 Generate Quiz",
    use_container_width=True
):

    all_text = "\n\n".join(

        item["chunk"]
        for item in chunks

    )

    with st.spinner(
        "Generating your quiz..."
    ):

        try:

            quiz_data = generate_quiz(
                all_text,
                num_questions=quiz_num_questions,
                difficulty=quiz_difficulty
            )

            st.session_state.quiz_questions = (
                quiz_data["questions"]
            )

            st.session_state.quiz_current = 0

            st.session_state.quiz_answers = {}

            st.session_state.quiz_submitted = False

            st.session_state.quiz_completed = False

            st.session_state.quiz_score = 0

            st.session_state.quiz_document_id = (
                st.session_state.active_document
            )

            st.success(
                "Quiz generated successfully!"
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Could not generate quiz: {e}"
            )


# ============================================================
# DISPLAY QUIZ
# ============================================================

if st.session_state.quiz_questions:

    # --------------------------------------------------------
    # CHECK QUIZ DOCUMENT
    # --------------------------------------------------------

    if (
        st.session_state.quiz_document_id
        != st.session_state.active_document
    ):

        reset_quiz()

        st.info(
            "The selected document changed. "
            "Please generate a new quiz."
        )

    else:

        questions = (
            st.session_state.quiz_questions
        )

        current_index = (
            st.session_state.quiz_current
        )

        total_questions = len(
            questions
        )

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        if st.session_state.quiz_completed:

            score = st.session_state.quiz_score

            percentage = (
                score / total_questions
            ) * 100

            st.subheader(
                "🎉 Quiz Completed!"
            )

            st.metric(
                "Your Score",
                f"{score} / {total_questions}"
            )

            st.metric(
                "Percentage",
                f"{percentage:.1f}%"
            )

            if percentage >= 80:

                st.success(
                    "Great work! You have a strong "
                    "understanding of this document."
                )

            elif percentage >= 50:

                st.info(
                    "Good attempt! Review the questions "
                    "you missed and try again."
                )

            else:

                st.warning(
                    "Keep studying the document and "
                    "try the quiz again."
                )

            st.divider()

            st.subheader(
                "📋 Quiz Review"
            )

            for index, question_data in enumerate(
                questions
            ):

                user_answer = (
                    st.session_state.quiz_answers.get(
                        index
                    )
                )

                correct_answer = question_data[
                    "correct_answer"
                ]

                st.markdown(
                    f"### Question {index + 1}"
                )

                st.write(
                    question_data["question"]
                )

                if user_answer is None:

                    st.write(
                        "Your answer: Not answered"
                    )

                else:

                    st.write(
                        f"Your answer: "
                        f"{question_data['options'][user_answer]}"
                    )

                st.write(
                    f"Correct answer: "
                    f"{question_data['options'][correct_answer]}"
                )

                st.info(
                    question_data["explanation"]
                )

                st.divider()

            if st.button(
                "🔄 Restart Quiz",
                use_container_width=True
            ):

                reset_quiz()

                st.rerun()

        # ----------------------------------------------------
        # CURRENT QUESTION
        # ----------------------------------------------------

        else:

            question_data = questions[
                current_index
            ]

            st.progress(
                (
                    current_index + 1
                ) / total_questions
            )

            st.write(
                f"**Question "
                f"{current_index + 1} "
                f"of {total_questions}**"
            )

            st.subheader(
                question_data["question"]
            )

            option_labels = [

                f"A. {question_data['options'][0]}",
                f"B. {question_data['options'][1]}",
                f"C. {question_data['options'][2]}",
                f"D. {question_data['options'][3]}"

            ]

            selected_option = st.radio(

                "Select your answer:",

                option_labels,

                key=f"quiz_option_{current_index}"

            )

            selected_index = option_labels.index(
                selected_option
            )

            # ------------------------------------------------
            # SUBMIT ANSWER
            # ------------------------------------------------

            if not st.session_state.quiz_submitted:

                if st.button(
                    "✅ Submit Answer",
                    use_container_width=True
                ):

                    st.session_state.quiz_answers[
                        current_index
                    ] = selected_index

                    st.session_state.quiz_submitted = (
                        True
                    )

                    if (
                        selected_index
                        == question_data[
                            "correct_answer"
                        ]
                    ):

                        st.session_state.quiz_score += 1

                    st.rerun()

            # ------------------------------------------------
            # ANSWER FEEDBACK
            # ------------------------------------------------

            else:

                correct_index = (
                    question_data[
                        "correct_answer"
                    ]
                )

                if (
                    selected_index
                    == correct_index
                ):

                    st.success(
                        "✅ Correct answer!"
                    )

                else:

                    st.error(
                        "❌ Incorrect answer."
                    )

                    st.write(
                        f"**Correct answer:** "
                        f"{option_labels[correct_index]}"
                    )

                st.info(
                    f"**Explanation:** "
                    f"{question_data['explanation']}"
                )

                # --------------------------------------------
                # NEXT QUESTION
                # --------------------------------------------

                if (
                    current_index
                    < total_questions - 1
                ):

                    if st.button(
                        "➡️ Next Question",
                        use_container_width=True
                    ):

                        st.session_state.quiz_current += 1

                        st.session_state.quiz_submitted = (
                            False
                        )

                        st.rerun()

                else:

                    if st.button(
                        "🏁 Finish Quiz",
                        use_container_width=True
                    ):

                        st.session_state.quiz_completed = (
                            True
                        )

                        st.session_state.quiz_submitted = (
                            False
                        )

                        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PDF → Multi-Document Management → "
    "Page-aware Chunks → Embeddings → "
    "Semantic Search → Conversational RAG → "
    "Query Rewriting → Similarity Filtering → "
    "Page Citations → Study Mode → "
    "PDF Study Material → Quiz Mode → "
    "Conversation Export → "
    "Clear Conversation → AI Answer"
)