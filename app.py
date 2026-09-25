import hashlib

import streamlit as st
from sentence_transformers import SentenceTransformer

from src.pdf_processor import (
    extract_text_from_pdf,
    extract_pages_from_pdf,
    split_pages_into_chunks
)

from src.rag import (
    create_embeddings,
    retrieve_relevant_chunks
)

from src.model import generate_answer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LocalAI StudyMate",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# ============================================================
# CACHE DOCUMENT PROCESSING
# ============================================================

@st.cache_data(show_spinner=False)
def process_document(pdf_bytes):

    """
    Process the uploaded PDF and cache the result.

    The same PDF will not need to be extracted,
    chunked, and embedded again on every Streamlit rerun.
    """

    # --------------------------------------------------------
    # EXTRACT COMPLETE TEXT
    # --------------------------------------------------------

    full_text, page_count = extract_text_from_pdf(
        pdf_bytes
    )


    # --------------------------------------------------------
    # EXTRACT PAGE-WISE TEXT
    # --------------------------------------------------------

    pages = extract_pages_from_pdf(
        pdf_bytes
    )


    # --------------------------------------------------------
    # CREATE PAGE-AWARE CHUNKS
    # --------------------------------------------------------

    chunk_data = split_pages_into_chunks(
        pages,
        chunk_size=1000,
        overlap=200
    )


    # --------------------------------------------------------
    # CREATE PLAIN CHUNK LIST
    # --------------------------------------------------------

    chunks = [
        item["chunk"]
        for item in chunk_data
    ]


    # --------------------------------------------------------
    # CREATE DOCUMENT EMBEDDINGS
    # --------------------------------------------------------

    document_embeddings = create_embeddings(
        chunks,
        embedding_model
    )


    # --------------------------------------------------------
    # RETURN PROCESSED DOCUMENT
    # --------------------------------------------------------

    return (
        full_text,
        page_count,
        chunk_data,
        chunks,
        document_embeddings
    )


# ============================================================
# TITLE
# ============================================================

st.title("🧠 LocalAI StudyMate")

st.write(
    "An AI-powered study assistant for reading "
    "and understanding your study materials."
)

st.divider()


# ============================================================
# PDF UPLOAD
# ============================================================

st.header("📄 Upload Your Study Material")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # READ PDF BYTES
    # ========================================================

    pdf_bytes = uploaded_file.getvalue()

    if len(pdf_bytes) == 0:

        st.error(
            "❌ The uploaded PDF is empty."
        )

        st.stop()


    # ========================================================
    # CREATE UNIQUE DOCUMENT ID
    # ========================================================

    document_id = hashlib.sha256(
        pdf_bytes
    ).hexdigest()


    # ========================================================
    # CHECK FOR NEW DOCUMENT
    # ========================================================

    if (
        "document_id" not in st.session_state
        or
        st.session_state.document_id != document_id
    ):

        st.session_state.document_id = document_id

        st.session_state.messages = []


    # ========================================================
    # SHOW UPLOADED FILE
    # ========================================================

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )


    # ========================================================
    # PROCESS / LOAD CACHED DOCUMENT
    # ========================================================

    try:

        # ----------------------------------------------------
        # CHECK WHETHER DOCUMENT IS ALREADY CACHED
        # ----------------------------------------------------

        cache_key = (
            f"document_cache_{document_id}"
        )

        if cache_key not in st.session_state:

            with st.spinner(
                "Processing PDF and creating embeddings..."
            ):

                (
                    full_text,
                    page_count,
                    chunk_data,
                    chunks,
                    document_embeddings
                ) = process_document(
                    pdf_bytes
                )

            st.session_state[cache_key] = (
                full_text,
                page_count,
                chunk_data,
                chunks,
                document_embeddings
            )

            st.session_state.document_cached = False

        else:

            (
                full_text,
                page_count,
                chunk_data,
                chunks,
                document_embeddings
            ) = st.session_state[cache_key]

            st.session_state.document_cached = True


    except ValueError as error:

        st.error(
            f"❌ {error}"
        )

        st.stop()


    # ========================================================
    # CACHE STATUS
    # ========================================================

    if st.session_state.document_cached:

        st.info(
            "⚡ Document loaded from cache. "
            "PDF processing and embeddings were reused."
        )

    else:

        st.success(
            "✅ Document processed and cached successfully."
        )


    # ========================================================
    # DOCUMENT INFORMATION
    # ========================================================

    st.subheader(
        "📚 Document Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Pages",
            page_count
        )

    with col2:

        st.metric(
            "Characters",
            len(full_text)
        )

    with col3:

        st.metric(
            "Text Chunks",
            len(chunks)
        )

    with col4:

        st.metric(
            "Embedding Size",
            document_embeddings.shape[1]
        )


    st.divider()


    # ========================================================
    # EXTRACTED TEXT
    # ========================================================

    st.subheader(
        "📖 Extracted Text"
    )

    with st.expander(
        "Click to view complete extracted text"
    ):

        st.text(
            full_text
        )


    # ========================================================
    # TEXT CHUNKS
    # ========================================================

    st.subheader(
        "🧩 Text Chunks"
    )

    st.write(
        f"The document contains {len(chunks)} text chunks."
    )


    for i, item in enumerate(
        chunk_data,
        start=1
    ):

        with st.expander(
            f"Chunk {i} — Page {item['page']}"
        ):

            st.write(
                item["chunk"]
            )


    st.divider()


    # ========================================================
    # CHAT WITH YOUR STUDY MATERIAL
    # ========================================================

    st.header(
        "💬 Chat With Your Study Material"
    )


    # ========================================================
    # INITIALIZE CHAT HISTORY
    # ========================================================

    if "messages" not in st.session_state:

        st.session_state.messages = []


    # ========================================================
    # DISPLAY PREVIOUS CHAT MESSAGES
    # ========================================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # ========================================================
    # CHAT INPUT
    # ========================================================

    question = st.chat_input(
        "Ask a question about your document..."
    )


    # ========================================================
    # PROCESS USER QUESTION
    # ========================================================

    if question:

        # ====================================================
        # DISPLAY USER QUESTION
        # ====================================================

        with st.chat_message("user"):

            st.markdown(
                question
            )


        # ====================================================
        # SAVE USER QUESTION
        # ====================================================

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # ====================================================
        # RETRIEVE RELEVANT DOCUMENT CHUNKS
        # ====================================================

        with st.spinner(
            "Searching your document..."
        ):

            results = retrieve_relevant_chunks(
                query=question,
                chunks=chunks,
                document_embeddings=document_embeddings,
                embedding_model=embedding_model,
                top_k=3
            )


        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        if not results:

            answer = (
                "I could not find relevant information "
                "in the uploaded document."
            )

        else:

            # ------------------------------------------------
            # COMBINE RELEVANT CHUNKS
            # ------------------------------------------------

            context = "\n\n".join(
                result["chunk"]
                for result in results
            )


            # ------------------------------------------------
            # BUILD PREVIOUS CHAT HISTORY
            # ------------------------------------------------

            previous_messages = (
                st.session_state.messages[:-1]
            )

            chat_history_parts = []


            for message in previous_messages:

                role = message["role"].capitalize()

                content = message["content"]

                chat_history_parts.append(
                    f"{role}: {content}"
                )


            chat_history = "\n".join(
                chat_history_parts
            )


            # ------------------------------------------------
            # GENERATE CONTEXT-AWARE ANSWER
            # ------------------------------------------------

            with st.spinner(
                "Generating answer..."
            ):

                answer = generate_answer(
                    question,
                    context,
                    chat_history
                )


        # ====================================================
        # DISPLAY AI ANSWER
        # ====================================================

        with st.chat_message("assistant"):

            st.markdown(
                answer
            )


        # ====================================================
        # SAVE AI ANSWER
        # ====================================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        # ====================================================
        # DISPLAY PAGE NUMBER CITATIONS
        # ====================================================

        if results:

            st.markdown(
                "### 📚 Sources Used"
            )


            for rank, result in enumerate(
                results,
                start=1
            ):

                score = result["score"]

                chunk_index = result["index"]

                page_number = chunk_data[
                    chunk_index
                ]["page"]


                with st.expander(
                    f"📄 Source {rank} — "
                    f"Page {page_number} — "
                    f"Chunk {chunk_index + 1} — "
                    f"Similarity: {score:.4f}"
                ):

                    st.markdown(
                        f"**📄 Page:** {page_number}"
                    )

                    st.markdown(
                        f"**🧩 Chunk:** "
                        f"{chunk_index + 1}"
                    )

                    st.markdown(
                        f"**🎯 Similarity Score:** "
                        f"{score:.4f}"
                    )

                    st.divider()

                    st.write(
                        result["chunk"]
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "LocalAI StudyMate | "
    "PDF → Page-aware Chunks → Embeddings → "
    "Semantic Search → Conversational RAG → "
    "Caching → Page Citations → AI Answer"
)