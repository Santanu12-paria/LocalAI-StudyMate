import streamlit as st
import hashlib

from src.pdf_processor import (
    extract_text_from_pdf,
    extract_pages_from_pdf,
    split_pages_into_chunks
)

from src.rag import (
    create_embeddings,
    retrieve_relevant_chunks
)

from src.model import (
    generate_answer,
    generate_search_query
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LocalAI StudyMate",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 LocalAI StudyMate")

st.write(
    "Upload a PDF and ask questions using "
    "a local AI-powered Retrieval-Augmented Generation system."
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

@st.cache_data(show_spinner=False)
def process_document(pdf_bytes):

    # --------------------------------------------------------
    # Extract complete text
    # --------------------------------------------------------

    full_text, page_count = extract_text_from_pdf(
        pdf_bytes
    )


    # --------------------------------------------------------
    # Extract pages separately
    # --------------------------------------------------------

    pages = extract_pages_from_pdf(
        pdf_bytes
    )


    # --------------------------------------------------------
    # Split pages into chunks
    # --------------------------------------------------------

    chunk_data = split_pages_into_chunks(
        pages,
        chunk_size=1000,
        overlap=200
    )


    # --------------------------------------------------------
    # Extract chunk text
    # --------------------------------------------------------

    chunks = [
        item["chunk"]
        for item in chunk_data
    ]


    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    document_embeddings = create_embeddings(
        chunks,
        embedding_model
    )


    return (
        full_text,
        page_count,
        chunk_data,
        chunks,
        document_embeddings
    )


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Read PDF bytes
    # --------------------------------------------------------

    pdf_bytes = uploaded_file.getvalue()


    # --------------------------------------------------------
    # Create unique document ID
    # --------------------------------------------------------

    document_id = hashlib.sha256(
        pdf_bytes
    ).hexdigest()


    # --------------------------------------------------------
    # Cache key
    # --------------------------------------------------------

    cache_key = (
        f"document_cache_{document_id}"
    )


    # --------------------------------------------------------
    # Check document cache
    # --------------------------------------------------------

    if cache_key in st.session_state:

        (
            full_text,
            page_count,
            chunk_data,
            chunks,
            document_embeddings
        ) = st.session_state[cache_key]

        cache_status = (
            "⚡ Loaded from document cache"
        )


    else:

        # ----------------------------------------------------
        # Process document
        # ----------------------------------------------------

        with st.spinner(
            "🔄 Processing your PDF..."
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


        # ----------------------------------------------------
        # Store in session cache
        # ----------------------------------------------------

        st.session_state[cache_key] = (
            full_text,
            page_count,
            chunk_data,
            chunks,
            document_embeddings
        )

        cache_status = (
            "✅ Document processed and cached"
        )


    # ========================================================
    # DETECT NEW DOCUMENT
    # ========================================================

    previous_document_id = st.session_state.get(
        "current_document_id"
    )


    if previous_document_id != document_id:

        # ----------------------------------------------------
        # Clear chat when a different PDF is uploaded
        # ----------------------------------------------------

        st.session_state.messages = []

        st.session_state.current_document_id = (
            document_id
        )


    # ========================================================
    # CACHE STATUS
    # ========================================================

    st.success(
        cache_status
    )


    # ========================================================
    # DOCUMENT INFORMATION
    # ========================================================

    st.subheader(
        "📊 Document Information"
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

        if len(document_embeddings.shape) > 1:

            embedding_size = (
                document_embeddings.shape[1]
            )

        else:

            embedding_size = 0


        st.metric(
            "Embedding Size",
            embedding_size
        )


    # ========================================================
    # EXTRACTED TEXT
    # ========================================================

    with st.expander(
        "📖 View Extracted Text"
    ):

        st.text_area(
            "Complete PDF Text",
            full_text,
            height=300
        )


    # ========================================================
    # VIEW CHUNKS
    # ========================================================

    with st.expander(
        "🧩 View Text Chunks"
    ):

        for index, item in enumerate(
            chunk_data,
            start=1
        ):

            st.markdown(
                f"**Chunk {index} — Page {item['page']}**"
            )

            st.write(
                item["chunk"]
            )

            st.divider()


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    if "messages" not in st.session_state:

        st.session_state.messages = []


    # --------------------------------------------------------
    # Display previous messages
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # ========================================================
    # USER QUESTION
    # ========================================================

    question = st.chat_input(
        "Ask a question about your PDF..."
    )


    if question:

        # ----------------------------------------------------
        # Display user question
        # ----------------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )


        # ----------------------------------------------------
        # Store user message
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # ====================================================
        # BUILD CHAT HISTORY
        # ====================================================

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


        # ====================================================
        # FEATURE 7 — QUERY REWRITING
        # ====================================================

        with st.spinner(
            "🔎 Understanding your question..."
        ):

            search_query = generate_search_query(
                question,
                chat_history
            )


        # ----------------------------------------------------
        # Show rewritten query
        # ----------------------------------------------------

        if (
            search_query.strip()
            != question.strip()
        ):

            with st.expander(
                "🔎 Retrieval Query"
            ):

                st.markdown(
                    "**Original Question:**"
                )

                st.write(
                    question
                )


                st.markdown(
                    "**Rewritten Search Query:**"
                )

                st.write(
                    search_query
                )


        # ====================================================
        # RETRIEVE RELEVANT CHUNKS
        # ====================================================

        with st.spinner(
            "🔍 Searching relevant sections..."
        ):

            relevant_chunks = (
                retrieve_relevant_chunks(
                    query=search_query,
                    chunks=chunks,
                    document_embeddings=document_embeddings,
                    embedding_model=embedding_model,
                    top_k=3
                )
            )


        # ====================================================
        # CHECK RETRIEVAL RESULTS
        # ====================================================

        if not relevant_chunks:

            answer = (
                "I could not find relevant information "
                "in the uploaded document."
            )

            source_information = []


        else:

            # ------------------------------------------------
            # Build context
            # ------------------------------------------------

            context_parts = []


            for result in relevant_chunks:

                chunk_index = result[
                    "index"
                ]

                chunk_text = result[
                    "chunk"
                ]

                page_number = chunk_data[
                    chunk_index
                ]["page"]


                context_parts.append(
                    f"""
Page {page_number}:

{chunk_text}
"""
                )


            context = "\n\n".join(
                context_parts
            )


            # =================================================
            # GENERATE ANSWER
            # =================================================

            with st.spinner(
                "🤖 Generating answer..."
            ):

                answer = generate_answer(
                    question,
                    context,
                    chat_history
                )


            source_information = (
                relevant_chunks
            )


        # ====================================================
        # DISPLAY AI RESPONSE
        # ====================================================

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                answer
            )


            # ------------------------------------------------
            # Source citations
            # ------------------------------------------------

            if source_information:

                st.markdown(
                    "### 📚 Sources"
                )


                for rank, result in enumerate(
                    source_information,
                    start=1
                ):

                    chunk_index = result[
                        "index"
                    ]

                    similarity = result[
                        "score"
                    ]

                    page_number = chunk_data[
                        chunk_index
                    ]["page"]


                    st.markdown(
                        f"""
**Source {rank}**

- 📄 Page: {page_number}
- 🧩 Chunk: {chunk_index + 1}
- 🎯 Similarity: {similarity:.4f}
"""
                    )


        # ====================================================
        # STORE ASSISTANT RESPONSE
        # ====================================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


    # ========================================================
    # FEATURE 9 — EXPORT CHAT HISTORY
    # ========================================================

    if st.session_state.get("messages"):

        st.divider()

        st.subheader(
            "📥 Export Conversation"
        )


        export_lines = []


        export_lines.append(
            "LocalAI StudyMate - Conversation"
        )


        export_lines.append(
            f"Document: {uploaded_file.name}"
        )


        export_lines.append(
            "=" * 60
        )


        export_lines.append("")


        # ----------------------------------------------------
        # Add every conversation message
        # ----------------------------------------------------

        for message in st.session_state.messages:

            if message["role"] == "user":

                export_lines.append(
                    "USER:"
                )

                export_lines.append(
                    message["content"]
                )

                export_lines.append("")


            elif message["role"] == "assistant":

                export_lines.append(
                    "ASSISTANT:"
                )

                export_lines.append(
                    message["content"]
                )

                export_lines.append("")


        # ----------------------------------------------------
        # Create downloadable text
        # ----------------------------------------------------

        export_text = "\n".join(
            export_lines
        )


        st.download_button(
            label="📄 Download Conversation",
            data=export_text,
            file_name=(
                "LocalAI_StudyMate_Conversation.txt"
            ),
            mime="text/plain"
        )


# ============================================================
# NO PDF UPLOADED
# ============================================================

else:

    st.info(
        "👆 Upload a PDF to start asking questions."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "LocalAI StudyMate | "
    "PDF → Page-aware Chunks → Embeddings → "
    "Semantic Search → Conversational RAG → "
    "Caching → Query Rewriting → "
    "Similarity Filtering → Page Citations → "
    "Conversation Export → AI Answer"
)