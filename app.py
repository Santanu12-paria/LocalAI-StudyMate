import streamlit as st
from sentence_transformers import SentenceTransformer

from src.pdf_processor import (
    extract_text_from_pdf,
    split_text
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

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # --------------------------------------------------------
    # Read uploaded PDF
    # --------------------------------------------------------

    pdf_bytes = uploaded_file.getvalue()


    # --------------------------------------------------------
    # Check empty PDF
    # --------------------------------------------------------

    if len(pdf_bytes) == 0:

        st.error(
            "❌ The uploaded PDF is empty."
        )

        st.stop()


    # ========================================================
    # EXTRACT TEXT
    # ========================================================

    try:

        full_text, page_count = extract_text_from_pdf(
            pdf_bytes
        )

    except ValueError as error:

        st.error(
            f"❌ {error}"
        )

        st.stop()


    # ========================================================
    # SPLIT TEXT INTO CHUNKS
    # ========================================================

    chunks = split_text(
        full_text,
        chunk_size=1000,
        overlap=200
    )


    # ========================================================
    # CREATE EMBEDDINGS
    # ========================================================

    with st.spinner(
        "Creating document embeddings..."
    ):

        document_embeddings = create_embeddings(
            chunks,
            embedding_model
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


    st.success(
        f"✅ Created {len(chunks)} document embeddings."
    )


    # ========================================================
    # EXTRACTED TEXT
    # ========================================================

    st.divider()

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


    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        with st.expander(
            f"Chunk {i}"
        ):

            st.write(
                chunk
            )


    # ========================================================
    # ASK YOUR STUDY MATERIAL
    # ========================================================

    st.divider()

    st.header(
        "🔍 Ask Your Study Material"
    )

    question = st.text_input(
        "Enter your question:",
        placeholder="Example: What is software design?"
    )


    # ========================================================
    # RETRIEVAL + AI ANSWER
    # ========================================================

    if question:

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
        # DISPLAY RELEVANT RESULTS
        # ====================================================

        st.subheader(
            "🎯 Most Relevant Sections"
        )


        if not results:

            st.warning(
                "No relevant sections were found."
            )

        else:

            # ------------------------------------------------
            # DISPLAY RETRIEVED CHUNKS
            # ------------------------------------------------

            for rank, result in enumerate(
                results,
                start=1
            ):

                st.markdown(
                    f"### Result {rank}"
                )

                st.write(
                    f"**Similarity Score:** "
                    f"{result['score']:.4f}"
                )

                with st.expander(
                    f"View Result {rank}"
                ):

                    st.write(
                        result["chunk"]
                    )


            # =================================================
            # BUILD CONTEXT FOR THE AI MODEL
            # =================================================

            context = "\n\n".join(
                result["chunk"]
                for result in results
            )


            # =================================================
            # GENERATE AI ANSWER
            # =================================================

            st.subheader(
                "🤖 AI Answer"
            )

            with st.spinner(
                "Generating answer..."
            ):

                answer = generate_answer(
                    question,
                    context
                )


            st.write(
                answer
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "LocalAI StudyMate | "
    "PDF → Chunks → Embeddings → Semantic Search → AI Answer"
)