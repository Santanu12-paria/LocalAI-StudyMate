import numpy as np


def create_embeddings(
    texts,
    embedding_model
):
    """
    Create embeddings for a list of text chunks.

    Parameters:
        texts: List of text chunks
        embedding_model: SentenceTransformer model

    Returns:
        NumPy array containing embeddings
    """

    if not texts:
        return np.array([])

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings


def create_query_embedding(
    query,
    embedding_model
):
    """
    Create an embedding for the user's question.
    """

    if not query or not query.strip():
        raise ValueError(
            "The question cannot be empty."
        )

    embedding = embedding_model.encode(
        query,
        convert_to_numpy=True
    )

    return embedding


def cosine_similarity(
    query_embedding,
    document_embeddings
):
    """
    Calculate cosine similarity between
    a query embedding and document embeddings.
    """

    query_norm = np.linalg.norm(
        query_embedding
    )

    document_norms = np.linalg.norm(
        document_embeddings,
        axis=1
    )

    query_norm = max(
        query_norm,
        1e-10
    )

    document_norms = np.maximum(
        document_norms,
        1e-10
    )

    similarities = np.dot(
        document_embeddings,
        query_embedding
    ) / (
        document_norms * query_norm
    )

    return similarities


def retrieve_relevant_chunks(
    query,
    chunks,
    document_embeddings,
    embedding_model,
    top_k=3,
    similarity_threshold=0.30
):
    """
    Retrieve the most relevant chunks
    for a user query.
    """

    if not chunks:
        return []

    query_embedding = create_query_embedding(
        query,
        embedding_model
    )

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings
    )

    top_k = min(
        top_k,
        len(chunks)
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        score = float(
            similarities[index]
        )

        if score >= similarity_threshold:

            results.append(
                {
                    "chunk": chunks[index],
                    "score": score,
                    "index": int(index)
                }
            )

    return results