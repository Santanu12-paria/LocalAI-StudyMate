import fitz


def extract_text_from_pdf(pdf_bytes):
    """
    Extract text from a PDF file.

    Parameters:
        pdf_bytes: PDF file data in bytes

    Returns:
        extracted_text: Complete text from the PDF
        page_count: Number of pages
    """

    if not pdf_bytes:
        raise ValueError("The PDF file is empty.")

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    all_text = []

    for page in document:

        text = page.get_text()

        if text:
            all_text.append(text)

    page_count = len(document)

    document.close()

    extracted_text = "\n".join(all_text)

    if not extracted_text.strip():

        raise ValueError(
            "No readable text was found in the PDF."
        )

    return extracted_text, page_count


def split_text(
    text,
    chunk_size=1000,
    overlap=200
):
    """
    Split extracted text into overlapping chunks.

    Parameters:
        text: Extracted PDF text
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters

    Returns:
        List of text chunks
    """

    if not text or not text.strip():

        return []

    if overlap >= chunk_size:

        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        if end >= text_length:

            break

        start = end - overlap

    return chunks


# ============================================================
# PAGE-AWARE TEXT EXTRACTION
# ============================================================

def extract_pages_from_pdf(pdf_bytes):
    """
    Extract text from each PDF page separately.

    Returns:
        List of dictionaries containing:
        - page number
        - page text
    """

    if not pdf_bytes:

        raise ValueError(
            "The PDF file is empty."
        )

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text().strip()

        if text:

            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    document.close()

    if not pages:

        raise ValueError(
            "No readable text was found in the PDF."
        )

    return pages


# ============================================================
# PAGE-AWARE CHUNKING
# ============================================================

def split_pages_into_chunks(
    pages,
    chunk_size=1000,
    overlap=200
):
    """
    Split each PDF page into chunks while preserving
    the original page number.

    Returns:
        List of dictionaries containing:
        - chunk
        - page
    """

    if not pages:

        return []

    if overlap >= chunk_size:

        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]

        text = page_data["text"]

        start = 0

        text_length = len(text)

        while start < text_length:

            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:

                chunks.append(
                    {
                        "chunk": chunk,
                        "page": page_number
                    }
                )

            if end >= text_length:

                break

            start = end - overlap

    return chunks