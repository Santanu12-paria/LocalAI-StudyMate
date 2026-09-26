# 🧠 LocalAI StudyMate

> A local AI-powered document study assistant that uses Retrieval-Augmented Generation (RAG) to answer questions, generate study material, and create quizzes from uploaded documents.

## 📌 Project Overview

**LocalAI StudyMate** is a local AI-powered study assistant designed to help students interact with their study materials using natural language.

The system allows users to upload PDF documents and ask questions about their content. Instead of relying on a cloud-based AI service for answer generation, the project uses a local Large Language Model (LLM) through **Ollama**.

The application combines:

- PDF text extraction
- Page-aware document processing
- Text chunking
- Semantic embeddings
- Vector-based similarity search
- Retrieval-Augmented Generation (RAG)
- Conversational chat history
- Query rewriting
- Similarity threshold filtering
- Page citations
- Multi-document management
- Study Mode
- Study material generation
- Quiz Mode

## 🎯 Key Features

### 📄 1. PDF Upload & Text Extraction

Users can upload PDF documents and extract their text for further processing.

The system preserves page information so retrieved information can be linked back to the original document page.

### ✂️ 2. Text Chunking

Large documents are divided into smaller overlapping chunks, making it easier to retrieve relevant sections during semantic search.

### 🧠 3. Semantic Embeddings

The project uses the **Sentence Transformers** model:

`all-MiniLM-L6-v2`

Document chunks and user queries are converted into numerical vector representations.

### 🔎 4. Semantic Search

When a user asks a question, the question is converted into an embedding and compared with document chunk embeddings using cosine similarity.

The most relevant chunks are selected as context for the AI model.

### 🤖 5. Retrieval-Augmented Generation (RAG)

The complete pipeline is:

```text
PDF
 ↓
Text Extraction
 ↓
Page-Aware Chunking
 ↓
Embeddings
 ↓
Semantic Search
 ↓
Relevant Context
 ↓
Ollama LLM
 ↓
Answer
```

This allows the AI to answer questions based on the uploaded documents.

### 💬 6. Conversational Chat

The application maintains conversation history so users can ask follow-up questions naturally.

Example:

```text
User: What is abstraction?

User: Why is it important?

User: Give me an example.
```

### 🔄 7. Query Rewriting

The system can rewrite user questions into clearer search queries before document retrieval.

This helps with follow-up questions and references such as:

```text
"What does this mean?"
"Why is it important?"
"Explain that concept."
```

### 📊 8. Similarity Threshold

Retrieved document chunks are filtered using a similarity threshold to reduce weakly related context.

### 📑 9. Page Citations

Retrieved information includes the source page from the uploaded document, helping users identify where the information came from.

### 📚 10. Study Mode

Study Mode provides:

- Summary
- Key Points
- Study Questions
- Important Definitions
- Generate All

### 📥 11. Study Material PDF

Generated study material can be downloaded as a PDF for revision and offline study.

### 📂 12. Multi-Document Management

Users can upload and manage multiple documents and select the document they want to study.

### 📝 13. Conversation Export

Users can export their conversation for future reference.

### 🧹 14. Clear Conversation

Users can clear the current conversation and start a fresh study session.

### 🧪 15. Quiz Mode

The application can generate multiple-choice quizzes from uploaded documents.

Users can select:

- Number of questions: 5, 10, or 15
- Difficulty: Easy, Medium, or Hard

Quiz Mode provides:

- Multiple-choice questions
- Four answer options
- Answer checking
- Explanations
- Score calculation
- Percentage
- Quiz review
- Restart Quiz

The system also calculates the correct answer index programmatically from the generated correct answer text to reduce answer-mapping errors from the local LLM.

## 🏗️ System Architecture

```text
                    ┌──────────────────┐
                    │   User / Student │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Streamlit     │
                    │       UI         │
                    └────────┬─────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
        ┌─────────────────┐     ┌─────────────────┐
        │  PDF Processing │     │ Conversation    │
        │                 │     │ History         │
        └────────┬────────┘     └────────┬────────┘
                 │                       │
                 ▼                       ▼
        ┌─────────────────┐     ┌─────────────────┐
        │ Page-Aware      │     │ Query Rewriting │
        │ Chunking        │     └────────┬────────┘
        └────────┬────────┘              │
                 │                       │
                 ▼                       │
        ┌─────────────────┐              │
        │ Sentence        │              │
        │ Transformers    │              │
        └────────┬────────┘              │
                 │                       │
                 └───────────┬───────────┘
                             ▼
                ┌─────────────────────────┐
                │  Semantic Similarity    │
                │        Search           │
                └────────────┬────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Relevant Context│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Ollama          │
                    │ Llama 3.2 3B    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Answer / Study  │
                    │ Material / Quiz │
                    └─────────────────┘
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| NumPy | Vector and similarity calculations |
| Ollama | Local LLM execution |
| Llama 3.2 3B | Local language model |
| ReportLab | Study material PDF generation |
| Git & GitHub | Version control and project hosting |

## 📁 Project Structure

```text
LocalAI-StudyMate/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── documents/
│
└── src/
    ├── model.py
    ├── pdf_processor.py
    └── rag.py
```

## ⚙️ Requirements

- Windows
- Python 3.12+
- Ollama
- Git
- Internet connection for installing Python packages and downloading required models

The language model inference is performed locally through Ollama.

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Santanu12-paria/LocalAI-StudyMate.git
```

### 2. Open the project

```bash
cd LocalAI-StudyMate
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Install and download the Ollama model

```bash
ollama pull llama3.2:3b
```

## ▶️ Running the Application

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Run the application:

```powershell
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## 🔐 Local AI & Privacy

LocalAI StudyMate is designed around local AI processing.

The project uses:

```text
Ollama
+
Llama 3.2 3B
```

for local language-model inference, allowing the application to process study documents locally rather than depending on a cloud AI API for answer generation.

## 🎓 Use Cases

LocalAI StudyMate can be useful for:

- College students
- Exam preparation
- Research paper reading
- Technical documentation
- Lecture notes
- Textbook study
- PDF-based question answering
- Creating study material
- Quiz preparation

## 🚧 Future Improvements

Possible future improvements include:

- OCR support for scanned PDFs
- Better table extraction
- Persistent vector databases
- Improved document ranking
- Support for additional document formats
- Improved multilingual support
- Voice-based interaction
- Advanced answer evaluation

## 👨‍💻 Author

**Santanu Paria**

GitHub Repository:

https://github.com/Santanu12-paria/LocalAI-StudyMate

## 📜 License

This project is created for educational, academic, and portfolio purposes.