# HR Policy Assistant RAG

A beginner-friendly Retrieval-Augmented Generation (RAG) web application built with Python and Streamlit.

The application lets a user:

- Upload an HR Policy PDF.
- Extract text from every PDF page using PyMuPDF.
- Preserve the original PDF page number for every extracted chunk.
- Detect invalid, empty, or image-only/scanned PDFs gracefully.
- Split policy text into configurable overlapping chunks.
- Create normalized Sentence Transformer embeddings using `all-MiniLM-L6-v2`.
- Store those embeddings in an in-memory FAISS similarity index.
- Configure the number of retrieved chunks with Top-K.
- Ask questions using Streamlit's chat interface.
- Retrieve the most relevant HR-policy chunks for each question.
- Send only the retrieved policy context and the current question to Google Gemini.
- Require Gemini to answer only from the uploaded HR policy.
- Return the exact fallback message when the answer is unavailable.
- Display the PDF pages and retrieved source chunks used for an answer.
- Maintain chat history with Streamlit session state.
- Clear the conversation without rebuilding the PDF index.
- Cache the embedding model and processed FAISS index for reuse.

## Architecture

The application follows this RAG pipeline:

```text
                    ┌───────────────────┐
                    │   HR Policy PDF   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   PyMuPDF / fitz  │
                    │  Extract every    │
                    │       page        │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   Text Chunking   │
                    │ size + overlap    │
                    │ + page metadata   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Sentence          │
                    │ Transformers      │
                    │ all-MiniLM-L6-v2  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Normalized        │
                    │ Embeddings        │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │       FAISS       │
                    │  IndexFlatIP      │
                    └─────────┬─────────┘
                              │
                 User question
                              │
                              ▼
                    ┌───────────────────┐
                    │ Question          │
                    │ embedding         │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ FAISS Top-K       │
                    │ similarity search │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Retrieved HR      │
                    │ policy context    │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Google Gemini     │
                    │ gemini-3.8-flash  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Policy-grounded   │
                    │ answer + sources  │
                    └───────────────────┘
