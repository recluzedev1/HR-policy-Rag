"""
HR Policy Assistant
-------------------
A beginner-friendly RAG application using:

- Streamlit              -> Web interface
- PyMuPDF                -> PDF text extraction
- SentenceTransformers   -> Text embeddings
- FAISS                   -> Vector similarity search
- Gemini                  -> LLM answer generation

RAG pipeline:

PDF
 ↓
Text extraction
 ↓
Text chunks
 ↓
Embeddings
 ↓
FAISS index
 ↓
User question
 ↓
Question embedding
 ↓
Similarity search
 ↓
Relevant chunks
 ↓
Gemini LLM
 ↓
Answer
"""

import os
from typing import List, Dict

import fitz
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

# Sentence Transformer model used for creating embeddings.
MODEL_NAME = "all-MiniLM-L6-v2"

# Gemini model used for generating the final answer.
GEMINI_MODEL = "gemini-2.5-flash"

# Default RAG settings.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5

# Message shown when the answer cannot be found
# in the retrieved HR policy context.
FALLBACK_MESSAGE = (
    "The information could not be found in the uploaded HR policy."
)


# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f4e79;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666666;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .source-box {
            background-color: #f7f9fc;
            border-left: 4px solid #1f77b4;
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 5px;
        }

        .success-box {
            background-color: #eef8f0;
            border-left: 4px solid #2e8b57;
            padding: 10px 15px;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Cached embedding model
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    """
    Load the Sentence Transformer model once.

    Streamlit caches this resource so we don't download/load
    the model again for every question.
    """
    return SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> List[Dict]:
    """
    Extract text from every page of a PDF.

    Returns:
        A list of dictionaries containing:
        - page number
        - page text

    Page numbers are 1-based for user-friendly display.
    """

    pages = []

    try:
        pdf_document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        if pdf_document.page_count == 0:
            pdf_document.close()
            raise ValueError("The PDF contains no pages.")

        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)

            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "page": page_number + 1,
                        "text": text,
                    }
                )

        pdf_document.close()

    except Exception as exc:
        raise ValueError(
            f"Could not read the PDF. Please make sure it is a valid PDF file."
        ) from exc

    return pages


# ---------------------------------------------------------
# Text chunking
# ---------------------------------------------------------

def chunk_text(
    pages: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Split extracted PDF text into overlapping chunks.

    Each chunk keeps its page number so that we can show
    the user where the information came from.

    This is a simple character-based chunking strategy,
    which is easy for beginners to understand.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"]

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "page": page_number,
                    }
                )

            if end >= len(text):
                break

            start = end - chunk_overlap

    return chunks


# ---------------------------------------------------------
# Create embeddings
# ---------------------------------------------------------

def create_embeddings(
    texts: List[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Convert text chunks into numerical vectors.

    normalize_embeddings=True makes cosine similarity
    equivalent to inner-product similarity when using
    FAISS IndexFlatIP.
    """

    if not texts:
        raise ValueError("There is no text to embed.")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.astype("float32")


# ---------------------------------------------------------
# Build FAISS index
# ---------------------------------------------------------

def build_faiss_index(
    embeddings: np.ndarray,
) -> faiss.Index:
    """
    Create a FAISS similarity-search index.

    IndexFlatIP performs inner-product similarity.
    Because our embeddings are normalized, this behaves
    like cosine similarity.
    """

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ---------------------------------------------------------
# Retrieve relevant chunks
# ---------------------------------------------------------

def retrieve_relevant_chunks(
    question: str,
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: List[Dict],
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict]:
    """
    Embed the user's question and search the FAISS index.

    Returns the top matching chunks with similarity scores.
    """

    if not question.strip():
        return []

    if index.ntotal == 0:
        return []

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    number_to_retrieve = min(top_k, index.ntotal)

    scores, indices = index.search(
        question_embedding,
        number_to_retrieve,
    )

    results = []

    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0:
            continue

        chunk = chunks[index_position].copy()
        chunk["score"] = float(score)

        results.append(chunk)

    return results


# ---------------------------------------------------------
# Build context for LLM
# ---------------------------------------------------------

def build_context(retrieved_chunks: List[Dict]) -> str:
    """
    Convert retrieved chunks into a clearly labelled context
    that can be sent to the LLM.
    """

    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"""
--- SOURCE {number} ---
Page: {chunk['page']}
Content:
{chunk['text']}
--- END SOURCE {number} ---
""".strip()
        )

    return "\n\n".join(context_parts)


# ---------------------------------------------------------
# Get Gemini client
# ---------------------------------------------------------

def get_gemini_client():
    """
    Create a Gemini client using GEMINI_API_KEY.

    The key can come from:
    1. Streamlit secrets
    2. Environment variables / .env
    """

    api_key = None

    # First try Streamlit secrets.
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    # Then try environment variable.
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    # Stop with a helpful message if no key is available.
    if not api_key:
        raise ValueError(
            "Gemini API key is missing. Add GEMINI_API_KEY "
            "to your .env file or Streamlit secrets."
        )

    # Create and return the Gemini client.
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------
# Generate answer using Gemini
# ---------------------------------------------------------

def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
) -> str:
    """
    Send retrieved HR policy context and the question to Gemini.

    The model is explicitly instructed to answer ONLY from
    the retrieved policy context.
    """

    # If FAISS did not find anything, don't call Gemini.
    if not retrieved_chunks:
        return FALLBACK_MESSAGE

    # Create the Gemini client.
    client = get_gemini_client()

    # Convert retrieved chunks into context for Gemini.
    context = build_context(retrieved_chunks)

    # System instruction that controls Gemini's behavior.
    system_prompt = f"""
You are an HR Policy Assistant.

Your job is to answer questions ONLY using the HR policy
context provided below.

STRICT RULES:

1. Use only information contained in the provided context.
2. Do not use your general knowledge.
3. Do not guess.
4. Do not invent policy rules, numbers, dates, benefits,
   requirements, or exceptions.
5. If the answer cannot be determined from the provided
   context, respond exactly with:

"The information could not be found in the uploaded HR policy."

6. Keep the answer clear, concise, and beginner-friendly.
7. When useful, mention the relevant policy page number.
8. If multiple policy sections are relevant, combine them
   carefully without adding information that isn't present.
9. Treat the uploaded policy as the source of truth.

HR POLICY CONTEXT:

{context}
"""

    try:
        # Send the question and HR policy context to Gemini.
        #
        # Gemini's current Python SDK uses
        # client.models.generate_content().
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=question,
            config={
                "system_instruction": system_prompt,
                "temperature": 0,
                "max_output_tokens": 1024,
            },
        )

        # Get the generated text from Gemini.
        answer = response.text

        # If Gemini returned no text, use our fallback message.
        if not answer:
            return FALLBACK_MESSAGE

        return answer.strip()

    except Exception as exc:
        raise RuntimeError(
            "The Gemini API could not generate an answer. "
            "Please check your Gemini API key and internet connection."
        ) from exc


# ---------------------------------------------------------
# Display sources
# ---------------------------------------------------------

def display_sources(retrieved_chunks: List[Dict]) -> None:
    """
    Display the pages/chunks retrieved by FAISS.
    """

    if not retrieved_chunks:
        return

    st.markdown("### 📌 Sources")

    for number, chunk in enumerate(retrieved_chunks, start=1):
        score = chunk.get("score", 0)

        with st.expander(
            f"Source {number} — Page {chunk['page']} "
            f"(similarity: {score:.3f})"
        ):
            st.write(chunk["text"])


# ---------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "pages" not in st.session_state:
    st.session_state.pages = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">📚 HR Policy Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Upload an HR policy PDF and ask questions about its contents.
    The assistant uses Retrieval-Augmented Generation (RAG) to
    retrieve relevant policy sections before generating an answer.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    chunk_size = st.slider(
        "Chunk size",
        min_value=500,
        max_value=2000,
        value=DEFAULT_CHUNK_SIZE,
        step=100,
        help="Approximate number of characters in each text chunk.",
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=500,
        value=DEFAULT_CHUNK_OVERLAP,
        step=50,
        help="Number of characters shared between neighboring chunks.",
    )

    top_k = st.slider(
        "Retrieved chunks (k)",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
        help="Number of relevant chunks retrieved from FAISS.",
    )

    st.divider()

    st.markdown("### 🔄 RAG Pipeline")

    st.markdown(
        """
        **1.** Upload PDF  
        ↓  
        **2.** Extract text  
        ↓  
        **3.** Create chunks  
        ↓  
        **4.** Generate embeddings  
        ↓  
        **5.** Build FAISS index  
        ↓  
        **6.** Retrieve relevant chunks  
        ↓  
        **7.** Ask Gemini LLM  
        ↓  
        **8.** Generate answer
        """
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------
# PDF uploader
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload your HR Policy PDF",
    type=["pdf"],
    help="Upload a PDF containing your organization's HR policy.",
)


# ---------------------------------------------------------
# Process uploaded PDF
# ---------------------------------------------------------

if uploaded_file is not None:

    # Detect a new file.
    if uploaded_file.name != st.session_state.document_name:

        with st.spinner(
            "Processing PDF... This may take a moment."
        ):
            try:
                pdf_bytes = uploaded_file.getvalue()

                if not pdf_bytes:
                    st.error("The uploaded PDF is empty.")
                    st.stop()

                # 1. Extract text
                pages = extract_text_from_pdf(pdf_bytes)

                if not pages:
                    st.error(
                        "No extractable text was found in this PDF. "
                        "The PDF may be scanned/image-based. "
                        "Please upload a text-based PDF."
                    )
                    st.stop()

                # 2. Chunk text
                chunks = chunk_text(
                    pages,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                if not chunks:
                    st.error(
                        "No usable text chunks could be created "
                        "from the PDF."
                    )
                    st.stop()

                # 3. Load embedding model
                embedding_model = load_embedding_model()

                # 4. Generate embeddings
                texts = [chunk["text"] for chunk in chunks]

                embeddings = create_embeddings(
                    texts,
                    embedding_model,
                )

                # 5. Build FAISS index
                index = build_faiss_index(embeddings)

                # Store everything in session state.
                st.session_state.document_name = uploaded_file.name
                st.session_state.pages = pages
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.document_ready = True

                # Clear old conversation because a new policy
                # has been uploaded.
                st.session_state.messages = []

                st.success(
                    f"✅ {uploaded_file.name} processed successfully."
                )

            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            except Exception:
                st.error(
                    "Something went wrong while processing the PDF. "
                    "Please try another PDF."
                )
                st.stop()


# ---------------------------------------------------------
# Document status
# ---------------------------------------------------------

if st.session_state.document_ready:

    st.markdown(
        f"""
        <div class="success-box">
        <strong>📄 Current policy:</strong>
        {st.session_state.document_name}<br>
        <strong>📑 Pages with text:</strong>
        {len(st.session_state.pages)}<br>
        <strong>🧩 Chunks:</strong>
        {len(st.session_state.chunks)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")


else:

    st.info(
        "👆 Upload an HR Policy PDF to start asking questions."
    )


# ---------------------------------------------------------
# Display previous chat messages
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Show sources for assistant messages.
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            display_sources(message["sources"])


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about the HR policy..."
)


# ---------------------------------------------------------
# Handle user question
# ---------------------------------------------------------

if question:

    if not st.session_state.document_ready:

        st.warning(
            "Please upload an HR Policy PDF before asking a question."
        )
        st.stop()

    # Display user message.
    with st.chat_message("user"):
        st.markdown(question)

    # Save user message.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    try:

        embedding_model = load_embedding_model()

        # Retrieve relevant chunks.
        retrieved_chunks = retrieve_relevant_chunks(
            question=question,
            model=embedding_model,
            index=st.session_state.index,
            chunks=st.session_state.chunks,
            top_k=top_k,
        )

        # Generate answer.
        with st.chat_message("assistant"):

            with st.spinner("Searching the policy..."):

                answer = generate_answer(
                    question,
                    retrieved_chunks,
                )

            st.markdown(answer)

            # Display source chunks.
            display_sources(retrieved_chunks)

        # Save assistant response.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": retrieved_chunks,
            }
        )

    except ValueError as exc:

        st.error(str(exc))

    except RuntimeError as exc:

        st.error(str(exc))

    except Exception:

        st.error(
            "An unexpected error occurred while answering "
            "your question. Please try again."
        )
