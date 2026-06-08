import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from google import genai

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Smart PDF Q&A Chatbot")
st.write("Upload a PDF and ask questions about its contents.")

# -----------------------------
# GEMINI CLIENT
# -----------------------------
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# -----------------------------
# BUILD VECTOR DATABASE
# -----------------------------
@st.cache_resource
def create_vector_db(pdf_path):

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return db


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    with st.spinner("Processing PDF..."):
        db = create_vector_db(pdf_path)

    st.success("PDF processed successfully!")

    question = st.text_input(
        "Ask a question about the document"
    )

    if question:

        with st.spinner("Searching document..."):

            results = db.similarity_search(
                question,
                k=5
            )

            context = "\n\n".join(
                [doc.page_content for doc in results]
            )

            prompt = f"""
You are a helpful PDF Question Answering Assistant.

Use ONLY the provided context.

If the answer is not found in the context, say:
"I could not find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

            try:

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                st.subheader("Answer")
                st.write(response.text)

                st.subheader("Sources")

                pages = []

                for doc in results:
                    page = doc.metadata.get("page", "Unknown")

                    if page not in pages:
                        pages.append(page)

                for page in pages:
                    st.write(f"📄 Page {page + 1}")

            except Exception as e:
                st.error(f"Error: {e}")