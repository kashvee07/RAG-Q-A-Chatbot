import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
import tempfile

# Gemini Client
client = genai.Client(api_key="AQ.Ab8RN6JkqNwrRlyLZZJrMMqXcAUapOHLIURpLkDf19elh2v_ng")

st.title("📄 PDF Q&A Chatbot")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Vector DB
    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    question = st.text_input(
        "Ask a question"
    )

    if question:

        results = db.similarity_search(
            question,
            k=5
        )

        context = "\n\n".join(
            [doc.page_content for doc in results]
        )

        prompt = f"""
        Answer only using the context.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        st.write("### Answer")
        st.write(response.text)

        st.write("### Sources")

        for doc in results:
            st.write(
                f"Page {doc.metadata.get('page', 'Unknown')}"
            )