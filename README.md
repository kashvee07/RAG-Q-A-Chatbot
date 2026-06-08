# RAG-Q-A-Chatbot
# 📄 Smart PDF Q&A Chatbot

This project is a Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload a PDF and ask questions about its contents. The chatbot retrieves relevant information from the document and uses Gemini to generate answers based on that information.

## Features

* Upload PDF files
* Ask questions about the document
* Semantic search using vector embeddings
* Answers generated using Gemini
* Displays source page numbers
* Simple Streamlit interface

## Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* HuggingFace Embeddings
* Google Gemini API
* PyPDF

## How It Works

1. The uploaded PDF is converted into text.
2. The text is split into smaller chunks.
3. Each chunk is converted into embeddings using a HuggingFace model.
4. The embeddings are stored in a FAISS vector database.
5. When a user asks a question, the most relevant chunks are retrieved.
6. These chunks are sent to Gemini along with the question.
7. Gemini generates an answer using the retrieved context.

## Project Structure

```text
rag-q-a-chatbot/
│
├── app.py
├── requirements.txt
└── README.md
```

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Create a `.streamlit/secrets.toml` file and add your Gemini API key:

```toml
GEMINI_API_KEY = "YOUR_API_KEY"
```

Run the application:

```bash
streamlit run app.py
```

## What I Learned

* Basics of Retrieval-Augmented Generation (RAG)
* Working with vector databases
* Generating embeddings for semantic search
* Integrating LLMs with external knowledge sources
* Deploying applications using Streamlit

## Future Improvements

* Support for multiple PDFs
* Chat history and conversation memory
* Better source citations
* Document summarization
* Hybrid search (FAISS + keyword search)

## Demo

Add your deployed Streamlit link here.

## Author

Kashvee Singh
