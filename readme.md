# 🏦 CitiBank RAG Assistant

An AI-powered Retrieval Augmented Generation (RAG) system for querying CitiBank documents. Built entirely with **open-source models** — no paid API dependencies for embeddings or LLM inference beyond free-tier services.

> Data source: [CitiBank Business FAQs](https://businessaccess.citibank.citigroup.com/cbusol/faq/faq.action#12)

---

## 🔧 Tech Stack

| Component | Technology | Notes |
|---|---|---|
| **LLM** | `meta-llama/llama-4-maverick-17b-128e-instruct` | Open-source, via Groq (free tier) |
| **Embeddings** | `mixedbread-ai/mxbai-embed-large-v1` | Open-source, runs locally on CPU |
| **Vector DB** | Pinecone (Serverless) | Free tier, cosine similarity |
| **Table Extraction** | Camelot | Extracts tables from PDFs as structured text |
| **Text Extraction** | PyMuPDF (fitz) | Extracts raw text from PDF pages |
| **Framework** | LangChain | RAG chain, MultiQueryRetriever, prompt templates |
| **Frontend** | Flask + Vanilla JS | Dark-themed glassmorphism UI |

## 📄 PDF Processing

- **Text** is extracted page-by-page using `PyMuPDF`
- **Tables** are extracted using `Camelot` (`camelot.read_pdf`) so structured tabular data isn't lost during chunking
- Documents are split using `RecursiveCharacterTextSplitter` (chunk size: 1500, overlap: 250)
- All chunks are embedded and stored in Pinecone with source + page metadata

## 🧠 RAG Pipeline

1. User query → **MultiQueryRetriever** generates multiple query variants for broader retrieval
2. Top-10 chunks retrieved via similarity search from Pinecone
3. Duplicate chunks are deduplicated
4. Context is formatted with source citations (filename + page number)
5. **Llama 4** generates the answer strictly from the retrieved context

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install flask flask-cors langchain langchain-groq langchain-huggingface langchain-pinecone pinecone-client python-dotenv pymupdf camelot-py[cv] ghostscript

# 2. Set up .env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key

# 3. Run
python app.py
```

Open `http://localhost:5000` in your browser.

## 📁 Project Structure

```
├── app.py                 # Flask server + RAG pipeline
├── chat.py                # CLI-based RAG query mode
├── rag_sys_pinecone.py    # Standalone Pinecone ingestion script
├── templates/
│   └── index.html         # Chat UI
├── static/
│   ├── style.css          # Dark glassmorphism theme
│   └── script.js          # Chat logic + particle background
├── Data/                  # CitiBank PDF documents (16 files)
└── .env                   # API keys (not committed)
```

## 🚧 Pending

- [ ] **Redis integration** — Caching layer for faster repeated queries and session management

## 📝 License

For educational / demo purposes only. Not affiliated with Citibank.
