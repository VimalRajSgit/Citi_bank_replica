import os
import time

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import camelot
import fitz
from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ── Flask App ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── 1. Environment ────────────────────────────────────────
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY missing.")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY missing.")

# ── 2. Constants ──────────────────────────────────────────
DATA_DIR = r"C:\projects\Citi Bank rag_reddis\Data"
INDEX_NAME = "citibank-rag"
DIMENSION = 1024
EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
LLM_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"

# Count PDFs
pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
PDF_COUNT = len(pdf_files)
PDF_NAMES = [os.path.splitext(f)[0] for f in pdf_files]

# ── 3. Embeddings + LLM ──────────────────────────────────
print("[1/4] Loading embedding model...")
embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("  ✅ Embedding model loaded")

print("[2/4] Initializing LLM...")
llm = ChatGroq(temperature=0, model_name=LLM_MODEL_NAME, api_key=groq_api_key)
print("  ✅ LLM ready")

# ── 4. Pinecone ───────────────────────────────────────────
print("[3/4] Connecting to Pinecone...")
pc = Pinecone(api_key=pinecone_api_key)
existing_indexes = [i.name for i in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)
total_vectors = index.describe_index_stats().get("total_vector_count", 0)

if total_vectors == 0:
    print("  Index empty — ingesting PDFs...")

    def extract_from_pdf(pdf_path):
        docs = []
        filename = os.path.basename(pdf_path)
        try:
            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": filename, "page": i + 1, "type": "text"}))
        except Exception as e:
            print(f"  Error reading {pdf_path}: {e}")
        try:
            tables = camelot.read_pdf(pdf_path, pages="all")
            for i, table in enumerate(tables):
                df = table.df.replace("\n", " ", regex=True)
                table_text = f"Table {i + 1} (Page {table.parsing_report['page']}):\n" + df.to_string(index=False)
                docs.append(Document(page_content=table_text, metadata={"source": filename, "page": table.parsing_report["page"], "type": "table"}))
        except Exception:
            pass
        return docs

    all_docs = []
    for f in pdf_files:
        all_docs.extend(extract_from_pdf(os.path.join(DATA_DIR, f)))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    docs = splitter.split_documents(all_docs)

    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embedding_function,
        index_name=INDEX_NAME,
        pinecone_api_key=pinecone_api_key,
        batch_size=100,
    )
else:
    print(f"  ✅ Using existing vectorstore ({total_vectors} vectors)")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embedding_function,
        pinecone_api_key=pinecone_api_key,
    )

# ── 5. RAG Chain ──────────────────────────────────────────
print("[4/4] Building RAG chain...")

base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10},
)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever, llm=llm
)


def deduplicate_docs(docs):
    """Remove duplicate chunks returned by MultiQueryRetriever."""
    seen = set()
    unique = []
    for doc in docs:
        key = doc.page_content[:200]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def retrieve_and_format(question: str) -> str:
    docs = multi_retriever.invoke(question)
    docs = deduplicate_docs(docs)
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


prompt_template = """You are a highly accurate CitiBank document assistant.
Your job is to answer questions using ONLY the provided context extracted from official CitiBank PDF documents.

Instructions:
1. Read ALL the context carefully before answering.
2. Synthesize information from multiple sources when relevant.
3. Quote or closely paraphrase the original text to support your answer.
4. Cite the source document and page number, e.g. (Source: filename.pdf, Page 3).
5. If the context does not contain enough information, say:
   "The provided documents do not contain sufficient information to answer this question."
6. Do NOT make up information or use outside knowledge.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
rag_chain = (
    {"context": retrieve_and_format, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("  ✅ RAG chain ready — server starting!\n")


# ── Routes ────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html", pdf_count=PDF_COUNT, pdf_names=PDF_NAMES)


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        start = time.time()
        answer = rag_chain.invoke(question)
        elapsed = round(time.time() - start, 1)
        return jsonify({"answer": answer, "time": elapsed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/info", methods=["GET"])
def info():
    return jsonify({
        "pdf_count": PDF_COUNT,
        "pdf_names": PDF_NAMES,
        "index_name": INDEX_NAME,
        "llm_model": LLM_MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
