import os
import time

from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

print("=" * 60)
print("       CITIBANK RAG - QUERY MODE")
print("=" * 60)

# --- 1. Load Environment ---
print("\n[1/4] Loading environment...")
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY missing.")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY missing.")
print("  ✅ API keys loaded")

# --- 2. Constants ---
INDEX_NAME = "citibank-rag"
EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
LLM_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"

# --- 3. Load Models ---
print(f"\n[2/4] Loading embedding model...")
print("  ⏳ Please wait...")
start = time.time()
embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print(f"  ✅ Embeddings ready in {time.time() - start:.1f}s")

print(f"\n[3/4] Connecting to Pinecone index '{INDEX_NAME}'...")
pc = Pinecone(api_key=pinecone_api_key)

# Verify index exists and has data
index = pc.Index(INDEX_NAME)
total_vectors = index.describe_index_stats().get("total_vector_count", 0)

if total_vectors == 0:
    raise ValueError(f"Index '{INDEX_NAME}' is empty! Run the ingestion script first.")

print(f"  ✅ Connected! Index has {total_vectors} vectors ready")

# Load vectorstore (no uploading, just connecting)
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedding_function,
    pinecone_api_key=pinecone_api_key,
)

# --- 4. Build RAG Chain ---
print(f"\n[4/4] Building RAG chain...")
llm = ChatGroq(temperature=0, model_name=LLM_MODEL_NAME, api_key=groq_api_key)

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
print("  ✅ RAG chain ready")

print("\n" + "=" * 60)
print("       READY! ASK YOUR CITIBANK QUESTIONS")
print("=" * 60)

# --- 5. Query Loop ---
if __name__ == "__main__":
    while True:
        q = input("\n💬 Ask (or 'exit'): ")
        if q.lower() == "exit":
            print("Goodbye!")
            break
        if not q.strip():
            continue
        print("  ⏳ Thinking...")
        start = time.time()
        answer = rag_chain.invoke(q)
        print(f"\n  ✅ Answer (in {time.time() - start:.1f}s):\n")
        print(answer)
