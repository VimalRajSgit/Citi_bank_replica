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

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), llm=llm
)

prompt_template = """
Answer strictly using the provided context.
If the answer is not present, say:
"The provided context does not contain the answer to this question."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
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
