import os
import time

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

print("=" * 60)
print("       CITIBANK RAG SYSTEM - STARTING UP")
print("=" * 60)

# --- 1. Load Environment ---
print("\n[1/7] Loading environment variables...")
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY missing.")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY missing.")
print("  ✅ API keys loaded successfully")

# --- 2. Paths & Constants ---
DATA_DIR = r"C:\projects\Citi Bank rag_reddis\Data"
INDEX_NAME = "citibank-rag"
DIMENSION = 1024

EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
LLM_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"

# --- 3. Embeddings + LLM ---
print(f"\n[2/7] Loading embedding model: '{EMBEDDING_MODEL_NAME}'...")
print("  ⏳ This may take a minute on first run (downloading model)...")
start = time.time()
embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print(f"  ✅ Embedding model loaded in {time.time() - start:.1f}s")

print(f"\n[3/7] Initializing LLM: '{LLM_MODEL_NAME}'...")
llm = ChatGroq(temperature=0, model_name=LLM_MODEL_NAME, api_key=groq_api_key)
print("  ✅ LLM ready")


# --- 4. Extract Documents from a Single PDF ---
def extract_from_pdf(pdf_path):
    docs = []
    filename = os.path.basename(pdf_path)

    # Extract text
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        text_count = 0
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename, "page": i + 1, "type": "text"},
                    )
                )
                text_count += 1
        print(f"    📄 Text: {text_count}/{page_count} pages extracted")
    except Exception as e:
        print(f"    ❌ Error reading text: {e}")

    # Extract tables
    try:
        print(f"    🔍 Scanning for tables...")
        tables = camelot.read_pdf(pdf_path, pages="all")
        table_count = tables.n
        for i, table in enumerate(tables):
            df = table.df.replace("\n", " ", regex=True)
            table_text = f"Table {i + 1} (Page {table.parsing_report['page']}):\n"
            table_text += df.to_string(index=False)
            docs.append(
                Document(
                    page_content=table_text,
                    metadata={
                        "source": filename,
                        "page": table.parsing_report["page"],
                        "type": "table",
                    },
                )
            )
        print(f"    📊 Tables: {table_count} tables extracted")
    except Exception as e:
        print(f"    ⚠️  No tables found or error: {e}")

    return docs


# --- 5. Extract from ALL PDFs ---
def load_all_documents(data_dir):
    all_docs = []
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    total_files = len(pdf_files)

    print(f"  📁 Found {total_files} PDF files in '{data_dir}'")
    print("-" * 50)

    for idx, file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(data_dir, file)
        print(f"\n  [{idx}/{total_files}] Processing: {file}")
        start = time.time()
        extracted = extract_from_pdf(pdf_path)
        elapsed = time.time() - start
        print(f"    ✅ Done — {len(extracted)} chunks in {elapsed:.1f}s")
        all_docs.extend(extracted)

    print("-" * 50)
    print(f"  📦 Total raw chunks from all PDFs: {len(all_docs)}")
    return all_docs


# --- 6. Setup Pinecone Index ---
print(f"\n[4/7] Connecting to Pinecone...")
pc = Pinecone(api_key=pinecone_api_key)
existing_indexes = [i.name for i in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"  ⚙️  Index '{INDEX_NAME}' not found. Creating...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"  ✅ Index '{INDEX_NAME}' created (dim={DIMENSION}, metric=cosine)")
else:
    print(f"  ✅ Index '{INDEX_NAME}' already exists")

# --- 7. Build or Load Vectorstore ---
print(f"\n[5/7] Checking index status...")
index = pc.Index(INDEX_NAME)
total_vectors = index.describe_index_stats().get("total_vector_count", 0)
print(f"  📊 Vectors currently in index: {total_vectors}")

if total_vectors == 0:
    print("\n  Index is empty — starting full ingestion pipeline...")

    # Extract
    print(f"\n  ── Step A: Extracting from PDFs ──")
    raw_docs = load_all_documents(DATA_DIR)

    # Split
    print(f"\n  ── Step B: Splitting into chunks ──")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    docs = splitter.split_documents(raw_docs)
    print(f"  ✅ {len(raw_docs)} raw chunks → {len(docs)} chunks after splitting")

    # Embed + Upload
    print(f"\n  ── Step C: Embedding + Uploading to Pinecone ──")
    print(f"  ⏳ Generating embeddings on CPU (this is the slow part, ~5-15 mins)...")
    print(f"  ⏳ Uploading {len(docs)} chunks to Pinecone...")
    start = time.time()
    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embedding_function,
        index_name=INDEX_NAME,
        pinecone_api_key=pinecone_api_key,
        batch_size=100,
    )
    elapsed = time.time() - start
    print(f"  ✅ Upload complete in {elapsed:.1f}s!")

    # Verify
    final_count = (
        pc.Index(INDEX_NAME).describe_index_stats().get("total_vector_count", 0)
    )
    print(f"  ✅ Pinecone now has {final_count} vectors stored")

else:
    print(f"  ✅ Skipping ingestion — loading existing vectorstore")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embedding_function,
        pinecone_api_key=pinecone_api_key,
    )

# --- 8. RAG Chain ---
print(f"\n[6/7] Building RAG chain with MultiQueryRetriever...")
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

print(f"\n[7/7] System is ready!")
print("=" * 60)
print("       ASK YOUR CITIBANK QUESTIONS BELOW")
print("=" * 60)

# --- 9. Query Loop ---
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
        elapsed = time.time() - start
        print(f"\n  ✅ Answer (in {elapsed:.1f}s):\n")
        print(answer)
