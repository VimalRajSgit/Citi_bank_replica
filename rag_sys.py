import os

import camelot
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. Load Environment ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY missing.")


# --- 2. Paths ---
DATA_DIR = r"C:\projects\Citi Bank rag_reddis\Data"
CHROMA_PERSIST_DIR = "chroma_db"

EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
LLM_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"


# --- 3. Embeddings + LLM ---
embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

llm = ChatGroq(temperature=0, model_name=LLM_MODEL_NAME, api_key=groq_api_key)


# --- Extract Documents from a Single PDF ---
def extract_from_pdf(pdf_path):
    docs = []
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path, "page": i + 1, "type": "text"},
                    )
                )
    except Exception as e:
        print(f"Error reading text from {pdf_path}: {e}")

    # Extract tables
    try:
        tables = camelot.read_pdf(pdf_path, pages="all")
        for i, table in enumerate(tables):
            df = table.df.replace("\n", " ", regex=True)
            table_text = f"Table {i + 1} (Page {table.parsing_report['page']}):\n"
            table_text += df.to_string(index=False)

            docs.append(
                Document(
                    page_content=table_text,
                    metadata={
                        "source": pdf_path,
                        "page": table.parsing_report["page"],
                        "type": "table",
                    },
                )
            )
    except:
        pass

    return docs


# --- Extract from ALL PDFs in Data folder ---
def load_all_documents(data_dir):
    all_docs = []
    for file in os.listdir(data_dir):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file)
            print(f"Processing: {pdf_path}")
            extracted = extract_from_pdf(pdf_path)
            all_docs.extend(extracted)
    return all_docs


# --- 4. Build or Load Vectorstore ---
if not os.path.exists(CHROMA_PERSIST_DIR):
    print("Creating new DB...")

    raw_docs = load_all_documents(DATA_DIR)
    print("Total extracted:", len(raw_docs))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    docs = splitter.split_documents(raw_docs)
    print("Chunks:", len(docs))

    vectorstore = Chroma.from_documents(
        docs, embedding=embedding_function, persist_directory=CHROMA_PERSIST_DIR
    )
else:
    print("Loading existing DB...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_function
    )


# --- 5. RAG Chain ---
retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

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


# --- 6. Query Loop ---
if __name__ == "__main__":
    while True:
        q = input("\nAsk (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("Thinking...")
        print("\nAnswer:", rag_chain.invoke(q))
