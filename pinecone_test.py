import os

from dotenv import load_dotenv
from pinecone import Pinecone

# Load .env
load_dotenv()

# Read key
api_key = os.getenv("PINECONE_API_KEY")
print("API Key Loaded:", bool(api_key))

# Initialize Pinecone
pc = Pinecone(api_key=api_key)

# Test call - list indexes
try:
    indexes = pc.list_indexes()
    print("Connection OK! Indexes:", indexes)
except Exception as e:
    print("Connection FAILED:", e)
