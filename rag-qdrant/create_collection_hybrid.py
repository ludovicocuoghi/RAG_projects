from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
import os
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# creates a persistant index to disk
client = QdrantClient(host="localhost", port=6333)
aclient = AsyncQdrantClient(host="localhost", port=6333)


file_path = "default"
# Ask the user for the file path
#file_path = input("Please enter the full path to the document file: ")

#exmaples
#/Users/ludovicocuoghi/Documents/CodingFolder/rag-qdrant/datafolder/attention_is_all_you_need.txt
#/Users/ludovicocuoghi/Documents/CodingFolder/rag-qdrant/datafolder/company_meeting_ml.txt

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Determine the file type
file_extension = os.path.splitext(file_path)[1].lower()

# Load the document based on the file type
if file_extension == '.pdf':
    loader = PyPDFLoader(file_path)
    documents = loader.load()
elif file_extension == '.txt':
    loader = TextLoader(file_path)
    documents = loader.load()

# Check if documents need to be wrapped in a list
#if isinstance(documents, str):
#   documents = [documents]  # Ensure documents are in list format if a single string is returned


 # load documents
documents = SimpleDirectoryReader('data').load_data()

# create our vector store with hybrid indexing enabled
# batch_size controls how many nodes are encoded with sparse vectors at once
vector_store = QdrantVectorStore(
    "attention_is_all_you_need_hybrid",
    client=client,
    aclient=aclient,
    enable_hybrid=True,
    batch_size=20,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
Settings.chunk_size = 512

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)