from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
import os

# Ask the user for the file path
file_path = input("Please enter the full path to the document file: ")

#exmaples
#/Users/ludovicocuoghi/Documents/CodingFolder/rag-qdrant/datafolder/attention_is_all_you_need.txt
#/Users/ludovicocuoghi/Documents/CodingFolder/rag-qdrant/datafolder/company_meeting_ml.txt


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
if isinstance(documents, str):
    documents = [documents]  # Ensure documents are in list format if a single string is returned

# Split the text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Load the embedding model
embedding_model = FastEmbedEmbeddings()

# Extract collection name from the file path
collection_name = os.path.splitext(os.path.basename(file_path))[0]

# URL configuration
url = "http://localhost:6333"

# Create Qdrant collection with the same name as the file (without extension)
qdrant = Qdrant.from_documents(
    texts,
    embedding_model,
    url=url,
    prefer_grpc=False,
    collection_name=collection_name  # Using the file name as the collection name
)

print(f"Vector DB Successfully Created for collection '{collection_name}'!")
