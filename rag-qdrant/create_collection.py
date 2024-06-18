import os
import PyPDF2
from io import BytesIO
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document

def extract_text_from_pdf(pdf_data):
    """Extracts text from a PDF file given its data."""
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_data))
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_load_document(file_data, file_extension):
    """Loads a document based on its file type and returns its text content."""
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_data)
    elif file_extension in ['.txt', '.md', '.csv']:  # Handle other text-based formats
        return file_data.decode('utf-8')
    else:
        raise ValueError("Unsupported file type. Only .pdf, .txt, .md, and .csv files are supported.")

def create_qdrant_collection_from_upload(uploaded_file):
    """Creates a Qdrant collection from an uploaded file."""
    embedding_model = FastEmbedEmbeddings()
    collection_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    print(collection_name)
    print(file_extension)

    # Load and prepare the document
    document_text = extract_text_from_load_document(uploaded_file.getvalue(), file_extension)
    doc = Document(page_content=document_text)

    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents([doc])  # Ensure we pass a list of documents

    # Configuration for the Qdrant
    url = "http://localhost:6333"
    qdrant = Qdrant.from_documents(
        texts,
        embedding_model,
        url=url,
        prefer_grpc=False,
        collection_name=collection_name  # Use the file name as the collection name
    )
    print(f"Vector DB Successfully Created for collection '{collection_name}'!")

def split_text(doc):
    # Ensure documents are in list format if a single Document object is passed
    if isinstance(doc, Document):
        documents = [doc]
    else:
        documents = doc  # Assume it's already a list of Document objects

    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)  # This should be a list of Document objects
    return texts

def create_qdrant_collection_from_filepath(file_path):
    """Creates a Qdrant collection from a file path."""
    embedding_model = FastEmbedEmbeddings()
    file_extension = os.path.splitext(file_path)[1].lower()
    with open(file_path, 'rb') as file:
        file_data = file.read()

    document_text = extract_text_from_load_document(file_data, file_extension)
    documents = [Document(page_content=document_text)]

    texts = split_text(documents)
    collection_name = os.path.splitext(os.path.basename(file_path))[0]
    url = "http://localhost:6333"
    qdrant = Qdrant.from_documents(
        texts,
        embedding_model,
        url=url,
        prefer_grpc=False,
        collection_name=collection_name
    )
    print(f"Vector DB Successfully Created for collection '{collection_name}'!")

def main():
    file_path = input("Please enter the full path to the document file: ")
    create_qdrant_collection_from_filepath(file_path)

if __name__ == "__main__":
    main()
