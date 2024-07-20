import os
import PyPDF2
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

def splits_into_chunks(uploaded_file):
    """Splits input text data into chunks."""
    collection_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    print(collection_name)
    print(file_extension)

   # Load the document text based on its file type
    document_text = extract_text_from_load_document(uploaded_file.getvalue(), file_extension)
    
    # Initialize the document with the loaded text
    doc = Document(document_text)

    # Define a text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    # Split the document into chunks and return them
    document_chunks = text_splitter.split_documents([doc])  # Pass a list containing the document
    string_chunks = [str(chunk) if not isinstance(chunk, str) else chunk for chunk in document_chunks]
    return string_chunks