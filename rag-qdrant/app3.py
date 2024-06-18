from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import streamlit as st
import numpy as np
from reranking import reranking
from visualize_scores import plot_comparisons
from filter_chunks_reranked import filter_chunks_reranked
from create_response import create_response, estimate_gpt_cost,generate_full_prompt
from create_collection import create_qdrant_collection_from_upload,extract_text_from_load_document
import os

provide_response_retrieved = False
generate_response_flag = True

def collection_exists(client, collection_name):
    try:
        # Try to get collection information
        client.get_collection(collection_name=collection_name)
        return True  # If this succeeds, the collection exists
    except Exception:
        return False  # If any error occurs, assume the collection does not exist


def main():
    st.title("Retrieval-Augmented Generation with ReRanking (Hosted by Qdrant)")
    st.markdown("""
    Get started with our advanced document retrieval system that includes re-ranking capabilities. This combination helps provide more accurate and cost-efficient responses quickly. Adjust the re-ranking settings to fit your specific needs.
    """)

    st.sidebar.header("Configuration")
    st.sidebar.write("Select a Reranking Model Size:")
    model_options = {
        "ms-marco-MiniLM-L-12-v2": "23 MB - Balances performance and size effectively with the MiniLM architecture.",
        "ms-marco-TinyBERT-L-2-v2": "4 MB - Optimized for high efficiency in resource-constrained environments.",
    }

    choice = st.sidebar.selectbox("Choose Model", list(model_options.keys()))
    st.sidebar.markdown("### Model Details")
    st.sidebar.info(model_options[choice])

    example_questions = """
        1. What phase is Project Alpha currently in?
        2. How does the retrieval component of the RAG system work?
        3. What technique does Optuna use to optimize hyperparameters?
        4. What makes BERT different from other transformer models?
        5. What new feature has David's team created to improve the LightGBM classifier?
        6. What are the quantitative metrics used to evaluate the RAG system's performance?
        7. How has transfer learning been beneficial for fine-tuning the BERT model?
        8. What challenge is associated with training the BERT model?
        9. What approach has been used to improve the LightGBM classifier's performance?
        10. What is the primary use of the BERT model in Sarah's projects?
        11. How is the new module for Project Alpha expected to improve real-time analytics?
        12. What are the main challenges in real-time data processing as described by John?
        13. What potential collaboration did Sarah suggest between the BERT project and the RAG system?
        14. What addition is David's team looking to incorporate into their LightGBM model?
        15. What initiative is John planning to maintain a high level of expertise within the team?
    """
    st.sidebar.markdown("### Example Queries")
    st.sidebar.write(example_questions)

    uploaded_file = st.file_uploader("Upload a file")

    if uploaded_file is not None:
        # Display the file name
        st.write("Filename: ", uploaded_file.name)

        # Determine collection name (excluding file format)
        collection_name = os.path.splitext(uploaded_file.name)[0]
        
        # Check for existing collection
        client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
        if collection_exists(client, collection_name):
            st.success(f"Collection '{collection_name}' already exists. Loading...")
            db = Qdrant(client=client, embeddings=FastEmbedEmbeddings(), collection_name=collection_name)
            st.write(f"Loaded Dataset: {collection_name}")
        else:
            st.warning(f"Collection '{collection_name}' does not exist. Creating...")
            create_qdrant_collection_from_upload(uploaded_file)
            # Load the newly created collection
            st.success(f"Collection '{collection_name}' successfully created and loaded.")
            db = Qdrant(client=client, embeddings=FastEmbedEmbeddings(), collection_name=collection_name)
            st.write(f"Loaded Dataset: {collection_name}")

        with st.form(key='input_form'):
            input_query= st.text_area("Enter your query here:", height=150)
            submit_button = st.form_submit_button('Run Retrieval and Reranking')

        if submit_button:
            with st.spinner('Fetching data from Qdrant Client'):
                retrieved_entries = db.similarity_search_with_score(query=input_query, k=25)
                retrieved_results = [
                    {"id": doc.metadata['_id'], "text": doc.page_content, "cosine_similarity": score}
                    for doc, score in retrieved_entries
                ]
            with st.spinner('Reraking the fetched data..'):
                reranked_results = reranking(input_query, retrieved_results, choice)

            combined_filtered_context = " ".join(doc["text"] for doc in filter_chunks_reranked(reranked_results))
            
            if generate_response_flag:
                with st.spinner('Generating GPT responses, please wait... This may take a few moments depending on the complexity of the query and the load on the system.'):
                    generate_gpt_responses(input_query, combined_filtered_context)

def generate_gpt_responses(input_query, reranked_context):
    
    if not reranked_context:
        st.write("**Answer:** The requested information is not available in the provided file. Please refine your query or upload a more relevant document.")
    else:
        prompt, answer_reranked, cost_reranked = create_response(input_query, reranked_context)
        st.markdown("### Optimized GPT Response")
        st.write(f"**Answer:** {answer_reranked}")
if __name__ == "__main__":
    main()
