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
import pandas as pd

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

    # example_questions = """
    #     1. What phase is Project Alpha currently in?
    #     2. How does the retrieval component of the RAG system work?
    #     3. What technique does Optuna use to optimize hyperparameters?
    #     4. What makes BERT different from other transformer models?
    #     5. What new feature has David's team created to improve the LightGBM classifier?
    #     6. What are the quantitative metrics used to evaluate the RAG system's performance?
    #     7. How has transfer learning been beneficial for fine-tuning the BERT model?
    #     8. What challenge is associated with training the BERT model?
    #     9. What approach has been used to improve the LightGBM classifier's performance?
    #     10. What is the primary use of the BERT model in Sarah's projects?
    #     11. How is the new module for Project Alpha expected to improve real-time analytics?
    #     12. What are the main challenges in real-time data processing as described by John?
    #     13. What potential collaboration did Sarah suggest between the BERT project and the RAG system?
    #     14. What addition is David's team looking to incorporate into their LightGBM model?
    #     15. What initiative is John planning to maintain a high level of expertise within the team?
    # """
    example_questions = """ 
    1. What specific feature engineering techniques did James implement to improve the CatBoost model's handling of categorical data, and what impact did they have on the model's performance?
    2. How did Sarah integrate transfer learning into her NLP project, and what were the specific benefits in terms of training time and model performance?
    3. Describe the hybrid approach Jennifer used in her recommendation system, including how she combined collaborative filtering and content-based filtering, and the role of the real-time feedback loop.
    4. What ensemble methods is Kevin using for anomaly detection, including how isolation forests, autoencoders, and clustering algorithms contribute to detecting suspicious activities?
    5. What suggestions did Jennifer offer to James for improving his CatBoost model, and how do these suggestions integrate with James's existing strategies?
    6. How is James planning to address the issue of imbalanced data in his CatBoost model, including the techniques he has tried and the recommendations he received from other team members?
    7. What metrics is Sarah using to evaluate the performance of the chatbot model, and how do these metrics relate to improvements in customer satisfaction and reduction in negative feedback?
    8. Explain how Jennifer is utilizing graph neural networks in her recommendation system, including the initial results and the scaling process she mentioned.
    9. What role does personalization play in the different ML projects mentioned in the meeting, specifically in James's churn model, Sarah's chatbot, Jennifer's recommendation system, and Kevin's anomaly detection?
    10. What are the benefits of the microservices architecture that Jennifer adopted for her recommendation system, and how has it improved the system's ability to handle larger datasets and more users?
    """

    st.sidebar.markdown("### Example Queries")
    st.sidebar.write(example_questions)

    uploaded_file = st.file_uploader("Upload a file")

    if uploaded_file is not None:
        # Display the file name
        st.write("Filename: ", uploaded_file.name)

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        string_data = extract_text_from_load_document(uploaded_file.getvalue(), file_extension)

        st.text_area("File content", string_data, height=300)

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
            st.markdown("### Processing Your Query...")
            with st.spinner('Fetching data from Qdrant Client'):
                retrieved_entries = db.similarity_search_with_score(query=input_query, k=25)
                retrieved_results = [
                    {"id": doc.metadata['_id'], "text": doc.page_content, "cosine_similarity": score}
                    for doc, score in retrieved_entries
                ]
            with st.spinner('Reraking the fetched data..'):
                reranked_results = reranking(input_query, retrieved_results, choice)
            
            # Extracting 'id' from both original and reranked results
            original_ids = [result["id"] for result in retrieved_results]
            reranked_ids = [result["id"] for result in reranked_results]

            # Creating a DataFrame and comparing IDs before and after reranking
            comparison_df = pd.DataFrame({
                "Original IDs": original_ids,
                "Reranked IDs": reranked_ids
            })

            # Adjusting the DataFrame's index to use as a ranking number starting from 1
            comparison_df.index = range(1, len(original_ids) + 1)

            # Displaying the DataFrame to visually compare the IDs at the top 10 ranks before and after reranking
            st.write("Comparison of IDs at Top 10 Ranks Before and After Reranking:")
            st.write(comparison_df.head(10))
            
            st.subheader("Comparison of Similarity Scores")

            plot_comparisons(reranked_results, retrieved_results)

            combined_original_context = " ".join(doc["text"] for doc in retrieved_results)
            combined_filtered_context = " ".join(doc["text"] for doc in filter_chunks_reranked(reranked_results))
            display_responses(combined_original_context, combined_filtered_context)
            
            # if generate_response_flag:
            #     with st.spinner('Generating GPT responses, please wait... This may take a few moments depending on the complexity of the query and the load on the system.'):
            #         generate_gpt_responses(input_query, combined_original_context, combined_filtered_context)

def display_responses(original_context, filtered_context):
    st.markdown("## Response Analysis")

    # Display original response context with length
    original_length = len(original_context.split())  # Length in terms of words
    st.markdown(f"### Original Response Context - {original_length} words")
    st.text_area("Original Context", original_context, height=150)

    # Display optimized response context after filtering with length
    filtered_length = len(filtered_context.split())  # Length in terms of words
    st.markdown(f"### Optimized Response Context after Filtering - {filtered_length} words")
    st.text_area("Filtered Context", filtered_context, height=150)


def generate_gpt_responses(input_query, original_context, reranked_context):
    
    if not reranked_context:
        st.write("**Answer:** The requested information is not available in the provided file. Please refine your query or upload a more relevant document.")
    else:
        #answer_original, cost_original = "xx", 0.002
        #answer_reranked, cost_reranked = "yy", 0.0001
        prompt, answer_reranked, cost_reranked = create_response(input_query, reranked_context)
        if provide_response_retrieved:
            _, answer_original, cost_original = create_response(input_query, original_context)       
            st.markdown("## GPT-Generated Responses")
            st.markdown("### Original GPT Response")
            st.write(f"**Answer:** {answer_original}")
            st.write(f"**Estimated Cost:** {cost_original:.5f}$")
        else:
            #consdering that output of using full context is same size of that obtained using filtered
            original_prompt = generate_full_prompt(input_query, original_context)
            cost_original = estimate_gpt_cost(original_prompt, answer_reranked)
        st.markdown("### Optimized GPT Response")
        st.write(f"**Answer:** {answer_reranked}")
        st.write(f"**Estimated Cost:** {cost_reranked:.5f}$")

        display_cost_efficiency(cost_original, cost_reranked)

def display_cost_efficiency(original_cost, reranked_cost):
    savings = original_cost - reranked_cost
    cost_effectiveness = original_cost / reranked_cost if reranked_cost else float('inf')

    st.markdown("## Cost Efficiency Analysis")
    st.markdown(f"**You saved:** {savings:.4f}$ by opting for the optimized response.")
    st.markdown(f"**Cost Effectiveness:** The optimized response is {cost_effectiveness:.2f} times more cost-effective.")

if __name__ == "__main__":
    main()
