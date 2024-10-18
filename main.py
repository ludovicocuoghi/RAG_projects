from qdrant_client import QdrantClient
import streamlit as st
from reranking import reranking
from visualize_scores import plot_comparisons
from filter_chunks_reranked import filter_chunks_reranked
from create_response import create_response, estimate_gpt_cost,generate_full_prompt
from text_splitter import splits_into_chunks,extract_text_from_load_document
import os
import pandas as pd

provide_response_retrieved = False
generate_response_flag = True
#CHOOSE IF RUNNING THE VECTORDB ON MEMORY OR HOSTED ON QDRANT (VIA DOCKER)
#CLIENT_TYPE = "localhost:6333"
CLIENT_TYPE = ":memory:"

# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'input_query' not in st.session_state:
    st.session_state.input_query = None

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

    st.sidebar.markdown("### Example Queries for file ml_meeting.txt")
    st.sidebar.write(example_questions)

    # File uploader and processing
    uploaded_file = st.file_uploader("Upload a file")
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
        up_file = st.session_state.uploaded_file
        st.write("Filename:", up_file.name)

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        string_data = extract_text_from_load_document(uploaded_file.getvalue(), file_extension)
        st.text_area("File content", string_data, height=300)

        # Determine collection name (excluding file format)
        collection_name = os.path.splitext(uploaded_file.name)[0]

        db_client = QdrantClient(CLIENT_TYPE)
        # Check for existing collection
        if collection_exists(db_client, collection_name):
            st.warning(f"Collection '{collection_name}' already exists. Loading...")
        else:
            st.warning(f"Collection '{collection_name}' does not exist. Creating...")
            docs = splits_into_chunks(uploaded_file)
            db_client.add(collection_name=collection_name, documents=docs)  
            
        st.success(f"Collection '{collection_name}' Loaded succesfully!")
    
        with st.form(key='input_form'):
            input_query= st.text_area("Enter your query here:", height=150)
            submit_button = st.form_submit_button('Run Retrieval and Reranking')
        if submit_button:
            st.session_state.input_query = input_query
        if st.session_state.input_query:
            st.markdown("### Processing Your Query...")
            with st.spinner('Fetching data from Qdrant Client'):
                retrieved_entries = db_client.query(
                    collection_name=collection_name,
                    query_text=input_query,
                    limit=25 
                )
                retrieved_results = [
                    {"id": doc.id,
                    "text": doc.metadata.get("document", "").split('page_content=')[1].strip('"'),
                    "score": doc.score}
                    for doc in retrieved_entries
                ]
            with st.spinner('Reraking the fetched data..'):
                reranked_results = reranking(input_query, retrieved_results, choice)
            
            st.subheader(f"TOP 5 Chunks content after retrieval")

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
            
            if generate_response_flag:
                with st.spinner('Generating GPT responses, please wait... This may take a few moments depending on the complexity of the query and the load on the system.'):
                    generate_gpt_responses(input_query, combined_original_context, combined_filtered_context)

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
