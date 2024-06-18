from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import streamlit as st
import numpy as np
from reranking import reranking
from visualize_scores import plot_comparisons
from filter_chunks_reranked import filter_chunks_reranked
from create_response import create_response, estimate_gpt_cost,generate_full_prompt


provide_response_retrieved = False

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
        "rank-T5-flan": "110 MB - Enhanced T5 model fine-tuned on diverse NLP tasks, ideal for integrating with RAG systems due to its robust text processing and adaptability."
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

    with st.form(key='input_form'):
        input_query= st.text_area("Enter your query here:", height=150)
        submit_button = st.form_submit_button('Run Retrieval and Reranking')

    if submit_button:
        st.markdown("## Processing Your Query")
        with st.spinner('Fetching data from Qdrant Client and reranking...'):
            client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
            db = Qdrant(client=client, embeddings=FastEmbedEmbeddings(), collection_name="ml_meeting")
            st.write("Loaded Dataset: company_meeting_ml")
            retrieved_entries = db.similarity_search_with_score(query=input_query, k=25)
            retrieved_results = [
                {"id": doc.metadata['_id'], "text": doc.page_content, "cosine_similarity": score}
                for doc, score in retrieved_entries
            ]
            retrieved_sims = [np.round(result["cosine_similarity"], 2) for result in retrieved_results]
            reranked_results = reranking(input_query, retrieved_results, choice)
            st.subheader("Comparison of Similarity Scores")

            plot_comparisons(reranked_results, retrieved_sims)

            combined_original_context = " ".join(doc["text"] for doc in retrieved_results)
            combined_filtered_context = " ".join(doc["text"] for doc in filter_chunks_reranked(reranked_results))
            display_responses(combined_original_context, combined_filtered_context)
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
