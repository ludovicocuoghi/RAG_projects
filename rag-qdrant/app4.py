import streamlit as st
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

def main():
    st.title("Simple Query Interface with Qdrant")
    st.markdown("Enter your query to retrieve relevant documents from the Qdrant collection.")

    input_query = st.text_area("Query:", height=150)

    if st.button("Submit"):
        if input_query:
            results = query_qdrant(input_query)
            st.write(results)
        else:
            st.warning("Please enter a query.")

def query_qdrant(query):
    client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
    db = Qdrant(client=client, embeddings=FastEmbedEmbeddings(), collection_name="ml_meeting")
    retrieved_entries = db.similarity_search_with_score(query=query, k=25)

    retrieved_results = [{"id": doc.metadata['_id'], "text": doc.page_content, "cosine_similarity": score} for doc, score in retrieved_entries]

    return retrieved_results

if __name__ == "__main__":
    main()