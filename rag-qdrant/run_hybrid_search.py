
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import streamlit as st
from reranking import reranking
from visualize_scores import plot_comparisons
from filter_chunks_reranked import filter_chunks_reranked
import numpy as np
embedding_model = FastEmbedEmbeddings()

url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)

query_input = "What is self attention?"

db = Qdrant(client=client, embeddings=embedding_model, collection_name="lightgbm_opt_cv")

client = QdrantClient(url=url, prefer_grpc=False)
db = Qdrant(client=client, embeddings=embedding_model, collection_name="lightgbm_opt_cv")
retrieved_entries = db.similarity_search_with_score(query=query_input, k=25)
print(retrieved_entries)