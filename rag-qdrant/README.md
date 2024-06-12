# Retrieval-Augmented Generation with ReRanking (Hosted by Qdrant)

## Overview
This project utilizes a Retrieval-Augmented Generation (RAG) system enhanced by Qdrant to offer advanced document retrieval capabilities combined with effective reranking. It's designed to provide precise and cost-effective responses in real-time for a variety of applications.

## Features
- **Model Configuration**: Users can select different Reranking model sizes and configurations to best fit their application needs, with the default being `ms-marco-MiniLM-L-12-v2`.
- **Advanced Document Retrieval**: Combines the power of transformer models and vector databases to fetch relevant documents.
- **Dynamic ReRanking**: Implements reranking to refine the retrieval results based on cosine similarity scores, ensuring the highest relevance of the information returned.

## Video DEMO

[![Watch the video](https://img.youtube.com/vi/2vnvE0LP40c/0.jpg)](https://www.youtube.com/watch?v=2vnvE0LP40c)

## Example Queries
1. What phase is Project Alpha currently in?
2. How does the retrieval component of the RAG system work?
3. What technique does Optuna use to optimize hyperparameters?

## Technologies/Libraries
1. Langchain
2. Qdrant
3. HugginFace (Reranking)
4. FastEmbed (by Qdrant)
5. OPENAI GPT
