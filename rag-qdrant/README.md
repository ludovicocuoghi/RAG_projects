# Retrieval-Augmented Generation with ReRanking (Hosted by Qdrant)

## Overview
Our project introduces a cutting-edge Retrieval-Augmented Generation (RAG) system, powered by Qdrant, designed to enhance document retrieval and re-ranking capabilities. This integration not only improves precision but also optimizes response efficiency, making it ideal for real-time applications across various domains.

## Key Features
- **Model Configuration**: Choose from a selection of reranking models, tailored to meet diverse application demands. The default configuration is `ms-marco-MiniLM-L-12-v2`, renowned for its balance of performance and efficiency.
- **Advanced Document Retrieval**: Leverages transformer-based models and sophisticated vector database technology from Qdrant to retrieve highly relevant documents swiftly.
- **Dynamic ReRanking**: Utilizes cosine similarity scores to dynamically rerank search results, ensuring the delivery of the most pertinent information.

## Video Demo
Click below to watch a brief demonstration of our RAG system in action:

[[![Watch the video](https://img.youtube.com/vi/tDP2tAlZGFU/0.jpg)](https://www.youtube.com/watch?v=tDP2tAlZGFU)]



## Example Queries
Explore the capabilities of our system with these sample queries:
1. "What phase is Project Alpha currently in?"
2. "How does the retrieval component of the RAG system work?"
3. "What technique does Optuna use to optimize hyperparameters?"

## Technologies and Libraries
This project is built using several leading technologies and libraries:
- **Langchain**: For constructing language models and managing linguistic data.
- **Qdrant**: Employs a powerful vector database for efficient data storage and retrieval.
- **HuggingFace Transformers**: Provides a robust platform for reranking using state-of-the-art pre-trained models.
- **FastEmbed**: Developed by Qdrant, this tool enhances embedding functionalities.
- **OpenAI GPT**: Utilized for generating coherent and contextually relevant text responses.

## Getting Started
To get started with this project, clone the repository and follow the setup instructions in the installation guide. Ensure you have the required dependencies installed, as listed in the `requirements.txt` file.

## How to Contribute
We welcome contributions from the community. If you wish to contribute, please fork the repository and submit a pull request. For more details, check out our contributing guidelines in the `CONTRIBUTING.md` file.

## License
This project is licensed under the MIT License - see the `LICENSE` file for more details.
