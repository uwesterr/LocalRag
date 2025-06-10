# Adaptive RAG with Local LLMs

This project implements an Adaptive RAG (Retrieval Augmented Generation) system using local LLMs through Ollama. It leverages LangGraph to create a sophisticated workflow that dynamically routes queries to appropriate retrieval methods.

## Overview

Adaptive RAG combines query analysis with self-corrective RAG to optimize information retrieval. Based on the [Adaptive RAG paper](https://arxiv.org/abs/2403.14403), this implementation routes queries between:

- Web search (for questions about recent events)
- Self-corrective RAG (for questions related to the indexed documents)

## Architecture

The system uses a LangGraph workflow that:

1. Analyzes the user query to determine the appropriate data source
2. Routes to either web search or vector retrieval
3. Grades document relevance
4. Transforms queries when necessary
5. Generates answers
6. Validates answer quality and relevance

![Adaptive RAG Workflow](attachment:3755396d-c4a8-45bd-87d4-00cb56339fe5.png)

## Requirements

- Python 3.8+
- Ollama (for local LLM inference)
- Tavily API key (for web search)
- Nomic API key (for embeddings)

## Setup

1. Install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Pull a Mistral model: `ollama pull mistral`
3. Install required packages:

```bash
pip install -U langchain-nomic langchain_community tiktoken langchainhub chromadb langchain langgraph tavily-python nomic[local]
```

4. Set up API keys for Tavily and Nomic

## Key Components

- **Document Retrieval**: Uses Nomic embeddings and Chroma vector store
- **Query Routing**: Analyzes queries to determine optimal retrieval path
- **Document Grading**: Evaluates relevance of retrieved documents
- **Answer Generation**: Uses local LLM with RAG
- **Hallucination Detection**: Checks if generated answers are grounded in retrieved documents
- **Query Transformation**: Reformulates questions to improve retrieval results

## Usage

Run the Jupyter notebook `langgraph_adaptive_rag_local.ipynb` to explore the implementation.

## Example

The notebook includes an example query "What is the AlphaCodium paper about?" that demonstrates the full adaptive RAG workflow.

## LangSmith Integration

This project includes LangSmith integration for monitoring and debugging. You can view traces of the workflow execution for analysis.

## License

This project is provided as-is for educational and research purposes.
