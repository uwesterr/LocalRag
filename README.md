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
pip install -U langchain-nomic langchain_community tiktoken langchainhub chromadb langchain langgraph tavily-python nomic[local] python-dotenv
```

4. Create a `.env` file in the project root with the following content:
```
TAVILY_API_KEY=your_tavily_api_key_here
NOMIC_API_KEY=your_nomic_api_key_here
# Uncomment if using LangSmith
# LANGCHAIN_API_KEY=your_langchain_api_key_here
# LANGCHAIN_PROJECT=your_langchain_project_name_here
```

5. Replace the placeholder values with your actual API keys

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

This project includes LangSmith integration for monitoring, debugging, and analyzing the workflow. To enable tracing:

1. Create a [LangSmith](https://smith.langchain.com/) account
2. Get your API key from the LangSmith dashboard
3. Add the following environment variables to your `.env` file:
   ```
   LANGCHAIN_API_KEY=your_langchain_api_key_here
   LANGCHAIN_PROJECT=adaptive-rag-local
   # Optional: Specify organization ID if you're part of multiple organizations
   # LANGCHAIN_ORG_ID=your_org_id
   ```

With LangSmith tracing enabled, you can:
- Visualize the complete flow of your graph execution
- Debug issues in complex workflows
- Analyze performance metrics (latency, token usage)
- Provide feedback on runs to improve your application
- Share traces with others for collaboration

The notebook will display links to view traces in the LangSmith UI after each execution.

## License

This project is provided as-is for educational and research purposes.
