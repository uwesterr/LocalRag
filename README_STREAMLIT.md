# Adaptive RAG Streamlit App

This Streamlit application demonstrates an Adaptive RAG (Retrieval-Augmented Generation) system using local LLMs. It intelligently routes questions between web search and document retrieval based on the query context.

## Features

- **Adaptive Query Routing**: Routes questions to either web search or local document retrieval
- **Document Relevance Grading**: Filters out irrelevant documents to improve answer quality
- **Query Transformation**: Rewrites queries to improve retrieval results
- **Hallucination Detection**: Validates that generated answers are grounded in retrieved data
- **Answer Quality Checking**: Ensures answers actually address the user's question
- **LangSmith Integration**: Optional tracing and debugging with LangSmith
- **Document Indexing**: Dynamic document indexing with chromadb and Nomic embeddings

## Prerequisites

1. **Local LLM via Ollama**:
   - Install [Ollama](https://ollama.ai/)
   - Pull a model: `ollama pull mistral` (or another supported model)

2. **API Keys**:
   - TAVILY_API_KEY: For web search capabilities
   - NOMIC_API_KEY: For embeddings
   - LANGCHAIN_API_KEY (optional): For LangSmith tracing
   - LANGCHAIN_PROJECT (optional): LangSmith project name

## Installation

1. Create and activate a Python environment (recommended Python 3.12+):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the requirements:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```
TAVILY_API_KEY=your_tavily_key
NOMIC_API_KEY=your_nomic_key
LANGCHAIN_API_KEY=your_langchain_key  # Optional
LANGCHAIN_PROJECT=your_project_name   # Optional
```

## Running the App

Launch the Streamlit app:

```bash
streamlit run adaptive_rag_streamlit_app.py
```

## Usage

1. The app will automatically initialize the document index on startup
2. Enter your question in the text input
3. Click "Submit" to process the question
4. View the answer and execution path on the right
5. Use "Rebuild Index" in the sidebar if you add new documents

## Folder Structure

- `docs/`: Place your documents here (PDFs, text files, etc.)
- `db/`: Vector database storage location (will be created automatically)
- `.env`: Environment variables file for API keys

## Customization

- Change the local LLM model in the Streamlit sidebar
- Modify document directories in the sidebar
- Add new documents to the `docs/` folder and rebuild the index

## Credits

Built with:
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [LangGraph](https://python.langchain.com/docs/langgraph)
- [Nomic Embeddings](https://blog.nomic.ai/posts/nomic-embed-text-v1)
- [Ollama](https://ollama.ai/)
- [Tavily](https://tavily.com/)
