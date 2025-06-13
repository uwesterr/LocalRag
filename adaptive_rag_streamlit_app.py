import streamlit as st
import os
import glob
import uuid
import io
import base64
from pathlib import Path
from dotenv import load_dotenv
from pprint import pprint
import time

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.tracers import ConsoleCallbackHandler, LangChainTracer
from langchain import hub
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langsmith import Client
from IPython.display import Image as IPythonImage
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Adaptive RAG with Local LLM",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    div[data-testid="stExpander"] details summary p {
        font-size: 1.2rem;
    }
    .step-label {
        font-size: 0.8rem;
        color: #888;
    }
    .step-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .graph-node {
        border: 1px solid #3366cc;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        background-color: #f0f8ff;
        color: #000 !important;
        display: block;
        font-family: monospace;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Check for new documents
def check_for_new_documents(doc_dir):
    """Check if there are new documents in the directory that haven't been indexed yet"""
    if 'retriever' not in st.session_state or not st.session_state.retriever:
        return []
    
    all_files = glob.glob(os.path.join(doc_dir, "**", "*"), recursive=True)
    
    # Get existing documents from the vector store
    try:
        vs = st.session_state.retriever.vectorstore
        existing = {md.get("source") for md in vs.get()["metadatas"] or []}
        
        # Find new documents
        new_files = []
        for fp in all_files:
            if os.path.isfile(fp) and fp not in existing:
                new_files.append(os.path.basename(fp))
        
        if new_files:
            st.session_state.rebuild_index = True
        return new_files
    except:
        return []

# Sidebar: setup and configuration
with st.sidebar:
    st.title("🔍 Adaptive RAG")
    st.markdown("### Configuration")
    
    # Load environment variables
    load_dotenv()
    
    # LLM model selection
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        lines = result.stdout.strip().splitlines()
        if lines and "MODEL" in lines[0].upper() or "NAME" in lines[0].upper():
            lines = lines[1:]  # Skip header row
            
        # Parse model names exactly as they appear in Ollama list output
        available_models = []
        for line in lines:
            if line.strip():
                # Extract the full model name with tag (e.g., "llama3:latest")
                model_name = line.split()[0]
                available_models.append({"display": model_name, "value": model_name})
        
        # If no models were found, use defaults
        if not available_models:
            available_models = [
                {"display": "mistral:latest", "value": "mistral:latest"},
                {"display": "mixtral:latest", "value": "mixtral:latest"},
                {"display": "llama3:latest", "value": "llama3:latest"},
                {"display": "command-r:latest", "value": "command-r:latest"}
            ]
    except Exception as e:
        st.warning(f"Error getting Ollama models: {str(e)}")
        available_models = [
            {"display": "mistral:latest", "value": "mistral:latest"},
            {"display": "mixtral:latest", "value": "mixtral:latest"},
            {"display": "llama3:latest", "value": "llama3:latest"},
            {"display": "command-r:latest", "value": "command-r:latest"}
        ]
    
    # Create options for the selectbox with display names
    model_options = [m["display"] for m in available_models]
    selected_display = st.selectbox("Select Local LLM Model", model_options, index=0)
    
    # Get the actual model value for the selected display name
    local_llm = next((m["value"] for m in available_models if m["display"] == selected_display), selected_display)
    
    # Check and display API keys status
    st.markdown("### API Keys Status")
    required_keys = ["TAVILY_API_KEY", "NOMIC_API_KEY", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"]
    for key in required_keys:
        if os.environ.get(key):
            st.success(f"✅ {key}: Available")
        else:
            st.error(f"❌ {key}: Missing")
    
    # Index management
    st.markdown("### Document Index")
    doc_dir = st.text_input("Documents Directory", value="docs")
    db_dir = st.text_input("Database Directory", value="./db")
    
    # Display index statistics
    if 'total_docs' in st.session_state and st.session_state.total_docs > 0:
        st.info(f"📚 Current index contains {st.session_state.total_docs} document chunks")
        
        # Show last update timestamp if available
        if st.session_state.last_index_update:
            st.caption(f"Last updated: {st.session_state.last_index_update}")
    
    # Display pending docs notification
    if 'pending_docs' in st.session_state and st.session_state.pending_docs:
        st.warning(f"⚠️ {len(st.session_state.pending_docs)} document(s) not indexed yet. Click 'Rebuild Index' to include them.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Check for New Docs"):
            new_files = check_for_new_documents(doc_dir)
            if new_files:
                st.session_state.pending_docs = new_files
                st.warning(f"Found {len(new_files)} new document(s): {', '.join(new_files)}")
            else:
                st.success("No new documents found")
                st.session_state.pending_docs = []
    
    with col2:
        if st.button("Rebuild Index"):
            with st.spinner("Rebuilding index..."):
                st.session_state.rebuild_index = True
                st.session_state.pending_docs = []
    
    # Display LangSmith status
    st.markdown("### LangSmith Integration")
    langsmith_enabled = bool(os.environ.get("LANGCHAIN_API_KEY"))
    st.write(f"LangSmith Tracing: {'Enabled' if langsmith_enabled else 'Disabled'}")
    if langsmith_enabled:
        st.write(f"Project: {os.environ.get('LANGCHAIN_PROJECT', 'default')}")
    
    # Display graph visualization in sidebar
    st.markdown("### System Architecture")
    if 'graph_image' in st.session_state and st.session_state.graph_image:
        # Display a smaller version of the graph in the sidebar
        encoded_image = base64.b64encode(st.session_state.graph_image).decode('utf-8')
        st.markdown(f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{encoded_image}" alt="RAG Graph" style="max-width: 100%;">
        </div>
        """, unsafe_allow_html=True)
        st.caption("Click 'How It Works' below for a larger view")
    else:
        st.info("Graph visualization will appear after initialization")
        if 'app' in st.session_state and st.session_state.app:
            st.caption("Basic flow: Question → Route → Retrieve/Search → Grade → Generate → Validate")
            st.caption("See 'How It Works' for details")

    with st.expander("How It Works"):
        if 'graph_image' in st.session_state and st.session_state.graph_image:
            st.image(st.session_state.graph_image, caption="Adaptive RAG Workflow Graph", use_container_width=True)
            st.download_button(
                label="Download Graph Image",
                data=st.session_state.graph_image,
                file_name="adaptive_rag_graph.png",
                mime="image/png"
            )
        else:
            st.info("Graph visualization will appear after initialization")

# App title and description
st.title("🧠 Adaptive RAG with Local LLMs")
st.markdown("""
This application demonstrates an Adaptive RAG (Retrieval-Augmented Generation) system using local LLMs.
It uses query analysis to route between web search for recent events and RAG for document-based queries.
""")

# Initialize session state variables
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'app' not in st.session_state:
    st.session_state.app = None
if 'rebuild_index' not in st.session_state:
    st.session_state.rebuild_index = False
if 'question_history' not in st.session_state:
    st.session_state.question_history = []
if 'answer_history' not in st.session_state:
    st.session_state.answer_history = []
if 'execution_trace' not in st.session_state:
    st.session_state.execution_trace = []
if 'graph_image' not in st.session_state:
    st.session_state.graph_image = None
if 'new_docs_added' not in st.session_state:
    st.session_state.new_docs_added = []
if 'total_docs' not in st.session_state:
    st.session_state.total_docs = 0
if 'pending_docs' not in st.session_state:
    st.session_state.pending_docs = []
if 'last_index_update' not in st.session_state:
    st.session_state.last_index_update = None

# Create or load index
@st.cache_resource
def initialize_index(doc_dir, db_dir, rebuild=False):
    # 1) Find all files
    all_files = glob.glob(os.path.join(doc_dir, "**", "*"), recursive=True)
    
    # 2) Init embeddings
    emb = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
    
    # 3) Create or load vector store
    if rebuild:
        # Clear existing DB if rebuild requested
        if os.path.exists(db_dir):
            import shutil
            shutil.rmtree(db_dir)
        
        vs = Chroma.from_documents([], emb, persist_directory=db_dir, collection_name="rag-chroma")
        existing = set()
    else:
        try:
            vs = Chroma(persist_directory=db_dir, embedding_function=emb, collection_name="rag-chroma")
            existing = {md.get("source") for md in vs.get()["metadatas"] or []}
        except:
            os.makedirs(db_dir, exist_ok=True)
            vs = Chroma(persist_directory=db_dir, embedding_function=emb, collection_name="rag-chroma")
            existing = set()
    
    # 4) Loader factory
    def load_path(fp):
        ext = Path(fp).suffix.lower()
        if ext == ".pdf":
            return PyPDFLoader(fp).load()
        else:
            # try UTF-8 and fall back to latin-1
            try:
                return TextLoader(fp, encoding="utf-8").load()
            except:
                return TextLoader(fp, encoding="latin-1").load()
    
    # 5) Load only new docs
    new_docs = []
    new_file_paths = []
    for fp in all_files:
        if os.path.isfile(fp) and fp not in existing:
            new_file_paths.append(os.path.basename(fp))
            new_docs += load_path(fp)
    
    if new_docs:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
        chunks = splitter.split_documents(new_docs)
        if chunks:
            vs.add_documents(chunks)
            vs.persist()
            existing_count = len(existing) if existing else 0
            doc_count = existing_count + len(chunks)
            st.session_state.new_docs_added = new_file_paths
            st.session_state.total_docs = doc_count
            st.session_state.last_index_update = time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            doc_count = len(existing) if existing else 0
            st.session_state.new_docs_added = []
            st.session_state.total_docs = doc_count
    else:
        doc_count = sum(1 for _ in existing)
        st.session_state.new_docs_added = []
        st.session_state.total_docs = doc_count
        
        # Update timestamp only if it's a rebuild
        if rebuild:
            st.session_state.last_index_update = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 9) Build retriever
    retriever = vs.as_retriever()
    
    return retriever, doc_count

# Initialize the components for the LangGraph
@st.cache_resource(show_spinner=False)
def initialize_langgraph(local_llm):
    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)
    
    # Router
    router_prompt = PromptTemplate(
        template="""You are an expert at routing a user question to a vectorstore or web search. \n
        Use the vectorstore for questions on laser comm, budget, orbits etc.. \n
        You do not need to be stringent with the keywords in the question related to these topics. \n
        Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
        Return the a JSON with a single key 'datasource' and no premable or explanation. \n
        Question to route: {question}""",
        input_variables=["question"],
    )
    question_router = router_prompt | llm | JsonOutputParser()
    
    # Retrieval Grader
    grader_prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["question", "document"],
    )
    retrieval_grader = grader_prompt | llm | JsonOutputParser()
    
    # Generate
    rag_prompt = hub.pull("rlm/rag-prompt")
    gen_llm = ChatOllama(model=local_llm, temperature=0)
    rag_chain = rag_prompt | gen_llm | StrOutputParser()
    
    # Hallucination Grader
    hallucination_prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "documents"],
    )
    hallucination_grader = hallucination_prompt | llm | JsonOutputParser()
    
    # Answer Grader
    answer_prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "question"],
    )
    answer_grader = answer_prompt | llm | JsonOutputParser()
    
    # Question Re-writer
    rewrite_prompt = PromptTemplate(
        template="""You are a question re-writer that converts an input question to a better version that is optimized \n 
         for vectorstore retrieval. Look at the initial and formulate an improved question. \n
         Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
        input_variables=["question"],
    )
    question_rewriter = rewrite_prompt | llm | StrOutputParser()
    
    # Web Search Tool
    web_search_tool = TavilySearchResults(k=3)
    
    # Define the GraphState
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents
        """
        question: str
        generation: str
        documents: List[str]
    
    # Define the nodes
    def retrieve(state):
        """Retrieve documents from vector store"""
        st.session_state.execution_trace.append("retrieve")
        question = state["question"]
        documents = st.session_state.retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}

    def generate(state):
        """Generate answer based on documents and question"""
        st.session_state.execution_trace.append("generate")
        question = state["question"]
        documents = state["documents"]
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(state):
        """Grade document relevance to question"""
        st.session_state.execution_trace.append("grade_documents")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score["score"]
            if grade == "yes":
                filtered_docs.append(d)
        return {"documents": filtered_docs, "question": question}

    def transform_query(state):
        """Transform query for better retrieval"""
        st.session_state.execution_trace.append("transform_query")
        question = state["question"]
        documents = state["documents"]
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def web_search(state):
        """Perform web search for question"""
        st.session_state.execution_trace.append("web_search")
        question = state["question"]
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        # Downstream components expect a list of Documents
        return {"documents": [web_results], "question": question}

    # Define conditional edges
    def route_question(state):
        """Route question to web search or vectorstore"""
        st.session_state.execution_trace.append("route_question")
        question = state["question"]
        source = question_router.invoke({"question": question})
        if source["datasource"] == "web_search":
            return "web_search"
        elif source["datasource"] == "vectorstore":
            return "vectorstore"

    def decide_to_generate(state):
        """Decide whether to generate answer or transform query"""
        st.session_state.execution_trace.append("decide_to_generate")
        filtered_documents = state["documents"]
        if not filtered_documents:
            return "transform_query"
        else:
            return "generate"

    def grade_generation_v_documents_and_question(state):
        """Grade generation against documents and question"""
        st.session_state.execution_trace.append("grade_generation")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score["score"]
        
        if grade == "yes":
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score["score"]
            if grade == "yes":
                return "useful"
            else:
                return "not useful"
        else:
            return "not supported"
    
    # Build the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    
    # Build graph edges
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )
    
    # Compile the graph
    app = workflow.compile(
        name="Adaptive RAG with Local LLM",
        checkpointer=None,
    )
    
    # Generate the graph visualization
    try:
        graph_img_bytes = app.get_graph().draw_mermaid_png()
    except Exception as e:
        print(f"Warning: Could not generate graph visualization: {str(e)}")
        graph_img_bytes = None
    
    return app, question_router, retrieval_grader, rag_chain, hallucination_grader, answer_grader, question_rewriter, graph_img_bytes

# Initialize the document index and langgraph
try:
    # Check if we need to rebuild the index
    if st.session_state.rebuild_index:
        retriever, doc_count = initialize_index(doc_dir, db_dir)
        st.session_state.retriever = retriever
        st.session_state.doc_count = doc_count
        st.session_state.rebuild_index = False
        
        # Display new document indicator after rebuild
        if st.session_state.new_docs_added:
            st.success(f"✨ Added {len(st.session_state.new_docs_added)} new document(s) to index: {', '.join(st.session_state.new_docs_added)}")
        st.success(f"Successfully rebuilt index with {doc_count} document chunks")
    elif not st.session_state.retriever:
        retriever, doc_count = initialize_index(doc_dir, db_dir)
        st.session_state.retriever = retriever
        st.session_state.doc_count = doc_count
        
        # Display new document indicator
        if st.session_state.new_docs_added:
            st.success(f"✨ Added {len(st.session_state.new_docs_added)} new document(s) to index: {', '.join(st.session_state.new_docs_added)}")
        st.info(f"Loaded index with {doc_count} document chunks")
    
    # Initialize the LangGraph components
    if st.session_state.retriever and not st.session_state.app:
        with st.spinner("Initializing Adaptive RAG components..."):
            app, *_, graph_img_bytes = initialize_langgraph(local_llm)
            st.session_state.app = app
            st.session_state.graph_image = graph_img_bytes
            st.success("Successfully initialized Adaptive RAG components")
except Exception as e:
    st.error(f"Error initializing components: {str(e)}")

# Main interface - Question input
question = st.text_input("Ask a question:", placeholder="e.g., What is the orbit of Eagle-1?")

# Custom Streamlit callback handler for tracing
class StreamlitTracer(ConsoleCallbackHandler):
    def __init__(self):
        super().__init__()
        self.messages = []
    
    def on_text(self, text, **kwargs):
        self.messages.append(text)
        return super().on_text(text, **kwargs)

# Query processing
col1, col2 = st.columns([1, 1])
trace_url = None

with col1:
    # Submit button
    if st.button("Submit", type="primary", disabled=not (st.session_state.app and question)):
        # Clear previous execution trace
        st.session_state.execution_trace = []
        
        # Set up callbacks
        callbacks = []
        tracer = StreamlitTracer()
        callbacks.append(tracer)
        
        # Initialize LangSmith tracer if available
        if os.environ.get("LANGCHAIN_API_KEY"):
            langchain_tracer = LangChainTracer(
                project_name=os.environ.get("LANGCHAIN_PROJECT", "default")
            )
            callbacks.append(langchain_tracer)
        
        # Invoke the graph
        with st.spinner("Processing your question..."):
            # Create metadata for tracing
            metadata = {
                "user_id": "streamlit_user",
                "session_id": str(uuid.uuid4()),
                "query_type": "user_query",
            }
            
            # Run with tracing
            try:
                result = st.session_state.app.invoke(
                    {"question": question},
                    {
                        "callbacks": callbacks,
                        "metadata": metadata,
                        "run_name": f"Streamlit Query: {question[:30]}...",
                    }
                )
                
                # Store in history
                st.session_state.question_history.append(question)
                st.session_state.answer_history.append(result["generation"])
                
                # Get LangSmith trace URL if available
                trace_url = None
                if os.environ.get("LANGCHAIN_API_KEY"):
                    run_id = result.get("run_id", "")
                    if run_id:
                        trace_url = f"https://smith.langchain.com/o/{os.environ.get('LANGCHAIN_ORG_ID', 'default')}/projects/p/{os.environ.get('LANGCHAIN_PROJECT', 'default')}/r/{run_id}"
                
                st.success("Question processed successfully!")
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.session_state.answer_history.append(f"Error: {str(e)}")

# Display answer and execution history
if st.session_state.question_history:
    with col1:
        st.markdown("### Answer")
        st.write(st.session_state.answer_history[-1])
        
        # Display trace URL if available
        if trace_url:
            st.markdown(f"[View detailed trace in LangSmith]({trace_url})")
    
    with col2:
        st.markdown("### Execution Path")
        if st.session_state.execution_trace:
            # First show total steps count for clarity
            st.info(f"Total steps: {len(st.session_state.execution_trace)}")
            
            # Create container with better scrolling and no height limit
            execution_container = st.container()
            with execution_container:
                # Display each step with custom styling
                for i, step in enumerate(st.session_state.execution_trace):
                    st.markdown(
                        f"""
                        <div class="graph-node" style="margin-bottom: 10px;">
                            <b>Step {i+1}:</b> {step}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    # Add arrow between steps except for the last one
                    if i < len(st.session_state.execution_trace) - 1:
                        st.markdown(
                            '<div style="text-align:center; margin:5px 0;">⬇️</div>', 
                            unsafe_allow_html=True
                        )
        else:
            st.info("No execution steps to display.")

# History tab
with st.expander("Question History"):
    if st.session_state.question_history:
        for i, (q, a) in enumerate(zip(st.session_state.question_history, st.session_state.answer_history)):
            st.markdown(f"**Question {i+1}**: {q}")
            st.markdown(f"**Answer {i+1}**: {a}")
            st.markdown("---")
    else:
        st.info("No questions asked yet.")

# Documentation tab
with st.expander("How It Works"):
    st.markdown("""
    ### Adaptive RAG System Architecture
    
    This Streamlit app implements an Adaptive RAG system that:
    
    1. **Analyzes user queries** to determine the best data source
    2. **Routes** to either web search (for recent events) or vectorstore (for documents)
    3. **Evaluates document relevance** to filter out irrelevant results
    4. **Transforms queries** when necessary to improve retrieval
    5. **Checks for hallucinations** in generated answers
    6. **Validates answer quality** against the original question
    
    The system uses LangGraph to orchestrate the workflow between these components.
    """)
    
    # Display the graph visualization
    st.markdown("### Graph Visualization")
    st.info("See the graph visualization in the sidebar.")
    
    st.markdown("""
    ### Requirements
    
    - Local LLM via Ollama (mistral, mixtral, etc.)
    - Nomic embeddings for vector search
    - Tavily API for web search
    - LangSmith for tracing and debugging (optional)
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain, and LangGraph using local LLMs")
