import streamlit as st
import tempfile
import os
from io import BytesIO
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# We use ChatOpenAI/OpenAIEmbeddings for compatibility, configuring the base_url for different providers.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma

# --- Configuration ---
PROVIDER_OPTIONS = ["OpenAI", "OpenRouter (for Multi-Model Access including Google)"]

# Model mapping including popular models for each service
MODEL_MAPPING = {
    "OpenAI": [
        "gpt-4o-mini", 
        "gpt-4o", 
        "gpt-3.5-turbo"
    ],
    "OpenRouter (for Multi-Model Access including Google)": [
        "google/gemini-2.5-flash", # Directly using a Google model via OpenRouter
        "meta-llama/llama-3.2-3b-instruct:free",
        "mistralai/mixtral-8x7b-instruct-v0.1",
        "openai/gpt-4o-mini"
    ]
}

def get_config_for_provider(provider):
    """Returns the base_url, key name, and model list for the given provider."""
    if "OpenRouter" in provider:
        return {
            "base_url": "https://openrouter.ai/api/v1",
            "key_name": "OPENROUTER_API_KEY",
            "placeholder": "Enter your OpenRouter API Key (sk-...)",
            "models": MODEL_MAPPING[provider]
        }
    else: # Default to OpenAI
        return {
            "base_url": None, # Use default OpenAI endpoint
            "key_name": "OPENAI_API_KEY",
            "placeholder": "Enter your OpenAI API Key (sk-...)",
            "models": MODEL_MAPPING["OpenAI"]
        }

# Set page config
st.set_page_config(
    page_title="365 Q&A Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
# Initialize dynamic state variables
if "selected_provider" not in st.session_state:
    st.session_state.selected_provider = PROVIDER_OPTIONS[0]
if "api_key_input_value" not in st.session_state:
     st.session_state.api_key_input_value = ""

# --- Utility Functions for Config ---

def get_api_config():
    """
    Retrieves the API key and base URL based on the selected provider.
    Prioritizes user input from the sidebar, then falls back to Streamlit secrets.
    """
    provider = st.session_state.selected_provider
    config = get_config_for_provider(provider)
    
    # 1. Check sidebar input value
    user_key = st.session_state.api_key_input_value
    if user_key:
        api_key = user_key
    else:
        # 2. Check Streamlit secrets using the provider-specific key name
        try:
            api_key = st.secrets[config["key_name"]]
        except KeyError:
            api_key = None

    return api_key, config["base_url"]

# --- Cached RAG Pipeline Functions ---

@st.cache_resource(show_spinner="Processing PDF and building knowledge base...")
def process_pdf_and_create_vectorstore(_uploaded_file, api_key, base_url):
    """
    Process uploaded PDF and create the Chroma vector store.
    Caches the vector store based on file ID, API key, and base URL.
    """
    if not api_key:
        st.error("❌ API key is missing. Please enter it in the sidebar or secrets.")
        return None

    # Use BytesIO to handle the uploaded file content
    bytes_data = _uploaded_file.getvalue()
    
    # Create temporary file to pass to PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(bytes_data)
        tmp_file_path = tmp_file.name

    try:
        # 1. Load the document
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        # 2. Split the document 
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # 3. Create Embeddings and Vector Store
        # NOTE: We use OpenAIEmbeddings but set the base_url dynamically.
        # OpenRouter supports this configuration for embeddings as well.
        embeddings = OpenAIEmbeddings(api_key=api_key, base_url=base_url)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        return vectorstore
        
    except Exception as e:
        st.error(f"❌ Error during PDF processing or vector store creation: {e}")
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@st.cache_resource(show_spinner=False)
def create_rag_chain(_vectorstore, model_name, api_key, base_url):
    """
    Creates the complete RAG chain runnable. Caches based on vectorstore (implicit),
    model name, API key, and base URL.
    """
    if _vectorstore is None:
        return None
    
    # 1. Define Retriever
    retriever = _vectorstore.as_retriever()

    # 2. Define LLM
    # Use ChatOpenAI and dynamically set the base_url for provider routing
    chat = ChatOpenAI(
        model=model_name, 
        temperature=0, 
        api_key=api_key, 
        base_url=base_url
    )

    # 3. Define Prompt Template
    system_message = (
        "You are an expert Q&A assistant for a document. Your primary source of information "
        "is the provided context. If the answer is not contained within the context, "
        "state that you cannot find the answer in the document, but do not guess."
        "\n\nContext: {context}"
    )
    chat_prompt_template_retrieving = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_message),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    # 4. Define Output Parser
    str_output_parser = StrOutputParser()

    # 5. Define RAG Chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_prompt_template_retrieving
        | chat
        | str_output_parser
    )
    return rag_chain


# --- Main Application Logic ---

def main():
    st.title("📄 365 Q&A Chatbot")
    st.caption("A Streamlit application for Retrieval-Augmented Generation (RAG) over your PDF files using LangChain and multiple LLM providers.")

    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.header("Configuration")
        
        # 1. Provider Selection
        st.session_state.selected_provider = st.selectbox(
            "Select LLM Provider",
            options=PROVIDER_OPTIONS,
            key="provider_selector",
            on_change=lambda: st.session_state.update(api_key_input_value="") # Clear key input when provider changes
        )
        
        current_config = get_config_for_provider(st.session_state.selected_provider)
        
        # 2. API Key Input
        key_name = current_config["key_name"]
        
        # Check secrets first
        default_key = st.secrets.get(key_name, "")
        if default_key:
            placeholder_text = f"Key loaded from secrets.toml ({key_name})"
        else:
            placeholder_text = current_config["placeholder"]
            
        st.session_state.api_key_input_value = st.text_input(
            f"{st.session_state.selected_provider} API Key", 
            value=st.session_state.api_key_input_value if st.session_state.api_key_input_value else default_key, 
            type="password", 
            placeholder=placeholder_text,
            key="api_key_input_widget"
        )
        
        # 3. Model Selection (Dynamically filtered by provider)
        selected_model = st.selectbox(
            "Select LLM Model",
            options=current_config["models"],
            index=0 # Always select the first model in the list
        )
        
        st.markdown("---")
        st.subheader("2. Upload Document")
        
        # 4. File Uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            accept_multiple_files=False,
            key="pdf_uploader"
        )
        
        # Button to trigger processing
        process_button = st.button("Process PDF")

    # --- Main Content Area ---
    
    current_api_key, current_base_url = get_api_config()
    
    if not current_api_key:
        st.warning(f"Please provide your API Key for **{st.session_state.selected_provider}** in the sidebar.")
        #uploaded_file = None # Keep uploaded file, but prevent process_button from working without key

    if process_button and uploaded_file and current_api_key:
        # Clear previous session state and reset flags
        st.session_state.chat_history = []
        st.session_state.pdf_processed = False
        
        # Process the PDF and create the vector store
        vectorstore = process_pdf_and_create_vectorstore(uploaded_file, current_api_key, current_base_url)
        
        if vectorstore:
            # Create the RAG chain
            rag_chain = create_rag_chain(vectorstore, selected_model, current_api_key, current_base_url)
            
            if rag_chain:
                # Store the chain and update the flag
                st.session_state.rag_chain = rag_chain
                st.session_state.pdf_processed = True
                st.success(f"PDF processed and RAG chain initialized with **{selected_model}** via **{st.session_state.selected_provider}**!")
            else:
                st.session_state.pdf_processed = False
        else:
             st.session_state.pdf_processed = False
    
    # --- Chat Interface ---
    
    if st.session_state.pdf_processed:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        prompt = st.chat_input("Ask a question about your PDF...")
        
        if prompt:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking..."):
                        # Stream the response using the cached chain
                        response = st.session_state.rag_chain.stream(prompt)
                        
                        # Stream the response
                        response_text = ""
                        response_placeholder = st.empty()
                        for chunk in response:
                            response_text += chunk
                            response_placeholder.markdown(response_text) 
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response. Check API Key/Model selection: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    else:
        st.info("👆 Please configure your API key, select a provider and model, upload a PDF file in the sidebar, and click 'Process PDF' to begin chatting!")

if __name__ == "__main__":
    main()
