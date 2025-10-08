# 365 Q&A Chatbot

A Streamlit application that enables users to upload a PDF document and chat with an AI assistant that answers questions using only the content from that PDF. This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain and ChromaDB.

## Features

- **PDF Upload**: Upload any PDF document to build a knowledge base
- **Multi-Provider Support**: Choose between OpenAI and OpenRouter for LLM access
- **Multiple Models**: Access to various models including:
  - **OpenAI**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
  - **OpenRouter**: Google Gemini 2.5 Flash, Llama 3.2, Mixtral 8x7B, and more
- **RAG Pipeline**: Uses LangChain with OpenAI embeddings and ChromaDB vector store
- **Interactive Chat**: Ask questions about your PDF content with a conversational interface
- **Context-Aware**: AI responses are based solely on the uploaded PDF content
- **Streaming Responses**: Real-time response streaming for better user experience
- **Dynamic Configuration**: Automatically configures API endpoints and models based on provider selection

## Tech Stack

- **Frontend**: Streamlit
- **LLM Providers**: OpenAI, OpenRouter (supporting Google Gemini, Llama, Mixtral, etc.)
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: ChromaDB
- **Document Processing**: LangChain, PyPDF

## Prerequisites

⚠️ **Important**: You need a PDF file and at least one API key (OpenAI or OpenRouter) to use this application.

### Required Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (for direct OpenAI models)
- `OPENROUTER_API_KEY`: Your OpenRouter API key (for multi-model access including Google Gemini, Llama, Mixtral, etc.)

**Note**: You only need one API key depending on which provider you choose to use.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd 365-QnA-Chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Secrets

Create a `.streamlit/secrets.toml` file in your project root:

```toml
# OpenAI API Key (for direct OpenAI models)
OPENAI_API_KEY = "sk-your-openai-api-key-here"

# OpenRouter API Key (for multi-model access)
OPENROUTER_API_KEY = "sk-your-openrouter-api-key-here"
```

**Note**: Copy the `.streamlit/secrets.toml.example` file and replace the placeholders with your actual API keys. You only need one API key depending on which provider you choose to use.

### 4. Test Your Installation (Optional but Recommended)

Before running the app, test that all imports work correctly:

```bash
python test_imports.py
```

This will verify that all required packages are properly installed.

### 5. Run the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

1. **Select Provider**: Choose between OpenAI or OpenRouter in the sidebar
2. **Enter API Key**: Provide your API key for the selected provider (or configure in secrets.toml)
3. **Select Model**: Choose from available models for your selected provider
4. **Upload PDF**: Use the file uploader to select a PDF document
5. **Process PDF**: Click "Process PDF" to build the knowledge base
6. **Start Chatting**: Once processing is complete, ask questions about your PDF content
7. **View History**: Your conversation history is maintained throughout the session

## How It Works

1. **Document Loading**: PDF is loaded using PyPDFLoader from LangChain
2. **Text Splitting**: Documents are split into chunks using TokenTextSplitter (1000 tokens per chunk, 100 token overlap)
3. **Embedding Creation**: Text chunks are converted to embeddings using OpenAI's text-embedding-3-small model
4. **Vector Store**: Embeddings are stored in ChromaDB for efficient similarity search
5. **Provider Configuration**: Based on your selection, the app configures the appropriate API endpoint and model
6. **Retrieval**: When you ask a question, relevant chunks are retrieved from the vector store
7. **Generation**: Retrieved context is passed to your selected model (GPT, Gemini, Llama, Mixtral, etc.) to generate responses

## Deployment

### Streamlit Community Cloud

1. Fork this repository
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/)
3. Connect your GitHub account and select your forked repository
4. Add your API keys (`OPENAI_API_KEY` and/or `OPENROUTER_API_KEY`) in the secrets section
5. Deploy!

### Local Deployment

For production deployment, consider using:
- Docker containers
- Cloud platforms (AWS, GCP, Azure)
- VPS with proper security configurations

## File Structure

```
365-QnA-Chatbot/
├── streamlit_app.py               # Main Streamlit application
├── test_imports.py                # Import testing script
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
├── .streamlit/
│   └── secrets.toml.example       # Secrets template
└── temp_vectorstore/              # Generated vector store (auto-created)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: If you get `ModuleNotFoundError` during deployment:
   - Run `python test_imports.py` locally to test imports
   - Check that your `requirements.txt` has the correct package versions
   - Ensure all LangChain packages are properly installed

2. **API Key Error**: Ensure your API key is correctly set in `.streamlit/secrets.toml` or entered in the sidebar
3. **Provider Selection**: Make sure you've selected the correct provider (OpenAI or OpenRouter) that matches your API key
4. **Model Availability**: Some models may not be available on OpenRouter; try a different model if you encounter errors
5. **PDF Processing Error**: Make sure the uploaded file is a valid PDF
6. **Memory Issues**: Large PDFs may require more memory; consider reducing chunk size
7. **Rate Limiting**: API providers have rate limits; consider upgrading your plan for heavy usage

### Performance Tips

- Use smaller chunk sizes for faster processing
- Consider using GPU-accelerated embeddings for large documents
- Implement caching for frequently accessed documents

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information about your problem

---

**Happy Chatting! 🤖📚**
