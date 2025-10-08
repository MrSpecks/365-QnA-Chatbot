#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
Run this before deploying to Streamlit to catch import issues early
"""

def test_imports():
    """Test all required imports"""
    try:
        print("Testing imports...")
        
        print("✓ Importing streamlit...")
        import streamlit as st
        
        print("✓ Importing tempfile...")
        import tempfile
        
        print("✓ Importing os...")
        import os
        
        print("✓ Importing io...")
        from io import BytesIO
        
        print("✓ Importing PyPDFLoader...")
        from langchain_community.document_loaders.pdf import PyPDFLoader
        
        print("✓ Importing TokenTextSplitter...")
        from langchain_text_splitters import TokenTextSplitter
        
        print("✓ Importing StrOutputParser...")
        from langchain_core.output_parsers.string import StrOutputParser
        
        print("✓ Importing LangChain core components...")
        from langchain_core.messages import SystemMessage
        from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        
        print("✓ Importing OpenAI components...")
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        print("✓ Importing Chroma...")
        from langchain_chroma.vectorstores import Chroma
        
        print("\n🎉 All imports successful! Your environment is ready for deployment.")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Please check your package installations.")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
