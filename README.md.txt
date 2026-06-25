# RAG System - Production Ready

A modular, enterprise-grade RAG (Retrieval-Augmented Generation) system for document-based question answering.

## Features

✅ **PDF Processing** - Extract text and tables from PDFs  
✅ **Intelligent Chunking** - Semantic + token-based hybrid chunking  
✅ **Vector Search** - FAISS-powered efficient similarity search  
✅ **Hybrid Retrieval** - Combine vector + BM25 lexical search  
✅ **Smart Reranking** - Cross-encoder reranking for better results  
✅ **LLM Integration** - Generate grounded answers from context  
✅ **Evaluation** - Comprehensive metrics (Recall, Precision, MRR)  

## Quick Start

### 1. Clone and Install

```bash
# Create project directory
mkdir rag_project && cd rag_project

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Environment

```bash
# Copy and configure environment
cp .env.example .env

# Add your OpenAI API key (optional)
export OPENAI_API_KEY="your-key-here"
```

### 3. Add PDF

```bash
# Place your PDF in the data/raw directory
cp your_document.pdf data/raw/knowledge_base.pdf
```

### 4. Run System

```bash
# Interactive mode
python main.py --interactive

# Single query
python main.py --query "What is the price?"

# Demo mode (no arguments)
python main.py

# Using mock LLM (no API key needed)
python main.py --mock --interactive
```

## Project Structure