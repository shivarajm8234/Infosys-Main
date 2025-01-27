Contract Analysis Assistant

An AI-powered contract analysis tool that helps users analyze, understand, and evaluate legal documents using advanced natural language processing and machine learning techniques.

Features

- Document Upload*: Support for PDF and CSV file formats
- Smart Contract Analysis*: Comprehensive evaluation of contract terms and conditions
- Legal Compliance Check*: Automated assessment against industry standards and regulations
- Interactive Chat*: AI-powered Q&A system with confidence scoring
- Similarity Search*: Find similar documents and clauses
- Visual Analytics*: Clear presentation of analysis results with color-coded ratings

Prerequisites

- Python 3.8+
- Google Cloud API key (for Gemini Pro)

Installation

1. Clone the repository
2. Install the required dependencies:

pip install -r requirements.txt

3. Create a .env file in the project root and add your Google API key:

GOOGLE_API_KEY=your_api_key_here


Dependencies

- streamlit==1.31.0
- pandas==2.1.4
- google-generativeai==0.3.2
- python-dotenv==1.0.0
- PyPDF2==3.0.1
- scikit-learn==1.3.2
- langchain==0.1.0
- sentence-transformers==2.2.2
- fastapi==0.109.0
- chromadb==0.4.22
- uvicorn==0.27.0

Architecture

The application uses a modern tech stack:
- FastAPI*: Powers the backend API service
- ChromaDB*: Vector database for efficient document storage and retrieval
- KNN (K-Nearest Neighbors)*: Used for similarity search to find related documents
- Streamlit*: Provides the interactive frontend interface
- Gemini Pro*: Handles natural language understanding and generation

Usage

1. Start the FastAPI backend:
   
uvicorn api:app --reload

2. Start the Streamlit frontend:
 
streamlit run app.py

3. Navigate to the web interface (typically http://localhost:8501)

4. Use the following features:
   - Upload contracts in PDF format
   - Analyze legal compliance
   - Chat with the AI assistant about contract details
   - Search for similar documents
   - View detailed analysis reports

Features in Detail

Contract Analysis
- Extracts and processes text from PDF documents
- Generates comprehensive contract evaluations
- Provides section-by-section analysis

Legal Compliance
- Evaluates contracts against industry standards
- Checks compliance with various regulations
- Provides risk assessments and recommendations

Interactive Assistant
- Natural language Q&A about contract contents
- Confidence scoring for responses
- Maintains chat history for context

Similarity Search
- KNN-based document similarity using sentence embeddings
- Efficient vector storage and retrieval with ChromaDB
- Configurable similarity thresholds and search parameters
- Fast nearest neighbor search for quick document comparison
- Semantic understanding of document contents

Security Note

This application handles sensitive legal documents. Ensure you:
- Keep your API keys secure
- Review the privacy implications of using external AI services
- Handle document storage according to your security requirements
