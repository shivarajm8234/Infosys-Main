import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os
import PyPDF2
from typing import List, Dict
import tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Initialize SentenceTransformer
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize or load stored data
data_file = "vector_store.pkl"

if os.path.exists(data_file):
    with open(data_file, 'rb') as f:
        stored_data = pickle.load(f)
        embeddings = stored_data['embeddings']
        metadata_list = stored_data['metadata']
else:
    embeddings = []
    metadata_list = []

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def save_state():
    """Save embeddings and metadata to disk"""
    with open(data_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'metadata': metadata_list
        }, f)

def process_csv_data():
    """Process and store CSV data"""
    df = pd.read_csv("cleaned_dataset.csv")
    
    for _, row in df.iterrows():
        # Create combined text for embedding
        document_text = f"""
        Category: {row['Category']}
        Parties: {row['Parties']}
        Agreement Date: {row['Agreement Date']}
        Effective Date: {row['Effective Date']}
        Expiration Date: {row['Expiration Date']}
        Renewal Term: {row['Renewal Term']}
        """
        
        # Create metadata
        metadata = {
            "source": "csv",
            "text": document_text,
            "category": row['Category'],
            "governing_law": row['Governing Law'],
            "law_explanation": row['Law Explanation']
        }
        
        # Get embedding
        embedding = encoder.encode(document_text)
        
        # Store embedding and metadata
        embeddings.append(embedding)
        metadata_list.append(metadata)
    
    save_state()

def process_pdf_file(pdf_file, category: str, governing_law: str):
    """Process and store PDF data"""
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_file)
    
    # Create metadata
    metadata = {
        "source": "pdf",
        "text": text,
        "category": category,
        "governing_law": governing_law,
        "law_explanation": "User uploaded PDF document"
    }
    
    # Get embedding
    embedding = encoder.encode(text)
    
    # Store embedding and metadata
    embeddings.append(embedding)
    metadata_list.append(metadata)
    
    save_state()
    return len(metadata_list) - 1

def search_similar(query: str, k: int = 3):
    """Search for similar documents"""
    if not embeddings:
        return []
    
    # Get query embedding
    query_vector = encoder.encode(query)
    
    # Calculate similarities
    similarities = cosine_similarity([query_vector], embeddings)[0]
    
    # Get top k indices
    top_indices = similarities.argsort()[-k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "metadata": metadata_list[idx],
            "similarity": similarities[idx]
        })
    
    return results

# Streamlit UI
st.title("Contract Analysis Assistant")
st.sidebar.title("Options")

# Add tabs
tab1, tab2, tab3 = st.tabs(["Upload PDF", "Similarity Search", "Chat with AI"])

with tab1:
    st.header("Upload Contract PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    category = st.selectbox("Select Contract Category", [
        "Co_Branding", "Endorsement", "Development", "Transportation",
        "Maintenance", "Hosting", "IP", "Other"
    ])
    governing_law = st.selectbox("Select Governing Law", [
        "International Law", "Indian Law", "US Law", "Other"
    ])
    
    if uploaded_file and st.button("Upload and Process"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            doc_id = process_pdf_file(tmp_file.name, category, governing_law)
            st.success(f"PDF processed and stored with ID: {doc_id}")

with tab2:
    st.header("Contract Similarity Search")
    query = st.text_area("Enter your contract text or query:", height=200)
    
    if st.button("Search Similar Contracts"):
        if query:
            results = search_similar(query)
            
            # Display results
            for i, result in enumerate(results):
                metadata = result["metadata"]
                with st.expander(f"Result {i + 1} - {metadata['category']} Contract"):
                    st.write(f"**Similarity Score:** {result['similarity']:.4f}")
                    st.write(f"**Category:** {metadata['category']}")
                    st.write(f"**Governing Law:** {metadata['governing_law']}")
                    st.write(f"**Law Explanation:** {metadata['law_explanation']}")
                    st.write("**Contract Text Preview:**")
                    preview = metadata['text'][:500]
                    st.write(preview + "..." if len(metadata['text']) > 500 else preview)
        else:
            st.warning("Please enter a query text.")

with tab3:
    st.header("Chat with AI about Contracts")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about contracts..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate Gemini response
        try:
            # Get similar documents for context
            results = search_similar(prompt, k=2)
            
            # Create context from similar documents
            context = "\n\n".join([
                f"""Similar contract {i+1}:
                Category: {result['metadata']['category']}
                Governing Law: {result['metadata']['governing_law']}
                Content: {result['metadata']['text'][:500]}...
                """
                for i, result in enumerate(results)
            ])
            
            # Create prompt with context
            enhanced_prompt = f"""Context from similar contracts:
            {context}
            
            User question: {prompt}
            
            Please provide a detailed answer based on the context above and your knowledge about contracts. Focus on the legal aspects and practical implications."""

            response = model.generate_content(enhanced_prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Add sidebar information
with st.sidebar:
    st.info("""
    **How to use:**
    1. Upload PDF contracts with metadata
    2. Use Similarity Search to find similar contracts
    3. Chat with AI about contracts
    
    The system will search through both uploaded PDFs and existing CSV data.
    """)
    
    if st.button("Process CSV Data"):
        process_csv_data()
        st.success("CSV data processed and stored in vector database")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
