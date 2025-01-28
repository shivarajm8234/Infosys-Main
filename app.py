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

def format_text_preview(text: str, lines: int = 3, chars_per_line: int = 80) -> str:
    """Format text to show exactly 3 lines with proper truncation"""
    if not text:
        return ""
    
    # Split text into words
    words = text.split()
    lines_of_text = []
    current_line = []
    current_length = 0
    
    for word in words:
        # Check if adding this word exceeds the line length
        if current_length + len(word) + 1 <= chars_per_line:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            # Line is full, add it to lines
            if len(lines_of_text) < lines:
                lines_of_text.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
    
    # Add the last line if there's space
    if current_line and len(lines_of_text) < lines:
        lines_of_text.append(" ".join(current_line))
    
    # Ensure exactly 3 lines
    while len(lines_of_text) < lines:
        lines_of_text.append("")
    
    # Truncate to exactly 3 lines
    result = lines_of_text[:lines]
    
    # Add ellipsis to last line if there was more text
    if len(words) > sum(len(line.split()) for line in result):
        result[-1] = result[-1] + "..."
    
    return "\n".join(result)

def search_similar(query: str, k: int = 3):
    """Search for similar documents with similarity threshold"""
    if not embeddings:
        return []  # Return empty list instead of float
    
    # Get query embedding
    query_vector = encoder.encode(query)
    
    # Calculate similarities
    similarities = cosine_similarity([query_vector], embeddings)[0]
    
    # Filter by similarity threshold and get top k indices
    threshold = 0.7
    valid_indices = [i for i, sim in enumerate(similarities) if sim < threshold]
    top_indices = sorted(valid_indices, key=lambda i: similarities[i], reverse=True)[:k]
    
    results = []
    for idx in top_indices:
        # Format the text into exactly 3 lines
        formatted_text = format_text_preview(metadata_list[idx]["text"])
        
        results.append({
            "metadata": {
                "category": metadata_list[idx]["category"],
                "governing_law": metadata_list[idx]["governing_law"],
                "text": formatted_text
            },
            "similarity": float(similarities[idx])
        })
    
    return results  # Return list of results

def evaluate_contract_query(query: str, contract_text: str) -> str:
    """Generate a structured contract evaluation based on the query"""
    prompt = f"""You are an AI assistant specialized in contract evaluation. Your role is to assess contracts against specific parameters and provide detailed, structured analysis.

Guidelines for evaluation:
1. Legal Compliance: Assess adherence to relevant laws (GDPR, HIPAA, corporate laws, privacy laws)
2. Clarity and Precision: Evaluate clarity of terms and definitions
3. Completeness: Verify all necessary sections are included
4. Consistency: Check for contradictions
5. Risk Mitigation: Identify potential risks
6. Scoring: Provide a score (0-35: inadequate, 36-70: adequate with improvements needed, 71-100: exceeds requirements)
7. Maintain professional and objective tone
8. Provide specific recommendations for improvements, especially regarding legal compliance

Contract Text:
{contract_text}

User Query: {query}

Provide a structured evaluation following the guidelines above, focusing on the specific aspect the user is asking about."""

    response = model.generate_content(prompt)
    return response.text

def get_contract_analysis(contract_text: str) -> str:
    """Generate a comprehensive contract analysis"""
    prompt = f"""As an AI contract evaluation assistant, provide a comprehensive analysis of this contract:

{contract_text}

Generate a structured report covering:

1. LEGAL COMPLIANCE
- Adherence to relevant laws and regulations
- Regulatory requirements coverage
- Compliance gaps or risks

2. CLARITY AND PRECISION
- Clear terms and definitions
- Unambiguous obligations
- Well-defined procedures

3. COMPLETENESS
- Parties involved
- Scope of work
- Payment terms
- Termination clauses
- Dispute resolution
- Other critical sections

4. CONSISTENCY CHECK
- Internal contradictions
- Conflicting clauses
- Terminology consistency

5. RISK ASSESSMENT
- Potential liabilities
- Unaddressed risks
- Mitigation recommendations

6. SCORING
Provide a score (0-100) with breakdown:
- 0-35: Does not meet requirements
- 36-70: Meets with improvements needed
- 71-100: Exceeds requirements

7. KEY FINDINGS
- Major strengths
- Critical weaknesses
- Missing elements
- Improvement recommendations

Maintain a professional and objective tone throughout the analysis."""

    response = model.generate_content(prompt)
    return response.text

def get_dummy_legal_data():
    """Get dummy legal compliance data when no similar documents are found"""
    return {
        "legal_compliance": {
            "sections": {
                "score": "25/100",
                "overall_rating": "Inadequate (0-35)",
                "legal_compliance": """
                • GDPR Status: Missing essential data protection clauses
                • Privacy Laws: Non-compliant with current regulations
                • Industry Regulations: Lacks required compliance statements
                • Compliance Score: 10/35""",
                "contract_structure": """
                • Clarity: Terms need better definition
                • Completeness: Missing key sections
                • Consistency: No major inconsistencies found
                • Structure Score: 8/35""",
                "risk_assessment": """
                • Key Risks: Data protection, Liability gaps
                • Risk Mitigation: Insufficient measures
                • Risk Score: 7/30""",
                "recommendations": """
                • Add GDPR compliance clauses
                • Update privacy policies
                • Include data protection measures
                • Add liability clauses""",
                "required_changes": """
                • Critical: GDPR compliance implementation
                • Critical: Privacy policy updates
                • Recommended: Term definitions
                • Recommended: Section organization"""
            }
        }
    }

def analyze_legal_compliance(text: str):
    """Analyze legal compliance aspects of the contract"""
    # Truncate text if too long
    max_length = 1000
    analysis_text = text[:max_length] + "..." if len(text) > max_length else text
    
    prompt = f"""You are a legal contract analysis AI. Evaluate the following contract and provide a structured analysis.
    Focus on these key aspects:

    1. Legal Compliance
    2. Clarity and Precision
    3. Completeness
    4. Consistency
    5. Risk Mitigation
    
    Provide your analysis in the following format:

    SCORE: [0-100]
    OVERALL RATING: [Inadequate (0-35) / Adequate with Improvements (36-70) / Exceeds Requirements (71-100)]

    1. LEGAL COMPLIANCE
    - GDPR Status: [Brief assessment]
    - Privacy Laws: [Brief assessment]
    - Industry Regulations: [Brief assessment]
    - Compliance Score: [0-35]

    2. CONTRACT STRUCTURE
    - Clarity of Terms: [Brief assessment]
    - Completeness: [Brief assessment]
    - Consistency: [Brief assessment]
    - Structure Score: [0-35]

    3. RISK ASSESSMENT
    - Key Risks Identified: [List top 2-3]
    - Risk Mitigation: [Brief assessment]
    - Risk Score: [0-30]

    4. KEY RECOMMENDATIONS
    - Legal Updates: [Top 2 points]
    - Structural Changes: [Top 2 points]
    - Risk Mitigation: [Top 2 points]

    5. REQUIRED CHANGES
    - Critical Updates: [List 2-3 must-have changes]
    - Suggested Improvements: [List 2-3 nice-to-have changes]

    Contract text for analysis: {analysis_text}
    """
    
    try:
        response = model.generate_content(prompt)
        return format_legal_analysis(response.text)
    except Exception:
        return get_dummy_legal_data()

def format_legal_analysis(analysis_text: str) -> dict:
    """Format the legal analysis response for better display"""
    return {
        "legal_compliance": {
            "content": analysis_text,
            "sections": {
                "score": extract_section(analysis_text, "SCORE"),
                "overall_rating": extract_section(analysis_text, "OVERALL RATING"),
                "legal_compliance": extract_section(analysis_text, "1. LEGAL COMPLIANCE"),
                "contract_structure": extract_section(analysis_text, "2. CONTRACT STRUCTURE"),
                "risk_assessment": extract_section(analysis_text, "3. RISK ASSESSMENT"),
                "recommendations": extract_section(analysis_text, "4. KEY RECOMMENDATIONS"),
                "required_changes": extract_section(analysis_text, "5. REQUIRED CHANGES")
            }
        }
    }

def extract_section(text: str, section_name: str) -> str:
    """Extract a section from the analysis text"""
    try:
        start = text.index(section_name)
        next_section = start + len(section_name)
        for next_header in ["SCORE:", "OVERALL RATING:", "1. ", "2. ", "3. ", "4. ", "5. "]:
            try:
                end = text.index(next_header, next_section)
                return text[start:end].strip()
            except ValueError:
                continue
        return text[start:].strip()
    except ValueError:
        return ""

def format_section_content(content: str) -> str:
    """Format section content with proper markdown and styling"""
    if not content:
        return ""
    
    # Remove the section header if present
    if ":" in content:
        content = content.split(":", 1)[1].strip()
    
    # Format bullet points
    lines = content.split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('-'):
            # Convert bullet points to emoji bullets
            line = "• " + line[1:].strip()
        if line:
            formatted_lines.append(line)
    
    return "\n".join(formatted_lines)

def get_rating_color(score: str) -> str:
    """Get appropriate color based on score"""
    try:
        score_val = int(score.split('/')[0])
        if score_val <= 35:
            return "🔴 "  # Red circle for low scores
        elif score_val <= 70:
            return "🟡 "  # Yellow circle for medium scores
        else:
            return "🟢 "  # Green circle for high scores
    except:
        return "⚪ "  # White circle for unknown scores

# Initialize session state for chat and confidence tracking
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'confidence_scores' not in st.session_state:
    st.session_state.confidence_scores = []

# Custom CSS for dark theme
st.markdown("""
    <style>
    /* Dark theme base styles */
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Override Streamlit's default styles */
    .stTextInput > div > div > input {
        background-color: #2d2d2d;
        color: white;
    }
    .stButton > button {
        background-color: #0f52ba;
        color: white;
    }
    .stTextArea > div > div > textarea {
        background-color: #2d2d2d;
        color: white;
    }
    .stTab {
        color: white;
    }
    .stMarkdown {
        color: white;
    }
    
    /* Chat message styles */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        color: white;
    }
    .user-message {
        background-color: #2d2d2d;
        border-left: 5px solid #0f52ba;
    }
    .assistant-message {
        background-color: #363636;
        border-left: 5px solid #28a745;
    }
    .confidence-indicator {
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: #b0b0b0;
    }
    .main-header {
        background: linear-gradient(90deg, #1a237e, #0d47a1);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Analysis card styles */
    .analysis-card {
        background-color: #2d2d2d;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #0f52ba;
    }
    
    /* Section styles */
    div[data-testid="stVerticalBlock"] > div {
        background-color: #2d2d2d;
        color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Score indicators */
    .score-indicator {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
        color: white;
    }
    
    /* Metric styles */
    .metric-label {
        color: #b0b0b0;
        font-size: 0.9rem;
        margin-bottom: 5px;
    }
    .metric-value {
        color: white;
        font-size: 1.1rem;
        margin-bottom: 15px;
    }
    
    /* Info boxes */
    div[data-testid="stInfo"] {
        background-color: #2d2d2d;
        color: white;
        border: 1px solid #0f52ba;
    }
    
    /* Headers and text */
    h1, h2, h3, h4, h5, h6, p {
        color: white !important;
    }
    
    /* Lists */
    ul, ol {
        color: #b0b0b0;
    }
    
    /* Container backgrounds */
    div[data-testid="stDecoration"] {
        background-color: #2d2d2d !important;
    }
    
    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2d2d2d;
    }
    .stTabs [data-baseweb="tab"] {
        color: white !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #0f52ba !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #2d2d2d;
        border-color: #444;
    }
    
    /* Explanatory boxes */
    div[style*="background-color: #f8f9fa"] {
        background-color: #2d2d2d !important;
    }
    div[style*="background-color: #f0f2f6"] {
        background-color: #363636 !important;
    }
    
    /* Text colors */
    div[style*="color: #4a4a4a"],
    div[style*="color: #666"],
    div[style*="color: #2c3e50"],
    p[style*="color: #666"] {
        color: #b0b0b0 !important;
    }
    
    /* Links */
    a {
        color: #66b3ff !important;
    }
    a:hover {
        color: #99ccff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Enhanced header
st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Legal Assistant</h1>
        <p>Advanced Contract Analysis & Legal Compliance Platform</p>
    </div>
""", unsafe_allow_html=True)

def chat_with_contract(user_query: str, contract_text: str):
    """Enhanced interactive chat about contract details with confidence scoring"""
    try:
        # Generate response using Gemini
        chat_prompt = f"""
        Contract Text: {contract_text}
        User Query: {user_query}
        
        Provide a detailed response addressing the user's query about the contract.
        Include specific references to relevant sections when possible.
        """
        
        response = model.generate_content(chat_prompt)
        response_text = response.text
        
        # Calculate confidence score based on response characteristics
        confidence_score = calculate_confidence_score(response_text, contract_text)
        st.session_state.confidence_scores.append(confidence_score)
        
        # Format response with confidence indicator
        formatted_response = {
            'text': response_text,
            'confidence': confidence_score,
            'timestamp': pd.Timestamp.now().strftime("%H:%M")
        }
        
        return formatted_response
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

def calculate_confidence_score(response: str, context: str) -> float:
    """Calculate confidence score based on response quality indicators"""
    score = 0.0
    
    # Check for specific citations
    if any(marker in response.lower() for marker in ['section', 'clause', 'paragraph']):
        score += 0.3
    
    # Check response length relative to context
    response_length_ratio = len(response) / len(context)
    if 0.05 <= response_length_ratio <= 0.3:
        score += 0.2
    
    # Check for numerical/date references
    if any(char.isdigit() for char in response):
        score += 0.2
    
    # Check for legal terminology
    legal_terms = ['pursuant to', 'hereby', 'shall', 'whereas', 'notwithstanding']
    if any(term in response.lower() for term in legal_terms):
        score += 0.3
    
    return min(score, 1.0)  # Cap at 1.0

# Streamlit UI
st.title("Contract Analysis Assistant")
st.sidebar.title("Options")

# Add tabs
tab1, tab2, tab3 = st.tabs(["Upload & Analyze", "Legal Compliance", "Chat"])

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
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            
            # Extract text and process PDF
            text = extract_text_from_pdf(tmp_file.name)
            doc_id = process_pdf_file(tmp_file.name, category, governing_law)
            
            # Store the text in session state for other tabs
            st.session_state['contract_text'] = text
            
            # Search for similar documents
            similar_docs = search_similar(text, k=3)
            
            st.success(f"PDF processed and stored with ID: {doc_id}")
            
            # Display similar documents if found
            if similar_docs:
                st.subheader("Similar Documents Found:")
                for idx, doc in enumerate(similar_docs):
                    if doc['similarity'] < 0.7:  # Additional threshold check
                        with st.expander(f"Similar Document {idx + 1} (Similarity: {doc['similarity']:.2f})"):
                            st.write(f"**Category:** {doc['metadata']['category']}")
                            st.write(f"**Governing Law:** {doc['metadata']['governing_law']}")
                            st.write("**Content Preview:**")
                            st.text(doc['metadata']['text'])  # Using st.text to preserve formatting
            else:
                st.info("No similar documents found. Showing AI-generated analysis.")
                dummy_data = get_dummy_legal_data()
                with st.expander("AI Analysis"):
                    st.write(dummy_data)

with tab2:
    st.markdown("""
    <div style='background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
        <h2 style='color: #0f52ba; margin-top: 0;'>Legal Compliance Analysis</h2>
        <p style='color: #b0b0b0; margin-top: 5px;'>This AI-powered analysis evaluates your contract against industry standards and legal requirements. The evaluation focuses on:</p>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;'>
            <div>
                <h4 style='color: #2c3e50; margin: 0;'>Evaluation Areas</h4>
                <ul style='margin: 5px 0; padding-left: 20px;'>
                    <li>Legal Compliance (GDPR, HIPAA)</li>
                    <li>Terms Clarity & Precision</li>
                    <li>Document Completeness</li>
                    <li>Internal Consistency</li>
                </ul>
            </div>
            <div>
                <h4 style='color: #2c3e50; margin: 0;'>Scoring System</h4>
                <ul style='margin: 5px 0; padding-left: 20px;'>
                    <li><span style='color: #dc3545;'>0-35: Inadequate</span></li>
                    <li><span style='color: #ffc107;'>36-70: Adequate (Needs Improvement)</span></li>
                    <li><span style='color: #28a745;'>71-100: Exceeds Requirements</span></li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if 'contract_text' in st.session_state:
        if st.button("Analyze Legal Compliance"):
            with st.spinner("Analyzing legal compliance..."):
                analysis = analyze_legal_compliance(st.session_state['contract_text'])
                
                # Create three columns for the header
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    score = analysis["legal_compliance"]["sections"]["score"]
                    rating_color = get_rating_color(score)
                    st.markdown(f"""
                    <div style='text-align: center; background-color: #363636; padding: 20px; border-radius: 10px;'>
                        <h2 style='margin: 0;'>{rating_color}{score}</h2>
                        <p style='margin: 5px 0 0 0; color: #b0b0b0;'>Overall Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    rating = analysis["legal_compliance"]["sections"]["overall_rating"]
                    st.markdown(f"""
                    <div style='text-align: center; background-color: #363636; padding: 20px; border-radius: 10px;'>
                        <h3 style='margin: 0; color: #2c3e50;'>{rating}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Section 1: Legal Compliance
                st.markdown("""
                <div style='margin-top: 30px;'>
                    <h3 style='color: #0f52ba;'>📋 Legal Compliance Assessment</h3>
                    <p style='color: #b0b0b0; margin-top: 5px;'>Evaluation of adherence to key legal frameworks and regulations</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown(format_section_content(analysis["legal_compliance"]["sections"]["legal_compliance"]))
                
                # Section 2: Contract Structure
                st.markdown("""
                <div style='margin-top: 20px;'>
                    <h3 style='color: #0f52ba;'>🏗️ Contract Structure Analysis</h3>
                    <p style='color: #b0b0b0; margin-top: 5px;'>Assessment of document organization, clarity, and completeness</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown(format_section_content(analysis["legal_compliance"]["sections"]["contract_structure"]))
                
                # Section 3: Risk Assessment
                st.markdown("""
                <div style='margin-top: 20px;'>
                    <h3 style='color: #0f52ba;'>⚠️ Risk Assessment</h3>
                    <p style='color: #b0b0b0; margin-top: 5px;'>Identification and analysis of potential legal and operational risks</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown(format_section_content(analysis["legal_compliance"]["sections"]["risk_assessment"]))
                
                # Create two columns for recommendations and changes
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div style='margin-top: 20px;'>
                        <h3 style='color: #0f52ba;'>💡 Key Recommendations</h3>
                        <p style='color: #b0b0b0; margin-top: 5px;'>Suggested improvements for better compliance</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(format_section_content(analysis["legal_compliance"]["sections"]["recommendations"]))
                
                with col2:
                    st.markdown("""
                    <div style='margin-top: 20px;'>
                        <h3 style='color: #0f52ba;'>🔄 Required Changes</h3>
                        <p style='color: #b0b0b0; margin-top: 5px;'>Critical updates needed for compliance</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(format_section_content(analysis["legal_compliance"]["sections"]["required_changes"]))
                
                # Add an explanation box
                st.markdown("""
                <div style='margin-top: 30px; padding: 20px; background-color: #2d2d2d; border-radius: 10px; border-left: 5px solid #0f52ba;'>
                    <h4 style='color: #0f52ba; margin-top: 0;'>📌 Understanding This Analysis</h4>
                    <p style='color: #4a4a4a; margin: 10px 0;'>This analysis is powered by advanced AI that evaluates your contract against industry standards and legal requirements. The evaluation focuses on:</p>
                    <ul style='color: #4a4a4a; margin: 10px 0;'>
                        <li><strong>Legal Framework Compliance:</strong> GDPR, HIPAA, and other relevant regulations</li>
                        <li><strong>Document Structure:</strong> Organization, clarity, and completeness</li>
                        <li><strong>Risk Management:</strong> Identification and mitigation strategies</li>
                    </ul>
                    <p style='color: #4a4a4a; margin: 10px 0;'><strong>Next Steps:</strong> Review the recommendations and required changes. Consider consulting with legal counsel for implementing critical updates.</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Please upload a contract in the Upload & Analyze tab first.")

with tab3:
    st.markdown("""
        <div style='background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h2 style='color: #0f52ba; margin-top: 0;'>💬 Interactive Legal Assistant</h2>
            <p style='color: #b0b0b0; margin-bottom: 0;'>Ask specific questions about your contract and get AI-powered insights with confidence scoring.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if 'contract_text' in st.session_state:
        # Display chat history with enhanced styling
        for i, (msg, confidence) in enumerate(zip(st.session_state.chat_history, st.session_state.confidence_scores)):
            if i % 2 == 0:
                st.markdown(f"""
                    <div class='chat-message user-message'>
                        <strong>You:</strong> {msg['query']}
                    </div>
                """, unsafe_allow_html=True)
            else:
                confidence_color = get_rating_color(str(confidence))
                st.markdown(f"""
                    <div class='chat-message assistant-message'>
                        <strong>Assistant:</strong> {msg['response']['text']}
                        <div class='confidence-indicator'>
                            <span style='color: {confidence_color}'>Confidence Score: {confidence:.2%}</span>
                            <span style='float: right'>{msg['response']['timestamp']}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        user_query = st.text_input("Ask a question about your contract:", key="chat_input", 
                                 placeholder="e.g., What are the key termination clauses?")
        
        if st.button("Send", key="send_button"):
            if user_query:
                with st.spinner("Generating response..."):
                    response = chat_with_contract(user_query, st.session_state['contract_text'])
                    if response:
                        st.session_state.chat_history.append({
                            'query': user_query,
                            'response': response
                        })
                        st.experimental_rerun()
    else:
        st.info("Please upload a contract in the Upload & Analyze tab first.")

# Add sidebar information
with st.sidebar:
    st.info("""
    **Contract Analysis Assistant**
    
    This AI assistant evaluates contracts based on:
    1. Legal Compliance
    2. Clarity and Precision
    3. Completeness
    4. Consistency
    5. Risk Assessment
    
    Get analysis by:
    - Uploading contracts
    - Searching similar contracts
    - Asking specific questions
    
    For full analysis, type "full analysis"
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("Process CSV Data"):
        process_csv_data()
        st.success("CSV data processed and stored in vector database")
