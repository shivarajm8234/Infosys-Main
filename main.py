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
import requests

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    st.error("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
    raise ValueError("Missing Google API key")

genai.configure(api_key=api_key)
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

print("API Key:", os.getenv('GOOGLE_API_KEY'))  # Will help verify the key is loaded

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

def send_to_slack(analysis_result):
    """
    Send analysis results to Slack using webhook
    """
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')  # Get webhook URL from environment variable
    
    if not webhook_url:
        st.error("Slack webhook URL not configured. Please set SLACK_WEBHOOK_URL environment variable.")
        return
    
    # Format the analysis result for better readability in Slack
    message = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🤖 Contract Analysis Results"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Score:* {analysis_result['legal_compliance']['sections']['score']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Rating:* {analysis_result['legal_compliance']['sections']['overall_rating']}"
                    }
                ]
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Legal Compliance Summary:*\n{analysis_result['legal_compliance']['sections']['legal_compliance'][:1000]}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Key Recommendations:*\n{analysis_result['legal_compliance']['sections']['recommendations']}"
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Analysis completed at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(webhook_url, json=message)
        response.raise_for_status()
        st.success("Analysis results sent to Slack successfully")
    except Exception as e:
        st.error(f"Failed to send message to Slack: {str(e)}")

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
        analysis_result = format_legal_analysis(response.text)
        
        # Send analysis results to Slack
        send_to_slack(analysis_result)
        
        return analysis_result
    except Exception:
        dummy_result = get_dummy_legal_data()
        send_to_slack(dummy_result)  # Send dummy data if analysis fails
        return dummy_result

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

def get_rating_color(confidence: str) -> str:
    """Get appropriate emoji indicator based on confidence score"""
    try:
        score = float(confidence)
        if score <= 0.4:
            return "🔴"  # Low confidence
        elif score <= 0.7:
            return "🟡"  # Medium confidence
        else:
            return "🟢"  # High confidence
    except:
        return "⚪"  # Unknown confidence

# Initialize session state for chat and confidence tracking
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'confidence_scores' not in st.session_state:
    st.session_state.confidence_scores = []

# Add custom CSS for better styling
st.markdown("""
    <style>
        /* Modern color scheme */
        :root {
            --primary-color: #1a73e8; /* Update this color if needed */
            --secondary-color: #34a853;
            --background-color: #121212; /* Dark background */
            --text-color: #e0e0e0; /* Light text color */
            --card-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Main container styling */
        .main-container {
            padding: 2rem;
            background-color: var(--background-color);
            border-radius: 12px;
            margin-bottom: 2rem;
        }

        /* Card styling */
        .analysis-card {
            background-color: #1e1e1e; /* Dark card background */
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: var(--card-shadow);
            margin-bottom: 1.5rem;
            border-left: 4px solid var(--primary-color);
            transition: transform 0.2s ease;
        }

        .analysis-card:hover {
            transform: translateY(-2px);
        }

        /* Metric cards */
        .metric-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 2rem;
        }

        .metric-card {
            background-color: #1e1e1e; /* Dark card background */
            padding: 1rem;
            border-radius: 8px;
            box-shadow: var(--card-shadow);
            text-align: center;
            flex: 1;
            margin: 0 0.5rem;
        }

        /* Typography */
        h1, h2, h3 {
            color: var(--text-color);
            font-weight: 600;
        }

        .section-title {
            color: var(--primary-color);
            font-size: 1.25rem;
            margin-bottom: 1rem;
        }

        .score-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary-color);
        }

        .rating-text {
            color: #999; /* Light grey text */
            font-size: 1rem;
        }

        /* Lists and content */
        .analysis-content {
            color: #ccc; /* Light grey text */
            line-height: 1.6;
        }

        .analysis-list {
            list-style-type: none;
            padding-left: 0;
        }

        .analysis-list li {
            margin-bottom: 0.5rem;
            padding-left: 1.5rem;
            position: relative;
        }

        .analysis-list li:before {
            content: "•";
            color: var(--primary-color);
            position: absolute;
            left: 0;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .metric-container {
                flex-direction: column;
            }
            
            .metric-card {
                margin: 0.5rem 0;
            }
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
        As a legal assistant, analyze this contract excerpt and answer the question.
        
        Contract Text:
        {contract_text[:2000]}  # Limit context to avoid token limits
        
        User Question: {user_query}
        
        Provide a clear, detailed response that:
        1. Directly addresses the user's question
        2. References specific sections when relevant
        3. Uses professional legal terminology
        4. Provides concrete examples or explanations
        """
        
        response = model.generate_content(chat_prompt)
        response_text = response.text
        
        # Calculate confidence score
        confidence_score = calculate_confidence_score(response_text, contract_text)
        
        return {
            'text': response_text,
            'confidence': confidence_score,
            'timestamp': pd.Timestamp.now().strftime("%H:%M")
        }
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
tab1, tab2, tab3 = st.tabs(["Upload & Analyze", ' ', "Chat"])

with tab1:
    st.header("Upload Contract PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            
            # Extract text and process PDF
            text = extract_text_from_pdf(tmp_file.name)
            
            # Store the text in session state
            st.session_state['contract_text'] = text
            
            # Automatically detect category and governing law
            prompt = f"""
            Analyze the following contract text and determine:
            1. The category (choose one): Co_Branding, Endorsement, Development, Transportation, Maintenance, Hosting, IP, Other
            2. The governing law (choose one): International Law, Indian Law, US Law, Other

            Respond ONLY with a JSON object in this exact format:
            {{"category": "selected_category", "governing_law": "selected_law"}}

            Contract Text:
            {text[:1000]}  # Only analyze first 1000 chars for classification
            """
            
            try:
                response = model.generate_content(prompt)
                # Clean the response text to ensure it only contains the JSON
                response_text = response.text.strip()
                if response_text.startswith('json'):
                    response_text = response_text[7:-3]  # Remove json and ``` markers
                detected_data = json.loads(response_text)
                category = detected_data.get("category", "Other")
                governing_law = detected_data.get("governing_law", "Other")
            except Exception as e:
                st.warning("Using default classification due to detection error")
                category = "Other"
                governing_law = "Other"
            
            # Process the PDF file
            process_pdf_file(tmp_file.name, category, governing_law)
            
            # Automatically perform legal compliance analysis
            with st.spinner("Analyzing legal compliance..."):
                st.session_state['analysis_result'] = analyze_legal_compliance(st.session_state['contract_text'])
                analysis_result = st.session_state['analysis_result']
            
            # Create three columns for the header
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.markdown("<h3 style='text-align: center;'>Category</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 18px;'>{category}</p>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<h3 style='text-align: center;'>Analysis Results</h3>", unsafe_allow_html=True)
                score = analysis_result["legal_compliance"]["sections"]["score"]
                rating = analysis_result["legal_compliance"]["sections"]["overall_rating"]
                st.markdown(f"<p style='text-align: center; font-size: 18px;'>Score: {score}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 18px;'>Rating: {rating}</p>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<h3 style='text-align: center;'>Governing Law</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 18px;'>{governing_law}</p>", unsafe_allow_html=True)
            
            # Display analysis results in a modern layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)

category = st.session_state.get('category', 'Not Specified')
governing_law = st.session_state.get('governing_law', 'Not Specified')

# Metrics row
if 'analysis_result' not in st.session_state:
    st.error("Please analyze a contract first.")
else:
    analysis_result = st.session_state['analysis_result']
    st.markdown("""
        <div class="metric-container">
            <div class="metric-card">
                <h3>Category</h3>
                <div class="score-value">{}</div>
            </div>
            <div class="metric-card">
                <h3>Analysis Score</h3>
                <div class="score-value">{}</div>
                <div class="rating-text">{}</div>
            </div>
            <div class="metric-card">
                <h3>Governing Law</h3>
                <div class="score-value">{}</div>
            </div>
        </div>
    """.format(
        category or 'Not Specified',
        analysis_result["legal_compliance"]["sections"]["score"].replace('SCORE:', '').strip(),
        analysis_result["legal_compliance"]["sections"]["overall_rating"].replace('OVERALL RATING:', '').strip(),
        governing_law or 'Not Specified'
    ), unsafe_allow_html=True)

    # Analysis sections
    sections = [
        ("Legal Compliance", "legal_compliance", "⚖️"),
        ("Contract Structure", "contract_structure", "📄"),
        ("Risk Assessment", "risk_assessment", "⚠️"),
        ("Key Recommendations", "recommendations", "💡"),
        ("Required Changes", "required_changes", "✔️")
    ]

    if 'analysis_result' in st.session_state:
        for title, key, icon in sections:
            st.markdown(f"""
                <div class="analysis-card">
                    <h3 class="section-title">{icon} {title}</h3>
                    <div class="analysis-content">
                        {analysis_result["legal_compliance"]["sections"][key]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("""
        <div style='background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h2 style='color: #0f52ba; margin-top: 0;'>💬 Interactive Legal Assistant</h2>
            <p style='color: #b0b0b0; margin-bottom: 0;'>Ask questions about your contract and get AI-powered insights.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if 'contract_text' in st.session_state:
        # Chat input at the bottom
        st.markdown("<div style='position: fixed; bottom: 0; width: 100%; padding: 20px; background-color: #1e1e1e;'>", unsafe_allow_html=True)
        
        # Chat input and send button in the same row
        col1, col2 = st.columns([5,1])
        with col1:
            user_query = st.text_input(
                "Ask a question:",
                key="chat_input",
                placeholder="e.g., What are the key termination clauses?"
            )
        with col2:
            send_button = st.button("Send", key="send_button")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add some space at the bottom to prevent overlap with fixed input
        st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
        
        if send_button and user_query:
            with st.spinner("Generating response..."):
                # Add user message to chat history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_query
                })
                
                # Get AI response
                response = chat_with_contract(user_query, st.session_state['contract_text'])
                
                if response:
                    # Add AI response to chat history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response
                    })
                    st.session_state.confidence_scores.append(response['confidence'])
                    st.experimental_rerun()
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"""
                    <div style='background-color: #1e1e1e; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                        <strong>You:</strong> {msg['content']}
                    </div>
                """, unsafe_allow_html=True)
            else:
                confidence = msg['content']['confidence']
                confidence_color = get_rating_color(str(confidence))
                st.markdown(f"""
                    <div style='background-color: #2d2d2d; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                        <strong>Assistant:</strong> {msg['content']['text']}
                        <div style='margin-top: 5px; font-size: 0.8em; color: #888;'>
                            {confidence_color} Confidence: {confidence:.1%} | {msg['content']['timestamp']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Clear chat button in a less prominent position
        if st.session_state.chat_history:
            if st.button("Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.session_state.confidence_scores = []
                st.experimental_rerun()
    else:
        st.info("Please upload a contract in the Upload & Analyze tab first.")

# Add sidebar information
with st.sidebar:
    st.info("""
    *Contract Analysis Assistant*
    
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
    
    if st.button("Process CSV Data"):
        process_csv_data()
        st.success("CSV data processed and stored in vector database")

