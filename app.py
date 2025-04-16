import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from agents import create_investment_graph, InvestmentState
import logging
from typing import Dict, Any
from openai import OpenAI
import os
from dotenv import load_dotenv
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and validate environment variables
def load_api_key() -> tuple[str | None, str | None]:
    """Load and validate the API key from .env file."""
    try:
        # Get the absolute path to the .env file
        env_path = Path('.env').resolve()
        
        if not env_path.exists():
            return None, "No .env file found. Please create one with your ALPHA_VANTAGE_API_KEY."
        
        # Load environment variables
        load_dotenv(dotenv_path=env_path, override=True)
        
        # Debug logging
        logger.info(f"Loading API key from: {env_path}")
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        logger.info(f"API key found: {'Yes' if api_key else 'No'}")
        
        if not api_key:
            # Try reading the file directly as a fallback
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('ALPHA_VANTAGE_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        os.environ['ALPHA_VANTAGE_API_KEY'] = api_key
                        break
        
        if not api_key:
            return None, "ALPHA_VANTAGE_API_KEY not found in .env file."
        
        api_key = api_key.strip()
        if len(api_key) < 10:  # Basic validation for key format
            return None, "ALPHA_VANTAGE_API_KEY appears to be invalid. Please check your .env file."
        
        return api_key, None
        
    except Exception as e:
        logger.error(f"Error loading API key: {str(e)}")
        return None, f"Error loading API key: {str(e)}"

# Load API key and error message if any
ALPHA_VANTAGE_API_KEY, api_key_error = load_api_key()

# Predefined options
POPULAR_STOCKS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc. (Google)",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc. (Facebook)",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "WMT": "Walmart Inc."
}

INVESTMENT_GOALS = {
    "growth": "Long-term capital growth",
    "income": "Regular dividend income",
    "balanced": "Balance between growth and income",
    "retirement": "Retirement savings",
    "education": "Education fund",
    "wealth_preservation": "Preserve wealth with moderate growth",
    "speculative": "High-risk, high-reward opportunities",
    "sector_focus": "Focus on specific industry sectors",
    "esg": "Environmentally and socially responsible investing"
}

# Page config
st.set_page_config(
    page_title="Investment Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        font-size: 1.1em;
        background-color: #1f77b4;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .analysis-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .analysis-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stSelectbox > div {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 0.5rem;
    }
    .section-spacing {
        margin: 2rem 0;
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chart-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
    }
    .header-icon {
        font-size: 1.8em;
        margin-right: 0.8rem;
        color: #1f77b4;
    }
    .sidebar-section {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar-section:last-child {
        margin-bottom: 0;
    }
    .sidebar-title {
        color: #1f77b4;
        font-size: 1.2em;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .sidebar-divider {
        height: 1px;
        background-color: #e6e6e6;
        margin: 0.8rem 0;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.9em;
    }
    .sidebar-input {
        margin: 0.8rem 0;
    }
    .sidebar-button {
        margin-top: 1rem;
    }
    .stRadio > div {
        flex-direction: row !important;
        gap: 0.8rem;
        margin-bottom: 0.5rem;
    }
    .stRadio > div > div {
        flex: 1;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
        padding: 0.4rem;
    }
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        padding: 0.4rem;
        min-height: 100px;
    }
    .stSelectbox > div > div > div {
        border-radius: 8px;
        padding: 0.4rem;
    }
    .stButton > button {
        margin-top: 0.6rem !important;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .user-message {
        background-color: #f0f2f6;
        margin-left: 1rem;
    }
    .assistant-message {
        background-color: #e6f3ff;
        margin-right: 1rem;
    }
    .chat-input-container {
        margin-top: 2rem;
        padding: 1.5rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-history-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Sidebar specific styles */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1rem;
    }
    .sidebar-section {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar-title {
        color: #1f77b4;
        font-size: 1.2em;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sidebar-subtitle {
        color: #666;
        font-size: 0.9em;
        margin: 0.8rem 0 0.4rem 0;
        font-weight: 500;
    }
    .sidebar-info {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        font-size: 0.9em;
        border-left: 4px solid #1f77b4;
    }
    /* Radio button enhancements */
    .stRadio > div {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .stRadio [role="radiogroup"] {
        gap: 0.3rem !important;
    }
    .stRadio > div > div > label {
        background-color: #ffffff;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        border: 1px solid #e6e6e6;
        transition: all 0.2s ease;
    }
    .stRadio > div > div > label:hover {
        background-color: #f0f7ff;
        border-color: #1f77b4;
    }
    /* Select box enhancements */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    .stSelectbox > div > div:hover {
        border-color: #1f77b4;
    }
    /* Text input enhancements */
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 0.5rem;
        transition: all 0.2s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #1f77b4;
        box-shadow: 0 0 0 1px #1f77b4;
    }
    /* Button enhancements */
    .stButton > button {
        margin-top: 1rem;
        background: linear-gradient(45deg, #1f77b4, #2196f3);
        border-radius: 8px;
        padding: 0.6rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.3);
    }
    /* Auto-scroll for chat container */
    .chat-history-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        scroll-behavior: smooth;
    }
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    .loading-dots {
        animation: pulse 1.5s infinite;
        display: inline-block;
    }
    /* Enhanced message styling */
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        opacity: 0;
        animation: fadeIn 0.5s ease forwards;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background-color: #f0f2f6;
        margin-left: 1rem;
        border-bottom-right-radius: 4px;
    }
    .assistant-message {
        background-color: #e6f3ff;
        margin-right: 1rem;
        border-bottom-left-radius: 4px;
    }
    /* Input area enhancements */
    .chat-input-area {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .input-hint {
        color: #666;
        font-size: 0.8em;
        margin-top: 0.5rem;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'current_state' not in st.session_state:
        st.session_state.current_state = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = "AAPL"
    if 'selected_goal' not in st.session_state:
        st.session_state.selected_goal = "growth"
    if 'custom_goal' not in st.session_state:
        st.session_state.custom_goal = ""

def create_metric_card(title, value, change=None):
    """Create a metric card with optional change indicator."""
    st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #262730;">{title}</h3>
            <h2 style="margin: 0.5rem 0; color: #262730;">{value}</h2>
            {f'<p style="margin: 0; color: {"#00cc00" if change >= 0 else "#ff4b4b"};">{change:+.2f}%</p>' if change is not None else ''}
        </div>
    """, unsafe_allow_html=True)

def create_analysis_card(title, content):
    """Create an analysis card with title and content."""
    st.markdown(f"""
        <div class="analysis-card">
            <h3 style="margin: 0 0 1rem 0; color: #262730;">{title}</h3>
            <p style="margin: 0; color: #262730;">{content}</p>
        </div>
    """, unsafe_allow_html=True)

def plot_stock_data(data):
    """Create an interactive stock price chart."""
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Price',
        line=dict(color='#1f77b4')
    ))
    
    # Add moving averages if available
    if 'moving_avg_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['moving_avg_50'],
            name='50-day MA',
            line=dict(color='#ff7f0e', dash='dash')
        ))
    
    if 'moving_avg_200' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['moving_avg_200'],
            name='200-day MA',
            line=dict(color='#2ca02c', dash='dash')
        ))
    
    fig.update_layout(
        title='Stock Price History',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        hovermode='x unified',
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def create_filing_card(filing: Dict[str, Any]) -> None:
    """Create a card for an SEC filing."""
    st.markdown(f"""
        <div class="analysis-card">
            <h4 style="margin: 0 0 0.5rem 0; color: #262730;">{filing['type']} - {filing['date']}</h4>
            <p style="margin: 0 0 0.5rem 0; color: #666666;">{filing['title']}</p>
            <div style="margin: 0; color: #262730;">{filing['analysis']}</div>
        </div>
    """, unsafe_allow_html=True)

def create_chatbot_interface(state: InvestmentState) -> None:
    """Create an enhanced chatbot interface for investment analysis."""
    st.markdown('<div class="section-spacing">', unsafe_allow_html=True)
    st.markdown("""
        <div class="header-container">
            <span class="header-icon">💬</span>
            <h2>Investment Analysis Chat</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history if not exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "is_typing" not in st.session_state:
        st.session_state.is_typing = False
    
    # Create context from investment analysis
    context = f"""
    Investment Analysis Context:
    - Stock: {state['stock_symbol']}
    - Current Price: ${state['metrics'].get('current_price', 'N/A'):.2f}
    - 50-day MA: ${state['metrics'].get('moving_avg_50', 'N/A'):.2f}
    - 200-day MA: ${state['metrics'].get('moving_avg_200', 'N/A'):.2f}
    - Risk Tolerance: {state['risk_tolerance']}
    - Investment Goals: {state['investment_goals']}
    """
    
    # Chat history display with better formatting
    st.markdown("### Conversation History")
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-history-container" id="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Show typing indicator when generating response
        if st.session_state.is_typing:
            st.markdown("""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> <span class="loading-dots">Thinking...</span>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # User input with better styling
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    st.markdown("### Ask a Question")
    
    # Initialize the keys if they don't exist
    if "temp_input" not in st.session_state:
        st.session_state.temp_input = ""
    
    def clear_input():
        st.session_state.temp_input = ""
    
    def handle_send():
        if st.session_state.temp_input.strip():
            # Check if this message is already in the chat history to prevent duplicates
            user_message = st.session_state.temp_input.strip()
            
            # Only add the message if it's not the last user message
            if not st.session_state.chat_history or \
               st.session_state.chat_history[-1]["role"] != "user" or \
               st.session_state.chat_history[-1]["content"] != user_message:
                
                st.session_state.chat_history.append({"role": "user", "content": user_message})
                st.session_state.is_typing = True
                st.rerun()
                
                # Generate response
                response = generate_chat_response(user_message, context, st.session_state.chat_history)
                
                # Update chat history and clear typing indicator
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.session_state.is_typing = False
                clear_input()
                st.rerun()
    
    # Text input for user message
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_area(
            "Type your question here...",
            key="temp_input",
            height=100,
            placeholder="Ask about stock analysis, investment strategies, or market insights..."
        )
        st.markdown('<p class="input-hint">Press Ctrl+Enter to send</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="height: 25px"></div>', unsafe_allow_html=True)  # Spacing to align with text area
        send_button = st.button(
            "Send",
            key="send_button",
            use_container_width=True
        )
    
    # Handle message sending
    if user_input and (send_button or (user_input and user_input.endswith('\n'))):
        handle_send()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # JavaScript for auto-scrolling
    st.markdown("""
        <script>
            const observer = new MutationObserver((mutations) => {
                const chatContainer = document.querySelector('#chat-container');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            });
            
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        </script>
    """, unsafe_allow_html=True)

def generate_chat_response(question: str, context: str, chat_history: list) -> str:
    """Generate a response using OpenAI with enhanced context and guidelines."""
    try:
        client = OpenAI()
        # Prepare the conversation with system message
        messages = [
            {
                "role": "system",
                "content": f"""You are an expert investment advisor. Use the following context to provide detailed, 
                personalized advice. Always consider the user's risk tolerance and investment goals.
                
                Context:
                {context}
                
                Guidelines:
                1. Provide clear, actionable insights
                2. Explain technical terms in simple language
                3. Consider both technical and fundamental analysis
                4. Include specific recommendations when appropriate
                5. Highlight risks and opportunities
                6. Reference the user's investment goals and risk tolerance
                7. Use bullet points for clarity when appropriate
                8. Include relevant metrics from the analysis
                """
            }
        ]
        
        # Add chat history
        messages.extend(chat_history)
        
        # Add the current question
        messages.append({"role": "user", "content": question})
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}")
        return f"I apologize, but I encountered an error while processing your question. Please try again later."

def export_chat_to_pdf(chat_history: list, context: str) -> None:
    """Export chat history to PDF with professional formatting."""
    try:
        from fpdf import FPDF
        
        pdf = FPDF()
        pdf.add_page()
        
        # Add title and context
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Investment Analysis Chat History", ln=True, align="C")
        pdf.ln(10)
        
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, context)
        pdf.ln(10)
        
        # Add chat history
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Conversation History", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", "", 12)
        for message in chat_history:
            role = "You" if message["role"] == "user" else "Assistant"
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"{role}:", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, message["content"])
            pdf.ln(5)
        
        # Save the PDF
        pdf_output = pdf.output(dest="S").encode("latin1")
        st.download_button(
            label="📥 Download PDF",
            data=pdf_output,
            file_name="investment_chat_history.pdf",
            mime="application/pdf"
        )
        
    except Exception as e:
        logger.error(f"Error exporting to PDF: {str(e)}")
        st.error("Failed to export chat history to PDF")

def export_chat_to_csv(chat_history: list) -> None:
    """Export chat history to CSV format."""
    try:
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["Role", "Message"])
        
        for message in chat_history:
            writer.writerow([message["role"], message["content"]])
        
        csv_data = output.getvalue()
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="investment_chat_history.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {str(e)}")
        st.error("Failed to export chat history to CSV")

def validate_state(state: dict) -> tuple[bool, str]:
    """Validate the state structure and return (is_valid, error_message)."""
    required_fields = {
        'stock_symbol': str,
        'investment_goals': str,
        'risk_tolerance': str,
        'stock_analysis': str,
        'risk_analysis': str,
        'metrics': dict,
        'messages': list,
        'stock_data': pd.DataFrame,
        'returns': pd.Series
    }
    
    for field, expected_type in required_fields.items():
        if field not in state:
            return False, f"Missing required field: {field}"
        if not isinstance(state[field], expected_type):
            return False, f"Invalid type for {field}. Expected {expected_type.__name__}, got {type(state[field]).__name__}"
    
    return True, ""

def initialize_state(stock_symbol: str, investment_goals: str, risk_tolerance: str) -> dict:
    """Initialize and validate the state dictionary."""
    # Initialize empty DataFrame with correct structure for stock data
    stock_data = pd.DataFrame(columns=['Close', 'moving_avg_50', 'moving_avg_200'])
    
    # Create initial state
    state = {
        'stock_symbol': stock_symbol.upper(),
        'investment_goals': investment_goals,
        'risk_tolerance': risk_tolerance.lower(),
        'stock_analysis': "",
        'risk_analysis': "",
        'metrics': {},
        'messages': [],
        'stock_data': stock_data,
        'returns': pd.Series(dtype=float)
    }
    
    # Validate state
    is_valid, error_message = validate_state(state)
    if not is_valid:
        raise ValueError(f"Invalid state initialization: {error_message}")
    
    return state

def main():
    initialize_session_state()
    
    # Sidebar for user input
    with st.sidebar:
        # Display API key status only if there's an error
        if api_key_error:
            st.error(f"⚠️ API Key Error: {api_key_error}")
            st.info("The analysis will proceed using Yahoo Finance as the primary data source.")
        
        # Stock Selection Section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📈 Stock Selection</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-subtitle">Selection Method</div>', unsafe_allow_html=True)
        stock_selection = st.radio(
            "Stock Selection Method",
            ["Select from popular stocks", "Enter custom symbol"],
            horizontal=True,
            key="stock_selection",
            label_visibility="collapsed"
        )
        
        if stock_selection == "Select from popular stocks":
            st.markdown('<div class="sidebar-subtitle">Choose Stock</div>', unsafe_allow_html=True)
            stock_options = [f"{symbol} - {name}" for symbol, name in POPULAR_STOCKS.items()]
            selected_option = st.selectbox(
                "Select Stock",
                stock_options,
                index=stock_options.index(f"{st.session_state.selected_stock} - {POPULAR_STOCKS[st.session_state.selected_stock]}"),
                help="Select a stock from popular options",
                label_visibility="collapsed"
            )
            stock_symbol = selected_option.split(" - ")[0]
        else:
            st.markdown('<div class="sidebar-subtitle">Enter Symbol</div>', unsafe_allow_html=True)
            stock_symbol = st.text_input(
                "Stock Symbol",
                value=st.session_state.selected_stock,
                placeholder="e.g., AAPL",
                help="Enter the stock symbol you want to analyze",
                label_visibility="collapsed"
            )
        
        st.session_state.selected_stock = stock_symbol
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Investment Goals Section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🎯 Investment Profile</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-subtitle">Goal Setting Method</div>', unsafe_allow_html=True)
        goal_selection = st.radio(
            "Goal Setting Method",
            ["Select predefined goal", "Enter custom goal"],
            horizontal=True,
            key="goal_selection",
            label_visibility="collapsed"
        )
        
        if goal_selection == "Select predefined goal":
            st.markdown('<div class="sidebar-subtitle">Investment Goal</div>', unsafe_allow_html=True)
            goal_options = [f"{key} - {value}" for key, value in INVESTMENT_GOALS.items()]
            selected_goal_option = st.selectbox(
                "Investment Goal",
                goal_options,
                index=goal_options.index(f"{st.session_state.selected_goal} - {INVESTMENT_GOALS[st.session_state.selected_goal]}"),
                help="Select your investment goal from predefined options",
                label_visibility="collapsed"
            )
            selected_goal = selected_goal_option.split(" - ")[0]
            investment_goals = INVESTMENT_GOALS[selected_goal]
            st.session_state.selected_goal = selected_goal
        else:
            st.markdown('<div class="sidebar-subtitle">Custom Goal Description</div>', unsafe_allow_html=True)
            investment_goals = st.text_area(
                "Custom Investment Goals",
                value=st.session_state.custom_goal,
                placeholder="Describe your investment goals in detail...",
                help="Enter your specific investment goals",
                label_visibility="collapsed"
            )
            st.session_state.custom_goal = investment_goals
        
        st.markdown('<div class="sidebar-subtitle">Risk Tolerance</div>', unsafe_allow_html=True)
        risk_tolerance = st.selectbox(
            "Risk Tolerance Level",
            options=["Conservative", "Moderate", "Aggressive"],
            help="Select your risk tolerance level",
            label_visibility="collapsed"
        )
        
        st.markdown(f'<div class="sidebar-info">⚖️ Risk Profile: {risk_tolerance}</div>', unsafe_allow_html=True)
        
        # Add a small space before the button
        st.markdown('<div style="height: 1rem"></div>', unsafe_allow_html=True)
        
        analyze_button = st.button(
            "Start Analysis",
            type="primary",
            use_container_width=True,
            key="analyze_button"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    st.title("Investment Analysis Dashboard")
    
    if analyze_button and stock_symbol:
        with st.spinner("Analyzing investment opportunity..."):
            try:
                # Initialize and validate state
                initial_state = initialize_state(
                    stock_symbol=stock_symbol,
                    investment_goals=investment_goals,
                    risk_tolerance=risk_tolerance
                )
                
                # Run analysis
                workflow = create_investment_graph()
                result = workflow.invoke(initial_state)
                
                # Update session state
                st.session_state.analysis_complete = True
                st.session_state.current_state = result
                
                # Display results
                st.success("Analysis complete!")
                
                # Display data source information
                if ALPHA_VANTAGE_API_KEY:
                    st.info("📊 Analysis performed using both Yahoo Finance and Alpha Vantage data sources.")
                else:
                    st.info("📊 Analysis performed using Yahoo Finance as the primary data source.")
                    
            except ValueError as ve:
                logger.error(f"State validation error: {str(ve)}")
                st.error(f"""
                    An error occurred during state initialization: {str(ve)}
                    
                    Please try:
                    1. Refreshing the page
                    2. Selecting a different stock symbol
                    3. Contacting support if the issue persists
                """)
            except Exception as e:
                logger.error(f"Error during analysis: {str(e)}")
                error_message = str(e)
                if "Invalid state update" in error_message:
                    st.error("""
                        An error occurred during the analysis. The system is unable to process the state update.
                        This might be a temporary issue. Please try again.
                        
                        If the problem persists, please try:
                        1. Selecting a different stock symbol
                        2. Checking your internet connection
                        3. Waiting a few minutes before trying again
                    """)
                else:
                    st.error(f"""
                        An error occurred during the analysis. This might be due to:
                        - Network connectivity issues
                        - API rate limits
                        - Invalid stock symbol
                        
                        Please try again in a few moments. If the problem persists, try a different stock symbol.
                    """)
                logger.debug(f"Detailed error: {error_message}")
                return
    
    if st.session_state.analysis_complete and st.session_state.current_state:
        state = st.session_state.current_state
        
        # Metrics section
        st.markdown('<div class="section-spacing">', unsafe_allow_html=True)
        st.markdown("""
            <div class="header-container">
                <span class="header-icon">📊</span>
                <h2>Key Metrics</h2>
            </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_metric_card(
                "Current Price",
                f"${state['metrics'].get('current_price', 0):.2f}"
            )
        
        with col2:
            create_metric_card(
                "50-day MA",
                f"${state['metrics'].get('moving_avg_50', 0):.2f}"
            )
        
        with col3:
            create_metric_card(
                "200-day MA",
                f"${state['metrics'].get('moving_avg_200', 0):.2f}"
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Stock data visualization
        if not state['stock_data'].empty:
            st.markdown('<div class="section-spacing">', unsafe_allow_html=True)
            st.markdown("""
                <div class="header-container">
                    <span class="header-icon">📈</span>
                    <h2>Price History</h2>
                </div>
            """, unsafe_allow_html=True)
            plot_stock_data(state['stock_data'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis section
        st.markdown('<div class="section-spacing">', unsafe_allow_html=True)
        st.markdown("""
            <div class="header-container">
                <span class="header-icon">📋</span>
                <h2>Analysis</h2>
            </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            create_analysis_card("Stock Analysis", state['stock_analysis'])
        
        with col2:
            create_analysis_card("Risk Analysis", state['risk_analysis'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add chatbot interface
        create_chatbot_interface(state)

if __name__ == "__main__":
    main() 