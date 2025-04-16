# Stock Investment Advisor

A LangGraph-based application that uses AI agents to analyze stock market investments and provide recommendations.

## Features

- Two specialized AI agents for comprehensive investment analysis:
  - Stock Performance Analysis Agent
  - Risk Analysis Agent
- Interactive Streamlit interface
- Human-in-the-loop mechanism for user intervention
- Real-time stock data analysis using yfinance
- OpenAI-powered insights and recommendations

## Setup

1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Make sure your virtual environment is activated
2. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open your browser and navigate to the provided local URL
4. Enter your investment criteria in the sidebar:
   - Stock symbol
   - Investment goals
   - Risk tolerance level
5. Click "Analyze Investment" to get AI-powered insights
6. Use the "Have Questions?" section to interact with the analysis

## Architecture

The application is built using:
- LangGraph for agent orchestration
- OpenAI's GPT models for analysis
- Streamlit for the user interface
- yfinance for stock data
- Python-dotenv for environment management

## Contributing

Feel free to submit issues and enhancement requests! 