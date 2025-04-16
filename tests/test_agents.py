import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import StockAnalysisAgent, RiskAnalysisAgent, InvestmentState

@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    prices = np.random.normal(100, 10, len(dates))
    return pd.DataFrame({
        'Close': prices,
        'Open': prices + np.random.normal(0, 1, len(dates)),
        'High': prices + np.random.normal(2, 1, len(dates)),
        'Low': prices - np.random.normal(2, 1, len(dates)),
        'Volume': np.random.randint(1000000, 2000000, len(dates))
    }, index=dates)

@pytest.fixture
def investment_state():
    """Create a sample investment state."""
    return {
        "stock_symbol": "AAPL",
        "investment_goals": "Long-term growth",
        "risk_tolerance": "moderate",
        "stock_analysis": "",
        "risk_analysis": "",
        "metrics": {},
        "messages": [],
        "stock_data": pd.DataFrame(),
        "returns": pd.Series()
    }

def test_stock_analysis_agent_initialization():
    """Test StockAnalysisAgent initialization."""
    agent = StockAnalysisAgent()
    assert agent.model_name == "gpt-4-turbo-preview"
    assert hasattr(agent, 'client')

@patch('yfinance.Ticker')
def test_analyze_stock_performance_success(mock_ticker, sample_stock_data, investment_state):
    """Test successful stock analysis."""
    # Setup mock
    mock_ticker.return_value.history.return_value = sample_stock_data
    
    # Create agent and analyze
    agent = StockAnalysisAgent()
    result = agent.analyze_stock_performance(investment_state)
    
    # Assertions
    assert result["stock_symbol"] == "AAPL"
    assert "metrics" in result
    assert "current_price" in result["metrics"]
    assert "moving_avg_50" in result["metrics"]
    assert "moving_avg_200" in result["metrics"]
    assert len(result["messages"]) > 0

def test_analyze_stock_performance_no_symbol(investment_state):
    """Test stock analysis with no symbol."""
    investment_state["stock_symbol"] = ""
    agent = StockAnalysisAgent()
    result = agent.analyze_stock_performance(investment_state)
    
    assert result["stock_symbol"] == ""
    assert "Error" in result["stock_analysis"]

@patch('yfinance.Ticker')
def test_analyze_stock_performance_empty_data(mock_ticker, investment_state):
    """Test stock analysis with empty data."""
    mock_ticker.return_value.history.return_value = pd.DataFrame()
    agent = StockAnalysisAgent()
    result = agent.analyze_stock_performance(investment_state)
    
    assert "No data available" in result["stock_analysis"]

def test_risk_analysis_agent_initialization():
    """Test RiskAnalysisAgent initialization."""
    agent = RiskAnalysisAgent()
    assert agent.model_name == "gpt-4-turbo-preview"
    assert hasattr(agent, 'client')

def test_evaluate_risk_success(investment_state):
    """Test successful risk evaluation."""
    # Add sample returns data
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    investment_state["returns"] = returns
    
    agent = RiskAnalysisAgent()
    result = agent.evaluate_risk(investment_state)
    
    assert result["stock_symbol"] == "AAPL"
    assert "volatility" in result["metrics"]
    assert len(result["messages"]) > 0

def test_evaluate_risk_no_returns(investment_state):
    """Test risk evaluation with no returns data."""
    agent = RiskAnalysisAgent()
    result = agent.evaluate_risk(investment_state)
    
    assert result["metrics"]["volatility"] == 0.0

def test_evaluate_risk_no_symbol(investment_state):
    """Test risk evaluation with no symbol."""
    investment_state["stock_symbol"] = ""
    agent = RiskAnalysisAgent()
    result = agent.evaluate_risk(investment_state)
    
    assert result["stock_symbol"] == ""
    assert "Error" in result["risk_analysis"]

@patch('openai.ChatCompletion.create')
def test_generate_analysis(mock_openai, investment_state):
    """Test analysis generation with OpenAI."""
    # Create a mock response object
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "Test analysis"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_openai.return_value = mock_response
    
    agent = StockAnalysisAgent()
    result = agent._generate_analysis(
        "AAPL",
        {"current_price": 100, "moving_avg_50": 95, "moving_avg_200": 90},
        []
    )
    
    assert result == "Test analysis"
    mock_openai.assert_called_once()

def test_calculate_metrics(sample_stock_data):
    """Test metrics calculation."""
    agent = StockAnalysisAgent()
    metrics = agent._calculate_metrics(sample_stock_data)
    
    assert "current_price" in metrics
    assert "moving_avg_50" in metrics
    assert "moving_avg_200" in metrics
    assert isinstance(metrics["current_price"], float)
    assert isinstance(metrics["moving_avg_50"], float)
    assert isinstance(metrics["moving_avg_200"], float) 