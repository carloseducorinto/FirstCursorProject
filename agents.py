from typing import Dict, Any, List, TypedDict, Annotated, Optional
from langgraph.graph import Graph, StateGraph
from openai import OpenAI
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from functools import lru_cache
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvestmentState(TypedDict):
    stock_symbol: str
    investment_goals: str
    risk_tolerance: str
    stock_analysis: str
    risk_analysis: str
    metrics: Dict[str, Any]
    messages: List[Dict[str, str]]
    stock_data: pd.DataFrame
    returns: pd.Series

class DataSource:
    """Base class for data sources."""
    def __init__(self):
        self._last_request_time = 0
        self._min_request_interval = 1.0  # Minimum seconds between requests

    def _rate_limit(self) -> None:
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def get_stock_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Get stock data from the source."""
        raise NotImplementedError

class YahooFinanceSource(DataSource):
    """Yahoo Finance data source."""
    def get_stock_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        try:
            self._rate_limit()
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            return hist if not hist.empty else None
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {str(e)}")
            return None

class AlphaVantageSource(DataSource):
    """Alpha Vantage data source."""
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def get_stock_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        try:
            self._rate_limit()
            # Map period to Alpha Vantage interval
            interval_map = {
                "1d": "TIME_SERIES_INTRADAY",
                "5d": "TIME_SERIES_DAILY",
                "1mo": "TIME_SERIES_DAILY",
                "3mo": "TIME_SERIES_DAILY",
                "6mo": "TIME_SERIES_DAILY",
                "1y": "TIME_SERIES_DAILY",
                "2y": "TIME_SERIES_DAILY",
                "5y": "TIME_SERIES_DAILY",
                "10y": "TIME_SERIES_DAILY"
            }
            
            function = interval_map.get(period, "TIME_SERIES_DAILY")
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full" if period in ["5y", "10y"] else "compact"
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if "Error Message" in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return None
                
            # Convert Alpha Vantage data to DataFrame
            time_series = data.get(f"Time Series ({'Daily' if function == 'TIME_SERIES_DAILY' else '1min'})", {})
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {str(e)}")
            return None

class StockAnalysisAgent:
    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        self.client = OpenAI()
        self.model_name = model_name
        self.data_sources = [
            YahooFinanceSource(),
            # Add your Alpha Vantage API key here
            AlphaVantageSource("OE11Q5QSWNR4L0CD")
        ]

    def _get_stock_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Try multiple data sources until successful."""
        for source in self.data_sources:
            try:
                data = source.get_stock_data(symbol, period)
                if data is not None and not data.empty:
                    logger.info(f"Successfully retrieved data from {source.__class__.__name__}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to get data from {source.__class__.__name__}: {str(e)}")
                continue
        
        logger.error(f"All data sources failed for {symbol}")
        return None

    def _calculate_metrics(self, hist: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Calculate stock metrics with validation."""
        try:
            if len(hist) < 1:
                logger.warning("Not enough data points for calculations")
                return None
                
            current_price = hist['Close'].iloc[-1]
            metrics = {"current_price": current_price}
            
            # Calculate moving averages if enough data is available
            if len(hist) >= 50:
                metrics["moving_avg_50"] = hist['Close'].rolling(window=50).mean().iloc[-1]
            else:
                metrics["moving_avg_50"] = current_price
                logger.warning("Not enough data for 50-day MA, using current price")
            
            if len(hist) >= 200:
                metrics["moving_avg_200"] = hist['Close'].rolling(window=200).mean().iloc[-1]
            else:
                metrics["moving_avg_200"] = current_price
                logger.warning("Not enough data for 200-day MA, using current price")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return None

    def _generate_analysis(self, symbol: str, metrics: Dict[str, float], messages: List[Dict[str, str]]) -> str:
        """Generate analysis using OpenAI."""
        prompt = f"""
        Analyze the following stock data for {symbol}:
        - Current Price: ${metrics['current_price']:.2f}
        - 50-day Moving Average: ${metrics['moving_avg_50']:.2f}
        - 200-day Moving Average: ${metrics['moving_avg_200']:.2f}
        
        Provide insights on:
        1. Price trends and momentum
        2. Technical indicators
        3. Recent performance
        4. Potential future outlook
        """
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            return f"Error generating analysis: {str(e)}"

    def _create_error_state(self, state: InvestmentState, error_msg: str) -> Dict[str, Any]:
        """Create an error state with proper structure."""
        messages = state.get("messages", [])
        messages.append({"role": "system", "content": error_msg})
        
        return {
            "stock_symbol": state.get("stock_symbol", ""),
            "investment_goals": state.get("investment_goals", ""),
            "risk_tolerance": state.get("risk_tolerance", ""),
            "stock_analysis": error_msg,
            "risk_analysis": "",
            "metrics": {},
            "messages": messages,
            "stock_data": pd.DataFrame(),
            "returns": pd.Series()
        }

    def analyze_stock_performance(self, state: InvestmentState) -> Dict[str, Any]:
        """Analyze stock performance and trends with proper error handling."""
        stock_symbol = state.get("stock_symbol")
        if not stock_symbol:
            logger.warning("No stock symbol provided")
            return self._create_error_state(state, "Error: No stock symbol provided")
            
        messages = state.get("messages", [])
        messages.append({"role": "system", "content": f"Starting stock analysis for {stock_symbol}"})
            
        try:
            # Try different periods with caching
            periods = ["1y", "6mo", "3mo", "1mo"]
            hist = None
            
            for period in periods:
                hist = self._get_stock_data(stock_symbol, period)
                if hist is not None:
                    logger.info(f"Successfully retrieved {stock_symbol} data for period {period}")
                    break
            
            if hist is None:
                logger.error(f"No data available for {stock_symbol}")
                return self._create_error_state(
                    state,
                    f"Error: No data available for stock symbol: {stock_symbol}. Please check the symbol and try again."
                )
            
            # Validate and calculate metrics
            metrics = self._calculate_metrics(hist)
            if not metrics:
                return self._create_error_state(
                    state,
                    f"Error: Failed to calculate metrics for {stock_symbol}"
                )
            
            # Generate analysis
            analysis_text = self._generate_analysis(stock_symbol, metrics, messages)
            
            return {
                "stock_symbol": stock_symbol,
                "investment_goals": state.get("investment_goals", ""),
                "risk_tolerance": state.get("risk_tolerance", ""),
                "stock_analysis": analysis_text,
                "risk_analysis": "",
                "metrics": metrics,
                "messages": messages,
                "stock_data": hist,
                "returns": hist['Close'].pct_change().dropna()
            }
            
        except Exception as e:
            logger.error(f"Error in stock analysis: {str(e)}")
            return self._create_error_state(state, f"Error: Failed to analyze stock {stock_symbol}: {str(e)}")

class RiskAnalysisAgent:
    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        self.client = OpenAI()
        self.model_name = model_name
        
    def evaluate_risk(self, state: InvestmentState) -> Dict[str, Any]:
        """Evaluate risk, diversification, and investment goals."""
        stock_symbol = state.get("stock_symbol")
        investment_goals = state.get("investment_goals", "")
        risk_tolerance = state.get("risk_tolerance", "moderate")
        
        if not stock_symbol:
            logger.warning("No stock symbol provided for risk analysis")
            return {
                "stock_symbol": "",
                "investment_goals": investment_goals,
                "risk_tolerance": risk_tolerance,
                "stock_analysis": "",
                "risk_analysis": "Error: No stock symbol provided",
                "metrics": {},
                "messages": [{"role": "system", "content": "Error: No stock symbol provided"}],
                "stock_data": pd.DataFrame(),
                "returns": pd.Series()
            }
            
        messages = state.get("messages", [])
        messages.append({"role": "system", "content": f"Starting risk analysis for {stock_symbol}"})
            
        try:
            # Calculate volatility from returns
            returns = state.get("returns", pd.Series())
            if returns.empty:
                logger.warning("No returns data available for risk analysis")
                volatility = 0.0
            else:
                volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
            
            messages.append({
                "role": "system",
                "content": f"Calculated annualized volatility: {volatility:.2%}"
            })
            
            # Prepare risk analysis prompt
            prompt = f"""
            Evaluate the following investment scenario:
            - Stock: {stock_symbol}
            - Investment Goals: {investment_goals}
            - Risk Tolerance: {risk_tolerance}
            - Annualized Volatility: {volatility:.2%}
            
            Provide analysis on:
            1. Risk assessment
            2. Portfolio diversification recommendations
            3. Alignment with investment goals
            4. Risk mitigation strategies
            """
            
            messages.append({"role": "user", "content": prompt})
            
            # Get the response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            
            analysis_text = response.choices[0].message.content
            
            # Add final analysis message
            messages.append({
                "role": "assistant",
                "content": analysis_text
            })
            
            return {
                "stock_symbol": stock_symbol,
                "investment_goals": investment_goals,
                "risk_tolerance": risk_tolerance,
                "stock_analysis": state.get("stock_analysis", ""),
                "risk_analysis": analysis_text,
                "metrics": {
                    **state.get("metrics", {}),
                    "volatility": volatility,
                    "risk_tolerance": risk_tolerance
                },
                "messages": messages,
                "stock_data": state.get("stock_data", pd.DataFrame()),
                "returns": returns
            }
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {str(e)}")
            return {
                "stock_symbol": stock_symbol,
                "investment_goals": investment_goals,
                "risk_tolerance": risk_tolerance,
                "stock_analysis": state.get("stock_analysis", ""),
                "risk_analysis": f"Error in risk analysis: {str(e)}",
                "metrics": state.get("metrics", {}),
                "messages": messages,
                "stock_data": state.get("stock_data", pd.DataFrame()),
                "returns": returns
            }

def create_investment_graph() -> Graph:
    """Create the LangGraph workflow for investment analysis."""
    # Initialize agents
    stock_agent = StockAnalysisAgent()
    risk_agent = RiskAnalysisAgent()
    
    # Define the graph with state schema
    workflow = StateGraph(InvestmentState)
    
    # Add nodes with unique names
    workflow.add_node("analyze_performance", stock_agent.analyze_stock_performance)
    workflow.add_node("evaluate_risk", risk_agent.evaluate_risk)
    
    # Define edges
    workflow.add_edge("analyze_performance", "evaluate_risk")
    
    # Set entry point
    workflow.set_entry_point("analyze_performance")
    
    # Set finish point
    workflow.set_finish_point("evaluate_risk")
    
    # Compile the graph
    return workflow.compile() 