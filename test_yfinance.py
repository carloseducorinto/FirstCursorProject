import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
import time
import os
import shutil
import tempfile
import functools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_all_caches() -> None:
    """
    Clear all caches in the application, including:
    - yfinance cache
    - pandas cache
    - Python's functools cache
    - Temporary files
    """
    logger.info("Clearing all caches...")
    
    try:
        # Clear yfinance cache
        yf.pdr_override()
        logger.info("Cleared yfinance cache")
        
        # Clear pandas cache
        pd.core.common._maybe_cache_info.clear()
        pd.core.common._maybe_cache_warn.clear()
        logger.info("Cleared pandas cache")
        
        # Clear functools cache
        for func in [obj for obj in globals().values() if callable(obj)]:
            if hasattr(func, 'cache_clear'):
                func.cache_clear()
        logger.info("Cleared functools cache")
        
        # Clear temporary files
        temp_dir = tempfile.gettempdir()
        for item in os.listdir(temp_dir):
            if item.startswith('yfinance_') or item.startswith('pandas_'):
                try:
                    item_path = os.path.join(temp_dir, item)
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    logger.warning(f"Could not remove {item}: {str(e)}")
        logger.info("Cleared temporary files")
        
        # Clear memory cache
        import gc
        gc.collect()
        logger.info("Cleared memory cache")
        
    except Exception as e:
        logger.error(f"Error clearing caches: {str(e)}")

def test_stock_data_retrieval(symbol: str, period: str = "1y") -> None:
    """
    Test the retrieval of stock data from Yahoo Finance.
    
    Args:
        symbol (str): Stock symbol to test
        period (str): Time period for data retrieval (default: "1y")
    """
    logger.info(f"Testing data retrieval for {symbol} over {period} period")
    
    try:
        # Create Ticker object
        stock = yf.Ticker(symbol)
        
        # Get stock info
        info = stock.info
        logger.info(f"Stock Info: {info.get('longName', 'N/A')} ({symbol})")
        logger.info(f"Current Price: ${info.get('currentPrice', 'N/A')}")
        logger.info(f"Market Cap: ${info.get('marketCap', 'N/A'):,.2f}")
        
        # Get historical data
        start_time = time.time()
        hist = stock.history(period=period)
        end_time = time.time()
        
        # Log data retrieval metrics
        logger.info(f"Data retrieval took {end_time - start_time:.2f} seconds")
        logger.info(f"Retrieved {len(hist)} data points")
        
        if not hist.empty:
            # Display key metrics
            logger.info("\nKey Metrics:")
            logger.info(f"First Date: {hist.index[0]}")
            logger.info(f"Last Date: {hist.index[-1]}")
            logger.info(f"First Close: ${hist['Close'].iloc[0]:.2f}")
            logger.info(f"Last Close: ${hist['Close'].iloc[-1]:.2f}")
            logger.info(f"Average Volume: {hist['Volume'].mean():,.0f}")
            
            # Calculate and display moving averages
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            hist['MA200'] = hist['Close'].rolling(window=200).mean()
            
            logger.info(f"50-day MA: ${hist['MA50'].iloc[-1]:.2f}")
            logger.info(f"200-day MA: ${hist['MA200'].iloc[-1]:.2f}")
            
            # Check for data quality
            missing_data = hist.isnull().sum()
            if missing_data.any():
                logger.warning("\nMissing Data Points:")
                for col, count in missing_data[missing_data > 0].items():
                    logger.warning(f"{col}: {count} missing values")
            
            # Check for data consistency
            if len(hist) < 200:
                logger.warning(f"Warning: Less than 200 data points available for {period} period")
            
        else:
            logger.error("No data retrieved from Yahoo Finance")
            
    except Exception as e:
        logger.error(f"Error retrieving data for {symbol}: {str(e)}")

def test_multiple_symbols(symbols: list, period: str = "1y") -> None:
    """
    Test data retrieval for multiple stock symbols.
    
    Args:
        symbols (list): List of stock symbols to test
        period (str): Time period for data retrieval
    """
    logger.info(f"\nTesting multiple symbols: {', '.join(symbols)}")
    
    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        test_stock_data_retrieval(symbol, period)
        time.sleep(1)  # Add delay between requests

if __name__ == "__main__":
    # Clear all caches before starting tests
    clear_all_caches()
    
    # Test popular stocks
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    # Test with different periods
    periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]
    
    logger.info("Starting Yahoo Finance Connectivity Test")
    logger.info("="*50)
    
    # Test each period for the first symbol
    logger.info("\nTesting different time periods for AAPL:")
    for period in periods:
        logger.info(f"\nTesting period: {period}")
        test_stock_data_retrieval("AAPL", period)
        time.sleep(1)
    
    # Test multiple symbols with 1-year period
    test_multiple_symbols(test_symbols, "1y")
    
    # Clear caches after tests
    clear_all_caches()
    
    logger.info("\nTest completed") 