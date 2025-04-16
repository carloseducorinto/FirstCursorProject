from typing import Dict, Any, List, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import logging
from openai import OpenAI
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SECAnalysisAgent:
    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        self.client = OpenAI()
        self.model_name = model_name
        self._last_request_time = 0
        self._min_request_interval = 1.0  # Minimum seconds between requests
        self.sec_base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        self.sec_archive_url = "https://www.sec.gov/Archives/edgar/data"

    def _rate_limit(self) -> None:
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    @lru_cache(maxsize=100)
    def _get_company_cik(self, symbol: str) -> Optional[str]:
        """Get CIK number for a company using its stock symbol."""
        try:
            self._rate_limit()
            params = {
                "action": "getcompany",
                "CIK": symbol,
                "output": "atom"
            }
            response = requests.get(self.sec_base_url, params=params)
            soup = BeautifulSoup(response.content, 'xml')
            cik = soup.find('cik')
            return cik.text if cik else None
        except Exception as e:
            logger.error(f"Error getting CIK for {symbol}: {str(e)}")
            return None

    def _get_filing_urls(self, cik: str, filing_type: str, count: int = 5) -> List[Dict[str, str]]:
        """Get URLs for recent SEC filings."""
        try:
            self._rate_limit()
            params = {
                "action": "getcompany",
                "CIK": cik,
                "type": filing_type,
                "count": count,
                "output": "atom"
            }
            response = requests.get(self.sec_base_url, params=params)
            soup = BeautifulSoup(response.content, 'xml')
            
            filings = []
            for entry in soup.find_all('entry'):
                filing = {
                    'date': entry.find('updated').text,
                    'title': entry.find('title').text,
                    'link': entry.find('link')['href']
                }
                filings.append(filing)
            return filings
        except Exception as e:
            logger.error(f"Error getting filing URLs: {str(e)}")
            return []

    def _extract_filing_content(self, filing_url: str) -> Optional[str]:
        """Extract content from SEC filing."""
        try:
            self._rate_limit()
            response = requests.get(filing_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract the main content (this might need adjustment based on actual filing structure)
            content = soup.find('div', {'class': 'document'})
            return content.get_text() if content else None
        except Exception as e:
            logger.error(f"Error extracting filing content: {str(e)}")
            return None

    def _analyze_filing(self, content: str, filing_type: str) -> str:
        """Analyze SEC filing content using OpenAI."""
        try:
            prompt = f"""
            Analyze the following {filing_type} SEC filing and provide insights on:
            1. Key financial metrics and trends
            2. Operational highlights and challenges
            3. Risk factors and mitigation strategies
            4. Management discussion and analysis
            5. Forward-looking statements
            
            Filing Content:
            {content[:8000]}  # Limit content length for API
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in SEC filings."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error analyzing filing: {str(e)}")
            return f"Error analyzing filing: {str(e)}"

    def analyze_sec_filings(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze recent SEC filings for a company."""
        stock_symbol = state.get("stock_symbol")
        if not stock_symbol:
            logger.warning("No stock symbol provided")
            return {
                "sec_analysis": "Error: No stock symbol provided",
                "filing_summaries": [],
                "messages": state.get("messages", [])
            }
        
        messages = state.get("messages", [])
        messages.append({"role": "system", "content": f"Starting SEC analysis for {stock_symbol}"})
        
        try:
            # Get company CIK
            cik = self._get_company_cik(stock_symbol)
            if not cik:
                return {
                    "sec_analysis": f"Error: Could not find CIK for {stock_symbol}",
                    "filing_summaries": [],
                    "messages": messages
                }
            
            # Get recent filings
            filing_types = ["10-K", "10-Q", "8-K"]
            filing_summaries = []
            
            for filing_type in filing_types:
                filings = self._get_filing_urls(cik, filing_type)
                for filing in filings:
                    content = self._extract_filing_content(filing['link'])
                    if content:
                        analysis = self._analyze_filing(content, filing_type)
                        filing_summaries.append({
                            "type": filing_type,
                            "date": filing['date'],
                            "title": filing['title'],
                            "analysis": analysis
                        })
            
            # Generate overall analysis
            sec_analysis = self._generate_overall_analysis(filing_summaries)
            
            return {
                "sec_analysis": sec_analysis,
                "filing_summaries": filing_summaries,
                "messages": messages
            }
            
        except Exception as e:
            logger.error(f"Error in SEC analysis: {str(e)}")
            return {
                "sec_analysis": f"Error analyzing SEC filings: {str(e)}",
                "filing_summaries": [],
                "messages": messages
            }

    def _generate_overall_analysis(self, filing_summaries: List[Dict[str, Any]]) -> str:
        """Generate overall analysis from multiple filings."""
        try:
            prompt = """
            Based on the following SEC filing summaries, provide a comprehensive analysis:
            1. Overall financial health and trends
            2. Key operational developments
            3. Major risks and their implications
            4. Management's outlook and strategy
            5. Investment implications
            
            Filing Summaries:
            """
            
            for summary in filing_summaries:
                prompt += f"\n\n{summary['type']} - {summary['date']}\n{summary['analysis']}"
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial analyst providing investment insights."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating overall analysis: {str(e)}")
            return f"Error generating overall analysis: {str(e)}" 