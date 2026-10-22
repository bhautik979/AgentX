# ============================================================================
# LIVE DATA PROVIDERS - Integration with Free APIs
# Finnhub, Alpha Vantage, Numbeo, and more
# ============================================================================

import logging
from typing import Dict, List, Optional, Tuple
import aiohttp
import asyncio
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class APIConfig:
    """API Keys and Configurations"""
    
    # Finnhub (Stocks & Commodities)
    FINNHUB_API_KEY = "d4c5ct9r01qoua32jb9gd4c5ct9r01qoua32jba0"  # Get from https://finnhub.io
    FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
    
    # Alpha Vantage (Stocks & Forex)
    ALPHA_VANTAGE_API_KEY = "8F5PNHOG03L5A3RQ"  # Get from https://www.alphavantage.co
    ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
    
    # Numbeo (Cost of Living)
    NUMBEO_API_KEY = "YOUR_NUMBEO_KEY"  # Get from https://www.numbeo.com/api/
    NUMBEO_BASE_URL = "https://www.numbeo.com/api/v2"
    
    # News API (Web Search Alternative)
    NEWS_API_KEY = "9e46aebf2cf54bc1aef10e141995dd3e"  # Get from https://newsapi.org
    
    # Serpstack (Web Search - requires subscription)
    SERPSTACK_API_KEY = "de080a3d694e86cc9a524bc9bbe060a1"  # Get from https://serpstack.com


class LiveDataProvider(ABC):
    """Abstract base class for all data providers"""
    
    @abstractmethod
    async def get_data(self, **kwargs) -> Dict:
        """Fetch data from API"""
        pass


class FinnhubProvider(LiveDataProvider):
    """Finnhub API Provider for stocks and commodities"""
    
    def __init__(self, api_key: str = APIConfig.FINNHUB_API_KEY):
        self.api_key = api_key
        self.base_url = APIConfig.FINNHUB_BASE_URL

    async def get_data(self, symbol: str) -> Dict:
        """Unified data fetch entrypoint required by abstract base class."""
        return await self.get_stock_price(symbol)
    
    async def get_stock_price(self, symbol: str) -> Dict:
        """
        Get current stock price
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
        
        Returns:
            Dict with price data
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/quote"
                params = {
                    "symbol": symbol.upper(),
                    "token": self.api_key
                }
                
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if "error" in data:
                        logger.warning(f"Finnhub error: {data['error']}")
                        return {}
                    
                    return {
                        "symbol": symbol.upper(),
                        "price": data.get("c", 0),
                        "high": data.get("h", 0),
                        "low": data.get("l", 0),
                        "open": data.get("o", 0),
                        "previous_close": data.get("pc", 0),
                        "timestamp": datetime.now().isoformat(),
                        "source": "Finnhub"
                    }
        
        except Exception as e:
            logger.error(f"Failed to get stock price from Finnhub: {e}")
            return {}
    
    async def get_company_profile(self, symbol: str) -> Dict:
        """
        Get company profile information
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dict with company info
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/stock/profile2"
                params = {
                    "symbol": symbol.upper(),
                    "token": self.api_key
                }
                
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    return {
                        "symbol": symbol.upper(),
                        "name": data.get("name", ""),
                        "industry": data.get("finnhubIndustry", ""),
                        "web_url": data.get("weburl", ""),
                        "logo": data.get("logo", ""),
                        "country": data.get("country", ""),
                        "timestamp": datetime.now().isoformat()
                    }
        
        except Exception as e:
            logger.error(f"Failed to get company profile: {e}")
            return {}
    
    async def get_commodity_price(self, commodity: str) -> Dict:
        """
        Get commodity prices (oil, gold, etc.)
        
        Args:
            commodity: Commodity type (e.g., "OIL", "GOLD")
        
        Returns:
            Dict with commodity data
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Map commodity names to Finnhub symbols
                commodity_symbols = {
                    "OIL": "CRUD.F",
                    "GOLD": "XAUUSD",
                    "SILVER": "XAGUSD",
                    "COPPER": "HG=F"
                }
                
                symbol = commodity_symbols.get(commodity.upper())
                if not symbol:
                    return {}
                
                return await self.get_stock_price(symbol)
        
        except Exception as e:
            logger.error(f"Failed to get commodity price: {e}")
            return {}


class AlphaVantageProvider(LiveDataProvider):
    """Alpha Vantage API Provider for stocks"""
    
    def __init__(self, api_key: str = APIConfig.ALPHA_VANTAGE_API_KEY):
        self.api_key = api_key
        self.base_url = APIConfig.ALPHA_VANTAGE_BASE_URL

    async def get_data(self, symbol: str) -> Dict:
        """Unified data fetch entrypoint required by abstract base class."""
        return await self.get_stock_price(symbol)
    
    async def get_stock_price(self, symbol: str) -> Dict:
        """
        Get current stock price from Alpha Vantage
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dict with price data
        """
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol.upper(),
                    "apikey": self.api_key
                }
                
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    
                    quote = data.get("Global Quote", {})
                    if not quote:
                        logger.warning(f"No data for {symbol}")
                        return {}
                    
                    return {
                        "symbol": symbol.upper(),
                        "price": float(quote.get("05. price", 0)),
                        "high": float(quote.get("03. high", 0)),
                        "low": float(quote.get("04. low", 0)),
                        "open": float(quote.get("02. open", 0)),
                        "change": float(quote.get("09. change", 0)),
                        "change_percent": quote.get("10. change percent", ""),
                        "timestamp": datetime.now().isoformat(),
                        "source": "Alpha Vantage"
                    }
        
        except Exception as e:
            logger.error(f"Failed to get stock price from Alpha Vantage: {e}")
            return {}
    
    async def get_intraday_data(self, symbol: str, interval: str = "5min") -> Dict:
        """
        Get intraday trading data
        
        Args:
            symbol: Stock symbol
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
        
        Returns:
            Dict with time series data
        """
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "function": f"TIME_SERIES_INTRADAY",
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "apikey": self.api_key
                }
                
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    
                    return {
                        "symbol": symbol.upper(),
                        "data": data,
                        "timestamp": datetime.now().isoformat()
                    }
        
        except Exception as e:
            logger.error(f"Failed to get intraday data: {e}")
            return {}


class NumbeoProvider(LiveDataProvider):
    """Numbeo API Provider for cost of living data"""
    
    def __init__(self, api_key: str = APIConfig.NUMBEO_API_KEY):
        self.api_key = api_key
        self.base_url = APIConfig.NUMBEO_BASE_URL

    async def get_data(self, symbol: str) -> Dict:
        """Unified data fetch entrypoint required by abstract base class."""
        return await self.get_cost_of_living(symbol)
    
    async def get_cost_of_living(self, city: str, country: str) -> Dict:
        """
        Get cost of living data for a city
        
        Args:
            city: City name
            country: Country name
        
        Returns:
            Dict with cost of living data
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/city"
                params = {
                    "query": f"{city}, {country}",
                    "api_key": self.api_key
                }
                
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if data.get("status") != "0":
                        logger.warning(f"Numbeo API error: {data}")
                        return {}
                    
                    return {
                        "city": city,
                        "country": country,
                        "groceries": data.get("groceries", {}),
                        "restaurants": data.get("restaurants", {}),
                        "housing": data.get("housing", {}),
                        "utilities": data.get("utilities", {}),
                        "transportation": data.get("transportation", {}),
                        "timestamps": datetime.now().isoformat()
                    }
        
        except Exception as e:
            logger.error(f"Failed to get cost of living: {e}")
            return {}


class WebSearchProvider(LiveDataProvider):
    """Web Search Provider - Multiple backends"""
    async def get_data(self, query: str) -> List[Dict]:
        """Unified entrypoint: defaults to news search."""
        return await self.search_news(query)
    
    async def search_news(self, query: str, api_key: str = APIConfig.NEWS_API_KEY) -> List[Dict]:
        """
        Search financial news using NewsAPI
        
        Args:
            query: Search query
            api_key: News API key
        
        Returns:
            List of news articles
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": query,
                    "sortBy": "publishedAt",
                    "apiKey": api_key
                }
                
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if not data.get("articles"):
                        return []
                    
                    return [
                        {
                            "title": article.get("title", ""),
                            "description": article.get("description", ""),
                            "url": article.get("url", ""),
                            "source": article.get("source", {}).get("name", ""),
                            "published_at": article.get("publishedAt", ""),
                            "image_url": article.get("urlToImage", "")
                        }
                        for article in data["articles"][:5]  # Top 5
                    ]
        
        except Exception as e:
            logger.error(f"Failed to search news: {e}")
            return []
    
    async def search_general(self, query: str) -> List[Dict]:
        """
        General web search (using Google Custom Search or Serpstack)
        
        Args:
            query: Search query
        
        Returns:
            List of search results
        """
        try:
            # Try multiple providers
            results = []
            
            # Option 1: Using Serpstack (requires API key)
            results.extend(await self._serpstack_search("fiance realted artical about:" + query))
            
            if not results:
                # Option 2: Fallback - parse from news API
                results = await self.search_news("fiance realted artical about:" + query)
            
            return results
        
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    async def _serpstack_search(self, query: str) -> List[Dict]:
        """Search using Serpstack API"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "http://api.serpstack.com/search"
                params = {
                    "access_key": APIConfig.SERPSTACK_API_KEY,
                    "query": query
                }
                
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if "organic_results" not in data:
                        return []
                    
                    return [
                        {
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "snippet": result.get("snippet", ""),
                            "position": result.get("position", 0)
                        }
                        for result in data["organic_results"][:5]
                    ]
        
        except Exception as e:
            logger.warning(f"Serpstack search failed: {e}")
            return []


class DataAggregator:
    """Aggregates data from multiple providers"""
    
    def __init__(self):
        self.finnhub = FinnhubProvider()
        self.alpha_vantage = AlphaVantageProvider()
        self.numbeo = NumbeoProvider()
        self.web_search = WebSearchProvider()
    
    async def get_stock_data(self, symbol: str) -> Dict:
        """
        Get stock data from best available source
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Stock data dict
        """
        # Try Finnhub first (more reliable)
        data = await self.finnhub.get_stock_price(symbol)
        
        if not data:
            # Fallback to Alpha Vantage
            data = await self.alpha_vantage.get_stock_price(symbol)
        
        # Add company profile if we have data
        if data:
            profile = await self.finnhub.get_company_profile(symbol)
            data.update(profile)
        
        return data
    
    async def get_commodity_data(self, commodity: str) -> Dict:
        """Get commodity price data"""
        return await self.finnhub.get_commodity_price(commodity)
    
    async def get_cost_of_living(self, city: str, country: str) -> Dict:
        """Get cost of living data"""
        return await self.numbeo.get_cost_of_living(city, country)
    
    async def search_financial_news(self, query: str) -> List[Dict]:
        """Search for financial news"""
        return await self.web_search.search_news(query)
    
    async def web_search(self, query: str) -> List[Dict]:
        """General web search"""
        return await self.web_search.search_general(query)


__all__ = [
    'APIConfig',
    'FinnhubProvider',
    'AlphaVantageProvider',
    'NumbeoProvider',
    'WebSearchProvider',
    'DataAggregator'
]
