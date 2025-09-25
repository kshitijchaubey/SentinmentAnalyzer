python
import json
import mlflow
import yfinance as yf
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import requests

from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_community.tools import BraveSearch
from langchain_core.tools import Tool

from config import Config

# Configure MLflow
mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

# Pydantic model for structured output
class MarketSentiment(BaseModel):
    company_name: str = Field(description="The name of the company")
    stock_code: str = Field(description="The stock ticker symbol")
    newsdesc: str = Field(description="Summary of the news analyzed")
    sentiment: str = Field(description="Overall sentiment: Positive, Negative, or Neutral")
    people_names: List[str] = Field(description="List of people mentioned in the news")
    places_names: List[str] = Field(description="List of places mentioned in the news")
    other_companies_referred: List[str] = Field(description="List of other companies mentioned")
    related_industries: List[str] = Field(description="List of related industries")
    market_implications: str = Field(description="Potential market implications")
    confidence_score: float = Field(description="Confidence score of the analysis (0.0 to 1.0)")

class StockSymbolLookup:
    """Tool to get stock symbol from company name"""
    
    @staticmethod
    def get_stock_symbol(company_name: str) -> str:
        with mlflow.start_span(name="stock_symbol_lookup") as span:
            span.set_tag("company_name", company_name)
            
            # Try common stock mappings first
            common_symbols = {
                "apple": "AAPL",
                "microsoft": "MSFT",
                "google": "GOOGL",
                "amazon": "AMZN",
                "tesla": "TSLA",
                "nvidia": "NVDA",
                "meta": "META",
                "netflix": "NFLX",
                "apple inc": "AAPL",
                "microsoft corporation": "MSFT",
                "alphabet": "GOOGL",
                "amazon.com": "AMZN",
                "tesla inc": "TSLA",
                "nvidia corporation": "NVDA",
                "meta platforms": "META"
            }
            
            lower_name = company_name.lower()
            if lower_name in common_symbols:
                symbol = common_symbols[lower_name]
                span.set_tag("symbol", symbol)
                span.set_tag("source", "static_mapping")
                return symbol
            
            # Fallback to Yahoo Finance search
            try:
                search_results = yf.Tickers(company_name)
                if search_results.symbols:
                    symbol = search_results.symbols[0]
                    span.set_tag("symbol", symbol)
                    span.set_tag("source", "yahoo_finance")
                    return symbol
            except:
                pass
            
            # If no symbol found, return the input as is (for LLM to handle)
            span.set_tag("symbol", "NOT_FOUND")
            return company_name

class NewsFetcher:
    """Tool to fetch recent news for a company"""
    
    @staticmethod
    def fetch_news(stock_symbol: str, max_results: int = 5) -> List[str]:
        with mlflow.start_span(name="news_fetching") as span:
            span.set_tag("stock_symbol", stock_symbol)
            span.set_tag("max_results", max_results)
            
            news_items = []
            
            # Method 1: Use Brave Search if API key is available
            if Config.BRAVE_API_KEY:
                try:
                    brave = BraveSearch.from_api_key(api_key=Config.BRAVE_API_KEY, search_kwargs={"count": max_results})
                    query = f"{stock_symbol} stock news latest"
                    results = brave.run(query)
                    
                    if results:
                        # Parse the results (Brave returns a string with multiple results)
                        lines = results.split('\n')
                        for line in lines[:max_results]:
                            if line.strip():
                                news_items.append(line.strip())
                        span.set_tag("source", "brave_search")
                except Exception as e:
                    span.set_tag("error", str(e))
            
            # Method 2: Fallback to Yahoo Finance news
            if not news_items:
                try:
                    ticker = yf.Ticker(stock_symbol)
                    news = ticker.news
                    for item in news[:max_results]:
                        if 'title' in item:
                            news_items.append(item['title'])
                            if 'summary' in item and item['summary']:
                                news_items.append(item['summary'])
                    span.set_tag("source", "yahoo_finance")
                except Exception as e:
                    span.set_tag("error", str(e))
            
            # Method 3: Fallback to manual search query
            if not news_items:
                try:
                    query = f"{stock_symbol} company latest news financial"
                    # Simple web search fallback (very basic)
                    news_items = [f"Recent developments for {stock_symbol}", 
                                 f"Market updates for {stock_symbol}"]
                    span.set_tag("source", "fallback")
                except Exception as e:
                    span.set_tag("error", str(e))
            
            span.set_tag("news_count", len(news_items))
            return news_items

class MarketSentimentAnalyzer:
    def __init__(self):
        # Initialize Azure OpenAI LLM
        self.llm = AzureChatOpenAI(
            azure_deployment=Config.AZURE_DEPLOYMENT_NAME,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            temperature=0.1
        )
        
        # Initialize output parser
        self.output_parser = JsonOutputParser(pydantic_object=MarketSentiment)
        
        # Define the sentiment analysis prompt
        self.sentiment_prompt = PromptTemplate(
            template="""Analyze the following news articles about {company_name} ({stock_symbol}) and provide a comprehensive market sentiment analysis.

News Articles:
{news_articles}

Please analyze this information and provide:
1. Overall sentiment (Positive, Negative, Neutral)
2. Key people mentioned
3. Important places referenced
4. Other companies mentioned
5. Related industries
6. Potential market implications
7. Confidence score for your analysis

Format your response as JSON with the following structure:
{format_instructions}

Be concise but thorough in your analysis.""",
            input_variables=["company_name", "stock_symbol", "news_articles"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        # Create the sentiment analysis chain
        self.sentiment_chain = LLMChain(
            llm=self.llm,
            prompt=self.sentiment_prompt,
            output_parser=self.output_parser,
            output_key="sentiment_analysis"
        )
        
        # Create the overall sequential chain
        self.overall_chain = SequentialChain(
            chains=[],
            input_variables=["company_name"],
            output_variables=["sentiment_analysis"],
            verbose=True
        )

    def analyze_sentiment(self, company_name: str) -> Dict[str, Any]:
        """Main method to analyze market sentiment for a company"""
        
        with mlflow.start_run(run_name=f"sentiment_analysis_{company_name}") as run:
            # Log input parameters
            mlflow.log_param("company_name", company_name)
            
            try:
                # Step 1: Get stock symbol
                with mlflow.start_span(name="stock_symbol_extraction"):
                    stock_symbol = StockSymbolLookup.get_stock_symbol(company_name)
                    mlflow.log_param("stock_symbol", stock_symbol)
                
                # Step 2: Fetch news
                with mlflow.start_span(name="news_fetching"):
                    news_articles = NewsFetcher.fetch_news(stock_symbol)
                    mlflow.log_param("news_count", len(news_articles))
                    mlflow.log_text("\n".join(news_articles), "news_articles.txt")
                
                # Step 3: Analyze sentiment
                with mlflow.start_span(name="sentiment_analysis"):
                    # Prepare input for the chain
                    input_data = {
                        "company_name": company_name,
                        "stock_symbol": stock_symbol,
                        "news_articles": "\n".join(news_articles[:5])  # Use first 5 articles
                    }
                    
                    # Log the prompt
                    prompt_text = self.sentiment_prompt.format(**input_data)
                    mlflow.log_text(prompt_text, "prompt.txt")
                    
                    # Execute the analysis
                    result = self.sentiment_chain.invoke(input_data)
                    
                    # Log the result
                    mlflow.log_metric("confidence_score", result["sentiment_analysis"].get("confidence_score", 0.0))
                    mlflow.log_dict(result["sentiment_analysis"], "sentiment_analysis.json")
                    
                    return result["sentiment_analysis"]
                    
            except Exception as e:
                mlflow.log_param("error", str(e))
                raise e

# Example usage and test function
def main():
    # Initialize the analyzer
    analyzer = MarketSentimentAnalyzer()
    
    # Test with a sample company
    companies = ["Microsoft", "Apple", "Tesla", "Nvidia"]
    
    for company in companies:
        try:
            print(f"\nAnalyzing sentiment for {company}...")
            result = analyzer.analyze_sentiment(company)
            
            print(f"\nResults for {company}:")
            print(f"Sentiment: {result.get('sentiment', 'Unknown')}")
            print(f"Confidence Score: {result.get('confidence_score', 0.0):.2f}")
            print(f"Market Implications: {result.get('market_implications', '')}")
            print(f"People Mentioned: {', '.join(result.get('people_names', []))}")
            print(f"Other Companies: {', '.join(result.get('other_companies_referred', []))}")
            
        except Exception as e:
            print(f"Error analyzing {company}: {str(e)}")

if __name__ == "__main__":
    main()
