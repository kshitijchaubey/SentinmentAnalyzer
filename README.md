markdown
# Real-Time Market Sentiment Analyzer

A LangChain-powered pipeline that analyzes market sentiment for companies by fetching recent news and processing it through Azure OpenAI GPT-4o.

## Features

- Company name to stock symbol conversion
- Real-time news fetching from multiple sources
- Sentiment analysis using Azure OpenAI GPT-4o
- Structured JSON output with comprehensive analysis
- MLflow integration for tracing and monitoring

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt

2.	**Environment Variables:**
Create a .env file with your credentials:
env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_DEPLOYMENT_NAME=gpt-4o

# MLflow (optional)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=market-sentiment-analysis

# Brave Search (optional)
BRAVE_API_KEY=your_brave_api_key

3.	**Run MLflow UI (optional):**
bash
mlflow ui
Usage
python
from sentiment_analyzer import MarketSentimentAnalyzer

# Initialize analyzer
analyzer = MarketSentimentAnalyzer()

# Analyze sentiment for a company
result = analyzer.analyze_sentiment("Microsoft")
print(result)

Sample Output
json
{
  "company_name": "Microsoft Corporation",
  "stock_code": "MSFT",
  "newsdesc": "Microsoft reports strong quarterly earnings with cloud revenue growth",
  "sentiment": "Positive",
  "people_names": ["Satya Nadella", "Amy Hood"],
  "places_names": ["Redmond", "Washington"],
  "other_companies_referred": ["Azure", "LinkedIn", "GitHub"],
  "related_industries": ["Cloud Computing", "Software", "Technology"],
  "market_implications": "Positive earnings may lead to stock price increase and investor confidence",
  "confidence_score": 0.85
}
Configuration
•	LLM Model: Configured to use Azure OpenAI GPT-4o
•	News Sources: Brave Search (primary), Yahoo Finance (fallback)
•	Output Format: Structured JSON with Pydantic validation
•	Monitoring: MLflow for tracing and prompt debugging
Bonus Features
•	Multiple news source fallbacks
•	Comprehensive error handling
•	MLflow integration for observability
•	Configurable through environment variables
text

## 5. .env.example
```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_DEPLOYMENT_NAME=gpt-4o

# MLflow Configuration (optional)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=market-sentiment-analysis

# Brave Search API (optional)
BRAVE_API_KEY=your_brave_api_key_here
Key Features Implemented:
1.	Input Handling: Accepts company names and converts to stock symbols
2.	Stock Symbol Extraction: Uses static mapping + Yahoo Finance fallback
3.	News Fetching: Integrates Brave Search and Yahoo Finance
4.	Sentiment Analysis: Uses Azure OpenAI with structured output parsing
5.	MLflow Integration: Comprehensive tracing and monitoring
6.	Error Handling: Robust fallback mechanisms
7.	Structured Output: Pydantic models for consistent JSON formatting
To Run:
1.	Install dependencies: pip install -r requirements.txt
2.	Set up your .env file with Azure OpenAI credentials
3.	Run: python sentiment_analyzer.py
The system will analyze Microsoft as a test case and provide structured sentiment analysis with MLflow tracing enabled.


