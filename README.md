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
pip install -r requirements.txt ```

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

