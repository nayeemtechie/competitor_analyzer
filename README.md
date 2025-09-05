# requirements.txt
# Core dependencies
aiohttp>=3.8.0
beautifulsoup4>=4.11.0
pyyaml>=6.0
python-dateutil>=2.8.0

# LLM providers
openai>=1.0.0
anthropic>=0.18.0

# Data processing
pandas>=1.5.0
numpy>=1.24.0

# Report generation
reportlab>=4.0.0
python-docx>=0.8.11

# Optional dependencies for enhanced functionality
# Uncomment as needed:

# Advanced data analysis
# matplotlib>=3.6.0
# seaborn>=0.12.0

# PDF processing
# PyPDF2>=3.0.0

# Excel file processing
# openpyxl>=3.1.0

# Text processing and NLP
# textstat>=0.7.0
# nltk>=3.8

# HTTP session management
# requests>=2.28.0

# Environment variable management
# python-dotenv>=1.0.0

# README.md
# Ecommerce Search Competitor Analysis System

A comprehensive, modular competitor analysis system specifically designed for ecommerce search SaaS providers. Built to extend existing Search Intel architecture with advanced competitive intelligence capabilities.

## 🎯 Overview

This system analyzes competitors like Algolia, Constructor.io, Bloomreach, Elasticsearch, and Coveo across multiple dimensions:

- **Website Analysis**: Content scraping, feature extraction, pricing analysis
- **Funding Intelligence**: Crunchbase integration, investment tracking
- **Hiring Intelligence**: Job posting analysis from multiple sources
- **Media Monitoring**: News mentions, sentiment analysis
- **Technical Intelligence**: GitHub activity, technology stack analysis
- **R&D Intelligence**: Patent analysis for innovation insights

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the system files
# Ensure you have Python 3.8+ installed

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Configuration

```bash
# Create initial configuration
python main_competitor.py --update-config

# Or validate existing configuration
python main_competitor.py --validate-config

# List configured competitors
python main_competitor.py --list-competitors
```

### 3. Run Analysis

```bash
# Analyze all configured competitors
python main_competitor.py

# Analyze specific competitors
python main_competitor.py --competitors "Algolia" "Constructor.io"

# Generate specific report format
python main_competitor.py --format pdf

# Comprehensive analysis
python main_competitor.py --depth comprehensive
```

## 📋 Requirements

### Required Environment Variables

```bash
# LLM API Keys (choose one or more)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
PERPLEXITY_API_KEY=your_perplexity_key
```

### Optional Environment Variables

```bash
# Enhanced data collection (optional but recommended)
CRUNCHBASE_API_KEY=your_crunchbase_key
GITHUB_TOKEN=your_github_token
NEWS_API_KEY=your_news_api_key

# Social media APIs (if available)
TWITTER_BEARER_TOKEN=your_twitter_token
LINKEDIN_ACCESS_TOKEN=your_linkedin_token
```

## 🏗️ Architecture

### Directory Structure

```
src/competitor/
├── models.py              # Data schemas and models
├── config.py              # Configuration management
├── scraper.py             # Web scraping engine
├── analyzer.py            # Main orchestrator
├── collectors/            # Data collection modules
│   ├── base.py           # Base classes
│   ├── website.py        # Website analysis
│   ├── funding.py        # Funding data
│   ├── jobs.py           # Job postings
│   ├── news.py           # News mentions
│   ├── social.py         # Social media
│   ├── github_activity.py # GitHub analysis
│   └── patents.py        # Patent intelligence
├── analysis/             # AI-powered analysis
│   ├── company.py        # Company analysis
│   ├── features.py       # Feature comparison
│   ├── market.py         # Market positioning
│   ├── pricing.py        # Pricing analysis
│   ├── threats.py        # Threat assessment
│   └── innovation.py     # Innovation tracking
└── reports/              # Report generation
    ├── pdf_reports.py    # PDF generation
    ├── docx_reports.py   # Word documents
    └── json_reports.py   # Data exports
```

### Integration with Existing Systems

```python
# In your existing main.py
from competitor.analyzer import run_competitor_analysis

# Run competitive analysis
competitor_files = await run_competitor_analysis(
    competitor_names=['Algolia', 'Constructor.io']
)
```

## ⚙️ Configuration

### Main Configuration File (competitor_config.yaml)

```yaml
llm:
  provider: "openai"  # or "anthropic", "perplexity"
  models:
    analysis: "gpt-4o"
    summary: "gpt-4o-mini"
    comparison: "sonar-pro"

competitors:
  - name: "Algolia"
    website: "https://www.algolia.com"
    competitive_threat: "high"
    priority: "high"

data_sources:
  company_websites: true
  funding_data: 
    enabled: true
  job_boards:
    enabled: true
    sources: ["linkedin", "glassdoor", "indeed"]
  
output:
  formats: ["pdf", "json"]
  output_dir: "competitor_reports"
```

### LLM Provider Configuration

Switch between LLM providers easily:

```yaml
# OpenAI
llm:
  provider: "openai"
  models:
    analysis: "gpt-4o"
    summary: "gpt-4o-mini"

# Anthropic Claude
llm:
  provider: "anthropic"
  models:
    analysis: "claude-3-sonnet-20240229"
    summary: "claude-3-haiku-20240307"

# Perplexity
llm:
  provider: "perplexity"
  models:
    analysis: "sonar-pro"
    summary: "sonar-medium"
```

## 📊 Output Formats

### 1. PDF Reports
- Executive summaries for C-level executives
- Detailed competitor profiles
- Threat assessment matrices
- Strategic recommendations

### 2. Word Documents  
- Collaborative editing format
- Detailed analysis sections
- Charts and tables
- Customizable templates

### 3. JSON Data Exports
- Machine-readable data
- API integration ready
- Complete datasets
- Metric calculations

## 🔧 Advanced Usage

### Custom Analysis Depth

```bash
# Basic analysis (key metrics only)
python main_competitor.py --depth basic

# Standard analysis (default)
python main_competitor.py --depth standard

# Comprehensive analysis (deep dive)
python main_competitor.py --depth comprehensive
```

### Specific Data Sources

Configure which data sources to use:

```yaml
data_sources:
  company_websites: true
  funding_data: 
    enabled: true
    sources: ["crunchbase"]
  job_boards:
    enabled: true
    sources: ["linkedin", "glassdoor"]
    max_jobs_per_company: 20
  news_sources:
    enabled: true
    days_back: 90
  social_media:
    enabled: true
    platforms: ["linkedin", "twitter", "github"]
  github_repos:
    enabled: true
    analyze_public_repos: true
  patent_databases:
    enabled: false  # Requires specific API access
```

### Custom Output Settings

```yaml
output:
  formats: ["pdf", "docx", "json"]
  output_dir: "competitor_reports"
  pdf:
    include_executive_summary: true
    include_charts: true
  json:
    pretty_print: true
    include_raw_data: false
```

## 🤖 LLM Integration

### Multiple Provider Support

The system supports multiple LLM providers with easy switching:

- **OpenAI**: GPT-4, GPT-3.5 models
- **Anthropic**: Claude 3 models
- **Perplexity**: Sonar models for web-enhanced analysis

### Custom Prompts

Prompts are modular and customizable for different analysis types:

- Company analysis prompts
- Feature comparison prompts  
- Market positioning prompts
- Threat assessment prompts

## 📈 Business Value

### Strategic Benefits
- **Competitive Awareness**: Continuous monitoring of key competitors
- **Market Intelligence**: Understanding of competitive landscape  
- **Threat Assessment**: Early warning system for competitive threats
- **Strategic Planning**: Data-driven competitive positioning

### Operational Benefits
- **Automated Intelligence**: Reduces manual competitive research
- **Executive Reporting**: Ready-to-present insights for leadership
- **Sales Enablement**: Competitive battle cards and positioning
- **Product Strategy**: Feature gap analysis and roadmap priorities

## 🔒 Data Sources & APIs

### Primary Data Sources
- **Company Websites**: Direct scraping of competitor sites
- **Crunchbase**: Funding and company information
- **Job Boards**: LinkedIn, Glassdoor, Indeed
- **News APIs**: TechCrunch, VentureBeat, industry publications
- **GitHub**: Public repository analysis
- **Social Media**: LinkedIn, Twitter/X presence

### API Requirements
- Most functionality works without API keys
- Enhanced features require specific API access
- Graceful degradation when APIs unavailable
- Rate limiting and ethical scraping practices

## 🚨 Troubleshooting

### Common Issues

1. **Configuration Errors**
   ```bash
   python main_competitor.py --validate-config
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **API Key Issues**
   ```bash
   # Check environment variables
   python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
   ```

4. **Permission Errors**
   ```bash
   # Check output directory permissions
   ls -la competitor_reports/
   ```

### Debugging

```bash
# Verbose logging
python main_competitor.py --verbose

# Debug mode
python main_competitor.py --debug

# Dry run (no actual analysis)
python main_competitor.py --dry-run
```

## 📝 Example Workflows

### Daily Competitive Monitoring

```bash
# Quick analysis of high-priority competitors
python main_competitor.py --competitors "Algolia" "Constructor.io" --depth basic --format json
```

### Weekly Executive Reports

```bash
# Comprehensive analysis for executive review
python main_competitor.py --depth comprehensive --format pdf
```

### Monthly Strategic Review

```bash
# Full analysis with all data sources
python main_competitor.py --depth comprehensive --format docx
```

## 🤝 Contributing

### Extending the System

1. **Add New Data Sources**: Create collectors in `src/competitor/collectors/`
2. **Add Analysis Types**: Extend analyzers in `src/competitor/analysis/`
3. **Add Report Formats**: Create generators in `src/competitor/reports/`
4. **Add LLM Providers**: Extend LLM integration in existing provider system

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Check code quality
flake8 src/
black src/
```

## 📄 License

This system is designed to integrate with existing Search Intel architecture. Please ensure compliance with:

- Web scraping best practices and robots.txt
- API terms of service for all integrated services
- Data privacy regulations (GDPR, CCPA)
- Competitive intelligence legal guidelines

## 🆘 Support

For issues, questions, or feature requests:

1. Check the troubleshooting section
2. Review configuration validation
3. Enable debug logging for detailed error information
4. Consult API documentation for external services

---

**Note**: This system is designed for legitimate competitive intelligence and market research. Please ensure all usage complies with applicable laws and terms of service.