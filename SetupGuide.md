# Competitor Analysis System - Installation & Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd competitor-analysis

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install with setup.py
pip install -e .
```

### 2. Set Up API Keys

Create a `.env` file in the project root:

```bash
# Required - At least one LLM provider
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key

# Optional - Enhanced features
CRUNCHBASE_API_KEY=your_crunchbase_key
GITHUB_TOKEN=your_github_token
NEWS_API_KEY=your_news_api_key
TWITTER_BEARER_TOKEN=your_twitter_token
LINKEDIN_ACCESS_TOKEN=your_linkedin_token
GOOGLE_API_KEY=your_google_key
GOOGLE_SEARCH_CX=your_custom_search_engine_id
```

### 3. Configuration

The system uses `competitor_config.yaml` for configuration. A default config is created automatically.

```bash
# Validate configuration
python src/main_competitor.py --validate-config

# List configured competitors
python src/main_competitor.py --list-competitors
```

### 4. Run Analysis

```bash
# Analyze all competitors
python src/main_competitor.py

# Analyze specific competitors
python src/main_competitor.py --competitors "Algolia" "Constructor.io"

# Generate different formats
python src/main_competitor.py --format json
python src/main_competitor.py --format pdf --depth comprehensive
```

## Installation Issues & Fixes

### Issue 1: Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'openai'`

**Fix**:
```bash
pip install openai anthropic aiohttp beautifulsoup4 pyyaml python-dateutil reportlab python-docx
```

### Issue 2: Import Errors

**Error**: `ImportError: cannot import name 'timedelta'`

**Fix**: The models.py has been updated to include the missing import:
```python
from datetime import datetime, timedelta
```

### Issue 3: LLM Provider Issues

**Error**: `RuntimeError: No LLM providers available`

**Fix**: Ensure you have at least one API key set:
```bash
export OPENAI_API_KEY=your_key_here
# OR
export ANTHROPIC_API_KEY=your_key_here
```

### Issue 4: Report Generation Fails

**Error**: `ImportError: No module named 'reportlab'`

**Fix**: Install report generation dependencies:
```bash
pip install reportlab python-docx
```

## API Key Setup Guide

### OpenAI (Recommended)
1. Visit https://platform.openai.com/api-keys
2. Create new API key
3. Add to environment: `OPENAI_API_KEY=sk-...`

### Anthropic
1. Visit https://console.anthropic.com/
2. Create API key
3. Add to environment: `ANTHROPIC_API_KEY=sk-ant-...`

### Optional APIs

#### Crunchbase (for funding data)
1. Visit https://data.crunchbase.com/docs
2. Sign up for API access
3. Add: `CRUNCHBASE_API_KEY=your_key`

#### GitHub (for repository analysis)
1. Visit https://github.com/settings/tokens
2. Create personal access token
3. Add: `GITHUB_TOKEN=ghp_...`

## Common Configuration Issues

### Issue: Wrong file paths
**Fix**: Ensure you're running from the project root directory

### Issue: Permission errors
**Fix**: 
```bash
chmod +x src/main_competitor.py
```

### Issue: Output directory doesn't exist
**Fix**: The system creates it automatically, but ensure parent directories exist

## Troubleshooting

### Enable Debug Logging
```bash
python src/main_competitor.py --debug --verbose
```

### Test Configuration
```bash
python src/main_competitor.py --validate-config
```

### Dry Run
```bash
python src/main_competitor.py --dry-run
```

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes* | LLM for analysis |
| `ANTHROPIC_API_KEY` | Yes* | Alternative LLM |
| `PERPLEXITY_API_KEY` | No | Online search LLM |
| `CRUNCHBASE_API_KEY` | No | Funding data |
| `GITHUB_TOKEN` | No | GitHub analysis |
| `NEWS_API_KEY` | No | News aggregation |

*At least one LLM provider is required

## Performance Optimization

### Faster Analysis
- Use `--depth basic` for quick analysis
- Limit competitors: `--competitors "Company1" "Company2"`
- Use JSON format for faster report generation

### Resource Management
- The system respects rate limits automatically
- Concurrent requests are limited to avoid overloading APIs
- Caching is enabled by default to avoid duplicate requests

## Docker Setup (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "src/main_competitor.py"]
```

```bash
docker build -t competitor-analysis .
docker run -e OPENAI_API_KEY=your_key competitor-analysis
```

## Next Steps

1. Customize `competitor_config.yaml` with your competitors
2. Set up required API keys
3. Run your first analysis
4. Review generated reports
5. Schedule regular analysis updates

For advanced configuration and customization, see the full documentation.