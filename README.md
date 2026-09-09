# Project Structure

competitor-analysis/
├── README.md                          # Comprehensive documentation
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
├── Dockerfile                         # Docker containerization
├── docker-compose.yml                 # Docker Compose configuration
├── Makefile                           # Development shortcuts
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore patterns
├── competitor_config.yaml             # Main configuration file
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── main_competitor.py             # CLI entry point
│   │
│   ├── competitor/                    # Main package
│   │   ├── __init__.py
│   │   ├── analyzer.py                # Main orchestrator
│   │   ├── config.py                  # Configuration management
│   │   ├── models.py                  # Data models and schemas
│   │   ├── scraper.py                 # Web scraping engine
│   │   │
│   │   ├── analysis/                  # AI-powered analysis modules
│   │   │   ├── __init__.py
│   │   │   ├── company.py             # Company-level analysis
│   │   │   ├── features.py            # Feature comparison
│   │   │   ├── market.py              # Market positioning
│   │   │   ├── pricing.py             # Pricing strategy
│   │   │   ├── threats.py             # Threat assessment
│   │   │   └── innovation.py          # Innovation analysis
│   │   │
│   │   ├── collectors/                # Data collection modules
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Base collector classes
│   │   │   ├── website.py             # Website data collection
│   │   │   ├── funding.py             # Funding information
│   │   │   ├── jobs.py                # Job postings
│   │   │   ├── news.py                # News mentions
│   │   │   ├── social.py              # Social media presence
│   │   │   ├── github_activity.py     # GitHub activity data
│   │   │   └── patents.py             # Patent information
│   │   │
│   │   ├── reports/                   # Report generation
│   │   │   ├── __init__.py            # ReportGenerator class
│   │   │   └── templates/             # Report templates
│   │   │
│   │   └── utils/                     # Utility functions
│   │       ├── __init__.py
│   │       └── web_utils.py           # Web scraping utilities
│   │
│   └── llm/                           # LLM provider abstraction
│       ├── __init__.py
│       └── provider.py                # Multi-provider LLM interface
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── test_analyzer.py               # Main analyzer tests
│   ├── test_config.py                 # Configuration tests
│   ├── test_models.py                 # Data model tests
│   ├── test_collectors.py             # Collector tests
│   ├── test_llm_provider.py           # LLM provider tests
│   └── fixtures/                      # Test fixtures
│       └── sample_config.yaml
│
├── docs/                              # Documentation
│   ├── api_reference.md               # API documentation
│   ├── configuration.md               # Configuration guide
│   ├── deployment.md                  # Deployment instructions
│   └── examples/                      # Usage examples
│       ├── basic_analysis.py
│       ├── custom_collectors.py
│       └── report_customization.py
│
├── cache/                             # Data cache directory
│   ├── funding/
│   ├── news/
│   ├── social/
│   └── github/
│
└── competitor_reports/                # Output directory
    ├── competitor_analysis_20240101_120000.pdf
    ├── competitor_analysis_20240101_120000.json
    └── individual_profiles/
        ├── algolia_profile.json
        ├── constructor_profile.json
        └── bloomreach_profile.json

# File: .gitignore

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
competitor_reports/
cache/
logs/
*.log
temp/
tmp/

# Configuration files with secrets
competitor_config_production.yaml
.env.production
.env.local

# Database files
*.db
*.sqlite
*.sqlite3

# Backup files
*.bak
*.backup
*~

# File: LICENSE

MIT License

Copyright (c) 2024 Competitor Analysis System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# File: pytest.ini

[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
asyncio_mode = auto

# File: pyproject.toml

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "competitor-analysis"
version = "1.0.0"
description = "Comprehensive competitor analysis system for ecommerce search companies"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Company", email = "your.email@company.com"},
]
keywords = ["competitor-analysis", "business-intelligence", "market-research", "ai"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Office/Business",
    "Topic :: Scientific/Engineering :: Information Analysis",
]
dependencies = [
    "aiohttp>=3.8.0",
    "beautifulsoup4>=4.11.0",
    "pyyaml>=6.0",
    "python-dateutil>=2.8.0",
    "openai>=1.0.0",
    "anthropic>=0.25.0",
    "pandas>=1.5.0",
    "numpy>=1.20.0",
    "requests>=2.28.0",
    "lxml>=4.9.0",
    "reportlab>=3.6.0",
    "python-docx>=0.8.11",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
    "mypy>=1.0.0",
    "pre-commit>=2.20.0",
]
enhanced = [
    "crunchbase-api>=0.1.0",
    "pygithub>=1.58.0",
    "tweepy>=4.14.0",
    "linkedin-api>=2.0.0",
]

[project.urls]
Homepage = "https://github.com/your-org/competitor-analysis"
Documentation = "https://competitor-analysis.readthedocs.io"
Repository = "https://github.com/your-org/competitor-analysis.git"
"Bug Tracker" = "https://github.com/your-org/competitor-analysis/issues"

[project.scripts]
competitor-analysis = "main_competitor:main"

[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

# File: .pre-commit-config.yaml

repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-merge-conflict
      - id: debug-statements
      - id: check-docstring-first

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-PyYAML]

# File: docs/api_reference.md

# API Reference

## Core Classes

### CompetitorAnalyzer

Main orchestrator for competitive analysis operations.

```python
class CompetitorAnalyzer:
    def __init__(self, config_path: str = "competitor_config.yaml")
    async def analyze_all_competitors(self, competitor_names: Optional[List[str]] = None) -> CompetitorIntelligence
    async def analyze_single_competitor(self, competitor_config: Dict[str, Any]) -> Optional[CompetitorProfile]
    async def generate_reports(self, intelligence: CompetitorIntelligence) -> List[str]
    async def generate_executive_summary(self, intelligence: CompetitorIntelligence) -> str
```

### CompetitorProfile

Data model representing a single competitor's information.

```python
@dataclass
class CompetitorProfile:
    name: str
    website: str
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    target_markets: List[str] = field(default_factory=list)
    key_features: List[str] = field(default_factory=list)
    funding_info: Optional[FundingInfo] = None
    # ... additional fields
```

### AnalysisEngine

Coordinates AI-powered analysis across different modules.

```python
class AnalysisEngine:
    def __init__(self, config, llm_provider)
    async def analyze_competitor_profile(self, profile) -> CompetitorProfile
    async def perform_cross_competitor_analysis(self, intelligence) -> CompetitorIntelligence
    async def generate_executive_summary(self, intelligence) -> str
```

## Data Collection

### BaseCollector

Base class for all data collectors with caching and rate limiting.

```python
class BaseCollector(ABC):
    def __init__(self, config: Dict[str, Any], collector_type: str)
    async def collect(self, competitor_name: str, **kwargs) -> Union[Dict, List, None]
```

### Specific Collectors

- **FundingCollector**: Collects funding and financial data
- **JobCollector**: Collects job posting data
- **NewsCollector**: Collects news mentions and media coverage
- **SocialCollector**: Collects social media presence data
- **GitHubCollector**: Collects GitHub activity and repository data
- **PatentCollector**: Collects patent and IP data

## Analysis Modules

### Company Analysis
```python
class CompanyAnalyzer:
    async def analyze_company_profile(self, profile) -> Dict[str, Any]
    async def generate_executive_summary(self, intelligence) -> str
```

### Feature Analysis
```python
class FeatureAnalyzer:
    async def analyze_features(self, profile) -> Dict[str, Any]
    async def compare_features_across_competitors(self, profiles) -> Dict[str, Any]
```

### Market Analysis
```python
class MarketAnalyzer:
    async def analyze_market_position(self, profile) -> Dict[str, Any]
    async def analyze_competitive_landscape(self, profiles) -> Dict[str, Any]
```

### Threat Analysis
```python
class ThreatAnalyzer:
    async def assess_competitive_threat(self, profile) -> Dict[str, Any]
    async def create_threat_matrix(self, profiles) -> Dict[str, Any]
```

## Configuration

### CompetitorConfig

Manages system configuration from YAML files.

```python
class CompetitorConfig:
    def __init__(self, config_path: str = "competitor_config.yaml")
    def add_competitor(self, name: str, website: str, **kwargs) -> None
    def get_competitor_by_name(self, name: str) -> Optional[Dict[str, Any]]
    def is_data_source_enabled(self, source_name: str) -> bool
```

## LLM Integration

### LLMProvider

Multi-provider LLM interface with fallback support.

```python
class LLMProvider:
    def __init__(self)
    def chat(self, system: str, user: str, model: str = "gpt-4o", **kwargs) -> str
    async def achat(self, system: str, user: str, model: str = "gpt-4o", **kwargs) -> str
```

# File: docs/configuration.md

# Configuration Guide

## Configuration File Structure

The system uses a YAML configuration file (`competitor_config.yaml`) to control all aspects of analysis.

### LLM Configuration

```yaml
llm:
  provider: openai  # openai, anthropic, perplexity
  models:
    analysis: gpt-4o           # Primary analysis model
    summary: gpt-4o-mini       # Summary generation
    comparison: sonar-pro      # Comparative analysis
  temperature: 0.3             # Creativity vs consistency
  max_tokens: 4000            # Maximum response length
  fallback_model: gpt-4o-mini # Fallback if primary fails
```

### Competitor Definitions

```yaml
competitors:
  - name: "Algolia"
    website: "https://www.algolia.com"
    focus_areas: ["search", "autocomplete", "recommendations"]
    priority: high               # low, medium, high
    market_segment: ["enterprise", "mid-market"]
    competitive_threat: high     # low, medium, high, critical
    last_analyzed: null         # Auto-updated
```

### Analysis Settings

```yaml
analysis:
  depth_level: standard        # basic, standard, comprehensive
  scraping:
    delay_between_requests: 2   # Seconds between requests
    max_pages_per_site: 50     # Maximum pages to analyze
    timeout: 30                # Request timeout in seconds
    concurrent_requests: 3      # Parallel request limit
    rate_limit: 1.0            # Requests per second
    user_agent: "CompetitorAnalysis Bot 1.0"
    respect_robots_txt: true
  target_pages:               # Pages to analyze on each site
    - path: "/"
      name: "homepage"
      priority: "high"
    - path: "/pricing"
      name: "pricing"
      priority: "high"
    - path: "/products"
      name: "products"
      priority: "high"
```

### Data Sources

```yaml
data_sources:
  company_websites: true      # Enable website scraping
  
  funding_data:
    enabled: true
    sources: ["crunchbase"]   # Data sources to use
    cache_ttl_hours: 168     # 7 days cache
  
  job_boards:
    enabled: true
    sources: ["linkedin", "glassdoor", "indeed"]
    max_jobs_per_company: 20
    focus_departments: ["engineering", "product", "sales"]
  
  news_sources:
    enabled: true
    sources: ["techcrunch", "venturebeat", "searchengineland"]
    days_back: 90            # How far back to search
    
  social_media:
    enabled: true
    platforms: ["linkedin", "twitter", "github", "youtube"]
    
  github_repos:
    enabled: true
    analyze_public_repos: true
    track_contributions: true
    language_analysis: true
    
  patent_databases:
    enabled: false           # Requires additional setup
    sources: ["google_patents"]
```

### Output Configuration

```yaml
output:
  formats: ["pdf", "json"]    # pdf, docx, json
  output_dir: "competitor_reports"
  include_charts: true
  include_screenshots: false
  include_competitive_matrix: true
  include_swot_analysis: true
  
  pdf:
    brand_color: [52, 152, 219]  # RGB color
    include_executive_summary: true
    include_detailed_profiles: true
    include_appendix: true
    logo_path: null            # Path to company logo
    
  docx:
    template: null             # Path to Word template
    include_toc: true          # Table of contents
    
  json:
    pretty_print: true
    include_raw_data: false    # Include all collected data
```

## Environment Variables

### Required Variables

At least one LLM provider API key is required:

```bash
# OpenAI (recommended)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Perplexity (for web-enhanced analysis)
PERPLEXITY_API_KEY=your_perplexity_api_key_here
```

### Optional Enhanced Features

```bash
# Funding data
CRUNCHBASE_API_KEY=your_crunchbase_api_key

# GitHub analysis
GITHUB_TOKEN=your_github_token

# News monitoring
NEWS_API_KEY=your_news_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_SEARCH_CX=your_custom_search_engine_id

# Social media analysis
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
LINKEDIN_ACCESS_TOKEN=your_linkedin_access_token

# System overrides
COMPETITOR_OUTPUT_DIR=./custom_reports
COMPETITOR_CACHE_ENABLED=true
COMPETITOR_RATE_LIMIT=2.0
```

## Analysis Depth Levels

### Basic Analysis
- Core website data collection
- Basic company information
- Essential competitive metrics
- Fastest execution time

### Standard Analysis (Default)
- All basic features plus:
- Funding and financial analysis
- Job posting intelligence
- News mention tracking
- Social media presence
- Balanced speed vs depth

### Comprehensive Analysis
- All standard features plus:
- Deep content analysis
- Patent research
- Technical intelligence
- Detailed market positioning
- Maximum insight depth

## Customization Examples

### Adding Custom Competitors

```python
from competitor.config import CompetitorConfig

config = CompetitorConfig()
config.add_competitor(
    name="New Competitor",
    website="https://newcompetitor.com",
    focus_areas=["ai", "personalization"],
    priority="medium",
    market_segment=["mid-market"],
    competitive_threat="low"
)
```

### Custom Data Source Configuration

```yaml
data_sources:
  custom_source:
    enabled: true
    api_endpoint: "https://api.custom.com"
    cache_ttl_hours: 24
    rate_limit: 0.5
    custom_headers:
      Authorization: "Bearer ${CUSTOM_API_KEY}"
```

### Report Customization

```yaml
output:
  pdf:
    brand_color: [255, 87, 34]  # Custom orange
    logo_path: "assets/logo.png"
    custom_footer: "Confidential - Internal Use Only"
    font_family: "Arial"
    include_watermark: true
```

# File: docs/deployment.md

# Deployment Guide

## Local Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- At least one LLM API key (OpenAI, Anthropic, or Perplexity)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/competitor-analysis.git
cd competitor-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Validate configuration
python src/main_competitor.py --validate-config
```

### Development Workflow

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Code formatting
black src/ tests/
flake8 src/ tests/

# Type checking
mypy src/
```

## Running Tests

The project uses [pytest](https://pytest.org) for its test suite. After installing the dependencies, run:

```bash
pytest
```

## Docker Deployment

### Building Container

```bash
# Build image
docker build -t competitor-analysis:latest .

# Run with environment file
docker run --env-file .env \
  -v $(pwd)/competitor_reports:/app/competitor_reports \
  competitor-analysis:latest
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  competitor-analysis:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./competitor_reports:/app/competitor_reports
      - ./competitor_config.yaml:/app/competitor_config.yaml
      - ./cache:/app/cache
    command: python src/main_competitor.py --format pdf --depth standard

  # Optional: Redis for caching
  redis:
    image: redis:alpine
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

```bash
# Deploy with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f competitor-analysis

# Scale analysis workers
docker-compose up --scale competitor-analysis=3
```

## Cloud Deployment

### AWS Deployment

#### ECS Fargate

```json
{
  "family": "competitor-analysis",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "competitor-analysis",
      "image": "your-account.dkr.ecr.region.amazonaws.com/competitor-analysis:latest",
      "environment": [
        {"name": "OPENAI_API_KEY", "value": "your-key"},
        {"name": "AWS_DEFAULT_REGION", "value": "us-west-2"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/competitor-analysis",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Lambda Deployment

```python
# lambda_handler.py
import json
import asyncio
from competitor.analyzer import run_competitor_analysis

def lambda_handler(event, context):
    """AWS Lambda handler for competitor analysis"""
    
    # Parse event parameters
    competitors = event.get('competitors', None)
    format_type = event.get('format', 'json')
    depth = event.get('depth', 'standard')
    
    # Run analysis
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        report_files = loop.run_until_complete(
            run_competitor_analysis(
                competitor_names=competitors,
                formats=[format_type],
                output_dir='/tmp/reports'
            )
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'report_files': report_files,
                'competitors_analyzed': len(competitors) if competitors else 'all'
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }
```

### Google Cloud Platform

#### Cloud Run

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/competitor-analysis', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/competitor-analysis']
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'competitor-analysis'
      - '--image'
      - 'gcr.io/$PROJECT_ID/competitor-analysis'
      - '--platform'
      - 'managed'
      - '--region'
      - 'us-central1'
      - '--allow-unauthenticated'
```

```bash
# Deploy to Cloud Run
gcloud builds submit --config cloudbuild.yaml
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: competitor-analysis
spec:
  replicas: 2
  selector:
    matchLabels:
      app: competitor-analysis
  template:
    metadata:
      labels:
        app: competitor-analysis
    spec:
      containers:
      - name: competitor-analysis
        image: competitor-analysis:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai_key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: reports-volume
          mountPath: /app/competitor_reports
      volumes:
      - name: reports-volume
        persistentVolumeClaim:
          claimName: reports-pvc

---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: weekly-competitor-analysis
spec:
  schedule: "0 6 * * 1"  # Every Monday at 6 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: competitor-analysis
            image: competitor-analysis:latest
            args: ["python", "src/main_competitor.py", "--depth", "comprehensive"]
          restartPolicy: OnFailure
```

## Production Considerations

### Performance Optimization

```yaml
# High-performance configuration
analysis:
  scraping:
    concurrent_requests: 10      # Increase for faster collection
    rate_limit: 2.0             # Higher rate limit
    timeout: 60                 # Longer timeout for reliability
    
data_sources:
  funding_data:
    cache_ttl_hours: 168       # Week-long cache
  news_sources:
    cache_ttl_hours: 4         # 4-hour cache for news
```

### Security Best Practices

1. **API Key Management**
   - Use environment variables or secret management systems
   - Rotate API keys regularly
   - Monitor API usage and set up alerts

2. **Network Security**
   - Use VPN or private networks for sensitive analysis
   - Implement rate limiting and request throttling
   - Monitor for unusual API usage patterns

3. **Data Protection**
   - Encrypt sensitive data at rest and in transit
   - Implement access controls for reports
   - Regular security audits and updates

### Monitoring and Logging

```python
# logging_config.py
import logging
import sys

def setup_production_logging():
    """Configure production logging"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/var/log/competitor-analysis.log'),
            # Add CloudWatch/Stackdriver handler for cloud deployments
        ]
    )
    
    # Set specific log levels
    logging.getLogger('competitor.scraper').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.ERROR)
```

### Backup and Recovery

```bash
#!/bin/bash
# backup_script.sh

# Backup configuration
cp competitor_config.yaml backups/config_$(date +%Y%m%d).yaml

# Backup reports
tar -czf backups/reports_$(date +%Y%m%d).tar.gz competitor_reports/

# Backup cache (optional)
tar -czf backups/cache_$(date +%Y%m%d).tar.gz cache/

# Upload to cloud storage
aws s3 cp backups/ s3://your-backup-bucket/competitor-analysis/ --recursive
```

### Scaling Strategies

1. **Horizontal Scaling**
   - Multiple analysis workers
   - Load balancing across instances
   - Distributed caching with Redis

2. **Vertical Scaling**
   - Increase CPU/memory for complex analysis
   - SSD storage for faster I/O
   - Optimize database queries

3. **Cost Optimization**
   - Use spot instances for non-critical analysis
   - Implement intelligent caching
   - Schedule analysis during off-peak hours