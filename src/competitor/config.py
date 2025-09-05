# src/competitor/config.py
"""
Configuration management for competitor analysis
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from .models import AnalysisConfig, AnalysisDepth

class CompetitorConfig:
    """Manages configuration for competitor analysis"""
    
    def __init__(self, config_path: str = "competitor_config.yaml"):
        self.config_path = Path(config_path)
        self._config_data = None
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config_data = yaml.safe_load(file) or {}
        else:
            self._config_data = self.create_default_config()
            self.save_config()
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as file:
            yaml.dump(self._config_data, file, default_flow_style=False, sort_keys=False)
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            'llm': {
                'provider': 'openai',
                'models': {
                    'analysis': 'gpt-4o',
                    'summary': 'gpt-4o-mini',
                    'comparison': 'sonar-pro'
                },
                'temperature': 0.3,
                'max_tokens': 4000,
                'fallback_model': 'gpt-4o-mini'
            },
            'competitors': [
                {
                    'name': 'Algolia',
                    'website': 'https://www.algolia.com',
                    'focus_areas': ['search', 'autocomplete', 'recommendations', 'analytics'],
                    'priority': 'high',
                    'market_segment': ['enterprise', 'mid-market'],
                    'competitive_threat': 'high'
                },
                {
                    'name': 'Constructor.io',
                    'website': 'https://constructor.io',
                    'focus_areas': ['product_search', 'browse', 'recommendations', 'quiz'],
                    'priority': 'high',
                    'market_segment': ['enterprise', 'mid-market'],
                    'competitive_threat': 'high'
                },
                {
                    'name': 'Bloomreach',
                    'website': 'https://www.bloomreach.com',
                    'focus_areas': ['discovery', 'content', 'personalization'],
                    'priority': 'high',
                    'market_segment': ['enterprise'],
                    'competitive_threat': 'medium'
                },
                {
                    'name': 'Elasticsearch',
                    'website': 'https://www.elastic.co',
                    'focus_areas': ['search', 'analytics', 'logging', 'security'],
                    'priority': 'medium',
                    'market_segment': ['enterprise', 'developer'],
                    'competitive_threat': 'medium'
                },
                {
                    'name': 'Coveo',
                    'website': 'https://www.coveo.com',
                    'focus_areas': ['search', 'recommendations', 'personalization', 'AI'],
                    'priority': 'high',
                    'market_segment': ['enterprise'],
                    'competitive_threat': 'high'
                }
            ],
            'analysis': {
                'depth_level': 'standard',
                'scraping': {
                    'delay_between_requests': 2,
                    'max_pages_per_site': 50,
                    'timeout': 30,
                    'concurrent_requests': 3,
                    'rate_limit': 1.0,
                    'user_agent': 'CompetitorAnalysis Bot 1.0',
                    'respect_robots_txt': True
                },
                'target_pages': [
                    {'path': '/', 'name': 'homepage', 'priority': 'high'},
                    {'path': '/about', 'name': 'about', 'priority': 'medium'},
                    {'path': '/products', 'name': 'products', 'priority': 'high'},
                    {'path': '/pricing', 'name': 'pricing', 'priority': 'high'},
                    {'path': '/customers', 'name': 'customers', 'priority': 'medium'},
                    {'path': '/case-studies', 'name': 'case_studies', 'priority': 'high'},
                    {'path': '/blog', 'name': 'blog', 'priority': 'medium'},
                    {'path': '/careers', 'name': 'careers', 'priority': 'medium'},
                    {'path': '/api', 'name': 'api_docs', 'priority': 'high'},
                    {'path': '/documentation', 'name': 'documentation', 'priority': 'high'}
                ]
            },
            'data_sources': {
                'company_websites': True,
                'job_boards': {
                    'enabled': True,
                    'sources': ['linkedin', 'glassdoor', 'indeed'],
                    'max_jobs_per_company': 20,
                    'focus_departments': ['engineering', 'product', 'sales', 'marketing']
                },
                'funding_data': {
                    'enabled': True,
                    'sources': ['crunchbase']
                },
                'social_media': {
                    'enabled': True,
                    'platforms': ['linkedin', 'twitter', 'github', 'youtube']
                },
                'github_repos': {
                    'enabled': True,
                    'analyze_public_repos': True,
                    'track_contributions': True,
                    'language_analysis': True
                },
                'news_sources': {
                    'enabled': True,
                    'sources': ['techcrunch', 'venturebeat', 'searchengineland'],
                    'days_back': 90
                },
                'review_sites': {
                    'enabled': True,
                    'sources': ['g2', 'capterra', 'trustpilot']
                }
            },
            'output': {
                'formats': ['pdf', 'json'],
                'output_dir': 'competitor_reports',
                'include_charts': True,
                'include_screenshots': False,
                'include_competitive_matrix': True,
                'include_swot_analysis': True,
                'pdf': {
                    'brand_color': [52, 152, 219],
                    'include_executive_summary': True,
                    'include_detailed_profiles': True,
                    'include_appendix': True
                },
                'docx': {
                    'template': None,
                    'include_toc': True
                },
                'json': {
                    'pretty_print': True,
                    'include_raw_data': False
                }
            }
        }
    
    @property
    def competitors(self) -> List[Dict[str, Any]]:
        """Get list of competitors to analyze"""
        return self._config_data.get('competitors', [])
    
    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self._config_data.get('llm', {})
    
    @property
    def analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration"""
        return self._config_data.get('analysis', {})
    
    @property
    def data_sources_config(self) -> Dict[str, Any]:
        """Get data sources configuration"""
        return self._config_data.get('data_sources', {})
    
    @property
    def output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self._config_data.get('output', {})
    
    def get_competitor_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get competitor configuration by name"""
        for competitor in self.competitors:
            if competitor.get('name', '').lower() == name.lower():
                return competitor
        return None
    
    def get_high_priority_competitors(self) -> List[Dict[str, Any]]:
        """Get competitors marked as high priority"""
        return [c for c in self.competitors if c.get('priority') == 'high']
    
    def get_analysis_config_object(self) -> AnalysisConfig:
        """Get analysis configuration as AnalysisConfig object"""
        config_dict = {
            'depth_level': self.analysis_config.get('depth_level', 'standard'),
            'competitors': [c['name'] for c in self.competitors],
            'output_formats': self.output_config.get('formats', ['pdf']),
            'output_dir': self.output_config.get('output_dir', 'competitor_reports'),
            'max_pages_per_site': self.analysis_config.get('scraping', {}).get('max_pages_per_site', 50),
            'request_delay': self.analysis_config.get('scraping', {}).get('delay_between_requests', 2.0),
            'timeout': self.analysis_config.get('scraping', {}).get('timeout', 30),
            'concurrent_requests': self.analysis_config.get('scraping', {}).get('concurrent_requests', 3),
            'analyze_website': self.data_sources_config.get('company_websites', True),
            'analyze_funding': self.data_sources_config.get('funding_data', {}).get('enabled', True),
            'analyze_jobs': self.data_sources_config.get('job_boards', {}).get('enabled', True),
            'analyze_news': self.data_sources_config.get('news_sources', {}).get('enabled', True),
            'analyze_social': self.data_sources_config.get('social_media', {}).get('enabled', True),
            'analyze_github': self.data_sources_config.get('github_repos', {}).get('enabled', True),
            'analyze_patents': self.data_sources_config.get('patent_databases', {}).get('enabled', False),
            'news_days_back': self.data_sources_config.get('news_sources', {}).get('days_back', 90),
            'jobs_days_back': 30
        }
        return AnalysisConfig.from_dict(config_dict)
    
    def add_competitor(self, name: str, website: str, **kwargs) -> None:
        """Add a new competitor to the configuration"""
        competitor = {
            'name': name,
            'website': website,
            'focus_areas': kwargs.get('focus_areas', []),
            'priority': kwargs.get('priority', 'medium'),
            'market_segment': kwargs.get('market_segment', []),
            'competitive_threat': kwargs.get('competitive_threat', 'medium'),
            'last_analyzed': None
        }
        
        # Check if competitor already exists
        existing = self.get_competitor_by_name(name)
        if existing:
            # Update existing competitor
            existing.update(competitor)
        else:
            # Add new competitor
            self._config_data['competitors'].append(competitor)
        
        self.save_config()
    
    def update_competitor_analysis_date(self, name: str, date: str) -> None:
        """Update the last analyzed date for a competitor"""
        competitor = self.get_competitor_by_name(name)
        if competitor:
            competitor['last_analyzed'] = date
            self.save_config()
    
    def get_target_pages(self, priority_filter: Optional[str] = None) -> List[Dict[str, str]]:
        """Get target pages to analyze, optionally filtered by priority"""
        pages = self.analysis_config.get('target_pages', [])
        if priority_filter:
            pages = [p for p in pages if p.get('priority') == priority_filter]
        return pages
    
    def get_llm_model(self, model_type: str) -> str:
        """Get LLM model for specific type (analysis, summary, comparison)"""
        models = self.llm_config.get('models', {})
        return models.get(model_type, models.get('analysis', 'gpt-4o'))
    
    def get_data_source_config(self, source_name: str) -> Dict[str, Any]:
        """Get configuration for a specific data source"""
        return self.data_sources_config.get(source_name, {})
    
    def is_data_source_enabled(self, source_name: str) -> bool:
        """Check if a data source is enabled"""
        source_config = self.get_data_source_config(source_name)
        if isinstance(source_config, dict):
            return source_config.get('enabled', False)
        return bool(source_config)
    
    def override_from_cli_args(self, args) -> None:
        """Override configuration with command line arguments"""
        if hasattr(args, 'output_dir') and args.output_dir:
            self._config_data['output']['output_dir'] = args.output_dir
        
        if hasattr(args, 'format') and args.format:
            self._config_data['output']['formats'] = [args.format]
        
        if hasattr(args, 'depth') and args.depth:
            self._config_data['analysis']['depth_level'] = args.depth
        
        if hasattr(args, 'competitors') and args.competitors:
            # Filter competitors based on CLI args
            filtered_competitors = [
                comp for comp in self.competitors
                if comp['name'].lower() in [c.lower() for c in args.competitors]
            ]
            self._config_data['competitors'] = filtered_competitors

# Environment variable support
def get_env_or_config(key: str, config_value: Any, env_prefix: str = "COMPETITOR_") -> Any:
    """Get value from environment variable or fall back to config"""
    env_key = f"{env_prefix}{key.upper()}"
    env_value = os.getenv(env_key)
    
    if env_value is not None:
        # Try to convert to same type as config value
        if isinstance(config_value, bool):
            return env_value.lower() in ('true', '1', 'yes')
        elif isinstance(config_value, int):
            try:
                return int(env_value)
            except ValueError:
                return config_value
        elif isinstance(config_value, float):
            try:
                return float(env_value)
            except ValueError:
                return config_value
        else:
            return env_value
    
    return config_value