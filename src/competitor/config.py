# src/competitor/config.py
"""
Configuration management system for the competitor analysis application.
Handles YAML configuration loading, validation, and environment variable integration.
"""

import os
import yaml
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime
from copy import deepcopy

try:  # pragma: no cover - import fallback for alternative YAML implementations
    from yaml import YAMLError
except ImportError:  # pragma: no cover
    YAMLError = Exception

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "openai"
    models: Dict[str, str] = field(default_factory=lambda: {
        "analysis": "gpt-4o",
        "summary": "gpt-4o-mini",
        "comparison": "gpt-4o"
    })
    temperature: float = 0.3
    max_tokens: int = 4000
    fallback_model: str = "gpt-4o-mini"


@dataclass
class CompetitorInfo:
    """Individual competitor configuration."""
    name: str
    website: str
    focus_areas: List[str] = field(default_factory=list)
    priority: str = "medium"  # low, medium, high
    market_segment: List[str] = field(default_factory=list)
    competitive_threat: str = "medium"  # low, medium, high, critical
    last_analyzed: Optional[datetime] = None


@dataclass
class ScrapingConfig:
    """Web scraping configuration."""
    delay_between_requests: float = 2.0
    max_pages_per_site: int = 50
    timeout: int = 30
    concurrent_requests: int = 3
    rate_limit: float = 1.0
    user_agent: str = "CompetitorAnalysis Bot 1.0"
    respect_robots_txt: bool = True
    target_pages: List[Dict[str, str]] = field(default_factory=lambda: [
        {"path": "/", "name": "homepage", "priority": "high"},
        {"path": "/pricing", "name": "pricing", "priority": "high"},
        {"path": "/products", "name": "products", "priority": "high"}
    ])


@dataclass
class DataSourceConfig:
    """Data source configuration."""
    company_websites: bool = True
    funding_data: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "sources": ["crunchbase"],
        "cache_ttl_hours": 168
    })
    job_boards: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "sources": ["linkedin", "glassdoor", "indeed"],
        "max_jobs_per_company": 20,
        "focus_departments": ["engineering", "product", "sales"]
    })
    news_sources: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "sources": ["techcrunch", "venturebeat", "searchengineland"],
        "days_back": 90
    })
    social_media: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "platforms": ["linkedin", "twitter", "github", "youtube"]
    })
    github_repos: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "analyze_public_repos": True,
        "track_contributions": True,
        "language_analysis": True
    })
    patent_databases: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "sources": ["google_patents"]
    })


@dataclass
class OutputConfig:
    """Output and reporting configuration."""
    formats: List[str] = field(default_factory=lambda: ["pdf", "json"])
    output_dir: str = "competitor_reports"
    include_charts: bool = True
    include_screenshots: bool = False
    include_competitive_matrix: bool = True
    include_swot_analysis: bool = True
    pdf: Dict[str, Any] = field(default_factory=lambda: {
        "brand_color": [52, 152, 219],
        "include_executive_summary": True,
        "include_detailed_profiles": True,
        "include_appendix": True,
        "logo_path": None
    })
    json: Dict[str, Any] = field(default_factory=lambda: {
        "pretty_print": True,
        "include_raw_data": False
    })


@dataclass(frozen=True)
class _DepthLevel:
    """Simple wrapper providing a value attribute for depth level strings."""

    value: str


@dataclass(frozen=True)
class AnalysisConfigSummary:
    """Lightweight container describing active analysis configuration."""

    depth_level: _DepthLevel
    competitors: List[str]
    output_formats: List[str]
    target_pages: List[Dict[str, Any]]
    data_sources: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the summary."""
        return {
            "depth_level": self.depth_level.value,
            "competitors": self.competitors,
            "output_formats": self.output_formats,
            "target_pages": deepcopy(self.target_pages),
            "data_sources": deepcopy(self.data_sources)
        }


class CompetitorConfig:
    """Main configuration management class."""
    
    def __init__(self, config_path: str = "competitor_config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self._raw_config: Dict[str, Any] = {}
        self._competitors: List[CompetitorInfo] = []
        
        # Initialize configuration components
        self.llm: LLMConfig = LLMConfig()
        self.scraping: ScrapingConfig = ScrapingConfig()
        self.data_sources: DataSourceConfig = DataSourceConfig()
        self.output: OutputConfig = OutputConfig()
        self.analysis: Dict[str, Any] = {"depth_level": "standard"}
        
        # Load configuration if file exists
        if self.config_path.exists():
            self.load_config()
        else:
            logger.warning(f"Configuration file {config_path} not found. Using defaults.")
            self._create_default_config()
            # Parse the freshly created defaults so all properties are initialised
            try:
                self.load_config()
            except Exception:  # pragma: no cover - defensive guard, should not happen in tests
                logger.exception("Failed to load default configuration")

    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._raw_config = yaml.safe_load(file) or {}
            
            self._parse_config()
            self._load_environment_overrides()
            self._validate_config()
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            
        except YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise
    
    def _parse_config(self) -> None:
        """Parse the raw configuration into structured objects."""
        # Parse LLM configuration
        if "llm" in self._raw_config:
            llm_data = self._raw_config["llm"]
            self.llm = LLMConfig(
                provider=llm_data.get("provider", "openai"),
                models=llm_data.get("models", self.llm.models),
                temperature=llm_data.get("temperature", 0.3),
                max_tokens=llm_data.get("max_tokens", 4000),
                fallback_model=llm_data.get("fallback_model", "gpt-4o-mini")
            )
        
        # Parse competitors
        if "competitors" in self._raw_config:
            self._competitors = []
            for comp_data in self._raw_config["competitors"]:
                last_analyzed = comp_data.get("last_analyzed")
                if isinstance(last_analyzed, str):
                    normalized_value = (
                        last_analyzed.replace("Z", "+00:00")
                        if last_analyzed.endswith("Z")
                        else last_analyzed
                    )
                    try:
                        last_analyzed = datetime.fromisoformat(normalized_value)
                    except ValueError:
                        logger.warning(
                            "Invalid last_analyzed value '%s' for competitor '%s'",
                            last_analyzed,
                            comp_data.get("name", "<unknown>")
                        )
                        last_analyzed = None
                competitor = CompetitorInfo(
                    name=comp_data["name"],
                    website=comp_data["website"],
                    focus_areas=comp_data.get("focus_areas", []),
                    priority=comp_data.get("priority", "medium"),
                    market_segment=comp_data.get("market_segment", []),
                    competitive_threat=comp_data.get("competitive_threat", "medium"),
                    last_analyzed=last_analyzed
                )
                self._competitors.append(competitor)
        
        # Parse analysis configuration
        if "analysis" in self._raw_config:
            self.analysis = self._raw_config["analysis"]
        
        # Parse scraping configuration
        if "scraping" in self._raw_config:
            scraping_data = self._raw_config["scraping"]
            self.scraping = ScrapingConfig(
                delay_between_requests=scraping_data.get("delay_between_requests", 2.0),
                max_pages_per_site=scraping_data.get("max_pages_per_site", 50),
                timeout=scraping_data.get("timeout", 30),
                concurrent_requests=scraping_data.get("concurrent_requests", 3),
                rate_limit=scraping_data.get("rate_limit", 1.0),
                user_agent=scraping_data.get("user_agent", "CompetitorAnalysis Bot 1.0"),
                respect_robots_txt=scraping_data.get("respect_robots_txt", True),
                target_pages=scraping_data.get("target_pages", self.scraping.target_pages)
            )
        
        # Parse data sources configuration
        if "data_sources" in self._raw_config:
            ds_data = self._raw_config["data_sources"]
            self.data_sources = DataSourceConfig(
                company_websites=ds_data.get("company_websites", True),
                funding_data=ds_data.get("funding_data", self.data_sources.funding_data),
                job_boards=ds_data.get("job_boards", self.data_sources.job_boards),
                news_sources=ds_data.get("news_sources", self.data_sources.news_sources),
                social_media=ds_data.get("social_media", self.data_sources.social_media),
                github_repos=ds_data.get("github_repos", self.data_sources.github_repos),
                patent_databases=ds_data.get("patent_databases", self.data_sources.patent_databases)
            )
        
        # Parse output configuration
        if "output" in self._raw_config:
            output_data = self._raw_config["output"]
            self.output = OutputConfig(
                formats=output_data.get("formats", ["pdf", "json"]),
                output_dir=output_data.get("output_dir", "competitor_reports"),
                include_charts=output_data.get("include_charts", True),
                include_screenshots=output_data.get("include_screenshots", False),
                include_competitive_matrix=output_data.get("include_competitive_matrix", True),
                include_swot_analysis=output_data.get("include_swot_analysis", True),
                pdf=output_data.get("pdf", self.output.pdf),
                json=output_data.get("json", self.output.json)
            )
    
    def _load_environment_overrides(self) -> None:
        """Load environment variable overrides."""
        # API Keys (handled by LLM provider)
        env_overrides = {
            "COMPETITOR_OUTPUT_DIR": "output.output_dir",
            "COMPETITOR_RATE_LIMIT": "scraping.rate_limit",
            "COMPETITOR_CACHE_ENABLED": "data_sources.cache_enabled"
        }
        
        for env_var, config_path in env_overrides.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Apply the override (simplified implementation)
                if env_var == "COMPETITOR_OUTPUT_DIR":
                    self.output.output_dir = value
                elif env_var == "COMPETITOR_RATE_LIMIT":
                    self.scraping.rate_limit = float(value)
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate LLM provider
        valid_providers = ["openai", "anthropic", "perplexity"]
        if self.llm.provider not in valid_providers:
            logger.warning(
                "Using unrecognised LLM provider '%s'. Expected one of %s",
                self.llm.provider,
                valid_providers
            )
        
        # Validate competitor data
        if not self._competitors:
            logger.warning("No competitors defined in configuration")
        
        # Validate output formats
        valid_formats = ["pdf", "json", "docx"]
        for fmt in self.output.formats:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid output format: {fmt}. Must be one of {valid_formats}")
        
        # Validate scraping configuration
        if self.scraping.rate_limit <= 0:
            raise ValueError("Rate limit must be positive")
        if self.scraping.concurrent_requests < 1:
            raise ValueError("Concurrent requests must be at least 1")
    
    def _create_default_config(self) -> None:
        """Create a default configuration file."""
        default_config = {
            "llm": {
                "provider": "openai",
                "models": {
                    "analysis": "gpt-4o",
                    "summary": "gpt-4o-mini",
                    "comparison": "gpt-4o"
                },
                "temperature": 0.3,
                "max_tokens": 4000,
                "fallback_model": "gpt-4o-mini"
            },
            "competitors": [
                {
                    "name": "Example Competitor",
                    "website": "https://example.com",
                    "focus_areas": ["search", "ai"],
                    "priority": "high",
                    "market_segment": ["enterprise"],
                    "competitive_threat": "medium"
                }
            ],
            "analysis": {
                "depth_level": "standard"
            },
            "scraping": {
                "delay_between_requests": 2,
                "max_pages_per_site": 50,
                "timeout": 30,
                "concurrent_requests": 3,
                "rate_limit": 1.0,
                "user_agent": "CompetitorAnalysis Bot 1.0",
                "respect_robots_txt": True,
                "target_pages": [
                    {"path": "/", "name": "homepage", "priority": "high"},
                    {"path": "/pricing", "name": "pricing", "priority": "high"},
                    {"path": "/products", "name": "products", "priority": "high"}
                ]
            },
            "data_sources": {
                "company_websites": True,
                "funding_data": {
                    "enabled": True,
                    "sources": ["crunchbase"],
                    "cache_ttl_hours": 168
                },
                "job_boards": {
                    "enabled": True,
                    "sources": ["linkedin", "glassdoor"],
                    "max_jobs_per_company": 20
                },
                "news_sources": {
                    "enabled": True,
                    "sources": ["techcrunch", "venturebeat"],
                    "days_back": 90
                }
            },
            "output": {
                "formats": ["pdf", "json"],
                "output_dir": "competitor_reports",
                "include_charts": True,
                "include_competitive_matrix": True,
                "include_swot_analysis": True
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as file:
            yaml.dump(default_config, file, default_flow_style=False, indent=2)
        
        logger.info(f"Created default configuration file: {self.config_path}")
    
    def add_competitor(self, name: str, website: str, **kwargs) -> None:
        """
        Add a new competitor to the configuration.
        
        Args:
            name: Competitor name
            website: Competitor website URL
            **kwargs: Additional competitor attributes
        """
        # Check if competitor already exists
        existing = self.get_competitor_by_name(name)
        if existing:
            logger.warning(f"Competitor {name} already exists. Updating configuration.")
            self.remove_competitor(name)
        
        competitor = CompetitorInfo(
            name=name,
            website=website,
            focus_areas=kwargs.get("focus_areas", []),
            priority=kwargs.get("priority", "medium"),
            market_segment=kwargs.get("market_segment", []),
            competitive_threat=kwargs.get("competitive_threat", "medium")
        )
        
        self._competitors.append(competitor)
        logger.info(f"Added competitor: {name}")
    
    def remove_competitor(self, name: str) -> bool:
        """
        Remove a competitor from the configuration.

        Args:
            name: Competitor name to remove

        Returns:
            True if competitor was removed, False if not found
        """
        for i, competitor in enumerate(self._competitors):
            if competitor.name.lower() == name.lower():
                del self._competitors[i]
                logger.info(f"Removed competitor: {name}")
                return True

        logger.warning(f"Competitor not found: {name}")
        return False

    def update_competitor_analysis_date(
        self,
        name: str,
        analysis_date: Union[datetime, str, None]
    ) -> None:
        """Update the ``last_analyzed`` timestamp for a competitor and persist it.

        Args:
            name: Name of the competitor to update.
            analysis_date: Datetime object or ISO formatted string representing the
                most recent analysis time. ``None`` clears the stored value.
        """

        competitor = self.get_competitor_by_name(name)
        if not competitor:
            logger.warning(
                "Attempted to update analysis date for unknown competitor '%s'",
                name
            )
            return

        parsed_date: Optional[datetime]
        if analysis_date is None:
            parsed_date = None
        elif isinstance(analysis_date, datetime):
            parsed_date = analysis_date
        elif isinstance(analysis_date, str):
            try:
                normalized_value = (
                    analysis_date.replace("Z", "+00:00")
                    if analysis_date.endswith("Z")
                    else analysis_date
                )
                parsed_date = datetime.fromisoformat(normalized_value)
            except ValueError:
                logger.warning(
                    "Invalid analysis date '%s' provided for competitor '%s'",
                    analysis_date,
                    name
                )
                return
        else:
            logger.warning(
                "Unsupported analysis date type '%s' for competitor '%s'",
                type(analysis_date).__name__,
                name
            )
            return

        competitor.last_analyzed = parsed_date

        # Refresh serialised representation and persist the change
        self._raw_config.setdefault("competitors", [])
        self._raw_config["competitors"] = self.competitors

        try:
            self.save_config()
        except Exception:  # pragma: no cover - persistence errors are logged
            logger.exception(
                "Failed to save updated analysis date for competitor '%s'",
                name
            )

    def get_competitor_by_name(self, name: str) -> Optional[CompetitorInfo]:
        """
        Get competitor configuration by name.

        Args:
            name: Competitor name
            
        Returns:
            CompetitorInfo object or None if not found
        """
        for competitor in self._competitors:
            if competitor.name.lower() == name.lower():
                return competitor
        return None
    
    def get_all_competitors(self) -> List[CompetitorInfo]:
        """Get all configured competitors."""
        return self._competitors.copy()

    def get_competitors_by_priority(self, priority: str) -> List[CompetitorInfo]:
        """
        Get competitors filtered by priority level.

        Args:
            priority: Priority level (low, medium, high)
            
        Returns:
            List of competitors with specified priority
        """
        return [comp for comp in self._competitors if comp.priority == priority]
    
    def is_data_source_enabled(self, source_name: str) -> bool:
        """
        Check if a data source is enabled.
        
        Args:
            source_name: Name of the data source to check
            
        Returns:
            True if enabled, False otherwise
        """
        source_mapping = {
            "websites": self.data_sources.company_websites,
            "funding": self.data_sources.funding_data,
            "jobs": self.data_sources.job_boards,
            "news": self.data_sources.news_sources,
            "social": self.data_sources.social_media,
            "github": self.data_sources.github_repos,
            "patents": self.data_sources.patent_databases
        }

        value = source_mapping.get(source_name)

        if isinstance(value, dict):
            return bool(value.get("enabled", False))

        return bool(value)
    
    def save_config(self) -> None:
        """Save current configuration back to YAML file."""
        config_dict = {
            "llm": {
                "provider": self.llm.provider,
                "models": self.llm.models,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "fallback_model": self.llm.fallback_model
            },
            "competitors": [
                {
                    "name": comp.name,
                    "website": comp.website,
                    "focus_areas": comp.focus_areas,
                    "priority": comp.priority,
                    "market_segment": comp.market_segment,
                    "competitive_threat": comp.competitive_threat,
                    "last_analyzed": comp.last_analyzed.isoformat() if comp.last_analyzed else None
                }
                for comp in self._competitors
            ],
            "analysis": self.analysis,
            "scraping": {
                "delay_between_requests": self.scraping.delay_between_requests,
                "max_pages_per_site": self.scraping.max_pages_per_site,
                "timeout": self.scraping.timeout,
                "concurrent_requests": self.scraping.concurrent_requests,
                "rate_limit": self.scraping.rate_limit,
                "user_agent": self.scraping.user_agent,
                "respect_robots_txt": self.scraping.respect_robots_txt,
                "target_pages": self.scraping.target_pages
            },
            "data_sources": {
                "company_websites": self.data_sources.company_websites,
                "funding_data": self.data_sources.funding_data,
                "job_boards": self.data_sources.job_boards,
                "news_sources": self.data_sources.news_sources,
                "social_media": self.data_sources.social_media,
                "github_repos": self.data_sources.github_repos,
                "patent_databases": self.data_sources.patent_databases
            },
            "output": {
                "formats": self.output.formats,
                "output_dir": self.output.output_dir,
                "include_charts": self.output.include_charts,
                "include_screenshots": self.output.include_screenshots,
                "include_competitive_matrix": self.output.include_competitive_matrix,
                "include_swot_analysis": self.output.include_swot_analysis,
                "pdf": self.output.pdf,
                "json": self.output.json
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {self.config_path}")
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Validate that required API keys are available.
        
        Returns:
            Dictionary mapping provider names to availability status
        """
        required_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", 
            "perplexity": "PERPLEXITY_API_KEY"
        }
        
        availability = {}
        for provider, env_var in required_keys.items():
            availability[provider] = env_var in os.environ and bool(os.environ[env_var])
        
        # Check if the configured provider has its API key
        main_provider_available = availability.get(self.llm.provider, False)
        if not main_provider_available:
            logger.warning(f"API key for configured provider '{self.llm.provider}' not found")
        
        return availability

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Return the LLM configuration as a plain dictionary."""
        return {
            "provider": self.llm.provider,
            "models": dict(self.llm.models),
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
            "fallback_model": self.llm.fallback_model
        }

    @property
    def competitors(self) -> List[Dict[str, Any]]:
        """Return configured competitors as dictionaries for easy consumption."""
        competitors_list: List[Dict[str, Any]] = []

        for competitor in self._competitors:
            competitor_dict = {
                "name": competitor.name,
                "website": competitor.website,
                "focus_areas": list(competitor.focus_areas),
                "priority": competitor.priority,
                "market_segment": list(competitor.market_segment),
                "competitive_threat": competitor.competitive_threat,
                "last_analyzed": (
                    competitor.last_analyzed.isoformat()
                    if isinstance(competitor.last_analyzed, datetime)
                    else competitor.last_analyzed
                )
            }
            competitors_list.append(competitor_dict)

        return competitors_list

    @property
    def analysis_config(self) -> Dict[str, Any]:
        """Return a defensive copy of the analysis configuration."""
        return deepcopy(self.analysis)

    @property
    def output_config(self) -> Dict[str, Any]:
        """Return the reporting/output configuration as a dictionary."""
        return {
            "formats": list(self.output.formats),
            "output_dir": self.output.output_dir,
            "include_charts": self.output.include_charts,
            "include_screenshots": self.output.include_screenshots,
            "include_competitive_matrix": self.output.include_competitive_matrix,
            "include_swot_analysis": self.output.include_swot_analysis,
            "pdf": deepcopy(self.output.pdf),
            "json": deepcopy(self.output.json)
        }

    def get_llm_model(self, task: str) -> str:
        """Return the configured model for a particular task with fallback."""
        return self.llm.models.get(task, self.llm.fallback_model)

    def _normalize_data_source_config(self, name: str, value: Any) -> Dict[str, Any]:
        """Merge a data source configuration with sensible defaults."""
        base_config: Dict[str, Any] = {
            "enabled": True,
            "cache_enabled": True,
            "cache_ttl_hours": 24,
            "cache_dir": str(Path("cache"))
        }

        if name == "company_websites":
            base_config.update({
                "user_agent": self.scraping.user_agent,
                "rate_limit": self.scraping.rate_limit,
                "delay_between_requests": self.scraping.delay_between_requests,
            })

        if isinstance(value, dict):
            merged = {**base_config, **value}
        elif isinstance(value, bool):
            merged = {**base_config, "enabled": value}
        else:
            merged = base_config.copy()

        return deepcopy(merged)

    def get_data_source_config(self, source_name: str) -> Dict[str, Any]:
        """Return the configuration for a particular data source."""
        if not hasattr(self.data_sources, source_name):
            raise KeyError(f"Unknown data source '{source_name}'")

        value = getattr(self.data_sources, source_name)
        return self._normalize_data_source_config(source_name, value)

    def get_target_pages(self) -> List[Dict[str, Any]]:
        """Return configured target pages for website scraping."""
        analysis_section = self.analysis if isinstance(self.analysis, dict) else {}
        target_pages = analysis_section.get("target_pages")

        if target_pages:
            return deepcopy(target_pages)

        return deepcopy(getattr(self.scraping, "target_pages", []))

    def get_analysis_config_object(self) -> AnalysisConfigSummary:
        """Create a rich summary object used throughout the application."""
        depth = self.analysis.get("depth_level", "standard")

        data_sources_status = {
            "websites": self.is_data_source_enabled("websites"),
            "funding": self.is_data_source_enabled("funding"),
            "jobs": self.is_data_source_enabled("jobs"),
            "news": self.is_data_source_enabled("news"),
            "social": self.is_data_source_enabled("social"),
            "github": self.is_data_source_enabled("github"),
            "patents": self.is_data_source_enabled("patents")
        }

        return AnalysisConfigSummary(
            depth_level=_DepthLevel(depth),
            competitors=[comp["name"] for comp in self.competitors],
            output_formats=list(self.output_config["formats"]),
            target_pages=self.get_target_pages(),
            data_sources=data_sources_status
        )

    def override_from_cli_args(self, args: Any) -> None:
        """Apply runtime overrides provided by CLI arguments."""
        updated = False

        output_dir = getattr(args, "output", None) or getattr(args, "output_dir", None)
        if output_dir:
            self.output.output_dir = output_dir
            updated = True

        depth = getattr(args, "depth", None)
        if depth:
            self.analysis["depth_level"] = depth
            updated = True

        formats = getattr(args, "format", None) or getattr(args, "formats", None)
        if formats:
            if isinstance(formats, str):
                parsed_formats = [fmt.strip() for fmt in formats.split(',') if fmt.strip()]
            else:
                parsed_formats = list(formats)
            if parsed_formats:
                self.output.formats = parsed_formats
                updated = True

        if updated:
            logger.debug("Configuration overridden via CLI arguments")

    def __repr__(self) -> str:
        """String representation of the configuration."""
        return (f"CompetitorConfig(competitors={len(self._competitors)}, "
                f"provider={self.llm.provider}, "
                f"output_formats={self.output.formats})")


# Example usage and validation
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize configuration
    config = CompetitorConfig("competitor_config.yaml")
    
    # Add a competitor
    config.add_competitor(
        name="Algolia",
        website="https://www.algolia.com",
        focus_areas=["search", "autocomplete", "recommendations"],
        priority="high",
        market_segment=["enterprise", "mid-market"],
        competitive_threat="high"
    )
    
    # Validate API keys
    api_status = config.validate_api_keys()
    print(f"API Key Status: {api_status}")
    
    # Check data source status
    print(f"Website scraping enabled: {config.is_data_source_enabled('websites')}")
    print(f"Funding data enabled: {config.is_data_source_enabled('funding')}")
    
    # Save configuration
    config.save_config()
    
    print(f"Configuration: {config}")