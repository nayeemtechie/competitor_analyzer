# src/competitor/models.py
"""
Data models and schemas for competitor analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta  # Added missing timedelta import
from enum import Enum

class ThreatLevel(Enum):
    """Competitive threat level assessment"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AnalysisDepth(Enum):
    """Analysis depth levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

class Priority(Enum):
    """Competitor analysis priority"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class PricingTier:
    """Pricing tier information"""
    name: str
    price: Optional[str] = None
    features: List[str] = field(default_factory=list)
    target_segment: Optional[str] = None
    billing_period: Optional[str] = None
    popular: bool = False

@dataclass
class CaseStudy:
    """Customer case study information"""
    title: str
    customer_name: Optional[str] = None
    industry: Optional[str] = None
    summary: Optional[str] = None
    url: Optional[str] = None
    results: List[str] = field(default_factory=list)

@dataclass
class NewsItem:
    """News mention or article"""
    title: str
    source: str
    date: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None
    sentiment: Optional[str] = None  # positive, neutral, negative

@dataclass
class JobPosting:
    """Job posting information"""
    title: str
    department: str
    location: Optional[str] = None
    posted_date: Optional[str] = None
    url: Optional[str] = None
    requirements: List[str] = field(default_factory=list)

@dataclass
class SocialPresence:
    """Social media presence data"""
    platform: str
    handle: Optional[str] = None
    url: Optional[str] = None
    followers: Optional[int] = None
    activity_level: Optional[str] = None  # high, medium, low
    last_post_date: Optional[str] = None

@dataclass
class GitHubActivity:
    """GitHub activity metrics"""
    organization: Optional[str] = None
    public_repos: int = 0
    contributors: int = 0
    total_commits: int = 0
    languages: List[str] = field(default_factory=list)
    activity_score: Optional[float] = None
    last_updated: Optional[str] = None

@dataclass
class PatentData:
    """Patent information"""
    total_patents: int = 0
    recent_patents: List[Dict[str, str]] = field(default_factory=list)
    technology_areas: List[str] = field(default_factory=list)
    filing_trend: Optional[str] = None  # increasing, stable, decreasing

@dataclass
class FundingInfo:
    """Funding and financial data"""
    total_funding: Optional[str] = None
    last_round_amount: Optional[str] = None
    last_round_type: Optional[str] = None  # seed, series_a, series_b, etc.
    last_round_date: Optional[str] = None
    valuation: Optional[str] = None
    investors: List[str] = field(default_factory=list)
    funding_trend: Optional[str] = None

@dataclass
class WebsiteData:
    """Website analysis data"""
    pages_analyzed: List[str] = field(default_factory=list)
    key_pages: Dict[str, Any] = field(default_factory=dict)
    technology_stack: List[str] = field(default_factory=list)
    seo_metrics: Dict[str, Any] = field(default_factory=dict)
    content_themes: List[str] = field(default_factory=list)
    last_analyzed: Optional[str] = None

@dataclass
class CompetitorProfile:
    """Comprehensive competitor profile"""
    # Basic information
    name: str
    website: str
    founded: Optional[str] = None
    headquarters: Optional[str] = None
    employees: Optional[str] = None
    
    # Business information
    business_model: Optional[str] = None
    target_markets: List[str] = field(default_factory=list)
    market_segments: List[str] = field(default_factory=list)
    
    # Product information
    key_features: List[str] = field(default_factory=list)
    pricing_tiers: List[PricingTier] = field(default_factory=list)
    technology_stack: List[str] = field(default_factory=list)
    integrations: List[str] = field(default_factory=list)
    
    # Intelligence data
    funding_info: Optional[FundingInfo] = None
    website_data: Optional[WebsiteData] = None
    case_studies: List[CaseStudy] = field(default_factory=list)
    recent_news: List[NewsItem] = field(default_factory=list)
    job_postings: List[JobPosting] = field(default_factory=list)
    social_presence: List[SocialPresence] = field(default_factory=list)
    github_activity: Optional[GitHubActivity] = None
    patent_data: Optional[PatentData] = None
    
    # Analysis metadata
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    competitive_overlap: Optional[float] = None  # 0.0 to 1.0
    last_analyzed: Optional[str] = None
    analysis_version: str = "1.0"
    
    # Scoring and assessment
    feature_score: Optional[float] = None
    market_position_score: Optional[float] = None
    innovation_score: Optional[float] = None
    overall_threat_score: Optional[float] = None

@dataclass
class AnalysisConfig:
    """Configuration for competitor analysis"""
    depth_level: AnalysisDepth = AnalysisDepth.STANDARD
    competitors: List[str] = field(default_factory=list)
    output_formats: List[str] = field(default_factory=lambda: ["pdf"])
    output_dir: str = "competitor_reports"
    
    # Analysis settings
    max_pages_per_site: int = 50
    request_delay: float = 2.0
    timeout: int = 30
    concurrent_requests: int = 3
    
    # Data source settings
    analyze_website: bool = True
    analyze_funding: bool = True
    analyze_jobs: bool = True
    analyze_news: bool = True
    analyze_social: bool = True
    analyze_github: bool = True
    analyze_patents: bool = False
    
    # Time ranges
    news_days_back: int = 90
    jobs_days_back: int = 30
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisConfig':
        """Create config from dictionary"""
        return cls(
            depth_level=AnalysisDepth(data.get('depth_level', 'standard')),
            competitors=data.get('competitors', []),
            output_formats=data.get('output_formats', ['pdf']),
            output_dir=data.get('output_dir', 'competitor_reports'),
            max_pages_per_site=data.get('max_pages_per_site', 50),
            request_delay=data.get('request_delay', 2.0),
            timeout=data.get('timeout', 30),
            concurrent_requests=data.get('concurrent_requests', 3),
            analyze_website=data.get('analyze_website', True),
            analyze_funding=data.get('analyze_funding', True),
            analyze_jobs=data.get('analyze_jobs', True),
            analyze_news=data.get('analyze_news', True),
            analyze_social=data.get('analyze_social', True),
            analyze_github=data.get('analyze_github', True),
            analyze_patents=data.get('analyze_patents', False),
            news_days_back=data.get('news_days_back', 90),
            jobs_days_back=data.get('jobs_days_back', 30)
        )

@dataclass 
class CompetitorIntelligence:
    """Container for all competitor intelligence data"""
    profiles: List[CompetitorProfile] = field(default_factory=list)
    analysis_date: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Optional[AnalysisConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_by_name(self, name: str) -> Optional[CompetitorProfile]:
        """Get competitor profile by name"""
        for profile in self.profiles:
            if profile.name.lower() == name.lower():
                return profile
        return None
    
    def get_high_threat_competitors(self) -> List[CompetitorProfile]:
        """Get competitors with high threat level"""
        return [p for p in self.profiles if p.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]
    
    def get_recent_news(self, days: int = 30) -> List[NewsItem]:
        """Get recent news across all competitors"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_news = []
        
        for profile in self.profiles:
            for news in profile.recent_news:
                if news.date:
                    try:
                        # More robust date parsing
                        from dateutil import parser as date_parser
                        news_date = date_parser.parse(news.date)
                        if news_date >= cutoff_date:
                            recent_news.append(news)
                    except Exception:
                        continue
        
        return sorted(recent_news, key=lambda x: x.date or '', reverse=True)
    
    def get_funding_summary(self) -> Dict[str, Any]:
        """Get funding summary across all competitors"""
        total_funding = 0
        recent_rounds = []
        
        for profile in self.profiles:
            if profile.funding_info and profile.funding_info.total_funding:
                try:
                    # More robust funding parsing
                    funding_str = profile.funding_info.total_funding.upper()
                    amount_str = funding_str.replace('$', '').replace(',', '')
                    
                    if 'B' in amount_str:
                        amount = float(amount_str.replace('B', '')) * 1000  # Convert to millions
                        total_funding += amount
                    elif 'M' in amount_str:
                        amount = float(amount_str.replace('M', ''))
                        total_funding += amount
                except (ValueError, AttributeError):
                    pass
            
            if profile.funding_info and profile.funding_info.last_round_date:
                recent_rounds.append({
                    'company': profile.name,
                    'amount': profile.funding_info.last_round_amount,
                    'type': profile.funding_info.last_round_type,
                    'date': profile.funding_info.last_round_date
                })
        
        return {
            'total_market_funding': f"${total_funding:.1f}M",
            'recent_rounds': sorted(recent_rounds, key=lambda x: x['date'] or '', reverse=True)[:10]
        }