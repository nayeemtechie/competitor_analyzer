# src/competitor/models.py
"""
Core data models and schemas for the competitor analysis system.
Defines the structure for competitor profiles, analysis results, and reports.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import json
from pathlib import Path


class ThreatLevel(Enum):
    """Competitive threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Priority(Enum):
    """Analysis priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MarketSegment(Enum):
    """Market segment categories."""
    ENTERPRISE = "enterprise"
    MID_MARKET = "mid-market"
    SMB = "smb"
    STARTUP = "startup"
    CONSUMER = "consumer"


class DataSourceType(Enum):
    """Types of data sources."""
    WEBSITE = "website"
    FUNDING = "funding"
    JOBS = "jobs"
    NEWS = "news"
    SOCIAL = "social"
    GITHUB = "github"
    PATENTS = "patents"
    MANUAL = "manual"


@dataclass
class ContactInfo:
    """Contact information for a company."""
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    headquarters: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "email": self.email,
            "phone": self.phone,
            "address": self.address,
            "headquarters": self.headquarters
        }


@dataclass
class FundingRound:
    """Individual funding round information."""
    round_type: str  # seed, series_a, series_b, etc.
    amount: Optional[float] = None
    currency: str = "USD"
    date: Optional[datetime] = None
    investors: List[str] = field(default_factory=list)
    valuation: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_type": self.round_type,
            "amount": self.amount,
            "currency": self.currency,
            "date": self.date.isoformat() if self.date else None,
            "investors": self.investors,
            "valuation": self.valuation
        }


@dataclass
class FundingInfo:
    """Complete funding information for a company."""
    total_funding: Optional[float] = None
    currency: str = "USD"
    funding_rounds: List[FundingRound] = field(default_factory=list)
    last_funding_date: Optional[datetime] = None
    investors: List[str] = field(default_factory=list)
    current_valuation: Optional[float] = None
    ipo_status: Optional[str] = None  # private, public, acquired
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_funding": self.total_funding,
            "currency": self.currency,
            "funding_rounds": [round.to_dict() for round in self.funding_rounds],
            "last_funding_date": self.last_funding_date.isoformat() if self.last_funding_date else None,
            "investors": self.investors,
            "current_valuation": self.current_valuation,
            "ipo_status": self.ipo_status
        }


@dataclass
class JobPosting:
    """Individual job posting information."""
    title: str
    department: str
    location: str
    posted_date: Optional[datetime] = None
    source: Optional[str] = None
    requirements: List[str] = field(default_factory=list)
    salary_range: Optional[str] = None
    remote_option: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "department": self.department,
            "location": self.location,
            "posted_date": self.posted_date.isoformat() if self.posted_date else None,
            "source": self.source,
            "requirements": self.requirements,
            "salary_range": self.salary_range,
            "remote_option": self.remote_option
        }


@dataclass
class SocialMediaPresence:
    """Social media presence information."""
    platform: str
    url: Optional[str] = None
    followers: Optional[int] = None
    following: Optional[int] = None
    posts: Optional[int] = None
    engagement_rate: Optional[float] = None
    last_post_date: Optional[datetime] = None
    verified: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "url": self.url,
            "followers": self.followers,
            "following": self.following,
            "posts": self.posts,
            "engagement_rate": self.engagement_rate,
            "last_post_date": self.last_post_date.isoformat() if self.last_post_date else None,
            "verified": self.verified
        }


@dataclass
class GitHubActivity:
    """GitHub repository and activity information."""
    username: Optional[str] = None
    public_repos: Optional[int] = None
    followers: Optional[int] = None
    total_stars: Optional[int] = None
    primary_languages: List[str] = field(default_factory=list)
    recent_commits: Optional[int] = None
    active_contributors: Optional[int] = None
    last_activity: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "username": self.username,
            "public_repos": self.public_repos,
            "followers": self.followers,
            "total_stars": self.total_stars,
            "primary_languages": self.primary_languages,
            "recent_commits": self.recent_commits,
            "active_contributors": self.active_contributors,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }


@dataclass
class NewsItem:
    """News article or mention."""
    title: str
    url: str
    source: str
    published_date: Optional[datetime] = None
    summary: Optional[str] = None
    sentiment: Optional[str] = None  # positive, negative, neutral
    relevance_score: Optional[float] = None
    category: Optional[str] = None  # funding, product, partnership, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "summary": self.summary,
            "sentiment": self.sentiment,
            "relevance_score": self.relevance_score,
            "category": self.category
        }


@dataclass
class Patent:
    """Patent information."""
    patent_id: str
    title: str
    description: Optional[str] = None
    filing_date: Optional[datetime] = None
    publication_date: Optional[datetime] = None
    inventors: List[str] = field(default_factory=list)
    assignee: Optional[str] = None
    status: Optional[str] = None  # pending, granted, expired
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "patent_id": self.patent_id,
            "title": self.title,
            "description": self.description,
            "filing_date": self.filing_date.isoformat() if self.filing_date else None,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "inventors": self.inventors,
            "assignee": self.assignee,
            "status": self.status
        }


@dataclass
class WebsiteData:
    """Website content and metadata."""
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    meta_keywords: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    last_crawled: Optional[datetime] = None
    page_type: Optional[str] = None  # homepage, pricing, product, etc.
    load_time: Optional[float] = None
    mobile_friendly: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "content": self.content[:1000] if self.content else None,  # Truncate for serialization
            "meta_keywords": self.meta_keywords,
            "technologies": self.technologies,
            "last_crawled": self.last_crawled.isoformat() if self.last_crawled else None,
            "page_type": self.page_type,
            "load_time": self.load_time,
            "mobile_friendly": self.mobile_friendly
        }


@dataclass
class CompetitorProfile:
    """Complete profile for a single competitor."""
    # Basic Information
    name: str
    website: str
    description: Optional[str] = None
    founded_year: Optional[int] = None
    headquarters: Optional[str] = None
    employee_count: Optional[int] = None
    
    # Classification
    focus_areas: List[str] = field(default_factory=list)
    market_segments: List[MarketSegment] = field(default_factory=list)
    target_markets: List[str] = field(default_factory=list)
    business_model: Optional[str] = None
    
    # Competitive Analysis
    competitive_threat: ThreatLevel = ThreatLevel.MEDIUM
    priority: Priority = Priority.MEDIUM
    key_features: List[str] = field(default_factory=list)
    competitive_advantages: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    # Data Sources
    contact_info: Optional[ContactInfo] = None
    funding_info: Optional[FundingInfo] = None
    job_postings: List[JobPosting] = field(default_factory=list)
    social_presence: List[SocialMediaPresence] = field(default_factory=list)
    github_activity: Optional[GitHubActivity] = None
    news_items: List[NewsItem] = field(default_factory=list)
    patents: List[Patent] = field(default_factory=list)
    website_data: List[WebsiteData] = field(default_factory=list)
    
    # Metadata
    last_analyzed: Optional[datetime] = None
    analysis_depth: str = "standard"  # basic, standard, comprehensive
    data_sources_used: List[DataSourceType] = field(default_factory=list)
    confidence_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "website": self.website,
            "description": self.description,
            "founded_year": self.founded_year,
            "headquarters": self.headquarters,
            "employee_count": self.employee_count,
            "focus_areas": self.focus_areas,
            "market_segments": [seg.value for seg in self.market_segments],
            "target_markets": self.target_markets,
            "business_model": self.business_model,
            "competitive_threat": self.competitive_threat.value,
            "priority": self.priority.value,
            "key_features": self.key_features,
            "competitive_advantages": self.competitive_advantages,
            "weaknesses": self.weaknesses,
            "contact_info": self.contact_info.to_dict() if self.contact_info else None,
            "funding_info": self.funding_info.to_dict() if self.funding_info else None,
            "job_postings": [job.to_dict() for job in self.job_postings],
            "social_presence": [social.to_dict() for social in self.social_presence],
            "github_activity": self.github_activity.to_dict() if self.github_activity else None,
            "news_items": [news.to_dict() for news in self.news_items],
            "patents": [patent.to_dict() for patent in self.patents],
            "website_data": [site.to_dict() for site in self.website_data],
            "last_analyzed": self.last_analyzed.isoformat() if self.last_analyzed else None,
            "analysis_depth": self.analysis_depth,
            "data_sources_used": [source.value for source in self.data_sources_used],
            "confidence_score": self.confidence_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompetitorProfile':
        """Create CompetitorProfile from dictionary."""
        # Convert enum fields
        market_segments = [MarketSegment(seg) for seg in data.get("market_segments", [])]
        competitive_threat = ThreatLevel(data.get("competitive_threat", "medium"))
        priority = Priority(data.get("priority", "medium"))
        data_sources_used = [DataSourceType(source) for source in data.get("data_sources_used", [])]
        
        # Convert datetime fields
        last_analyzed = None
        if data.get("last_analyzed"):
            last_analyzed = datetime.fromisoformat(data["last_analyzed"])
        
        # Convert nested objects
        contact_info = None
        if data.get("contact_info"):
            contact_info = ContactInfo(**data["contact_info"])
        
        funding_info = None
        if data.get("funding_info"):
            funding_data = data["funding_info"]
            funding_rounds = []
            for round_data in funding_data.get("funding_rounds", []):
                if round_data.get("date"):
                    round_data["date"] = datetime.fromisoformat(round_data["date"])
                funding_rounds.append(FundingRound(**round_data))
            
            funding_data["funding_rounds"] = funding_rounds
            if funding_data.get("last_funding_date"):
                funding_data["last_funding_date"] = datetime.fromisoformat(funding_data["last_funding_date"])
            funding_info = FundingInfo(**funding_data)
        
        # Convert lists of objects
        job_postings = []
        for job_data in data.get("job_postings", []):
            if job_data.get("posted_date"):
                job_data["posted_date"] = datetime.fromisoformat(job_data["posted_date"])
            job_postings.append(JobPosting(**job_data))
        
        social_presence = []
        for social_data in data.get("social_presence", []):
            if social_data.get("last_post_date"):
                social_data["last_post_date"] = datetime.fromisoformat(social_data["last_post_date"])
            social_presence.append(SocialMediaPresence(**social_data))
        
        github_activity = None
        if data.get("github_activity"):
            github_data = data["github_activity"]
            if github_data.get("last_activity"):
                github_data["last_activity"] = datetime.fromisoformat(github_data["last_activity"])
            github_activity = GitHubActivity(**github_data)
        
        news_items = []
        for news_data in data.get("news_items", []):
            if news_data.get("published_date"):
                news_data["published_date"] = datetime.fromisoformat(news_data["published_date"])
            news_items.append(NewsItem(**news_data))
        
        patents = []
        for patent_data in data.get("patents", []):
            for date_field in ["filing_date", "publication_date"]:
                if patent_data.get(date_field):
                    patent_data[date_field] = datetime.fromisoformat(patent_data[date_field])
            patents.append(Patent(**patent_data))
        
        website_data = []
        for site_data in data.get("website_data", []):
            if site_data.get("last_crawled"):
                site_data["last_crawled"] = datetime.fromisoformat(site_data["last_crawled"])
            website_data.append(WebsiteData(**site_data))
        
        return cls(
            name=data["name"],
            website=data["website"],
            description=data.get("description"),
            founded_year=data.get("founded_year"),
            headquarters=data.get("headquarters"),
            employee_count=data.get("employee_count"),
            focus_areas=data.get("focus_areas", []),
            market_segments=market_segments,
            target_markets=data.get("target_markets", []),
            business_model=data.get("business_model"),
            competitive_threat=competitive_threat,
            priority=priority,
            key_features=data.get("key_features", []),
            competitive_advantages=data.get("competitive_advantages", []),
            weaknesses=data.get("weaknesses", []),
            contact_info=contact_info,
            funding_info=funding_info,
            job_postings=job_postings,
            social_presence=social_presence,
            github_activity=github_activity,
            news_items=news_items,
            patents=patents,
            website_data=website_data,
            last_analyzed=last_analyzed,
            analysis_depth=data.get("analysis_depth", "standard"),
            data_sources_used=data_sources_used,
            confidence_score=data.get("confidence_score")
        )


@dataclass
class CompetitiveMatrix:
    """Competitive comparison matrix."""
    features: List[str] = field(default_factory=list)
    competitors: List[str] = field(default_factory=list)
    matrix: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # competitor -> feature -> value
    
    def add_competitor(self, competitor_name: str, feature_scores: Dict[str, Any]):
        """Add a competitor's feature scores to the matrix."""
        self.matrix[competitor_name] = feature_scores
        if competitor_name not in self.competitors:
            self.competitors.append(competitor_name)
        
        # Update features list
        for feature in feature_scores.keys():
            if feature not in self.features:
                self.features.append(feature)
    
    def get_feature_comparison(self, feature: str) -> Dict[str, Any]:
        """Get all competitors' scores for a specific feature."""
        return {comp: self.matrix.get(comp, {}).get(feature) for comp in self.competitors}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "competitors": self.competitors,
            "matrix": self.matrix
        }


@dataclass
class SWOTAnalysis:
    """SWOT analysis for a competitor or market."""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "opportunities": self.opportunities,
            "threats": self.threats
        }


@dataclass
class MarketIntelligence:
    """Market-level competitive intelligence."""
    market_size: Optional[float] = None
    growth_rate: Optional[float] = None
    key_trends: List[str] = field(default_factory=list)
    market_leaders: List[str] = field(default_factory=list)
    emerging_players: List[str] = field(default_factory=list)
    technology_trends: List[str] = field(default_factory=list)
    regulatory_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_size": self.market_size,
            "growth_rate": self.growth_rate,
            "key_trends": self.key_trends,
            "market_leaders": self.market_leaders,
            "emerging_players": self.emerging_players,
            "technology_trends": self.technology_trends,
            "regulatory_factors": self.regulatory_factors
        }


@dataclass
class CompetitorIntelligence:
    """Complete competitive intelligence report."""
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    analysis_depth: str = "standard"
    total_competitors: int = 0
    
    # Core Data
    competitor_profiles: List[CompetitorProfile] = field(default_factory=list)
    competitive_matrix: Optional[CompetitiveMatrix] = None
    market_intelligence: Optional[MarketIntelligence] = None
    
    # Analysis Results
    executive_summary: Optional[str] = None
    key_insights: List[str] = field(default_factory=list)
    strategic_recommendations: List[str] = field(default_factory=list)
    threat_assessment: Dict[str, ThreatLevel] = field(default_factory=dict)
    
    # SWOT Analyses
    overall_swot: Optional[SWOTAnalysis] = None
    competitor_swots: Dict[str, SWOTAnalysis] = field(default_factory=dict)
    
    def add_competitor_profile(self, profile: CompetitorProfile):
        """Add a competitor profile to the intelligence."""
        self.competitor_profiles.append(profile)
        self.total_competitors = len(self.competitor_profiles)
        self.threat_assessment[profile.name] = profile.competitive_threat
    
    def get_competitor_by_name(self, name: str) -> Optional[CompetitorProfile]:
        """Get a specific competitor profile by name."""
        for profile in self.competitor_profiles:
            if profile.name.lower() == name.lower():
                return profile
        return None
    
    def get_competitors_by_threat_level(self, threat_level: ThreatLevel) -> List[CompetitorProfile]:
        """Get competitors filtered by threat level."""
        return [profile for profile in self.competitor_profiles 
                if profile.competitive_threat == threat_level]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "analysis_depth": self.analysis_depth,
            "total_competitors": self.total_competitors,
            "competitor_profiles": [profile.to_dict() for profile in self.competitor_profiles],
            "competitive_matrix": self.competitive_matrix.to_dict() if self.competitive_matrix else None,
            "market_intelligence": self.market_intelligence.to_dict() if self.market_intelligence else None,
            "executive_summary": self.executive_summary,
            "key_insights": self.key_insights,
            "strategic_recommendations": self.strategic_recommendations,
            "threat_assessment": {name: level.value for name, level in self.threat_assessment.items()},
            "overall_swot": self.overall_swot.to_dict() if self.overall_swot else None,
            "competitor_swots": {name: swot.to_dict() for name, swot in self.competitor_swots.items()}
        }
    
    def save_to_file(self, filepath: Union[str, Path]):
        """Save intelligence report to JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'CompetitorIntelligence':
        """Load intelligence report from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert datetime
        generated_at = datetime.fromisoformat(data["generated_at"])
        
        # Convert competitor profiles
        competitor_profiles = [
            CompetitorProfile.from_dict(profile_data) 
            for profile_data in data.get("competitor_profiles", [])
        ]
        
        # Convert competitive matrix
        competitive_matrix = None
        if data.get("competitive_matrix"):
            matrix_data = data["competitive_matrix"]
            competitive_matrix = CompetitiveMatrix(
                features=matrix_data["features"],
                competitors=matrix_data["competitors"],
                matrix=matrix_data["matrix"]
            )
        
        # Convert market intelligence
        market_intelligence = None
        if data.get("market_intelligence"):
            market_intelligence = MarketIntelligence(**data["market_intelligence"])
        
        # Convert threat assessment
        threat_assessment = {
            name: ThreatLevel(level) 
            for name, level in data.get("threat_assessment", {}).items()
        }
        
        # Convert SWOT analyses
        overall_swot = None
        if data.get("overall_swot"):
            overall_swot = SWOTAnalysis(**data["overall_swot"])
        
        competitor_swots = {
            name: SWOTAnalysis(**swot_data)
            for name, swot_data in data.get("competitor_swots", {}).items()
        }
        
        return cls(
            generated_at=generated_at,
            analysis_depth=data.get("analysis_depth", "standard"),
            total_competitors=data.get("total_competitors", 0),
            competitor_profiles=competitor_profiles,
            competitive_matrix=competitive_matrix,
            market_intelligence=market_intelligence,
            executive_summary=data.get("executive_summary"),
            key_insights=data.get("key_insights", []),
            strategic_recommendations=data.get("strategic_recommendations", []),
            threat_assessment=threat_assessment,
            overall_swot=overall_swot,
            competitor_swots=competitor_swots
        )


# Utility functions for data validation and processing
def validate_competitor_profile(profile: CompetitorProfile) -> List[str]:
    """Validate a competitor profile and return list of issues."""
    issues = []
    
    if not profile.name:
        issues.append("Competitor name is required")
    
    if not profile.website:
        issues.append("Website URL is required")
    elif not profile.website.startswith(('http://', 'https://')):
        issues.append("Website URL must include protocol (http:// or https://)")
    
    if profile.founded_year and (profile.founded_year < 1800 or profile.founded_year > datetime.now().year):
        issues.append("Founded year seems invalid")
    
    if profile.employee_count and profile.employee_count < 0:
        issues.append("Employee count cannot be negative")
    
    if profile.confidence_score and (profile.confidence_score < 0 or profile.confidence_score > 1):
        issues.append("Confidence score must be between 0 and 1")
    
    return issues


def create_sample_competitor() -> CompetitorProfile:
    """Create a sample competitor profile for testing."""
    return CompetitorProfile(
        name="Algolia",
        website="https://www.algolia.com",
        description="Search and discovery API platform",
        founded_year=2012,
        headquarters="San Francisco, CA",
        employee_count=500,
        focus_areas=["search", "autocomplete", "recommendations"],
        market_segments=[MarketSegment.ENTERPRISE, MarketSegment.MID_MARKET],
        target_markets=["e-commerce", "media", "saas"],
        business_model="API/SaaS",
        competitive_threat=ThreatLevel.HIGH,
        priority=Priority.HIGH,
        key_features=["real-time search", "typo tolerance", "faceted search"],
        competitive_advantages=["speed", "ease of integration", "analytics"],
        weaknesses=["pricing", "vendor lock-in"],
        funding_info=FundingInfo(
            total_funding=334000000,
            current_valuation=2250000000,
            ipo_status="public"
        ),
        data_sources_used=[DataSourceType.WEBSITE, DataSourceType.FUNDING, DataSourceType.NEWS],
        confidence_score=0.85
    )


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_competitor = create_sample_competitor()
    
    # Validate
    issues = validate_competitor_profile(sample_competitor)
    if issues:
        print(f"Validation issues: {issues}")
    else:
        print("Profile is valid")
    
    # Create intelligence report
    intelligence = CompetitorIntelligence()
    intelligence.add_competitor_profile(sample_competitor)
    intelligence.executive_summary = "Sample competitive intelligence report"
    intelligence.key_insights = [
        "Algolia dominates the API-first search market",
        "Strong developer adoption and ecosystem"
    ]
    
    # Save to file
    intelligence.save_to_file("sample_intelligence.json")
    print("Intelligence report saved")
    
    # Load from file
    loaded_intelligence = CompetitorIntelligence.load_from_file("sample_intelligence.json")
    print(f"Loaded report with {loaded_intelligence.total_competitors} competitors")