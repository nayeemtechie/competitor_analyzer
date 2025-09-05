# src/competitor/analysis/__init__.py
"""
Analysis modules for competitor intelligence
"""

from .company import CompanyAnalyzer
from .features import FeatureAnalyzer
from .market import MarketAnalyzer
from .pricing import PricingAnalyzer
from .threats import ThreatAnalyzer
from .innovation import InnovationAnalyzer

class AnalysisEngine:
    """Main analysis engine that coordinates all analysis modules"""
    
    def __init__(self, config, llm_provider):
        self.config = config
        self.llm_provider = llm_provider
        
        # Initialize analyzers
        self.company_analyzer = CompanyAnalyzer(config, llm_provider)
        self.feature_analyzer = FeatureAnalyzer(config, llm_provider)
        self.market_analyzer = MarketAnalyzer(config, llm_provider)
        self.pricing_analyzer = PricingAnalyzer(config, llm_provider)
        self.threat_analyzer = ThreatAnalyzer(config, llm_provider)
        self.innovation_analyzer = InnovationAnalyzer(config, llm_provider)
    
    async def analyze_competitor_profile(self, profile):
        """Run comprehensive analysis on a competitor profile"""
        # Company analysis
        company_insights = await self.company_analyzer.analyze_company_profile(profile)
        
        # Feature analysis
        feature_insights = await self.feature_analyzer.analyze_features(profile)
        
        # Market analysis
        market_insights = await self.market_analyzer.analyze_market_position(profile)
        
        # Pricing analysis
        pricing_insights = await self.pricing_analyzer.analyze_pricing_strategy(profile)
        
        # Threat analysis
        threat_insights = await self.threat_analyzer.assess_competitive_threat(profile)
        
        # Innovation analysis
        innovation_insights = await self.innovation_analyzer.analyze_innovation_trajectory(profile)
        
        # Update profile with analysis results
        profile.feature_score = feature_insights.get('overall_score', 0.0)
        profile.market_position_score = market_insights.get('position_score', 0.0)
        profile.innovation_score = innovation_insights.get('innovation_score', 0.0)
        profile.overall_threat_score = threat_insights.get('threat_score', 0.0)
        
        return profile
    
    async def perform_cross_competitor_analysis(self, intelligence):
        """Perform cross-competitor comparative analysis"""
        if len(intelligence.profiles) < 2:
            return intelligence
        
        # Comparative feature analysis
        feature_comparison = await self.feature_analyzer.compare_features_across_competitors(
            intelligence.profiles
        )
        
        # Market positioning analysis
        market_comparison = await self.market_analyzer.analyze_competitive_landscape(
            intelligence.profiles
        )
        
        # Threat prioritization
        threat_matrix = await self.threat_analyzer.create_threat_matrix(
            intelligence.profiles
        )
        
        # Store comparative analysis in metadata
        intelligence.metadata.update({
            'feature_comparison': feature_comparison,
            'market_comparison': market_comparison,
            'threat_matrix': threat_matrix
        })
        
        return intelligence
    
    async def generate_executive_summary(self, intelligence):
        """Generate executive summary of competitive landscape"""
        return await self.company_analyzer.generate_executive_summary(intelligence)

__all__ = [
    'AnalysisEngine',
    'CompanyAnalyzer',
    'FeatureAnalyzer',
    'MarketAnalyzer',
    'PricingAnalyzer',
    'ThreatAnalyzer',
    'InnovationAnalyzer'
]