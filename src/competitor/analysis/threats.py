# src/competitor/analysis/threats.py
"""
Threat assessment for competitive intelligence
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ThreatAnalyzer:
    """Analyzes competitive threats and risks"""
    
    def __init__(self, config, llm_provider):
        self.config = config
        self.llm = llm_provider
    
    async def assess_competitive_threat(self, profile) -> Dict[str, Any]:
        """Assess competitive threat level for a single competitor"""
        threat_assessment = {
            'overall_threat_score': 0.0,
            'threat_categories': {},
            'immediate_threats': [],
            'long_term_threats': [],
            'threat_factors': {},
            'mitigation_strategies': []
        }
        
        # Calculate threat scores for different categories
        threat_categories = {
            'market_threat': self._assess_market_threat(profile),
            'technology_threat': self._assess_technology_threat(profile),
            'financial_threat': self._assess_financial_threat(profile),
            'customer_threat': self._assess_customer_threat(profile),
            'innovation_threat': self._assess_innovation_threat(profile)
        }
        
        threat_assessment['threat_categories'] = threat_categories
        
        # Calculate overall threat score (0.0 to 1.0)
        overall_score = sum(threat_categories.values()) / len(threat_categories)
        threat_assessment['overall_threat_score'] = round(overall_score, 2)
        
        # Identify specific threats
        threat_assessment['immediate_threats'] = await self._identify_immediate_threats(profile, threat_categories)
        threat_assessment['long_term_threats'] = await self._identify_long_term_threats(profile, threat_categories)
        
        # Detailed threat factors
        threat_assessment['threat_factors'] = self._analyze_threat_factors(profile)
        
        # Suggest mitigation strategies
        threat_assessment['mitigation_strategies'] = await self._suggest_mitigation_strategies(
            profile, threat_categories
        )
        
        return threat_assessment
    
    def _assess_market_threat(self, profile) -> float:
        """Assess market-based competitive threat (0.0 to 1.0)"""
        score = 0.0
        
        # Market position factors
        if profile.target_markets:
            # Direct market overlap threat
            overlapping_markets = ['enterprise', 'mid-market', 'ecommerce', 'saas']
            overlap_count = len([m for m in profile.target_markets if m.lower() in overlapping_markets])
            score += min(overlap_count * 0.15, 0.3)
        
        # Customer success threat
        if profile.case_studies:
            case_count = len(profile.case_studies)
            if case_count > 15:
                score += 0.25
            elif case_count > 8:
                score += 0.15
            elif case_count > 3:
                score += 0.1
        
        # Market momentum threat (from news)
        if profile.recent_news:
            recent_positive_news = [news for news in profile.recent_news 
                                  if news.sentiment == 'positive']
            if len(recent_positive_news) > 5:
                score += 0.2
            elif len(recent_positive_news) > 2:
                score += 0.1
        
        # Geographic expansion threat
        if profile.job_postings:
            international_jobs = [job for job in profile.job_postings 
                                if job.location and any(country in job.location.lower() 
                                for country in ['uk', 'europe', 'canada', 'singapore'])]
            if international_jobs:
                score += 0.15
        
        return min(score, 1.0)
    
    def _assess_technology_threat(self, profile) -> float:
        """Assess technology-based competitive threat"""
        score = 0.0
        
        # Advanced technology adoption
        if profile.technology_stack:
            advanced_tech = ['AI', 'Machine Learning', 'GraphQL', 'Kubernetes', 'Microservices']
            advanced_count = len([tech for tech in profile.technology_stack if tech in advanced_tech])
            score += min(advanced_count * 0.1, 0.3)
        
        # Feature sophistication
        if profile.key_features:
            ai_features = [f for f in profile.key_features 
                          if any(word in f.lower() for word in ['ai', 'ml', 'machine learning', 'neural'])]
            score += min(len(ai_features) * 0.05, 0.2)
            
            # Real-time capabilities
            realtime_features = [f for f in profile.key_features 
                               if 'real-time' in f.lower() or 'streaming' in f.lower()]
            if realtime_features:
                score += 0.15
        
        # GitHub activity (development velocity)
        if profile.github_activity:
            activity_score = profile.github_activity.activity_score or 0
            if activity_score > 80:
                score += 0.2
            elif activity_score > 50:
                score += 0.1
        
        # Patent protection
        if profile.patent_data:
            patent_count = profile.patent_data.total_patents
            if patent_count > 50:
                score += 0.2
            elif patent_count > 20:
                score += 0.1
            elif patent_count > 5:
                score += 0.05
        
        return min(score, 1.0)
    
    def _assess_financial_threat(self, profile) -> float:
        """Assess financial strength threat"""
        score = 0.0
        
        if not profile.funding_info:
            return 0.0
        
        # Total funding threat
        if profile.funding_info.total_funding:
            funding_str = profile.funding_info.total_funding
            
            # Parse funding amount
            if 'B' in funding_str:  # Billions
                score += 0.4
            elif 'M' in funding_str:
                try:
                    amount = float(funding_str.replace(',', '').replace('M', '').replace(',', ''))
                    if amount > 500:
                        score += 0.35
                    elif amount > 200:
                        score += 0.25
                    elif amount > 100:
                        score += 0.15
                    elif amount > 50:
                        score += 0.1
                except:
                    score += 0.1  # Some funding is better than none
        
        # Recent funding activity
        if profile.funding_info.last_round_date:
            try:
                last_round = datetime.fromisoformat(profile.funding_info.last_round_date.replace('Z', '+00:00'))
                days_ago = (datetime.now() - last_round).days
                
                if days_ago < 365:  # Within last year
                    score += 0.2
                elif days_ago < 730:  # Within last 2 years
                    score += 0.1
            except:
                pass
        
        # Funding trend
        if profile.funding_info.funding_trend == 'increasing':
            score += 0.15
        
        return min(score, 1.0)
    
    def _assess_customer_threat(self, profile) -> float:
        """Assess customer acquisition/retention threat"""
        score = 0.0
        
        # Customer portfolio strength
        if profile.case_studies:
            # Enterprise customer count
            enterprise_cases = [case for case in profile.case_studies 
                              if hasattr(case, 'industry') and case.industry]
            if len(enterprise_cases) > 10:
                score += 0.3
            elif len(enterprise_cases) > 5:
                score += 0.2
            elif len(enterprise_cases) > 2:
                score += 0.1
        
        # Sales team expansion (hiring threat)
        if profile.job_postings:
            sales_jobs = [job for job in profile.job_postings 
                         if 'sales' in job.department.lower() or 'account' in job.title.lower()]
            if len(sales_jobs) > 5:
                score += 0.2
            elif len(sales_jobs) > 2:
                score += 0.1
        
        # Market reputation (social presence)
        if profile.social_presence:
            total_followers = sum(p.followers for p in profile.social_presence if p.followers)
            if total_followers > 50000:
                score += 0.15
            elif total_followers > 10000:
                score += 0.1
        
        # Content marketing threat
        if profile.recent_news:
            content_indicators = [news for news in profile.recent_news 
                                if any(word in news.title.lower() 
                                for word in ['guide', 'best practices', 'how to', 'webinar'])]
            if len(content_indicators) > 3:
                score += 0.1
        
        return min(score, 1.0)
    
    def _assess_innovation_threat(self, profile) -> float:
        """Assess innovation and R&D threat"""
        score = 0.0
        
        # R&D investment indicators
        if profile.job_postings:
            rd_jobs = [job for job in profile.job_postings 
                      if any(dept in job.department.lower() 
                      for dept in ['engineering', 'research', 'product', 'data science'])]
            rd_ratio = len(rd_jobs) / len(profile.job_postings) if profile.job_postings else 0
            
            if rd_ratio > 0.6:
                score += 0.25
            elif rd_ratio > 0.4:
                score += 0.15
            elif rd_ratio > 0.2:
                score += 0.1
        
        # Patent activity
        if profile.patent_data:
            if profile.patent_data.filing_trend == 'increasing':
                score += 0.2
            elif profile.patent_data.total_patents > 10:
                score += 0.1
        
        # GitHub innovation indicators
        if profile.github_activity:
            if profile.github_activity.contributors > 50:
                score += 0.15
            
            # Language diversity as innovation indicator
            if len(profile.github_activity.languages) > 8:
                score += 0.1
        
        # Innovation in features
        if profile.key_features:
            cutting_edge_keywords = ['ai', 'ml', 'neural', 'semantic', 'vector', 'graph', 'federated']
            innovation_features = [f for f in profile.key_features 
                                 if any(keyword in f.lower() for keyword in cutting_edge_keywords)]
            score += min(len(innovation_features) * 0.05, 0.2)
        
        return min(score, 1.0)
    
    async def _identify_immediate_threats(self, profile, threat_categories: Dict[str, float]) -> List[str]:
        """Identify immediate competitive threats (next 6-12 months)"""
        threats = []
        
        # High financial threat + recent funding
        if (threat_categories['financial_threat'] > 0.6 and 
            profile.funding_info and profile.funding_info.last_round_date):
            threats.append("Well-funded competitor with recent capital raise poses acquisition threat")
        
        # High market threat + aggressive hiring
        if (threat_categories['market_threat'] > 0.5 and 
            profile.job_postings and len(profile.job_postings) > 15):
            threats.append("Aggressive hiring indicates rapid market expansion plans")
        
        # Technology threat + AI features
        if threat_categories['technology_threat'] > 0.6:
            threats.append("Advanced AI/ML capabilities may leapfrog current offerings")
        
        # Customer threat + sales hiring
        if threat_categories['customer_threat'] > 0.5:
            sales_jobs = [job for job in (profile.job_postings or []) 
                         if 'sales' in job.department.lower()]
            if len(sales_jobs) > 3:
                threats.append("Sales team expansion targets direct customer acquisition")
        
        # Recent positive news momentum
        if profile.recent_news:
            recent_positive = [news for news in profile.recent_news[-10:] 
                             if news.sentiment == 'positive']
            if len(recent_positive) > 3:
                threats.append("Strong recent media momentum increases market visibility")
        
        return threats[:5]  # Limit to top 5 immediate threats
    
    async def _identify_long_term_threats(self, profile, threat_categories: Dict[str, float]) -> List[str]:
        """Identify long-term strategic threats (1-3 years)"""
        threats = []
        
        # Innovation threat
        if threat_categories['innovation_threat'] > 0.5:
            threats.append("Strong R&D investment may result in breakthrough innovations")
        
        # Platform threat
        if profile.key_features and len(profile.key_features) > 20:
            threats.append("Comprehensive platform strategy may dominate multiple use cases")
        
        # Market expansion threat
        if profile.target_markets and len(profile.target_markets) > 2:
            threats.append("Multi-segment strategy may achieve dominant market position")
        
        # Technology stack modernization
        if profile.technology_stack:
            modern_tech_count = len([tech for tech in profile.technology_stack 
                                   if tech in ['Kubernetes', 'GraphQL', 'Microservices']])
            if modern_tech_count > 2:
                threats.append("Modern technology stack enables rapid scaling and innovation")
        
        # International expansion
        if profile.job_postings:
            international_jobs = [job for job in profile.job_postings 
                                if job.location and any(country in job.location.lower() 
                                for country in ['uk', 'europe', 'canada', 'asia'])]
            if len(international_jobs) > 2:
                threats.append("International expansion may establish global market presence")
        
        return threats[:4]  # Limit to top 4 long-term threats
    
    def _analyze_threat_factors(self, profile) -> Dict[str, Any]:
        """Analyze detailed threat factors"""
        factors = {
            'strengths': [],
            'vulnerabilities': [],
            'market_dynamics': {},
            'competitive_advantages': []
        }
        
        # Identify strengths
        if profile.funding_info and profile.funding_info.total_funding:
            if 'B' in profile.funding_info.total_funding or '500M' in profile.funding_info.total_funding:
                factors['strengths'].append("Strong financial position enables aggressive growth")
        
        if profile.key_features and len(profile.key_features) > 15:
            factors['strengths'].append("Comprehensive feature set provides broad market appeal")
        
        if profile.case_studies and len(profile.case_studies) > 8:
            factors['strengths'].append("Strong customer portfolio demonstrates market validation")
        
        if profile.github_activity and profile.github_activity.activity_score and profile.github_activity.activity_score > 70:
            factors['strengths'].append("High development velocity indicates rapid innovation")
        
        # Identify vulnerabilities
        if not profile.funding_info or not profile.funding_info.total_funding:
            factors['vulnerabilities'].append("Limited funding information suggests potential capital constraints")
        
        if not profile.case_studies or len(profile.case_studies) < 3:
            factors['vulnerabilities'].append("Limited customer success stories may hinder credibility")
        
        if not profile.patent_data or profile.patent_data.total_patents < 5:
            factors['vulnerabilities'].append("Weak IP portfolio provides limited competitive protection")
        
        if profile.target_markets and len(profile.target_markets) == 1:
            factors['vulnerabilities'].append("Single market focus creates concentration risk")
        
        # Market dynamics
        factors['market_dynamics'] = {
            'market_focus': ', '.join(profile.target_markets) if profile.target_markets else 'Unknown',
            'customer_segments': len(profile.case_studies) if profile.case_studies else 0,
            'technology_maturity': 'Advanced' if (profile.technology_stack and 
                                               len([t for t in profile.technology_stack 
                                                   if t in ['AI', 'ML', 'Kubernetes']]) > 1) else 'Standard'
        }
        
        # Competitive advantages
        if profile.technology_stack:
            ai_tech = [t for t in profile.technology_stack if t in ['AI', 'Machine Learning']]
            if ai_tech:
                factors['competitive_advantages'].append("AI/ML technology provides differentiation")
        
        if profile.patent_data and profile.patent_data.total_patents > 20:
            factors['competitive_advantages'].append("Strong IP portfolio creates defensive moat")
        
        return factors
    
    async def _suggest_mitigation_strategies(self, profile, threat_categories: Dict[str, float]) -> List[str]:
        """Suggest strategies to mitigate competitive threats"""
        strategies = []
        
        # High financial threat mitigation
        if threat_categories['financial_threat'] > 0.6:
            strategies.append("Accelerate fundraising to maintain competitive parity")
            strategies.append("Focus on capital efficiency and path to profitability")
        
        # High technology threat mitigation
        if threat_categories['technology_threat'] > 0.6:
            strategies.append("Increase R&D investment in AI/ML capabilities")
            strategies.append("Consider strategic partnerships for technology advancement")
        
        # High market threat mitigation
        if threat_categories['market_threat'] > 0.5:
            strategies.append("Strengthen customer retention through enhanced support")
            strategies.append("Accelerate product differentiation initiatives")
        
        # High customer threat mitigation
        if threat_categories['customer_threat'] > 0.5:
            strategies.append("Invest in sales team expansion and training")
            strategies.append("Develop competitive battle cards and positioning")
        
        # High innovation threat mitigation
        if threat_categories['innovation_threat'] > 0.5:
            strategies.append("Establish innovation labs or acquisition strategy")
            strategies.append("Increase patent filing and IP protection efforts")
        
        # General strategies based on overall threat
        overall_threat = sum(threat_categories.values()) / len(threat_categories)
        if overall_threat > 0.7:
            strategies.append("Consider strategic acquisition or merger opportunities")
            strategies.append("Implement continuous competitive monitoring program")
        
        return strategies[:6]  # Limit to top 6 strategies
    
    async def create_threat_matrix(self, profiles: List) -> Dict[str, Any]:
        """Create comprehensive threat assessment matrix"""
        if not profiles:
            return {'error': 'No profiles provided for threat matrix'}
        
        threat_matrix = {
            'threat_rankings': [],
            'threat_categories_summary': {},
            'high_priority_threats': [],
            'monitoring_priorities': [],
            'strategic_recommendations': []
        }
        
        # Assess each competitor
        competitor_threats = []
        for profile in profiles:
            threat_assessment = await self.assess_competitive_threat(profile)
            competitor_threats.append({
                'name': profile.name,
                'overall_score': threat_assessment['overall_threat_score'],
                'categories': threat_assessment['threat_categories'],
                'immediate_threats': threat_assessment['immediate_threats'],
                'long_term_threats': threat_assessment['long_term_threats']
            })
        
        # Sort by threat level
        threat_matrix['threat_rankings'] = sorted(
            competitor_threats, 
            key=lambda x: x['overall_score'], 
            reverse=True
        )
        
        # Summarize threat categories across all competitors
        category_averages = {}
        categories = ['market_threat', 'technology_threat', 'financial_threat', 
                     'customer_threat', 'innovation_threat']
        
        for category in categories:
            avg_score = sum(ct['categories'][category] for ct in competitor_threats) / len(competitor_threats)
            category_averages[category] = round(avg_score, 2)
        
        threat_matrix['threat_categories_summary'] = category_averages
        
        # Identify high priority threats (score > 0.7)
        high_priority = [ct for ct in competitor_threats if ct['overall_score'] > 0.7]
        threat_matrix['high_priority_threats'] = [ct['name'] for ct in high_priority]
        
        # Monitoring priorities
        threat_matrix['monitoring_priorities'] = self._determine_monitoring_priorities(competitor_threats)
        
        # Strategic recommendations
        threat_matrix['strategic_recommendations'] = await self._generate_strategic_recommendations(
            competitor_threats, category_averages
        )
        
        return threat_matrix
    
    def _determine_monitoring_priorities(self, competitor_threats: List[Dict]) -> List[Dict[str, str]]:
        """Determine monitoring priorities based on threat levels"""
        priorities = []
        
        for ct in competitor_threats:
            priority_level = 'low'
            monitoring_focus = []
            
            if ct['overall_score'] > 0.8:
                priority_level = 'critical'
                monitoring_focus = ['funding rounds', 'product launches', 'customer wins', 'hiring']
            elif ct['overall_score'] > 0.6:
                priority_level = 'high'
                monitoring_focus = ['product updates', 'market expansion', 'partnerships']
            elif ct['overall_score'] > 0.4:
                priority_level = 'medium'
                monitoring_focus = ['major announcements', 'funding news']
            else:
                priority_level = 'low'
                monitoring_focus = ['quarterly updates']
            
            priorities.append({
                'competitor': ct['name'],
                'priority_level': priority_level,
                'monitoring_focus': ', '.join(monitoring_focus)
            })
        
        return priorities
    
    async def _generate_strategic_recommendations(self, competitor_threats: List[Dict], 
                                                category_averages: Dict[str, float]) -> List[str]:
        """Generate strategic recommendations based on threat analysis"""
        recommendations = []
        
        # Overall threat level recommendations
        avg_overall_threat = sum(ct['overall_score'] for ct in competitor_threats) / len(competitor_threats)
        
        if avg_overall_threat > 0.7:
            recommendations.append("Market shows high competitive intensity - consider defensive strategies")
        elif avg_overall_threat > 0.5:
            recommendations.append("Moderate competitive pressure - focus on differentiation")
        else:
            recommendations.append("Lower competitive intensity - opportunity for aggressive growth")
        
        # Category-specific recommendations
        if category_averages['financial_threat'] > 0.6:
            recommendations.append("Competitors are well-funded - prioritize capital efficiency")
        
        if category_averages['technology_threat'] > 0.6:
            recommendations.append("Technology arms race detected - increase R&D investment")
        
        if category_averages['market_threat'] > 0.6:
            recommendations.append("High market overlap - strengthen customer retention")
        
        if category_averages['innovation_threat'] > 0.6:
            recommendations.append("Innovation pressure high - consider strategic partnerships")
        
        # Top threat specific recommendations
        top_threat = competitor_threats[0] if competitor_threats else None
        if top_threat and top_threat['overall_score'] > 0.8:
            recommendations.append(f"Monitor {top_threat['name']} closely - highest competitive threat")
        
        return recommendations[:5]  # Limit to top 5 recommendations