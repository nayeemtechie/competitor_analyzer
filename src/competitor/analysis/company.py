# src/competitor/analysis/company.py
"""
Company-level analysis for competitive intelligence
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CompanyAnalyzer:
    """Analyzes company-level competitive intelligence"""
    
    def __init__(self, config, llm_provider):
        self.config = config
        self.llm = llm_provider
        
    async def analyze_company_profile(self, profile) -> Dict[str, Any]:
        """Comprehensive company profile analysis"""
        analysis = {
            'business_model_analysis': await self._analyze_business_model(profile),
            'market_positioning': await self._analyze_market_positioning(profile),
            'financial_health': await self._analyze_financial_indicators(profile),
            'growth_trajectory': await self._analyze_growth_trajectory(profile),
            'strategic_focus': await self._identify_strategic_focus(profile),
            'competitive_advantages': await self._identify_competitive_advantages(profile),
            'vulnerability_assessment': await self._assess_vulnerabilities(profile)
        }
        
        return analysis
    
    async def _analyze_business_model(self, profile) -> Dict[str, str]:
        """Analyze business model and revenue streams"""
        analysis = {}
        
        # Infer business model from available data
        if profile.pricing_tiers:
            tier_count = len(profile.pricing_tiers)
            if tier_count > 3:
                analysis['pricing_strategy'] = "Multi-tier SaaS with extensive segmentation"
            elif tier_count > 1:
                analysis['pricing_strategy'] = "Tiered SaaS model"
            else:
                analysis['pricing_strategy'] = "Single-tier or custom pricing"
        
        # Market segment analysis
        if profile.target_markets:
            if 'enterprise' in profile.target_markets:
                analysis['market_focus'] = "Enterprise-focused with high ACV potential"
            elif 'mid-market' in profile.target_markets:
                analysis['market_focus'] = "Mid-market focused with scalable growth"
            else:
                analysis['market_focus'] = "Broad market approach"
        
        # Technology strategy
        if profile.technology_stack:
            modern_tech_count = sum(1 for tech in profile.technology_stack 
                                  if tech in ['AI', 'Machine Learning', 'GraphQL', 'Kubernetes'])
            if modern_tech_count > 2:
                analysis['technology_approach'] = "Cutting-edge technology adoption"
            else:
                analysis['technology_approach'] = "Proven technology stack"
        
        return analysis
    
    async def _analyze_market_positioning(self, profile) -> Dict[str, Any]:
        """Analyze market positioning strategy"""
        positioning = {
            'segment_strategy': self._determine_segment_strategy(profile),
            'differentiation_approach': await self._identify_differentiation(profile),
            'competitive_messaging': await self._extract_competitive_messaging(profile)
        }
        
        return positioning
    
    def _determine_segment_strategy(self, profile) -> str:
        """Determine market segmentation strategy"""
        if not profile.target_markets:
            return "Undefined market focus"
        
        segments = profile.target_markets
        if len(segments) > 2:
            return "Multi-segment approach across enterprise and mid-market"
        elif 'enterprise' in segments:
            return "Enterprise-first strategy with high-value accounts"
        elif 'mid-market' in segments:
            return "Mid-market focused with volume growth strategy"
        else:
            return f"Focused on {', '.join(segments)} segment"
    
    async def _identify_differentiation(self, profile) -> List[str]:
        """Identify key differentiation factors"""
        differentiators = []
        
        # Technology differentiation
        if profile.technology_stack:
            if 'AI' in profile.technology_stack or 'Machine Learning' in profile.technology_stack:
                differentiators.append("AI-powered capabilities")
            
            if 'GraphQL' in profile.technology_stack:
                differentiators.append("Modern API architecture")
        
        # Feature differentiation
        if profile.key_features:
            feature_text = ' '.join(profile.key_features).lower()
            if 'personalization' in feature_text:
                differentiators.append("Advanced personalization")
            if 'real-time' in feature_text or 'realtime' in feature_text:
                differentiators.append("Real-time processing")
            if 'analytics' in feature_text:
                differentiators.append("Advanced analytics")
        
        # Market differentiation
        if profile.case_studies and len(profile.case_studies) > 5:
            differentiators.append("Strong customer success track record")
        
        return differentiators[:5]  # Top 5 differentiators
    
    async def _extract_competitive_messaging(self, profile) -> Dict[str, str]:
        """Extract competitive messaging themes"""
        messaging = {}
        
        # Use LLM to analyze content if available
        if profile.website_data and profile.website_data.content_themes:
            themes_text = ', '.join(profile.website_data.content_themes)
            
            try:
                system_prompt = "Extract key competitive messaging themes from this content."
                user_prompt = f"Analyze competitive messaging for {profile.name} based on these themes: {themes_text}"
                
                model = self.config.get_llm_model('analysis')
                response = self.llm.chat(
                    system=system_prompt,
                    user=user_prompt,
                    model=model
                )
                
                messaging['primary_message'] = response[:200]  # Limit length
                
            except Exception as e:
                logger.warning(f"LLM messaging analysis failed: {e}")
                messaging['primary_message'] = "Unable to determine messaging"
        
        return messaging
    
    async def _analyze_financial_indicators(self, profile) -> Dict[str, Any]:
        """Analyze financial health indicators"""
        financial_analysis = {}
        
        if profile.funding_info:
            funding = profile.funding_info
            
            # Funding strength
            if funding.total_funding:
                try:
                    # Extract numeric value for analysis
                    funding_str = funding.total_funding.replace('$', '').replace('B', '000').replace('M', '')
                    funding_value = float(funding_str)
                    
                    if funding_value > 500:  # >$500M
                        financial_analysis['funding_strength'] = "Very strong funding position"
                    elif funding_value > 100:  # >$100M
                        financial_analysis['funding_strength'] = "Strong funding position"
                    elif funding_value > 50:   # >$50M
                        financial_analysis['funding_strength'] = "Adequate funding"
                    else:
                        financial_analysis['funding_strength'] = "Limited funding"
                        
                except (ValueError, AttributeError):
                    financial_analysis['funding_strength'] = "Funding amount unclear"
            
            # Recent funding activity
            if funding.last_round_date:
                try:
                    from datetime import datetime
                    last_round = datetime.fromisoformat(funding.last_round_date.replace('Z', '+00:00'))
                    days_ago = (datetime.now() - last_round).days
                    
                    if days_ago < 365:
                        financial_analysis['funding_recency'] = "Recent funding (strong momentum)"
                    elif days_ago < 730:
                        financial_analysis['funding_recency'] = "Moderate funding recency"
                    else:
                        financial_analysis['funding_recency'] = "Older funding (may need new round)"
                        
                except Exception:
                    financial_analysis['funding_recency'] = "Funding date unclear"
        
        return financial_analysis
    
    async def _analyze_growth_trajectory(self, profile) -> Dict[str, str]:
        """Analyze growth trajectory indicators"""
        growth_analysis = {}
        
        # Job posting growth signals
        if profile.job_postings:
            job_count = len(profile.job_postings)
            eng_jobs = len([j for j in profile.job_postings if 'engineering' in j.department.lower()])
            sales_jobs = len([j for j in profile.job_postings if 'sales' in j.department.lower()])
            
            if job_count > 20:
                growth_analysis['hiring_velocity'] = "Aggressive hiring (high growth signal)"
            elif job_count > 10:
                growth_analysis['hiring_velocity'] = "Steady hiring (moderate growth)"
            else:
                growth_analysis['hiring_velocity'] = "Limited hiring (stable or slow growth)"
            
            # Department focus
            if eng_jobs > job_count * 0.5:
                growth_analysis['growth_focus'] = "Engineering-heavy hiring (product scaling)"
            elif sales_jobs > job_count * 0.3:
                growth_analysis['growth_focus'] = "Sales-focused hiring (revenue scaling)"
            else:
                growth_analysis['growth_focus'] = "Balanced hiring across functions"
        
        # News momentum
        if profile.recent_news:
            news_count = len(profile.recent_news)
            if news_count > 10:
                growth_analysis['media_momentum'] = "High media attention (market momentum)"
            elif news_count > 5:
                growth_analysis['media_momentum'] = "Moderate media presence"
            else:
                growth_analysis['media_momentum'] = "Low media visibility"
        
        return growth_analysis
    
    async def _identify_strategic_focus(self, profile) -> Dict[str, List[str]]:
        """Identify strategic focus areas"""
        strategic_focus = {
            'product_strategy': [],
            'market_strategy': [],
            'technology_strategy': []
        }
        
        # Product strategy from features
        if profile.key_features:
            feature_text = ' '.join(profile.key_features).lower()
            
            if 'ai' in feature_text or 'machine learning' in feature_text:
                strategic_focus['product_strategy'].append("AI/ML advancement")
            if 'personalization' in feature_text:
                strategic_focus['product_strategy'].append("Personalization focus")
            if 'analytics' in feature_text or 'insights' in feature_text:
                strategic_focus['product_strategy'].append("Analytics capabilities")
            if 'api' in feature_text or 'integration' in feature_text:
                strategic_focus['product_strategy'].append("Integration ecosystem")
        
        # Market strategy from targets and case studies
        if profile.target_markets:
            if 'enterprise' in profile.target_markets:
                strategic_focus['market_strategy'].append("Enterprise market penetration")
            if len(profile.target_markets) > 2:
                strategic_focus['market_strategy'].append("Multi-segment expansion")
        
        if profile.case_studies:
            industries = set()
            for case in profile.case_studies:
                if hasattr(case, 'industry') and case.industry:
                    industries.add(case.industry.lower())
            
            if len(industries) > 3:
                strategic_focus['market_strategy'].append("Vertical market diversification")
        
        # Technology strategy from stack and GitHub
        if profile.technology_stack:
            if 'Kubernetes' in profile.technology_stack:
                strategic_focus['technology_strategy'].append("Cloud-native architecture")
            if 'GraphQL' in profile.technology_stack:
                strategic_focus['technology_strategy'].append("Modern API development")
        
        if profile.github_activity and profile.github_activity.activity_score > 70:
            strategic_focus['technology_strategy'].append("Open source engagement")
        
        return strategic_focus
    
    async def _identify_competitive_advantages(self, profile) -> List[str]:
        """Identify key competitive advantages"""
        advantages = []
        
        # Technology advantages
        if profile.technology_stack:
            modern_count = sum(1 for tech in profile.technology_stack 
                             if tech in ['AI', 'Machine Learning', 'Kubernetes', 'GraphQL'])
            if modern_count >= 2:
                advantages.append("Modern technology stack provides scalability edge")
        
        # Market advantages
        if profile.case_studies and len(profile.case_studies) > 8:
            advantages.append("Strong customer success portfolio builds credibility")
        
        if profile.funding_info and profile.funding_info.total_funding:
            funding_str = profile.funding_info.total_funding
            if 'B' in funding_str or (('M' in funding_str) and (int(''.join(filter(str.isdigit, funding_str))) > 200)):
                advantages.append("Strong financial position enables aggressive growth")
        
        # Innovation advantages
        if profile.patent_data and profile.patent_data.total_patents > 20:
            advantages.append("Significant IP portfolio provides defensive moat")
        
        if profile.github_activity and profile.github_activity.contributors > 50:
            advantages.append("Large developer community drives innovation")
        
        # Team advantages (from hiring patterns)
        if profile.job_postings:
            senior_roles = sum(1 for job in profile.job_postings 
                             if any(level in job.title.lower() 
                                   for level in ['senior', 'principal', 'staff', 'lead']))
            if senior_roles > len(profile.job_postings) * 0.4:
                advantages.append("Focus on senior talent acquisition")
        
        return advantages[:5]  # Top 5 advantages
    
    async def _assess_vulnerabilities(self, profile) -> List[str]:
        """Assess potential competitive vulnerabilities"""
        vulnerabilities = []
        
        # Financial vulnerabilities
        if not profile.funding_info or not profile.funding_info.total_funding:
            vulnerabilities.append("Limited funding information suggests potential capital constraints")
        elif profile.funding_info.funding_trend == "decreasing":
            vulnerabilities.append("Decreasing funding trend indicates investor confidence issues")
        
        # Technology vulnerabilities
        if not profile.technology_stack or len(profile.technology_stack) < 3:
            vulnerabilities.append("Limited technology diversity may hinder adaptation")
        
        # Market vulnerabilities
        if profile.target_markets and len(profile.target_markets) == 1:
            vulnerabilities.append("Single market focus creates concentration risk")
        
        if not profile.case_studies or len(profile.case_studies) < 3:
            vulnerabilities.append("Limited customer success stories may hurt credibility")
        
        # Innovation vulnerabilities
        if not profile.patent_data or profile.patent_data.total_patents < 5:
            vulnerabilities.append("Weak IP portfolio provides limited competitive protection")
        
        if profile.github_activity and profile.github_activity.activity_score < 20:
            vulnerabilities.append("Low open source activity may indicate limited innovation")
        
        # Team vulnerabilities (from hiring patterns)
        if not profile.job_postings or len(profile.job_postings) < 5:
            vulnerabilities.append("Limited hiring activity suggests slow growth or resource constraints")
        
        return vulnerabilities[:5]  # Top 5 vulnerabilities
    
    async def generate_executive_summary(self, intelligence) -> str:
        """Generate executive summary of competitive landscape"""
        if not intelligence.profiles:
            return "No competitor data available for executive summary."
        
        try:
            # Prepare data for LLM
            competitor_summaries = []
            for profile in intelligence.profiles:
                summary_data = {
                    'name': profile.name,
                    'threat_level': profile.threat_level.value if profile.threat_level else 'medium',
                    'funding': profile.funding_info.total_funding if profile.funding_info else 'Unknown',
                    'key_features_count': len(profile.key_features) if profile.key_features else 0,
                    'recent_news_count': len(profile.recent_news) if profile.recent_news else 0,
                    'job_postings_count': len(profile.job_postings) if profile.job_postings else 0
                }
                competitor_summaries.append(summary_data)
            
            # Create executive summary prompt
            system_prompt = """You are a senior business analyst creating an executive summary of the competitive landscape for an ecommerce search SaaS company. 
            Focus on strategic implications, key threats, and actionable recommendations for C-level executives."""
            
            user_prompt = f"""
            Create an executive summary based on analysis of {len(intelligence.profiles)} key competitors:
            
            Competitor Data:
            {json.dumps(competitor_summaries, indent=2)}
            
            Structure the summary with:
            1. Key Market Dynamics (2-3 sentences)
            2. Primary Competitive Threats (3 bullets)
            3. Strategic Opportunities (2-3 bullets) 
            4. Recommended Actions (3 bullets with timeframes)
            
            Keep it under 300 words, executive-friendly, and actionable.
            """
            
            model = self.config.get_llm_model('analysis')
            summary = self.llm.chat(
                system=system_prompt,
                user=user_prompt,
                model=model
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            
            # Fallback summary
            high_threat_count = len([p for p in intelligence.profiles 
                                   if p.threat_level and p.threat_level.value in ['high', 'critical']])
            
            return f"""
            COMPETITIVE LANDSCAPE EXECUTIVE SUMMARY
            
            Analyzed {len(intelligence.profiles)} key competitors in the ecommerce search market.
            
            Key Findings:
            • {high_threat_count} competitors pose high competitive threat requiring immediate attention
            • Market shows continued consolidation with active funding rounds
            • AI/ML capabilities becoming table stakes across leading vendors
            
            Strategic Implications:
            • Accelerated product development needed to maintain competitive parity
            • Consider strategic partnerships or acquisitions to strengthen market position
            • Pricing pressure likely to increase as market matures
            
            Recommended Actions:
            • Conduct detailed feature gap analysis within 90 days
            • Evaluate pricing strategy competitiveness by Q2
            • Assess acquisition targets for technology or market expansion
            """