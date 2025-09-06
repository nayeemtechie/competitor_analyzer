# src/competitor/analysis/market.py
"""
Market positioning analysis for competitive intelligence
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """Analyzes market positioning and competitive landscape"""
    
    def __init__(self, config, llm_provider):
        self.config = config
        self.llm = llm_provider
    
    async def analyze_market_position(self, profile) -> Dict[str, Any]:
        """Analyze market positioning for a single competitor"""
        analysis = {
            'target_market_analysis': self._analyze_target_markets(profile),
            'market_segment_strategy': self._analyze_market_segments(profile),
            'positioning_strategy': await self._analyze_positioning_strategy(profile),
            'competitive_moat': await self._identify_competitive_moat(profile),
            'market_expansion_indicators': self._analyze_expansion_indicators(profile)
        }
        
        return analysis
    
    def _analyze_target_markets(self, profile) -> Dict[str, Any]:
        """Analyze target market focus"""
        target_analysis = {
            'primary_markets': profile.target_markets or [],
            'market_focus': 'unknown',
            'market_breadth': 'narrow'
        }
        
        if profile.target_markets:
            market_count = len(profile.target_markets)
            
            if 'enterprise' in profile.target_markets:
                target_analysis['market_focus'] = 'enterprise-first'
            elif 'mid-market' in profile.target_markets:
                target_analysis['market_focus'] = 'mid-market focused'
            elif 'smb' in profile.target_markets:
                target_analysis['market_focus'] = 'smb-focused'
            else:
                target_analysis['market_focus'] = 'mixed market approach'
            
            if market_count > 2:
                target_analysis['market_breadth'] = 'broad'
            elif market_count == 2:
                target_analysis['market_breadth'] = 'moderate'
        
        return target_analysis
    
    def _analyze_market_segments(self, profile) -> Dict[str, str]:
        """Analyze market segmentation strategy"""
        segmentation = {
            'vertical_focus': 'horizontal',
            'customer_size_focus': 'mixed',
            'geographic_focus': 'unknown'
        }
        
        # Analyze case studies for vertical focus
        if profile.case_studies:
            industries = set()
            for case in profile.case_studies:
                if hasattr(case, 'industry') and case.industry:
                    industries.add(case.industry.lower())
            
            if len(industries) == 1:
                segmentation['vertical_focus'] = f"vertical - {list(industries)[0]}"
            elif len(industries) <= 3:
                segmentation['vertical_focus'] = "focused verticals"
            else:
                segmentation['vertical_focus'] = "horizontal - multiple industries"
        
        # Analyze customer size focus from target markets
        if profile.target_markets:
            if 'enterprise' in profile.target_markets and len(profile.target_markets) == 1:
                segmentation['customer_size_focus'] = 'enterprise-only'
            elif 'smb' in profile.target_markets and len(profile.target_markets) == 1:
                segmentation['customer_size_focus'] = 'smb-only'
            else:
                segmentation['customer_size_focus'] = 'multi-segment'
        
        return segmentation
    
    async def _analyze_positioning_strategy(self, profile) -> Dict[str, str]:
        """Analyze competitive positioning strategy"""
        positioning = {
            'differentiation_approach': 'unknown',
            'value_proposition': 'unknown',
            'competitive_messaging': 'unknown'
        }
        
        # Analyze features for differentiation
        if profile.key_features:
            feature_text = ' '.join(profile.key_features).lower()
            
            if 'ai' in feature_text or 'machine learning' in feature_text:
                positioning['differentiation_approach'] = 'ai-first technology'
            elif 'performance' in feature_text or 'speed' in feature_text:
                positioning['differentiation_approach'] = 'performance-focused'
            elif 'personalization' in feature_text:
                positioning['differentiation_approach'] = 'personalization-driven'
            elif 'enterprise' in feature_text or 'scale' in feature_text:
                positioning['differentiation_approach'] = 'enterprise-grade platform'
            else:
                positioning['differentiation_approach'] = 'feature completeness'
        
        # Analyze pricing for value proposition
        if profile.pricing_tiers:
            tier_count = len(profile.pricing_tiers)
            if tier_count > 3:
                positioning['value_proposition'] = 'flexible pricing for all segments'
            elif tier_count == 1:
                positioning['value_proposition'] = 'simple, transparent pricing'
            else:
                positioning['value_proposition'] = 'tiered value delivery'
        
        return positioning
    
    async def _identify_competitive_moat(self, profile) -> List[str]:
        """Identify competitive moats and barriers"""
        moats = []
        
        # Technology moat
        if profile.technology_stack:
            advanced_tech_count = len([tech for tech in profile.technology_stack 
                                     if tech in ['AI', 'Machine Learning', 'GraphQL', 'Kubernetes']])
            if advanced_tech_count >= 2:
                moats.append("Advanced technology stack creates technical barriers")
        
        # Patent moat
        if profile.patent_data and profile.patent_data.total_patents > 20:
            moats.append("Significant IP portfolio provides defensive protection")
        
        # Customer moat
        if profile.case_studies and len(profile.case_studies) > 10:
            moats.append("Strong customer success portfolio builds credibility barrier")
        
        # Network effects moat
        if profile.key_features:
            feature_text = ' '.join(profile.key_features).lower()
            if 'marketplace' in feature_text or 'ecosystem' in feature_text:
                moats.append("Platform ecosystem creates network effects")
        
        # Scale moat
        if profile.funding_info and profile.funding_info.total_funding:
            funding_str = profile.funding_info.total_funding
            if 'B' in funding_str or ('M' in funding_str and '500' in funding_str):
                moats.append("Large funding enables scale advantages")
        
        return moats
    
    def _analyze_expansion_indicators(self, profile) -> Dict[str, Any]:
        """Analyze indicators of market expansion"""
        expansion_indicators = {
            'hiring_growth': False,
            'international_expansion': False,
            'vertical_expansion': False,
            'product_expansion': False
        }
        
        # Hiring growth indicator
        if profile.job_postings and len(profile.job_postings) > 15:
            expansion_indicators['hiring_growth'] = True
        
        # International expansion from job locations
        if profile.job_postings:
            international_locations = [job for job in profile.job_postings 
                                     if job.location and any(country in job.location.lower() 
                                     for country in ['uk', 'europe', 'canada', 'singapore', 'australia'])]
            if international_locations:
                expansion_indicators['international_expansion'] = True
        
        # Product expansion from feature breadth
        if profile.key_features and len(profile.key_features) > 20:
            expansion_indicators['product_expansion'] = True
        
        # Vertical expansion from case study diversity
        if profile.case_studies:
            industries = set()
            for case in profile.case_studies:
                if hasattr(case, 'industry') and case.industry:
                    industries.add(case.industry)
            
            if len(industries) > 3:
                expansion_indicators['vertical_expansion'] = True
        
        return expansion_indicators
    
    async def analyze_competitive_landscape(self, profiles: List) -> Dict[str, Any]:
        """Analyze the overall competitive landscape"""
        if len(profiles) < 2:
            return {'error': 'Need at least 2 competitors for landscape analysis'}
        
        landscape = {
            'market_segments': self._analyze_market_segment_coverage(profiles),
            'positioning_clusters': self._identify_positioning_clusters(profiles),
            'market_gaps': self._identify_market_gaps(profiles),
            'competitive_intensity': self._assess_competitive_intensity(profiles)
        }
        
        return landscape
    
    def _analyze_market_segment_coverage(self, profiles: List) -> Dict[str, List[str]]:
        """Analyze which competitors target which market segments"""
        segment_coverage = {
            'enterprise': [],
            'mid-market': [],
            'smb': [],
            'developer': [],
            'retail': [],
            'b2b': []
        }
        
        for profile in profiles:
            if profile.target_markets:
                for market in profile.target_markets:
                    market_lower = market.lower()
                    if market_lower in segment_coverage:
                        segment_coverage[market_lower].append(profile.name)
        
        return {k: v for k, v in segment_coverage.items() if v}  # Remove empty segments
    
    def _identify_positioning_clusters(self, profiles: List) -> Dict[str, List[str]]:
        """Identify clusters of similarly positioned competitors"""
        clusters = {
            'ai_first': [],
            'enterprise_focused': [],
            'performance_leaders': [],
            'full_platform': []
        }
        
        for profile in profiles:
            if profile.key_features:
                feature_text = ' '.join(profile.key_features).lower()
                
                # AI-first cluster
                if len([f for f in profile.key_features if 'ai' in f.lower() or 'ml' in f.lower()]) > 2:
                    clusters['ai_first'].append(profile.name)
                
                # Enterprise-focused cluster
                if 'enterprise' in profile.target_markets or \
                   len([f for f in profile.key_features if 'enterprise' in f.lower()]) > 1:
                    clusters['enterprise_focused'].append(profile.name)
                
                # Performance leaders cluster
                if len([f for f in profile.key_features if any(word in f.lower() 
                       for word in ['performance', 'speed', 'fast', 'real-time'])]) > 1:
                    clusters['performance_leaders'].append(profile.name)
                
                # Full platform cluster
                if len(profile.key_features) > 15:
                    clusters['full_platform'].append(profile.name)
        
        return {k: v for k, v in clusters.items() if v}
    
    def _identify_market_gaps(self, profiles: List) -> List[str]:
        """Identify underserved market segments or needs"""
        # Analyze what's missing across the competitive landscape
        gaps = []
        
        # Check for vertical specialization gaps
        all_case_studies = []
        for profile in profiles:
            if profile.case_studies:
                all_case_studies.extend(profile.case_studies)
        
        covered_industries = set()
        for case in all_case_studies:
            if hasattr(case, 'industry') and case.industry:
                covered_industries.add(case.industry.lower())
        
        # Common industries not well covered
        target_industries = {'healthcare', 'fintech', 'automotive', 'manufacturing', 'education'}
        missing_industries = target_industries - covered_industries
        
        for industry in missing_industries:
            gaps.append(f"Underserved vertical: {industry.title()}")
        
        # Check for size segment gaps
        all_targets = set()
        for profile in profiles:
            if profile.target_markets:
                all_targets.update(profile.target_markets)
        
        if 'smb' not in [t.lower() for t in all_targets]:
            gaps.append("SMB market appears underserved")
        
        # Feature gaps
        all_features = []
        for profile in profiles:
            if profile.key_features:
                all_features.extend(profile.key_features)
        
        all_features_text = ' '.join(all_features).lower()
        
        if 'voice search' not in all_features_text:
            gaps.append("Voice search capabilities gap")
        
        if 'visual search' not in all_features_text:
            gaps.append("Visual search functionality gap")
        
        return gaps[:5]  # Return top 5 gaps
    
    def _assess_competitive_intensity(self, profiles: List) -> Dict[str, Any]:
        """Assess the intensity of competition in the market"""
        intensity_analysis = {
            'competitor_count': len(profiles),
            'funding_competition': self._analyze_funding_competition(profiles),
            'feature_competition': self._analyze_feature_competition(profiles),
            'market_maturity': self._assess_market_maturity(profiles),
            'overall_intensity': 'moderate'
        }
        
        # Calculate overall intensity score
        intensity_score = 0
        
        # Competitor count factor
        if len(profiles) > 5:
            intensity_score += 3
        elif len(profiles) > 3:
            intensity_score += 2
        else:
            intensity_score += 1
        
        # Funding factor
        well_funded_count = 0
        for profile in profiles:
            if profile.funding_info and profile.funding_info.total_funding:
                if 'B' in profile.funding_info.total_funding or \
                   ('M' in profile.funding_info.total_funding and '100' in profile.funding_info.total_funding):
                    well_funded_count += 1
        
        if well_funded_count > 2:
            intensity_score += 2
        elif well_funded_count > 0:
            intensity_score += 1
        
        # Feature competition factor
        avg_features = sum(len(p.key_features) if p.key_features else 0 for p in profiles) / len(profiles)
        if avg_features > 15:
            intensity_score += 2
        elif avg_features > 10:
            intensity_score += 1
        
        # Determine overall intensity
        if intensity_score >= 7:
            intensity_analysis['overall_intensity'] = 'high'
        elif intensity_score >= 4:
            intensity_analysis['overall_intensity'] = 'moderate'
        else:
            intensity_analysis['overall_intensity'] = 'low'
        
        return intensity_analysis
    
    def _analyze_funding_competition(self, profiles: List) -> Dict[str, Any]:
        """Analyze funding levels across competitors"""
        funding_analysis = {
            'total_market_funding': 0,
            'well_funded_competitors': [],
            'funding_leaders': []
        }
        
        funding_amounts = {}
        
        for profile in profiles:
            if profile.funding_info and profile.funding_info.total_funding:
                funding_str = profile.funding_info.total_funding
                try:
                    # Simple parsing - extract numeric value
                    import re
                    amount_match = re.search(r'[\d.]+', funding_str.replace(',', ''))
                    if amount_match:
                        amount = float(amount_match.group())
                        if 'B' in funding_str:
                            amount *= 1000  # Convert to millions
                        funding_amounts[profile.name] = amount
                        
                        if amount > 100:  # >$100M
                            funding_analysis['well_funded_competitors'].append(profile.name)
                except:
                    pass
        
        # Sort by funding amount
        if funding_amounts:
            sorted_funding = sorted(funding_amounts.items(), key=lambda x: x[1], reverse=True)
            funding_analysis['funding_leaders'] = sorted_funding[:3]
            funding_analysis['total_market_funding'] = sum(funding_amounts.values())
        
        return funding_analysis
    
    def _analyze_feature_competition(self, profiles: List) -> Dict[str, Any]:
        """Analyze feature competition intensity"""
        feature_analysis = {
            'average_feature_count': 0,
            'feature_leaders': [],
            'innovation_competition': 'low'
        }
        
        feature_counts = {}
        innovation_counts = {}
        
        for profile in profiles:
            feature_count = len(profile.key_features) if profile.key_features else 0
            feature_counts[profile.name] = feature_count
            
            # Count AI/innovation features
            if profile.key_features:
                innovation_features = [f for f in profile.key_features 
                                     if any(word in f.lower() for word in ['ai', 'ml', 'machine learning', 'neural'])]
                innovation_counts[profile.name] = len(innovation_features)
        
        if feature_counts:
            feature_analysis['average_feature_count'] = sum(feature_counts.values()) / len(feature_counts)
            sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
            feature_analysis['feature_leaders'] = sorted_features[:3]
        
        # Assess innovation competition
        if innovation_counts:
            avg_innovation = sum(innovation_counts.values()) / len(innovation_counts)
            if avg_innovation > 3:
                feature_analysis['innovation_competition'] = 'high'
            elif avg_innovation > 1:
                feature_analysis['innovation_competition'] = 'moderate'
        
        return feature_analysis
    
    def _assess_market_maturity(self, profiles: List) -> str:
        """Assess market maturity based on competitor characteristics"""
        maturity_indicators = {
            'established_players': 0,
            'well_funded_count': 0,
            'feature_sophistication': 0
        }
        
        for profile in profiles:
            # Established players (based on funding history or feature count)
            if profile.funding_info and profile.funding_info.total_funding:
                if 'B' in profile.funding_info.total_funding:
                    maturity_indicators['established_players'] += 1
                    maturity_indicators['well_funded_count'] += 1
                elif 'M' in profile.funding_info.total_funding:
                    maturity_indicators['well_funded_count'] += 1
            
            # Feature sophistication
            if profile.key_features and len(profile.key_features) > 15:
                maturity_indicators['feature_sophistication'] += 1
        
        total_competitors = len(profiles)
        
        # Calculate maturity score
        maturity_score = (
            (maturity_indicators['established_players'] / total_competitors) * 3 +
            (maturity_indicators['well_funded_count'] / total_competitors) * 2 +
            (maturity_indicators['feature_sophistication'] / total_competitors) * 2
        )
        
        if maturity_score > 4:
            return "mature - Established market with well-funded players"
        elif maturity_score > 2:
            return "developing - Growing market with some established players"
        else:
            return "emerging - Early-stage market with room for new entrants"