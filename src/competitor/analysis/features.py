# src/competitor/analysis/features.py
"""
Feature analysis for competitive intelligence
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureAnalyzer:
    """Analyzes product features for competitive comparison"""
    
    def __init__(self, config, llm_provider):
        self.config = config
        self.llm = llm_provider
    
    async def analyze_features(self, profile) -> Dict[str, Any]:
        """Analyze features for a single competitor"""
        analysis = {
            'feature_count': len(profile.key_features) if profile.key_features else 0,
            'feature_categories': await self._categorize_features(profile.key_features),
            'innovation_indicators': await self._identify_innovation_features(profile.key_features),
            'competitive_advantages': await self._identify_feature_advantages(profile),
            'feature_sophistication': await self._assess_feature_sophistication(profile.key_features),
            'integration_capabilities': await self._analyze_integration_features(profile)
        }
        
        return analysis
    
    async def _categorize_features(self, features: List[str]) -> Dict[str, List[str]]:
        """Categorize features into functional areas"""
        if not features:
            return {}
        
        categories = {
            'search_core': [],
            'ai_ml': [],
            'analytics': [],
            'personalization': [],
            'integration': [],
            'performance': [],
            'enterprise': [],
            'developer': []
        }
        
        # Feature categorization keywords
        category_keywords = {
            'search_core': ['search', 'indexing', 'query', 'relevance', 'ranking', 'autocomplete'],
            'ai_ml': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'neural', 'algorithm'],
            'analytics': ['analytics', 'insights', 'reporting', 'metrics', 'dashboard', 'tracking'],
            'personalization': ['personalization', 'recommendation', 'custom', 'tailored', 'individual'],
            'integration': ['api', 'integration', 'connect', 'sync', 'webhook', 'sdk'],
            'performance': ['performance', 'speed', 'latency', 'scale', 'optimization', 'cache'],
            'enterprise': ['security', 'compliance', 'sso', 'admin', 'governance', 'enterprise'],
            'developer': ['developer', 'api', 'sdk', 'documentation', 'tools', 'debug']
        }
        
        for feature in features:
            feature_lower = feature.lower()
            categorized = False
            
            for category, keywords in category_keywords.items():
                if any(keyword in feature_lower for keyword in keywords):
                    categories[category].append(feature)
                    categorized = True
                    break
            
            if not categorized:
                categories.setdefault('other', []).append(feature)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    async def _identify_innovation_features(self, features: List[str]) -> List[str]:
        """Identify innovative or cutting-edge features"""
        if not features:
            return []
        
        innovation_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'neural network',
            'deep learning', 'nlp', 'natural language', 'computer vision',
            'real-time', 'streaming', 'edge computing', 'serverless',
            'voice search', 'visual search', 'semantic search',
            'federated search', 'vector search', 'graph search'
        ]
        
        innovative_features = []
        for feature in features:
            feature_lower = feature.lower()
            if any(keyword in feature_lower for keyword in innovation_keywords):
                innovative_features.append(feature)
        
        return innovative_features
    
    async def _identify_feature_advantages(self, profile) -> List[str]:
        """Identify potential competitive feature advantages"""
        advantages = []
        
        if not profile.key_features:
            return advantages
        
        feature_count = len(profile.key_features)
        
        # Feature breadth advantage
        if feature_count > 20:
            advantages.append("Comprehensive feature set provides broad functionality")
        
        # AI/ML advantage
        ai_features = await self._identify_innovation_features(profile.key_features)
        if len(ai_features) > 3:
            advantages.append("Strong AI/ML capabilities for advanced functionality")
        
        # Integration advantage
        integration_features = [f for f in profile.key_features 
                              if any(word in f.lower() for word in ['api', 'integration', 'connect'])]
        if len(integration_features) > 2:
            advantages.append("Extensive integration capabilities")
        
        # Technology stack advantage
        if profile.technology_stack:
            modern_tech = ['GraphQL', 'Kubernetes', 'Microservices', 'React', 'Node.js']
            modern_count = len([tech for tech in profile.technology_stack if tech in modern_tech])
            if modern_count > 2:
                advantages.append("Modern technology stack enables rapid feature development")
        
        return advantages
    
    async def _assess_feature_sophistication(self, features: List[str]) -> str:
        """Assess overall feature sophistication level"""
        if not features:
            return "No feature data available"
        
        sophistication_indicators = {
            'advanced': ['ai', 'machine learning', 'neural', 'deep learning', 'semantic'],
            'enterprise': ['enterprise', 'governance', 'compliance', 'security', 'admin'],
            'developer': ['api', 'sdk', 'webhook', 'graphql', 'rest'],
            'performance': ['real-time', 'streaming', 'optimization', 'cache', 'scale']
        }
        
        sophistication_score = 0
        total_features = len(features)
        
        for category, keywords in sophistication_indicators.items():
            category_features = [f for f in features 
                               if any(keyword in f.lower() for keyword in keywords)]
            category_ratio = len(category_features) / total_features
            
            if category_ratio > 0.2:  # 20% of features in this category
                sophistication_score += 2
            elif category_ratio > 0.1:  # 10% of features
                sophistication_score += 1
        
        if sophistication_score >= 6:
            return "Highly sophisticated - Advanced features across multiple areas"
        elif sophistication_score >= 4:
            return "Sophisticated - Strong feature set with advanced capabilities"
        elif sophistication_score >= 2:
            return "Moderate - Standard features with some advanced capabilities"
        else:
            return "Basic - Fundamental features without advanced sophistication"
    
    async def _analyze_integration_features(self, profile) -> Dict[str, Any]:
        """Analyze integration and connectivity features"""
        integration_analysis = {
            'api_capabilities': [],
            'pre_built_integrations': [],
            'developer_tools': [],
            'integration_sophistication': 'basic'
        }
        
        if not profile.key_features:
            return integration_analysis
        
        # Analyze features for integration capabilities
        for feature in profile.key_features:
            feature_lower = feature.lower()
            
            if any(word in feature_lower for word in ['api', 'rest', 'graphql']):
                integration_analysis['api_capabilities'].append(feature)
            elif any(word in feature_lower for word in ['integration', 'connect', 'sync']):
                integration_analysis['pre_built_integrations'].append(feature)
            elif any(word in feature_lower for word in ['sdk', 'webhook', 'developer']):
                integration_analysis['developer_tools'].append(feature)
        
        # Assess sophistication
        total_integration_features = (
            len(integration_analysis['api_capabilities']) +
            len(integration_analysis['pre_built_integrations']) +
            len(integration_analysis['developer_tools'])
        )
        
        if total_integration_features > 5:
            integration_analysis['integration_sophistication'] = 'advanced'
        elif total_integration_features > 2:
            integration_analysis['integration_sophistication'] = 'moderate'
        
        return integration_analysis
    
    async def compare_features_across_competitors(self, profiles: List) -> Dict[str, Any]:
        """Compare features across multiple competitors"""
        if len(profiles) < 2:
            return {'error': 'Need at least 2 competitors for comparison'}
        
        comparison = {
            'feature_leaders': {},
            'feature_gaps': {},
            'innovation_leaders': [],
            'comprehensive_analysis': {}
        }
        
        # Analyze each competitor's features
        competitor_analyses = {}
        for profile in profiles:
            analysis = await self.analyze_features(profile)
            competitor_analyses[profile.name] = analysis
        
        # Identify feature leaders
        comparison['feature_leaders'] = self._identify_feature_leaders(competitor_analyses)
        
        # Identify common feature gaps
        comparison['feature_gaps'] = self._identify_feature_gaps(profiles)
        
        # Identify innovation leaders
        comparison['innovation_leaders'] = self._identify_innovation_leaders(competitor_analyses)
        
        # Generate comprehensive comparison matrix
        comparison['feature_matrix'] = await self._create_feature_matrix(profiles)
        
        return comparison
    
    def _identify_feature_leaders(self, analyses: Dict[str, Dict]) -> Dict[str, str]:
        """Identify leaders in different feature categories"""
        leaders = {}
        
        # Feature count leader
        feature_counts = {name: analysis['feature_count'] for name, analysis in analyses.items()}
        if feature_counts:
            leaders['feature_breadth'] = max(feature_counts.items(), key=lambda x: x[1])[0]
        
        # Innovation leader
        innovation_counts = {}
        for name, analysis in analyses.items():
            innovation_counts[name] = len(analysis.get('innovation_indicators', []))
        
        if innovation_counts:
            leaders['innovation'] = max(innovation_counts.items(), key=lambda x: x[1])[0]
        
        return leaders
    
    def _identify_feature_gaps(self, profiles: List) -> List[str]:
        """Identify features that are missing across competitors"""
        # This would require more sophisticated analysis
        # For now, return common gaps in ecommerce search
        common_gaps = [
            'Voice search capabilities',
            'Visual search functionality', 
            'Real-time personalization',
            'Advanced A/B testing',
            'Federated search across multiple sources',
            'Natural language query processing'
        ]
        
        return common_gaps[:3]  # Return top 3 gaps
    
    def _identify_innovation_leaders(self, analyses: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Identify companies leading in innovation"""
        innovation_scores = []
        
        for name, analysis in analyses.items():
            innovation_count = len(analysis.get('innovation_indicators', []))
            sophistication = analysis.get('feature_sophistication', '')
            
            score = innovation_count
            if 'highly sophisticated' in sophistication.lower():
                score += 3
            elif 'sophisticated' in sophistication.lower():
                score += 2
            
            innovation_scores.append({
                'company': name,
                'innovation_score': score,
                'innovation_features': analysis.get('innovation_indicators', [])
            })
        
        return sorted(innovation_scores, key=lambda x: x['innovation_score'], reverse=True)
    
    async def _create_feature_matrix(self, profiles: List) -> Dict[str, Dict[str, str]]:
        """Create feature comparison matrix"""
        # Common ecommerce search features to compare
        key_features_to_compare = [
            'Search relevance',
            'Autocomplete',
            'Faceted search',
            'Personalization',
            'A/B testing',
            'Analytics',
            'API access',
            'Mobile optimization',
            'AI/ML capabilities',
            'Real-time indexing'
        ]
        
        feature_matrix = {}
        
        for feature in key_features_to_compare:
            feature_matrix[feature] = {}
            
            for profile in profiles:
                # Simple heuristic to check if competitor has the feature
                has_feature = self._check_feature_presence(feature, profile)
                feature_matrix[feature][profile.name] = "✓" if has_feature else "✗"
        
        return feature_matrix
    
    def _check_feature_presence(self, feature_name: str, profile) -> bool:
        """Check if competitor has a specific feature"""
        if not profile.key_features:
            return False
        
        feature_keywords = {
            'Search relevance': ['relevance', 'ranking', 'scoring'],
            'Autocomplete': ['autocomplete', 'suggestion', 'typeahead'],
            'Faceted search': ['facet', 'filter', 'refinement'],
            'Personalization': ['personalization', 'recommendation', 'custom'],
            'A/B testing': ['a/b', 'split test', 'experiment'],
            'Analytics': ['analytics', 'reporting', 'metrics'],
            'API access': ['api', 'rest', 'graphql'],
            'Mobile optimization': ['mobile', 'responsive', 'app'],
            'AI/ML capabilities': ['ai', 'machine learning', 'ml', 'neural'],
            'Real-time indexing': ['real-time', 'live', 'instant', 'streaming']
        }
        
        keywords = feature_keywords.get(feature_name, [feature_name.lower()])
        
        for feature in profile.key_features:
            feature_lower = feature.lower()
            if any(keyword in feature_lower for keyword in keywords):
                return True
        
        return False

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

# src/competitor/analysis/pricing.py
"""
Pricing strategy analysis for competitive intelligence
"""

from typing import Dict, List, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)

class PricingAnalyzer:
    """Analyzes pricing strategies and models"""
    
    def __init__(self, config, llm_provider):
        self.config = config
        self.llm = llm_provider
    
    async def analyze_pricing_strategy(self, profile) -> Dict[str, Any]:
        """Analyze pricing strategy for a single competitor"""
        analysis = {
            'pricing_model': self._determine_pricing_model(profile),
            'tier_analysis': self._analyze_pricing_tiers(profile),
            'pricing_positioning': self._analyze_pricing_positioning(profile),
            'value_metrics': self._analyze_value_metrics(profile),
            'competitive_pricing': await self._assess_competitive_pricing(profile)
        }
        
        return analysis
    
    def _determine_pricing_model(self, profile) -> Dict[str, str]:
        """Determine the overall pricing model"""
        pricing_model = {
            'type': 'unknown',
            'structure': 'unknown',
            'billing_options': []
        }
        
        if not profile.pricing_tiers:
            return pricing_model
        
        tier_count = len(profile.pricing_tiers)
        
        # Determine pricing type
        if tier_count == 1:
            pricing_model['type'] = 'single-tier'
        elif 2 <= tier_count <= 3:
            pricing_model['type'] = 'simple-tiered'
        elif tier_count > 3:
            pricing_model['type'] = 'multi-tiered'
        
        # Analyze pricing structure from tier data
        has_usage_based = False
        has_flat_rate = False
        
        for tier in profile.pricing_tiers:
            if hasattr(tier, 'price') and tier.price:
                price_str = tier.price.lower()
                if any(word in price_str for word in ['per', 'usage', 'request', 'search']):
                    has_usage_based = True
                else:
                    has_flat_rate = True
        
        if has_usage_based and has_flat_rate:
            pricing_model['structure'] = 'hybrid'
        elif has_usage_based:
            pricing_model['structure'] = 'usage-based'
        elif has_flat_rate:
            pricing_model['structure'] = 'subscription'
        
        return pricing_model
    
    def _analyze_pricing_tiers(self, profile) -> Dict[str, Any]:
        """Analyze individual pricing tiers"""
        tier_analysis = {
            'tier_count': 0,
            'tier_strategy': 'unknown',
            'popular_tier': None,
            'enterprise_tier': False,
            'free_tier': False
        }
        
        if not profile.pricing_tiers:
            return tier_analysis
        
        tier_analysis['tier_count'] = len(profile.pricing_tiers)
        
        # Analyze tier strategy
        tier_names = [tier.name.lower() if hasattr(tier, 'name') and tier.name else '' 
                     for tier in profile.pricing_tiers]
        
        # Check for common tier patterns
        if any('free' in name or 'trial' in name for name in tier_names):
            tier_analysis['free_tier'] = True
        
        if any('enterprise' in name or 'custom' in name for name in tier_names):
            tier_analysis['enterprise_tier'] = True
        
        # Find popular tier
        for tier in profile.pricing_tiers:
            if hasattr(tier, 'popular') and tier.popular:
                tier_analysis['popular_tier'] = tier.name if hasattr(tier, 'name') else 'unnamed'
                break
        
        # Determine tier strategy
        if tier_analysis['free_tier'] and tier_analysis['enterprise_tier']:
            tier_analysis['tier_strategy'] = 'freemium-to-enterprise'
        elif tier_analysis['free_tier']:
            tier_analysis['tier_strategy'] = 'freemium'
        elif tier_analysis['enterprise_tier']:
            tier_analysis['tier_strategy'] = 'subscription-to-enterprise'
        else:
            tier_analysis['tier_strategy'] = 'pure-subscription'
        
        return tier_analysis
    
    def _analyze_pricing_positioning(self, profile) -> Dict[str, str]:
        """Analyze pricing positioning strategy"""
        positioning = {
            'price_point': 'unknown',
            'target_segment': 'unknown',
            'value_positioning': 'unknown'
        }
        
        if not profile.pricing_tiers:
            return positioning
        
        # Extract price ranges
        prices = []
        for tier in profile.pricing_tiers:
            if hasattr(tier, 'price') and tier.price:
                price_numbers = self._extract_price_numbers(tier.price)
                prices.extend(price_numbers)
        
        if prices:
            min_price = min(prices)
            max_price = max(prices)
            
            # Determine price point positioning
            if min_price < 50:
                positioning['price_point'] = 'budget-friendly'
            elif min_price < 200:
                positioning['price_point'] = 'mid-market'
            else:
                positioning['price_point'] = 'premium'
            
            # Determine target segment based on price range
            if max_price > 1000:
                positioning['target_segment'] = 'enterprise-focused'
            elif max_price > 500:
                positioning['target_segment'] = 'mid-market-focused'
            else:
                positioning['target_segment'] = 'smb-focused'
        
        # Analyze value positioning from tier features
        all_features = []
        for tier in profile.pricing_tiers:
            if hasattr(tier, 'features') and tier.features:
                all_features.extend(tier.features)
        
        if all_features:
            feature_text = ' '.join(all_features).lower()
            if 'support' in feature_text and 'sla' in feature_text:
                positioning['value_positioning'] = 'service-oriented'
            elif 'api' in feature_text and 'integration' in feature_text:
                positioning['value_positioning'] = 'platform-oriented'
            elif 'custom' in feature_text and 'enterprise' in feature_text:
                positioning['value_positioning'] = 'enterprise-solutions'
            else:
                positioning['value_positioning'] = 'feature-driven'
        
        return positioning
    
    def _extract_price_numbers(self, price_str: str) -> List[float]:
        """Extract numeric prices from price string"""
        if not price_str:
            return []
        
        # Common price patterns
        patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',  # $99, $1,000, $99.99
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:USD|dollars?)',  # 99 USD
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)'  # Just numbers
        ]
        
        prices = []
        for pattern in patterns:
            matches = re.findall(pattern, price_str, re.IGNORECASE)
            for match in matches:
                try:
                    price = float(match.replace(',', ''))
                    prices.append(price)
                except ValueError:
                    continue
        
        return prices
    
    def _analyze_value_metrics(self, profile) -> Dict[str, Any]:
        """Analyze value metrics and pricing dimensions"""
        value_metrics = {
            'pricing_dimensions': [],
            'value_drivers': [],
            'scalability_model': 'unknown'
        }
        
        if not profile.pricing_tiers:
            return value_metrics
        
        # Analyze pricing dimensions from tier features
        all_features = []
        for tier in profile.pricing_tiers:
            if hasattr(tier, 'features') and tier.features:
                all_features.extend(tier.features)
        
        if all_features:
            feature_text = ' '.join(all_features).lower()
            
            # Common pricing dimensions
            if 'requests' in feature_text or 'queries' in feature_text:
                value_metrics['pricing_dimensions'].append('API requests/queries')
            
            if 'records' in feature_text or 'documents' in feature_text:
                value_metrics['pricing_dimensions'].append('Indexed records')
            
            if 'users' in feature_text or 'seats' in feature_text:
                value_metrics['pricing_dimensions'].append('User seats')
            
            if 'bandwidth' in feature_text or 'traffic' in feature_text:
                value_metrics['pricing_dimensions'].append('Bandwidth/traffic')
            
            # Value drivers
            if 'support' in feature_text:
                value_metrics['value_drivers'].append('Support level')
            
            if 'sla' in feature_text or 'uptime' in feature_text:
                value_metrics['value_drivers'].append('SLA guarantees')
            
            if 'analytics' in feature_text or 'reporting' in feature_text:
                value_metrics['value_drivers'].append('Analytics capabilities')
            
            if 'customization' in feature_text or 'custom' in feature_text:
                value_metrics['value_drivers'].append('Customization options')
        
        # Determine scalability model
        if 'requests' in ' '.join(value_metrics['pricing_dimensions']):
            value_metrics['scalability_model'] = 'usage-based scaling'
        elif 'users' in ' '.join(value_metrics['pricing_dimensions']):
            value_metrics['scalability_model'] = 'per-seat scaling'
        elif len(profile.pricing_tiers) > 3:
            value_metrics['scalability_model'] = 'tier-based scaling'
        else:
            value_metrics['scalability_model'] = 'fixed pricing'
        
        return value_metrics
    
    async def _assess_competitive_pricing(self, profile) -> Dict[str, str]:
        """Assess competitive pricing position"""
        assessment = {
            'price_competitiveness': 'unknown',
            'differentiation_strategy': 'unknown',
            'market_positioning': 'unknown'
        }
        
        if not profile.pricing_tiers:
            return assessment
        
        # This would typically involve comparison with other competitors
        # For now, provide general assessment based on tier structure
        
        tier_count = len(profile.pricing_tiers)
        has_free_tier = any('free' in tier.name.lower() 
                           if hasattr(tier, 'name') and tier.name else False 
                           for tier in profile.pricing_tiers)
        
        if has_free_tier:
            assessment['price_competitiveness'] = 'aggressive - Freemium model'
            assessment['differentiation_strategy'] = 'customer acquisition focused'
        elif tier_count > 4:
            assessment['price_competitiveness'] = 'flexible - Multiple options'
            assessment['differentiation_strategy'] = 'segment optimization'
        elif tier_count <= 2:
            assessment['price_competitiveness'] = 'simple - Limited options'
            assessment['differentiation_strategy'] = 'simplicity focused'
        else:
            assessment['price_competitiveness'] = 'standard - Balanced approach'
            assessment['differentiation_strategy'] = 'market standard'
        
        return assessment

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
                    #amount = float(funding_str.replace(', '').replace('M', '').replace(',', ''))
                    amount = float(funding_str.replace(',', '').replace('M', ''))
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

# src/competitor/analysis/innovation.py
"""
Innovation trajectory analysis for competitive intelligence
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class InnovationAnalyzer:
    """Analyzes innovation trends and R&D activities"""
    
    def __init__(self, config, llm_provider):
        self.config = config
        self.llm = llm_provider
    
    async def analyze_innovation_trajectory(self, profile) -> Dict[str, Any]:
        """Analyze innovation trajectory for a single competitor"""
        analysis = {
            'innovation_score': 0.0,
            'innovation_indicators': {},
            'technology_trends': [],
            'rd_investment_signals': {},
            'innovation_velocity': 'unknown',
            'future_innovation_areas': []
        }
        
        # Calculate innovation indicators
        analysis['innovation_indicators'] = self._calculate_innovation_indicators(profile)
        
        # Calculate overall innovation score
        analysis['innovation_score'] = self._calculate_innovation_score(analysis['innovation_indicators'])
        
        # Analyze technology trends
        analysis['technology_trends'] = self._analyze_technology_trends(profile)
        
        # Assess R&D investment signals
        analysis['rd_investment_signals'] = self._analyze_rd_investment(profile)
        
        # Determine innovation velocity
        analysis['innovation_velocity'] = self._assess_innovation_velocity(profile)
        
        # Predict future innovation areas
        analysis['future_innovation_areas'] = await self._predict_innovation_areas(profile)
        
        return analysis
    
    def _calculate_innovation_indicators(self, profile) -> Dict[str, Any]:
        """Calculate various innovation indicators"""
        indicators = {
            'patent_activity': 0,
            'github_innovation': 0,
            'technology_adoption': 0,
            'feature_innovation': 0,
            'rd_hiring': 0,
            'research_publications': 0
        }
        
        # Patent activity indicator
        if profile.patent_data:
            patent_count = profile.patent_data.total_patents
            if patent_count > 50:
                indicators['patent_activity'] = 10
            elif patent_count > 20:
                indicators['patent_activity'] = 8
            elif patent_count > 10:
                indicators['patent_activity'] = 6
            elif patent_count > 5:
                indicators['patent_activity'] = 4
            elif patent_count > 0:
                indicators['patent_activity'] = 2
            
            # Bonus for increasing trend
            if profile.patent_data.filing_trend == 'increasing':
                indicators['patent_activity'] += 2
        
        # GitHub innovation indicator
        if profile.github_activity:
            github_score = 0
            
            # Language diversity
            lang_count = len(profile.github_activity.languages)
            if lang_count > 10:
                github_score += 4
            elif lang_count > 5:
                github_score += 2
            elif lang_count > 2:
                github_score += 1
            
            # Activity level
            activity = profile.github_activity.activity_score or 0
            if activity > 80:
                github_score += 4
            elif activity > 60:
                github_score += 3
            elif activity > 40:
                github_score += 2
            elif activity > 20:
                github_score += 1
            
            # Contributors (community innovation)
            contributors = profile.github_activity.contributors
            if contributors > 100:
                github_score += 2
            elif contributors > 50:
                github_score += 1
            
            indicators['github_innovation'] = min(github_score, 10)
        
        # Technology adoption indicator
        if profile.technology_stack:
            tech_score = 0
            
            # Modern/cutting-edge technologies
            cutting_edge_tech = ['AI', 'Machine Learning', 'GraphQL', 'Kubernetes', 
                                'Microservices', 'Serverless', 'Edge Computing']
            
            for tech in profile.technology_stack:
                if tech in cutting_edge_tech:
                    tech_score += 1
            
            indicators['technology_adoption'] = min(tech_score * 1.5, 10)
        
        # Feature innovation indicator
        if profile.key_features:
            feature_score = 0
            
            # Innovation keywords in features
            innovation_keywords = ['ai', 'machine learning', 'neural', 'semantic', 'natural language',
                                 'computer vision', 'real-time', 'streaming', 'federated', 'vector']
            
            for feature in profile.key_features:
                feature_lower = feature.lower()
                keyword_count = sum(1 for keyword in innovation_keywords if keyword in feature_lower)
                feature_score += keyword_count
            
            indicators['feature_innovation'] = min(feature_score * 0.5, 10)
        
        # R&D hiring indicator
        if profile.job_postings:
            rd_score = 0
            
            # R&D related job postings
            rd_departments = ['engineering', 'research', 'data science', 'ai', 'ml', 'product']
            rd_titles = ['research', 'scientist', 'ai', 'machine learning', 'data scientist', 
                        'principal engineer', 'staff engineer', 'architect']
            
            rd_jobs = []
            for job in profile.job_postings:
                if any(dept in job.department.lower() for dept in rd_departments):
                    rd_jobs.append(job)
                elif any(title in job.title.lower() for title in rd_titles):
                    rd_jobs.append(job)
            
            rd_ratio = len(rd_jobs) / len(profile.job_postings) if profile.job_postings else 0
            
            if rd_ratio > 0.6:
                rd_score = 10
            elif rd_ratio > 0.4:
                rd_score = 8
            elif rd_ratio > 0.3:
                rd_score = 6
            elif rd_ratio > 0.2:
                rd_score = 4
            elif rd_ratio > 0.1:
                rd_score = 2
            
            indicators['rd_hiring'] = rd_score
        
        return indicators
    
    def _calculate_innovation_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate overall innovation score (0.0 to 10.0)"""
        if not indicators:
            return 0.0
        
        # Weighted average of indicators
        weights = {
            'patent_activity': 0.25,
            'github_innovation': 0.20,
            'technology_adoption': 0.20,
            'feature_innovation': 0.15,
            'rd_hiring': 0.15,
            'research_publications': 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for indicator, score in indicators.items():
            if indicator in weights and score > 0:
                weighted_score += score * weights[indicator]
                total_weight += weights[indicator]
        
        if total_weight > 0:
            return round(weighted_score / total_weight * 10, 1)
        else:
            return 0.0
    
    def _analyze_technology_trends(self, profile) -> List[Dict[str, str]]:
        """Analyze technology adoption trends"""
        trends = []
        
        # Analyze technology stack for trends
        if profile.technology_stack:
            # AI/ML trend
            ai_technologies = [tech for tech in profile.technology_stack 
                             if tech in ['AI', 'Machine Learning', 'Neural Networks', 'TensorFlow', 'PyTorch']]
            if ai_technologies:
                trends.append({
                    'trend': 'AI/ML Adoption',
                    'technologies': ', '.join(ai_technologies),
                    'significance': 'High - Core competitive differentiator'
                })
            
            # Cloud-native trend
            cloud_technologies = [tech for tech in profile.technology_stack 
                                if tech in ['Kubernetes', 'Docker', 'Microservices', 'Serverless']]
            if cloud_technologies:
                trends.append({
                    'trend': 'Cloud-Native Architecture',
                    'technologies': ', '.join(cloud_technologies),
                    'significance': 'Medium - Scalability and operational efficiency'
                })
            
            # Modern API trend
            api_technologies = [tech for tech in profile.technology_stack 
                              if tech in ['GraphQL', 'REST API', 'gRPC']]
            if api_technologies:
                trends.append({
                    'trend': 'Modern API Development',
                    'technologies': ', '.join(api_technologies),
                    'significance': 'Medium - Developer experience and integration'
                })
        
        # Analyze features for emerging trends
        if profile.key_features:
            feature_text = ' '.join(profile.key_features).lower()
            
            # Real-time/streaming trend
            if any(keyword in feature_text for keyword in ['real-time', 'streaming', 'live', 'instant']):
                trends.append({
                    'trend': 'Real-time Processing',
                    'technologies': 'Real-time indexing, streaming analytics',
                    'significance': 'High - User experience and competitive advantage'
                })
            
            # Personalization trend
            if any(keyword in feature_text for keyword in ['personalization', 'recommendation', 'custom']):
                trends.append({
                    'trend': 'Advanced Personalization',
                    'technologies': 'ML-driven recommendations, behavioral analysis',
                    'significance': 'High - Customer engagement and conversion'
                })
            
            # Voice/conversational trend
            if any(keyword in feature_text for keyword in ['voice', 'conversational', 'nlp', 'natural language']):
                trends.append({
                    'trend': 'Conversational Interfaces',
                    'technologies': 'NLP, voice recognition, chatbots',
                    'significance': 'Medium - Next-generation user interfaces'
                })
        
        return trends
    
    def _analyze_rd_investment(self, profile) -> Dict[str, Any]:
        """Analyze R&D investment signals"""
        rd_signals = {
            'hiring_intensity': 'low',
            'research_focus_areas': [],
            'innovation_partnerships': [],
            'funding_allocation': 'unknown'
        }
        
        # Analyze hiring for R&D intensity
        if profile.job_postings:
            total_jobs = len(profile.job_postings)
            
            # R&D related hiring
            rd_jobs = [job for job in profile.job_postings 
                      if any(keyword in job.title.lower() 
                      for keyword in ['research', 'scientist', 'engineer', 'architect', 'ai', 'ml'])]
            
            rd_ratio = len(rd_jobs) / total_jobs if total_jobs > 0 else 0
            
            if rd_ratio > 0.5:
                rd_signals['hiring_intensity'] = 'very high'
            elif rd_ratio > 0.3:
                rd_signals['hiring_intensity'] = 'high'
            elif rd_ratio > 0.2:
                rd_signals['hiring_intensity'] = 'medium'
            elif rd_ratio > 0.1:
                rd_signals['hiring_intensity'] = 'low'
            else:
                rd_signals['hiring_intensity'] = 'minimal'
            
            # Identify research focus areas from job titles and requirements
            focus_areas = set()
            for job in rd_jobs:
                title_lower = job.title.lower()
                if 'ai' in title_lower or 'machine learning' in title_lower:
                    focus_areas.add('Artificial Intelligence/ML')
                if 'data' in title_lower:
                    focus_areas.add('Data Science/Analytics')
                if 'cloud' in title_lower or 'infrastructure' in title_lower:
                    focus_areas.add('Cloud Infrastructure')
                if 'security' in title_lower:
                    focus_areas.add('Security/Privacy')
                if 'mobile' in title_lower or 'frontend' in title_lower:
                    focus_areas.add('User Experience/Mobile')
            
            rd_signals['research_focus_areas'] = list(focus_areas)
        
        # Analyze news for partnership signals
        if profile.recent_news:
            partnership_news = [news for news in profile.recent_news 
                              if any(keyword in news.title.lower() 
                              for keyword in ['partnership', 'collaboration', 'acquisition', 'invest'])]
            
            for news in partnership_news:
                if any(keyword in news.title.lower() 
                      for keyword in ['university', 'research', 'lab', 'ai', 'innovation']):
                    rd_signals['innovation_partnerships'].append(news.title)
        
        return rd_signals
    
    def _assess_innovation_velocity(self, profile) -> str:
        """Assess the pace of innovation"""
        velocity_factors = []
        
        # GitHub activity velocity
        if profile.github_activity:
            activity_score = profile.github_activity.activity_score or 0
            if activity_score > 80:
                velocity_factors.append('high_github_activity')
            elif activity_score > 50:
                velocity_factors.append('medium_github_activity')
        
        # Patent filing velocity
        if profile.patent_data:
            if profile.patent_data.filing_trend == 'increasing':
                velocity_factors.append('increasing_patents')
            elif profile.patent_data.total_patents > 20:
                velocity_factors.append('active_patenting')
        
        # Feature development velocity
        if profile.key_features and len(profile.key_features) > 20:
            velocity_factors.append('comprehensive_features')
        
        # Hiring velocity for R&D
        if profile.job_postings:
            rd_jobs = [job for job in profile.job_postings 
                      if 'engineering' in job.department.lower() or 'research' in job.title.lower()]
            if len(rd_jobs) > 10:
                velocity_factors.append('aggressive_rd_hiring')
            elif len(rd_jobs) > 5:
                velocity_factors.append('steady_rd_hiring')
        
        # Recent funding for innovation
        if profile.funding_info and profile.funding_info.last_round_date:
            try:
                last_round = datetime.fromisoformat(profile.funding_info.last_round_date.replace('Z', '+00:00'))
                if (datetime.now() - last_round).days < 365:
                    velocity_factors.append('recent_funding')
            except:
                pass
        
        # Assess overall velocity
        high_velocity_count = len([f for f in velocity_factors 
                                 if f in ['high_github_activity', 'increasing_patents', 
                                         'aggressive_rd_hiring', 'recent_funding']])
        
        if high_velocity_count >= 3:
            return 'very high - Rapid innovation across multiple dimensions'
        elif high_velocity_count >= 2:
            return 'high - Strong innovation momentum'
        elif len(velocity_factors) >= 3:
            return 'medium - Steady innovation pace'
        elif len(velocity_factors) >= 1:
            return 'low - Limited innovation signals'
        else:
            return 'minimal - Little evidence of active innovation'
    
    async def _predict_innovation_areas(self, profile) -> List[str]:
        """Predict future innovation areas based on current trends"""
        future_areas = []
        
        # Analyze hiring trends for future focus
        if profile.job_postings:
            recent_jobs = profile.job_postings[-10:]  # Last 10 job postings
            
            # AI/ML hiring surge
            ai_jobs = [job for job in recent_jobs 
                      if any(keyword in job.title.lower() 
                      for keyword in ['ai', 'machine learning', 'data scientist', 'ml engineer'])]
            if len(ai_jobs) > 2:
                future_areas.append('Advanced AI/ML capabilities - Significant hiring in AI roles')
            
            # Infrastructure/platform hiring
            infra_jobs = [job for job in recent_jobs 
                         if any(keyword in job.title.lower() 
                         for keyword in ['devops', 'platform', 'infrastructure', 'cloud'])]
            if len(infra_jobs) > 1:
                future_areas.append('Platform scalability - Infrastructure team expansion')
            
            # Product/UX hiring
            product_jobs = [job for job in recent_jobs 
                           if any(keyword in job.title.lower() 
                           for keyword in ['product', 'ux', 'ui', 'design'])]
            if len(product_jobs) > 1:
                future_areas.append('User experience innovation - Product team growth')
        
        # Analyze technology stack for future directions
        if profile.technology_stack:
            # Edge computing trend
            if 'Kubernetes' in profile.technology_stack:
                future_areas.append('Edge computing deployment - Kubernetes infrastructure suggests edge readiness')
            
            # API-first strategy
            if 'GraphQL' in profile.technology_stack:
                future_areas.append('API ecosystem expansion - Modern API technologies indicate platform strategy')
        
        # Analyze patent areas for research direction
        if profile.patent_data and profile.patent_data.technology_areas:
            emerging_areas = profile.patent_data.technology_areas
            
            if 'Natural Language Processing' in emerging_areas:
                future_areas.append('Conversational interfaces - NLP patents suggest voice/chat capabilities')
            
            if 'Machine Learning' in emerging_areas:
                future_areas.append('Automated optimization - ML patents indicate self-improving systems')
        
        # Analyze recent news for strategic direction
        if profile.recent_news:
            recent_news_text = ' '.join([news.title + ' ' + (news.summary or '') 
                                       for news in profile.recent_news[-5:]])
            
            if any(keyword in recent_news_text.lower() 
                  for keyword in ['acquisition', 'acquire', 'partnership']):
                future_areas.append('Strategic acquisitions - Recent M&A activity suggests capability expansion')
            
            if any(keyword in recent_news_text.lower() 
                  for keyword in ['international', 'global', 'europe', 'asia']):
                future_areas.append('Global expansion - International focus indicates market scaling')
        
        return future_areas[:5]  # Limit to top 5 predicted areas