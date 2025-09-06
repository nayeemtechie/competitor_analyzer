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
    
    async def compare_pricing_strategies(self, profiles: List) -> Dict[str, Any]:
        """Compare pricing strategies across multiple competitors"""
        if len(profiles) < 2:
            return {'error': 'Need at least 2 competitors for pricing comparison'}
        
        comparison = {
            'pricing_models': self._compare_pricing_models(profiles),
            'price_positioning': self._analyze_price_positioning_across_competitors(profiles),
            'tier_strategies': self._compare_tier_strategies(profiles),
            'value_proposition_analysis': self._analyze_value_propositions(profiles),
            'pricing_gaps': self._identify_pricing_gaps(profiles)
        }
        
        return comparison
    
    def _compare_pricing_models(self, profiles: List) -> Dict[str, Any]:
        """Compare pricing models across competitors"""
        model_distribution = {
            'freemium': [],
            'tiered_subscription': [],
            'usage_based': [],
            'enterprise_only': [],
            'hybrid': []
        }
        
        for profile in profiles:
            if not profile.pricing_tiers:
                continue
                
            tier_analysis = self._analyze_pricing_tiers(profile)
            strategy = tier_analysis['tier_strategy']
            
            if 'freemium' in strategy:
                model_distribution['freemium'].append(profile.name)
            elif tier_analysis['enterprise_tier'] and not tier_analysis['free_tier']:
                model_distribution['enterprise_only'].append(profile.name)
            elif len(profile.pricing_tiers) > 1:
                model_distribution['tiered_subscription'].append(profile.name)
            else:
                model_distribution['hybrid'].append(profile.name)
        
        return model_distribution
    
    def _analyze_price_positioning_across_competitors(self, profiles: List) -> Dict[str, Any]:
        """Analyze price positioning across all competitors"""
        all_prices = []
        competitor_price_ranges = {}
        
        for profile in profiles:
            if not profile.pricing_tiers:
                continue
                
            profile_prices = []
            for tier in profile.pricing_tiers:
                if hasattr(tier, 'price') and tier.price:
                    prices = self._extract_price_numbers(tier.price)
                    profile_prices.extend(prices)
            
            if profile_prices:
                competitor_price_ranges[profile.name] = {
                    'min_price': min(profile_prices),
                    'max_price': max(profile_prices),
                    'avg_price': sum(profile_prices) / len(profile_prices)
                }
                all_prices.extend(profile_prices)
        
        if not all_prices:
            return {'analysis': 'No pricing data available for comparison'}
        
        market_stats = {
            'market_min': min(all_prices),
            'market_max': max(all_prices),
            'market_avg': sum(all_prices) / len(all_prices),
            'competitor_positioning': self._categorize_price_positioning(competitor_price_ranges, all_prices)
        }
        
        return market_stats
    
    def _categorize_price_positioning(self, competitor_ranges: Dict, all_prices: List) -> Dict[str, List[str]]:
        """Categorize competitors by price positioning"""
        market_avg = sum(all_prices) / len(all_prices)
        market_low = min(all_prices)
        market_high = max(all_prices)
        
        # Define thresholds
        budget_threshold = market_low + (market_avg - market_low) * 0.5
        premium_threshold = market_avg + (market_high - market_avg) * 0.5
        
        positioning = {
            'budget': [],
            'mid_market': [],
            'premium': [],
            'enterprise': []
        }
        
        for competitor, price_range in competitor_ranges.items():
            avg_price = price_range['avg_price']
            max_price = price_range['max_price']
            
            if max_price > premium_threshold:
                if max_price > market_high * 0.8:
                    positioning['enterprise'].append(competitor)
                else:
                    positioning['premium'].append(competitor)
            elif avg_price < budget_threshold:
                positioning['budget'].append(competitor)
            else:
                positioning['mid_market'].append(competitor)
        
        return positioning
    
    def _compare_tier_strategies(self, profiles: List) -> Dict[str, Any]:
        """Compare tier strategies across competitors"""
        tier_strategies = {
            'average_tiers': 0,
            'freemium_adoption': 0,
            'enterprise_tier_adoption': 0,
            'tier_distribution': {},
            'popular_tier_patterns': []
        }
        
        tier_counts = []
        freemium_count = 0
        enterprise_count = 0
        
        for profile in profiles:
            if not profile.pricing_tiers:
                continue
                
            tier_count = len(profile.pricing_tiers)
            tier_counts.append(tier_count)
            
            # Count tier distribution
            tier_strategies['tier_distribution'][tier_count] = \
                tier_strategies['tier_distribution'].get(tier_count, 0) + 1
            
            # Analyze tier characteristics
            tier_analysis = self._analyze_pricing_tiers(profile)
            
            if tier_analysis['free_tier']:
                freemium_count += 1
            
            if tier_analysis['enterprise_tier']:
                enterprise_count += 1
        
        if tier_counts:
            tier_strategies['average_tiers'] = sum(tier_counts) / len(tier_counts)
            tier_strategies['freemium_adoption'] = freemium_count / len(profiles)
            tier_strategies['enterprise_tier_adoption'] = enterprise_count / len(profiles)
        
        # Identify popular patterns
        most_common_tier_count = max(tier_strategies['tier_distribution'].items(), 
                                   key=lambda x: x[1])[0] if tier_strategies['tier_distribution'] else 0
        
        tier_strategies['popular_tier_patterns'] = [
            f"Most common: {most_common_tier_count} tiers",
            f"Freemium adoption: {tier_strategies['freemium_adoption']:.1%}",
            f"Enterprise tier adoption: {tier_strategies['enterprise_tier_adoption']:.1%}"
        ]
        
        return tier_strategies
    
    def _analyze_value_propositions(self, profiles: List) -> Dict[str, Any]:
        """Analyze value propositions across competitors"""
        value_analysis = {
            'common_value_drivers': [],
            'unique_differentiators': {},
            'pricing_dimension_adoption': {},
            'value_positioning_clusters': {}
        }
        
        all_value_drivers = []
        all_pricing_dimensions = []
        
        for profile in profiles:
            value_metrics = self._analyze_value_metrics(profile)
            
            all_value_drivers.extend(value_metrics['value_drivers'])
            all_pricing_dimensions.extend(value_metrics['pricing_dimensions'])
            
            # Track unique differentiators
            for driver in value_metrics['value_drivers']:
                if driver not in value_analysis['unique_differentiators']:
                    value_analysis['unique_differentiators'][driver] = []
                value_analysis['unique_differentiators'][driver].append(profile.name)
        
        # Identify common value drivers (used by multiple competitors)
        driver_counts = {}
        for driver in all_value_drivers:
            driver_counts[driver] = driver_counts.get(driver, 0) + 1
        
        value_analysis['common_value_drivers'] = [
            driver for driver, count in driver_counts.items() if count > 1
        ]
        
        # Analyze pricing dimension adoption
        dimension_counts = {}
        for dimension in all_pricing_dimensions:
            dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1
        
        value_analysis['pricing_dimension_adoption'] = dimension_counts
        
        return value_analysis
    
    def _identify_pricing_gaps(self, profiles: List) -> List[str]:
        """Identify pricing gaps and opportunities"""
        gaps = []
        
        # Analyze price ranges to find gaps
        all_prices = []
        for profile in profiles:
            if not profile.pricing_tiers:
                continue
                
            for tier in profile.pricing_tiers:
                if hasattr(tier, 'price') and tier.price:
                    prices = self._extract_price_numbers(tier.price)
                    all_prices.extend(prices)
        
        if not all_prices:
            return gaps
        
        all_prices.sort()
        
        # Look for price gaps (significant jumps in pricing)
        for i in range(1, len(all_prices)):
            price_jump = all_prices[i] - all_prices[i-1]
            if price_jump > all_prices[i-1] * 0.5:  # 50% jump
                gaps.append(f"Pricing gap between ${all_prices[i-1]:.0f} and ${all_prices[i]:.0f}")
        
        # Analyze market segments not well covered
        tier_analyses = []
        for profile in profiles:
            if profile.pricing_tiers:
                tier_analyses.append(self._analyze_pricing_tiers(profile))
        
        # Check for underserved pricing strategies
        freemium_count = sum(1 for analysis in tier_analyses if analysis['free_tier'])
        if freemium_count == 0:
            gaps.append("No freemium options available - potential customer acquisition gap")
        
        usage_based_count = 0
        for profile in profiles:
            pricing_model = self._determine_pricing_model(profile)
            if pricing_model['structure'] == 'usage-based':
                usage_based_count += 1
        
        if usage_based_count == 0:
            gaps.append("Limited usage-based pricing options - scalability gap for growing businesses")
        
        # Check for SMB vs Enterprise gaps
        enterprise_focused = 0
        smb_friendly = 0
        
        for profile in profiles:
            positioning = self._analyze_pricing_positioning(profile)
            if positioning['target_segment'] == 'enterprise-focused':
                enterprise_focused += 1
            elif positioning['target_segment'] in ['smb-focused', 'mid-market-focused']:
                smb_friendly += 1
        
        if enterprise_focused > smb_friendly * 2:
            gaps.append("Market skews heavily enterprise - SMB pricing gap opportunity")
        elif smb_friendly > enterprise_focused * 2:
            gaps.append("Limited enterprise options - high-value customer gap")
        
        return gaps[:5]  # Return top 5 gaps