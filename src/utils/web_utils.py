# src/utils/web_utils.py
"""
Web scraping utilities and helper functions for competitor analysis.
Provides URL validation, content analysis, and integration helpers.
"""

import re
import asyncio
import logging
from urllib.parse import urlparse, urljoin, quote, unquote
from typing import List, Dict, Optional, Tuple, Any, Set
from datetime import datetime
import tldextract
import hashlib
import json
from pathlib import Path

# Import our scraper
try:  # pragma: no cover - import resolution depends on installation context
    from ..competitor.scraper import WebScraper, ScrapingResult
    from ..competitor.models import WebsiteData, CompetitorProfile
except ImportError:  # Fallback for top-level package imports
    from competitor.scraper import WebScraper, ScrapingResult
    from competitor.models import WebsiteData, CompetitorProfile

logger = logging.getLogger(__name__)


class URLValidator:
    """Validates and normalizes URLs."""
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """
        Check if URL is valid and accessible.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL appears valid
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """
        Normalize URL for consistent processing.
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL
        """
        if not url:
            return url
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Parse and reconstruct
        parsed = urlparse(url)
        
        # Normalize components
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip('/') if parsed.path != '/' else '/'
        
        # Remove default ports
        if ':80' in netloc and scheme == 'http':
            netloc = netloc.replace(':80', '')
        elif ':443' in netloc and scheme == 'https':
            netloc = netloc.replace(':443', '')
        
        return f"{scheme}://{netloc}{path}"
    
    @staticmethod
    def extract_domain_info(url: str) -> Dict[str, str]:
        """
        Extract detailed domain information.
        
        Args:
            url: URL to analyze
            
        Returns:
            Dictionary with domain components
        """
        try:
            extracted = tldextract.extract(url)
            parsed = urlparse(url)
            
            return {
                'full_url': url,
                'protocol': parsed.scheme,
                'subdomain': extracted.subdomain,
                'domain': extracted.domain,
                'suffix': extracted.suffix,
                'full_domain': f"{extracted.domain}.{extracted.suffix}",
                'netloc': parsed.netloc,
                'path': parsed.path,
                'is_subdomain': bool(extracted.subdomain and extracted.subdomain != 'www')
            }
        except Exception as e:
            logger.error(f"Error extracting domain info from {url}: {e}")
            return {'full_url': url, 'error': str(e)}


class ContentAnalyzer:
    """Analyzes scraped content for competitive intelligence."""
    
    # Keywords for different types of analysis
    PRICING_KEYWORDS = [
        'price', 'pricing', 'cost', 'fee', 'subscription', 'plan', 'tier', 
        'premium', 'enterprise', 'starter', 'professional', 'basic',
        '$', '€', '£', 'usd', 'eur', 'gbp', 'per month', 'per year', 'annually',
        'free trial', 'money back', 'refund'
    ]
    
    FEATURE_KEYWORDS = [
        'feature', 'capability', 'functionality', 'solution', 'tool',
        'api', 'integration', 'dashboard', 'analytics', 'reporting',
        'real-time', 'automated', 'custom', 'advanced', 'enterprise-grade'
    ]
    
    COMPETITIVE_KEYWORDS = [
        'competitor', 'alternative', 'vs', 'versus', 'compared to',
        'better than', 'faster than', 'unlike', 'industry leading',
        'market leader', 'best in class', 'top rated'
    ]
    
    TECHNOLOGY_KEYWORDS = [
        'api', 'rest', 'graphql', 'sdk', 'javascript', 'python', 'react',
        'cloud', 'aws', 'azure', 'gcp', 'microservices', 'kubernetes',
        'machine learning', 'ai', 'artificial intelligence', 'ml', 'nlp'
    ]
    
    def analyze_content(self, result: ScrapingResult) -> Dict[str, Any]:
        """
        Analyze scraped content for competitive insights.
        
        Args:
            result: ScrapingResult to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not result.success or not result.content:
            return {'error': 'No content to analyze'}
        
        content_lower = result.content.lower()
        title_lower = (result.title or "").lower()
        
        analysis = {
            'url': result.url,
            'page_type': result.page_type,
            'analyzed_at': datetime.now().isoformat(),
            'content_length': len(result.content),
            'word_count': result.word_count,
            'title_analysis': self._analyze_title(result.title),
            'pricing_signals': self._find_pricing_signals(content_lower),
            'feature_mentions': self._find_feature_mentions(content_lower),
            'competitive_language': self._find_competitive_language(content_lower),
            'technology_stack': self._analyze_technology_mentions(content_lower, result.technologies),
            'key_phrases': self._extract_key_phrases(result.content),
            'sentiment_indicators': self._analyze_sentiment_indicators(content_lower),
            'cta_analysis': self._analyze_call_to_actions(result.content),
            'contact_info': self._extract_contact_info(result.content)
        }
        
        return analysis
    
    def _analyze_title(self, title: Optional[str]) -> Dict[str, Any]:
        """Analyze page title for insights."""
        if not title:
            return {'title': None, 'insights': []}
        
        title_lower = title.lower()
        insights = []
        
        # Check for company positioning
        if any(word in title_lower for word in ['leading', 'best', '#1', 'top']):
            insights.append('positioning_claim')
        
        if any(word in title_lower for word in self.PRICING_KEYWORDS):
            insights.append('pricing_focus')
        
        if any(word in title_lower for word in self.FEATURE_KEYWORDS):
            insights.append('feature_focus')
        
        return {
            'title': title,
            'length': len(title),
            'insights': insights,
            'word_count': len(title.split())
        }
    
    def _find_pricing_signals(self, content: str) -> Dict[str, Any]:
        """Find pricing-related information in content."""
        pricing_mentions = []
        for keyword in self.PRICING_KEYWORDS:
            count = content.count(keyword)
            if count > 0:
                pricing_mentions.append({'keyword': keyword, 'count': count})
        
        # Find potential price amounts
        price_patterns = [
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # $1,000.00
            r'\d+(?:,\d{3})*\s*(?:dollars?|usd|per month|/month|per year|/year)',
            r'(?:from|starting at|as low as)\s*\$\d+',
            r'free\s+(?:trial|plan|tier|version)'
        ]
        
        price_matches = []
        for pattern in price_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            price_matches.extend(matches)
        
        return {
            'pricing_keyword_mentions': pricing_mentions,
            'price_references': list(set(price_matches)),
            'total_pricing_signals': len(pricing_mentions),
            'has_pricing_content': len(pricing_mentions) > 5
        }
    
    def _find_feature_mentions(self, content: str) -> Dict[str, Any]:
        """Find feature-related mentions."""
        feature_mentions = []
        for keyword in self.FEATURE_KEYWORDS:
            count = content.count(keyword)
            if count > 0:
                feature_mentions.append({'keyword': keyword, 'count': count})
        
        # Look for feature lists
        feature_list_patterns = [
            r'features?\s*(?:include|:)',
            r'(?:our|key|main|core)\s+features?',
            r'capabilities?\s*(?:include|:)'
        ]
        
        feature_list_indicators = []
        for pattern in feature_list_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                feature_list_indicators.append(pattern)
        
        return {
            'feature_mentions': feature_mentions,
            'feature_list_indicators': feature_list_indicators,
            'total_feature_signals': len(feature_mentions),
            'has_feature_focus': len(feature_mentions) > 3
        }
    
    def _find_competitive_language(self, content: str) -> Dict[str, Any]:
        """Find competitive positioning language."""
        competitive_mentions = []
        for keyword in self.COMPETITIVE_KEYWORDS:
            count = content.count(keyword)
            if count > 0:
                competitive_mentions.append({'keyword': keyword, 'count': count})
        
        # Look for comparison tables or sections
        comparison_patterns = [
            r'comparison\s+(?:table|chart)',
            r'vs\.?\s+\w+',
            r'compared?\s+to',
            r'alternative\s+to',
            r'why\s+choose\s+us'
        ]
        
        comparison_indicators = []
        for pattern in comparison_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                comparison_indicators.extend(matches)
        
        return {
            'competitive_mentions': competitive_mentions,
            'comparison_indicators': comparison_indicators,
            'total_competitive_signals': len(competitive_mentions),
            'has_competitive_focus': len(competitive_mentions) > 2
        }
    
    def _analyze_technology_mentions(self, content: str, detected_technologies: List[str]) -> Dict[str, Any]:
        """Analyze technology stack and mentions."""
        tech_mentions = []
        for keyword in self.TECHNOLOGY_KEYWORDS:
            count = content.count(keyword)
            if count > 0:
                tech_mentions.append({'keyword': keyword, 'count': count})
        
        return {
            'detected_technologies': detected_technologies,
            'technology_mentions': tech_mentions,
            'total_tech_signals': len(tech_mentions),
            'tech_focus_areas': self._categorize_tech_mentions(tech_mentions)
        }
    
    def _categorize_tech_mentions(self, tech_mentions: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Categorize technology mentions."""
        categories = {
            'api_integration': ['api', 'rest', 'graphql', 'sdk', 'webhook'],
            'cloud_infrastructure': ['cloud', 'aws', 'azure', 'gcp', 'kubernetes'],
            'programming_languages': ['javascript', 'python', 'java', 'go', 'ruby'],
            'ai_ml': ['ai', 'ml', 'machine learning', 'artificial intelligence', 'nlp'],
            'frontend': ['react', 'vue', 'angular', 'javascript', 'typescript']
        }
        
        categorized = {}
        for category, keywords in categories.items():
            found = []
            for mention in tech_mentions:
                if mention['keyword'] in keywords:
                    found.append(mention['keyword'])
            if found:
                categorized[category] = found
        
        return categorized
    
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases from content."""
        if not content:
            return []
        
        # Simple phrase extraction (could be enhanced with NLP)
        sentences = re.split(r'[.!?]+', content)
        key_phrases = []
        
        # Look for sentences with competitive keywords
        for sentence in sentences[:20]:  # Limit to first 20 sentences
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:  # Reasonable length
                if any(keyword in sentence.lower() for keyword in 
                      self.COMPETITIVE_KEYWORDS + self.FEATURE_KEYWORDS[:5]):
                    key_phrases.append(sentence)
        
        return key_phrases[:5]  # Return top 5
    
    def _analyze_sentiment_indicators(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment indicators in content."""
        positive_words = [
            'best', 'leading', 'innovative', 'advanced', 'superior', 'excellent',
            'outstanding', 'proven', 'trusted', 'reliable', 'fast', 'easy',
            'powerful', 'comprehensive', 'cutting-edge', 'award-winning'
        ]
        
        negative_words = [
            'slow', 'difficult', 'complex', 'expensive', 'limited', 'basic',
            'outdated', 'legacy', 'complicated', 'unreliable'
        ]
        
        positive_count = sum(content.count(word) for word in positive_words)
        negative_count = sum(content.count(word) for word in negative_words)
        
        return {
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'sentiment_ratio': positive_count / max(1, negative_count),
            'overall_tone': 'positive' if positive_count > negative_count * 2 else 'neutral'
        }
    
    def _analyze_call_to_actions(self, content: str) -> Dict[str, Any]:
        """Analyze call-to-action elements."""
        if not content:
            return {}
        
        cta_patterns = [
            r'(?:get started|sign up|try (?:free|now)|start (?:free )?trial)',
            r'(?:contact us|talk to|speak with|schedule)',
            r'(?:request demo|book demo|see demo)',
            r'(?:learn more|find out|discover)',
            r'(?:download|get|access) (?:now|today|free)'
        ]
        
        ctas_found = []
        for pattern in cta_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            ctas_found.extend(matches)
        
        return {
            'cta_phrases': list(set(ctas_found)),
            'total_ctas': len(ctas_found),
            'cta_density': len(ctas_found) / max(1, len(content.split())) * 1000  # CTAs per 1000 words
        }
    
    def _extract_contact_info(self, content: str) -> Dict[str, Any]:
        """Extract contact information from content."""
        if not content:
            return {}
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        
        # Phone pattern (US format)
        phone_pattern = r'(?:\+1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phones = re.findall(phone_pattern, content)
        
        # Address patterns
        address_pattern = r'\d+\s+[\w\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|Lane|Ln)'
        addresses = re.findall(address_pattern, content, re.IGNORECASE)
        
        return {
            'emails': list(set(emails)),
            'phone_numbers': ['-'.join(phone) for phone in phones],
            'addresses': addresses[:3],  # Limit to first 3
            'has_contact_info': bool(emails or phones or addresses)
        }


class CompetitorWebsiteAnalyzer:
    """High-level analyzer for competitor websites."""
    
    def __init__(self, config=None):
        """
        Initialize analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.scraper = WebScraper(config)
        self.content_analyzer = ContentAnalyzer()
        self.url_validator = URLValidator()
    
    async def analyze_competitor_website(self, 
                                       competitor_url: str,
                                       pages_to_analyze: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a competitor's website.
        
        Args:
            competitor_url: Base URL of competitor
            pages_to_analyze: Specific pages to analyze
            
        Returns:
            Comprehensive analysis results
        """
        # Validate URL
        if not self.url_validator.is_valid_url(competitor_url):
            return {'error': f'Invalid URL: {competitor_url}'}
        
        # Normalize URL
        normalized_url = self.url_validator.normalize_url(competitor_url)
        domain_info = self.url_validator.extract_domain_info(normalized_url)
        
        # Determine pages to scrape
        if not pages_to_analyze:
            pages_to_analyze = [
                {"path": "/", "name": "homepage", "priority": "high"},
                {"path": "/pricing", "name": "pricing", "priority": "high"},
                {"path": "/products", "name": "products", "priority": "medium"},
                {"path": "/features", "name": "features", "priority": "medium"},
                {"path": "/about", "name": "about", "priority": "low"}
            ]
        
        analysis_results = {
            'competitor_url': competitor_url,
            'normalized_url': normalized_url,
            'domain_info': domain_info,
            'analyzed_at': datetime.now().isoformat(),
            'pages_analyzed': [],
            'overall_insights': {},
            'page_analyses': {},
            'summary_metrics': {},
            'technologies_detected': set(),
            'competitive_positioning': {}
        }
        
        try:
            async with self.scraper:
                # Scrape website pages
                scraping_results = await self.scraper.scrape_competitor_website(
                    normalized_url, pages_to_analyze
                )
                
                # Analyze each page
                for result in scraping_results:
                    if result.success:
                        page_analysis = self.content_analyzer.analyze_content(result)
                        analysis_results['page_analyses'][result.page_type] = page_analysis
                        analysis_results['pages_analyzed'].append(result.page_type)
                        
                        # Collect technologies
                        if result.technologies:
                            analysis_results['technologies_detected'].update(result.technologies)
                
                # Generate overall insights
                analysis_results['overall_insights'] = self._generate_overall_insights(
                    analysis_results['page_analyses']
                )
                
                # Calculate summary metrics
                analysis_results['summary_metrics'] = self._calculate_summary_metrics(
                    scraping_results, analysis_results['page_analyses']
                )
                
                # Determine competitive positioning
                analysis_results['competitive_positioning'] = self._analyze_competitive_positioning(
                    analysis_results['page_analyses']
                )
                
                # Convert sets to lists for JSON serialization
                analysis_results['technologies_detected'] = list(analysis_results['technologies_detected'])
        
        except Exception as e:
            logger.error(f"Error analyzing competitor website {competitor_url}: {e}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    def _generate_overall_insights(self, page_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall insights from page analyses."""
        insights = {
            'primary_focus': None,
            'positioning_strategy': [],
            'content_themes': [],
            'user_acquisition_strategy': [],
            'technical_sophistication': 'unknown'
        }
        
        if not page_analyses:
            return insights
        
        # Determine primary focus
        total_pricing_signals = sum(
            analysis.get('pricing_signals', {}).get('total_pricing_signals', 0)
            for analysis in page_analyses.values()
        )
        
        total_feature_signals = sum(
            analysis.get('feature_mentions', {}).get('total_feature_signals', 0)
            for analysis in page_analyses.values()
        )
        
        if total_pricing_signals > total_feature_signals:
            insights['primary_focus'] = 'pricing_driven'
        elif total_feature_signals > 10:
            insights['primary_focus'] = 'feature_driven'
        else:
            insights['primary_focus'] = 'brand_driven'
        
        # Analyze positioning strategy
        for page_type, analysis in page_analyses.items():
            competitive_lang = analysis.get('competitive_language', {})
            if competitive_lang.get('has_competitive_focus'):
                insights['positioning_strategy'].append('direct_comparison')
                break
        
        # Determine technical sophistication
        all_tech_mentions = []
        for analysis in page_analyses.values():
            tech_analysis = analysis.get('technology_stack', {})
            all_tech_mentions.extend(tech_analysis.get('technology_mentions', []))
        
        if len(all_tech_mentions) > 10:
            insights['technical_sophistication'] = 'high'
        elif len(all_tech_mentions) > 3:
            insights['technical_sophistication'] = 'medium'
        else:
            insights['technical_sophistication'] = 'low'
        
        return insights
    
    def _calculate_summary_metrics(self, scraping_results: List[ScrapingResult], 
                                 page_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary metrics."""
        successful_scrapes = [r for r in scraping_results if r.success]
        
        metrics = {
            'total_pages_attempted': len(scraping_results),
            'successful_scrapes': len(successful_scrapes),
            'success_rate': len(successful_scrapes) / len(scraping_results) if scraping_results else 0,
            'average_load_time': 0,
            'total_content_words': 0,
            'pages_with_pricing_focus': 0,
            'pages_with_competitive_language': 0,
            'unique_technologies': 0
        }
        
        if successful_scrapes:
            load_times = [r.load_time for r in successful_scrapes if r.load_time]
            if load_times:
                metrics['average_load_time'] = sum(load_times) / len(load_times)
            
            metrics['total_content_words'] = sum(
                r.word_count for r in successful_scrapes if r.word_count
            )
        
        # Count pages with specific characteristics
        for analysis in page_analyses.values():
            if analysis.get('pricing_signals', {}).get('has_pricing_content'):
                metrics['pages_with_pricing_focus'] += 1
            
            if analysis.get('competitive_language', {}).get('has_competitive_focus'):
                metrics['pages_with_competitive_language'] += 1
        
        return metrics
    
    def _analyze_competitive_positioning(self, page_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competitive positioning strategy."""
        positioning = {
            'strategy_type': 'unknown',
            'key_differentiators': [],
            'target_audience_signals': [],
            'pricing_strategy_indicators': [],
            'market_positioning': []
        }
        
        all_key_phrases = []
        all_cta_phrases = []
        
        for page_type, analysis in page_analyses.items():
            # Collect key phrases
            key_phrases = analysis.get('key_phrases', [])
            all_key_phrases.extend(key_phrases)
            
            # Collect CTA phrases
            cta_analysis = analysis.get('cta_analysis', {})
            cta_phrases = cta_analysis.get('cta_phrases', [])
            all_cta_phrases.extend(cta_phrases)
            
            # Analyze homepage specifically for positioning
            if page_type == 'homepage':
                title_analysis = analysis.get('title_analysis', {})
                if 'positioning_claim' in title_analysis.get('insights', []):
                    positioning['market_positioning'].append('market_leader_claim')
                
                sentiment = analysis.get('sentiment_indicators', {})
                if sentiment.get('overall_tone') == 'positive' and sentiment.get('positive_indicators', 0) > 5:
                    positioning['strategy_type'] = 'value_proposition_focused'
            
            # Analyze pricing page for strategy
            elif page_type == 'pricing':
                pricing_signals = analysis.get('pricing_signals', {})
                price_refs = pricing_signals.get('price_references', [])
                
                if any('free' in ref.lower() for ref in price_refs):
                    positioning['pricing_strategy_indicators'].append('freemium_model')
                
                if any('enterprise' in ref.lower() for ref in price_refs):
                    positioning['pricing_strategy_indicators'].append('enterprise_focused')
        
        # Determine overall strategy type
        if len(all_cta_phrases) > 5:
            positioning['strategy_type'] = 'conversion_focused'
        elif any('demo' in phrase.lower() for phrase in all_cta_phrases):
            positioning['strategy_type'] = 'sales_led'
        elif any('free' in phrase.lower() for phrase in all_cta_phrases):
            positioning['strategy_type'] = 'product_led'
        
        return positioning
    
    async def batch_analyze_competitors(self, competitor_urls: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple competitor websites in batch.
        
        Args:
            competitor_urls: List of competitor URLs
            
        Returns:
            Batch analysis results
        """
        batch_results = {
            'analyzed_at': datetime.now().isoformat(),
            'total_competitors': len(competitor_urls),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'competitor_analyses': {},
            'comparative_insights': {},
            'batch_summary': {}
        }
        
        # Analyze each competitor
        for url in competitor_urls:
            try:
                logger.info(f"Analyzing competitor: {url}")
                analysis = await self.analyze_competitor_website(url)
                
                if 'error' not in analysis:
                    batch_results['successful_analyses'] += 1
                    batch_results['competitor_analyses'][url] = analysis
                else:
                    batch_results['failed_analyses'] += 1
                    batch_results['competitor_analyses'][url] = {'error': analysis['error']}
                
            except Exception as e:
                logger.error(f"Failed to analyze {url}: {e}")
                batch_results['failed_analyses'] += 1
                batch_results['competitor_analyses'][url] = {'error': str(e)}
        
        # Generate comparative insights if we have successful analyses
        successful_analyses = {
            url: analysis for url, analysis in batch_results['competitor_analyses'].items()
            if 'error' not in analysis
        }
        
        if successful_analyses:
            batch_results['comparative_insights'] = self._generate_comparative_insights(successful_analyses)
            batch_results['batch_summary'] = self._generate_batch_summary(successful_analyses)
        
        return batch_results
    
    def _generate_comparative_insights(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights comparing multiple competitors."""
        insights = {
            'technology_landscape': {},
            'positioning_comparison': {},
            'content_strategy_comparison': {},
            'performance_comparison': {},
            'common_patterns': []
        }
        
        # Technology landscape
        all_technologies = {}
        for url, analysis in analyses.items():
            domain = self.url_validator.extract_domain_info(url)['domain']
            technologies = analysis.get('technologies_detected', [])
            all_technologies[domain] = technologies
        
        insights['technology_landscape'] = all_technologies
        
        # Find common technologies
        tech_counts = {}
        for technologies in all_technologies.values():
            for tech in technologies:
                tech_counts[tech] = tech_counts.get(tech, 0) + 1
        
        common_technologies = [tech for tech, count in tech_counts.items() 
                             if count > len(analyses) / 2]
        insights['common_patterns'].append(f"Common technologies: {common_technologies}")
        
        # Positioning comparison
        positioning_strategies = {}
        for url, analysis in analyses.items():
            domain = self.url_validator.extract_domain_info(url)['domain']
            positioning = analysis.get('competitive_positioning', {})
            positioning_strategies[domain] = positioning.get('strategy_type', 'unknown')
        
        insights['positioning_comparison'] = positioning_strategies
        
        # Performance comparison
        performance_data = {}
        for url, analysis in analyses.items():
            domain = self.url_validator.extract_domain_info(url)['domain']
            metrics = analysis.get('summary_metrics', {})
            performance_data[domain] = {
                'average_load_time': metrics.get('average_load_time', 0),
                'success_rate': metrics.get('success_rate', 0),
                'total_content_words': metrics.get('total_content_words', 0)
            }
        
        insights['performance_comparison'] = performance_data
        
        return insights
    
    def _generate_batch_summary(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of batch analysis."""
        summary = {
            'fastest_loading_site': None,
            'most_content_rich': None,
            'most_technically_advanced': None,
            'most_aggressive_positioning': None,
            'average_performance': {}
        }
        
        # Find fastest loading site
        fastest_site = None
        fastest_time = float('inf')
        
        most_content_site = None
        most_content_words = 0
        
        most_tech_site = None
        most_tech_count = 0
        
        all_load_times = []
        all_content_words = []
        
        for url, analysis in analyses.items():
            domain = self.url_validator.extract_domain_info(url)['domain']
            metrics = analysis.get('summary_metrics', {})
            
            # Load time comparison
            load_time = metrics.get('average_load_time', 0)
            if load_time > 0:
                all_load_times.append(load_time)
                if load_time < fastest_time:
                    fastest_time = load_time
                    fastest_site = domain
            
            # Content comparison
            content_words = metrics.get('total_content_words', 0)
            all_content_words.append(content_words)
            if content_words > most_content_words:
                most_content_words = content_words
                most_content_site = domain
            
            # Technology comparison
            tech_count = len(analysis.get('technologies_detected', []))
            if tech_count > most_tech_count:
                most_tech_count = tech_count
                most_tech_site = domain
        
        summary['fastest_loading_site'] = fastest_site
        summary['most_content_rich'] = most_content_site
        summary['most_technically_advanced'] = most_tech_site
        
        # Calculate averages
        if all_load_times:
            summary['average_performance']['load_time'] = sum(all_load_times) / len(all_load_times)
        
        if all_content_words:
            summary['average_performance']['content_words'] = sum(all_content_words) / len(all_content_words)
        
        return summary
    
    def convert_to_competitor_profile(self, analysis: Dict[str, Any], 
                                    competitor_name: Optional[str] = None) -> CompetitorProfile:
        """
        Convert website analysis to CompetitorProfile.
        
        Args:
            analysis: Website analysis results
            competitor_name: Optional competitor name override
            
        Returns:
            CompetitorProfile object
        """
        from ..competitor.models import CompetitorProfile, WebsiteData, DataSourceType, ThreatLevel
        
        # Extract basic info
        url = analysis.get('normalized_url', analysis.get('competitor_url', ''))
        domain_info = analysis.get('domain_info', {})
        
        if not competitor_name:
            competitor_name = domain_info.get('domain', 'Unknown Competitor')
        
        # Create competitor profile
        profile = CompetitorProfile(
            name=competitor_name,
            website=url,
            description=f"Competitor analyzed from {url}",
            last_analyzed=datetime.now(),
            data_sources_used=[DataSourceType.WEBSITE],
            analysis_depth="comprehensive"
        )
        
        # Extract technologies and features
        technologies = analysis.get('technologies_detected', [])
        profile.key_features = technologies[:10]  # Limit to top 10
        
        # Determine competitive threat based on analysis
        overall_insights = analysis.get('overall_insights', {})
        tech_sophistication = overall_insights.get('technical_sophistication', 'low')
        
        if tech_sophistication == 'high':
            profile.competitive_threat = ThreatLevel.HIGH
        elif tech_sophistication == 'medium':
            profile.competitive_threat = ThreatLevel.MEDIUM
        else:
            profile.competitive_threat = ThreatLevel.LOW
        
        # Create website data entries
        page_analyses = analysis.get('page_analyses', {})
        for page_type, page_analysis in page_analyses.items():
            website_data = WebsiteData(
                url=f"{url.rstrip('/')}/{page_type}" if page_type != 'homepage' else url,
                title=page_analysis.get('title_analysis', {}).get('title'),
                content=None,  # Don't store full content
                page_type=page_type,
                last_crawled=datetime.now(),
                technologies=technologies
            )
            profile.website_data.append(website_data)
        
        # Set focus areas based on analysis
        positioning = analysis.get('competitive_positioning', {})
        strategy_type = positioning.get('strategy_type', 'unknown')
        
        if strategy_type == 'product_led':
            profile.focus_areas = ['product', 'user_experience']
        elif strategy_type == 'sales_led':
            profile.focus_areas = ['enterprise', 'sales']
        elif strategy_type == 'conversion_focused':
            profile.focus_areas = ['marketing', 'growth']
        else:
            profile.focus_areas = ['general']
        
        # Add competitive advantages based on analysis
        summary_metrics = analysis.get('summary_metrics', {})
        if summary_metrics.get('average_load_time', 0) < 2.0:
            profile.competitive_advantages.append('fast_website_performance')
        
        if summary_metrics.get('pages_with_pricing_focus', 0) > 0:
            profile.competitive_advantages.append('transparent_pricing')
        
        if len(technologies) > 5:
            profile.competitive_advantages.append('advanced_technology_stack')
        
        return profile


# Utility functions for external use
async def quick_competitor_analysis(competitor_url: str, config=None) -> Dict[str, Any]:
    """Quick analysis of a single competitor website."""
    analyzer = CompetitorWebsiteAnalyzer(config)
    return await analyzer.analyze_competitor_website(competitor_url)


async def batch_competitor_analysis(competitor_urls: List[str], config=None) -> Dict[str, Any]:
    """Batch analysis of multiple competitor websites."""
    analyzer = CompetitorWebsiteAnalyzer(config)
    return await analyzer.batch_analyze_competitors(competitor_urls)


def validate_competitor_urls(urls: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate a list of competitor URLs.
    
    Args:
        urls: List of URLs to validate
        
    Returns:
        Tuple of (valid_urls, invalid_urls)
    """
    validator = URLValidator()
    valid_urls = []
    invalid_urls = []
    
    for url in urls:
        if validator.is_valid_url(url):
            valid_urls.append(validator.normalize_url(url))
        else:
            invalid_urls.append(url)
    
    return valid_urls, invalid_urls


def extract_competitor_insights(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key insights from competitor analysis results.
    
    Args:
        analysis_results: Results from competitor analysis
        
    Returns:
        Dictionary of key insights
    """
    insights = {
        'key_findings': [],
        'competitive_strengths': [],
        'potential_weaknesses': [],
        'technology_advantages': [],
        'strategic_recommendations': []
    }
    
    if 'error' in analysis_results:
        insights['key_findings'].append(f"Analysis failed: {analysis_results['error']}")
        return insights
    
    # Extract key findings
    overall_insights = analysis_results.get('overall_insights', {})
    primary_focus = overall_insights.get('primary_focus')
    if primary_focus:
        insights['key_findings'].append(f"Primary strategy appears to be {primary_focus}")
    
    # Technology advantages
    technologies = analysis_results.get('technologies_detected', [])
    if len(technologies) > 5:
        insights['technology_advantages'].append("Advanced technology stack detected")
        insights['competitive_strengths'].append("Technical sophistication")
    
    # Performance analysis
    summary_metrics = analysis_results.get('summary_metrics', {})
    avg_load_time = summary_metrics.get('average_load_time', 0)
    if avg_load_time > 0 and avg_load_time < 2.0:
        insights['competitive_strengths'].append("Fast website performance")
    elif avg_load_time > 5.0:
        insights['potential_weaknesses'].append("Slow website performance")
    
    # Content strategy
    if summary_metrics.get('pages_with_pricing_focus', 0) > 1:
        insights['competitive_strengths'].append("Clear pricing communication")
    
    if summary_metrics.get('pages_with_competitive_language', 0) > 0:
        insights['key_findings'].append("Uses competitive positioning language")
    
    return insights


# Example usage
if __name__ == "__main__":
    import sys
    import argparse
    
    async def main():
        parser = argparse.ArgumentParser(description="Competitor Website Analysis")
        parser.add_argument("url", help="Competitor URL to analyze")
        parser.add_argument("--batch", "-b", nargs="+", 
                          help="Multiple URLs for batch analysis")
        parser.add_argument("--output", "-o", help="Output file for results")
        parser.add_argument("--insights-only", action="store_true",
                          help="Show only key insights")
        parser.add_argument("--profile", action="store_true",
                          help="Convert to competitor profile")
        
        args = parser.parse_args()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
        if args.batch:
            # Batch analysis
            print("Performing batch competitor analysis...")
            results = await batch_competitor_analysis(args.batch)
            
            print(f"\nBatch Analysis Results:")
            print(f"Total competitors: {results['total_competitors']}")
            print(f"Successful analyses: {results['successful_analyses']}")
            print(f"Failed analyses: {results['failed_analyses']}")
            
            if results['successful_analyses'] > 0:
                comparative_insights = results.get('comparative_insights', {})
                print(f"\nComparative Insights:")
                
                tech_landscape = comparative_insights.get('technology_landscape', {})
                for domain, technologies in tech_landscape.items():
                    print(f"  {domain}: {', '.join(technologies[:5])}")
                
                batch_summary = results.get('batch_summary', {})
                if batch_summary.get('fastest_loading_site'):
                    print(f"  Fastest site: {batch_summary['fastest_loading_site']}")
                if batch_summary.get('most_technically_advanced'):
                    print(f"  Most advanced: {batch_summary['most_technically_advanced']}")
        
        else:
            # Single competitor analysis
            print(f"Analyzing competitor: {args.url}")
            analysis = await quick_competitor_analysis(args.url)
            
            if args.insights_only:
                # Show only key insights
                insights = extract_competitor_insights(analysis)
                print(f"\nKey Insights for {args.url}:")
                
                for finding in insights['key_findings']:
                    print(f"  • {finding}")
                
                if insights['competitive_strengths']:
                    print(f"\nStrengths:")
                    for strength in insights['competitive_strengths']:
                        print(f"  + {strength}")
                
                if insights['potential_weaknesses']:
                    print(f"\nPotential Weaknesses:")
                    for weakness in insights['potential_weaknesses']:
                        print(f"  - {weakness}")
            
            elif args.profile:
                # Convert to competitor profile
                analyzer = CompetitorWebsiteAnalyzer()
                profile = analyzer.convert_to_competitor_profile(analysis)
                
                print(f"\nCompetitor Profile:")
                print(f"  Name: {profile.name}")
                print(f"  Website: {profile.website}")
                print(f"  Threat Level: {profile.competitive_threat.value}")
                print(f"  Key Features: {', '.join(profile.key_features[:5])}")
                print(f"  Competitive Advantages: {', '.join(profile.competitive_advantages)}")
                print(f"  Focus Areas: {', '.join(profile.focus_areas)}")
            
            else:
                # Full analysis results
                if 'error' not in analysis:
                    print(f"\nAnalysis Results for {args.url}:")
                    print(f"  Pages analyzed: {len(analysis.get('pages_analyzed', []))}")
                    print(f"  Technologies detected: {len(analysis.get('technologies_detected', []))}")
                    
                    summary = analysis.get('summary_metrics', {})
                    print(f"  Success rate: {summary.get('success_rate', 0):.1%}")
                    print(f"  Average load time: {summary.get('average_load_time', 0):.2f}s")
                    print(f"  Total content words: {summary.get('total_content_words', 0):,}")
                    
                    insights = analysis.get('overall_insights', {})
                    print(f"  Primary focus: {insights.get('primary_focus', 'unknown')}")
                    print(f"  Technical sophistication: {insights.get('technical_sophistication', 'unknown')}")
                else:
                    print(f"Analysis failed: {analysis['error']}")
        
        # Save results if requested
        if args.output:
            output_data = results if args.batch else analysis
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nResults saved to: {args.output}")
    
    # Run the analyzer
    if sys.version_info >= (3, 7):
        asyncio.run(main())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())