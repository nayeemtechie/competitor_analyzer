# src/competitor/collectors/website.py
"""
Website data collector for competitor analysis
"""

import asyncio
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import logging

from .base import CachedCollector, RateLimitedSession
from ..models import WebsiteData, PricingTier, CaseStudy

logger = logging.getLogger(__name__)

class WebsiteCollector(CachedCollector):
    """Collects comprehensive website data"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "website")
        self.user_agent = config.get('user_agent', 'CompetitorAnalysis Bot 1.0')
        
    async def _collect_data(self, competitor_name: str, website: str, target_pages: List[Dict] = None) -> WebsiteData:
        """Collect website data"""
        if not target_pages:
            target_pages = [
                {'path': '/', 'name': 'homepage', 'priority': 'high'},
                {'path': '/pricing', 'name': 'pricing', 'priority': 'high'},
                {'path': '/products', 'name': 'products', 'priority': 'high'}
            ]
        
        website_data = WebsiteData()
        
        async with RateLimitedSession(rate_limit=self.rate_limit) as session:
            for page_info in target_pages:
                url = urljoin(website, page_info['path'])
                
                try:
                    content = await session.get(url)
                    if content:
                        soup = BeautifulSoup(content, 'html.parser')
                        page_data = await self._analyze_page(soup, page_info, url)
                        website_data.key_pages[page_info['name']] = page_data
                        website_data.pages_analyzed.append(url)
                        
                        # Extract page-specific data
                        if page_info['name'] == 'pricing':
                            pricing_data = await self._extract_pricing_data(soup)
                            if pricing_data:
                                page_data['pricing_tiers'] = pricing_data
                        
                        elif page_info['name'] == 'products':
                            features = await self._extract_product_features(soup)
                            if features:
                                page_data['features'] = features
                        
                        await asyncio.sleep(1)  # Be respectful
                        
                except Exception as e:
                    logger.warning(f"Failed to collect from {url}: {e}")
        
        # Extract derived insights
        website_data.technology_stack = await self._detect_technology_stack(website_data)
        website_data.content_themes = await self._extract_content_themes(website_data)
        
        return website_data
    
    async def _analyze_page(self, soup: BeautifulSoup, page_info: Dict, url: str) -> Dict[str, Any]:
        """Analyze individual page content"""
        return {
            'url': url,
            'title': soup.title.string if soup.title else '',
            'meta_description': self._get_meta_description(soup),
            'headings': [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])[:10]],
            'word_count': len(soup.get_text().split()),
            'key_content': self._extract_key_content_blocks(soup),
            'forms_count': len(soup.find_all('form')),
            'images_count': len(soup.find_all('img')),
            'external_links': len([a for a in soup.find_all('a', href=True) 
                                 if self._is_external_link(a['href'], url)])
        }
    
    def _get_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description"""
        meta = soup.find('meta', attrs={'name': 'description'})
        if not meta:
            meta = soup.find('meta', attrs={'property': 'og:description'})
        return meta.get('content', '') if meta else ''
    
    def _extract_key_content_blocks(self, soup: BeautifulSoup) -> List[str]:
        """Extract main content blocks"""
        content_blocks = []
        
        # Look for main content areas
        main_selectors = ['main', 'article', '.content', '.main-content', '#content']
        
        for selector in main_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if 100 < len(text) < 1000:  # Reasonable content length
                    content_blocks.append(text)
                    break  # One main content block per selector
        
        return content_blocks[:5]  # Limit to 5 blocks
    
    def _is_external_link(self, href: str, base_url: str) -> bool:
        """Check if link is external"""
        from urllib.parse import urlparse
        try:
            base_domain = urlparse(base_url).netloc
            link_domain = urlparse(href).netloc
            return link_domain and link_domain != base_domain and not href.startswith('#')
        except:
            return False
    
    async def _extract_pricing_data(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract pricing information"""
        pricing_tiers = []
        
        # Look for pricing elements
        pricing_elements = soup.find_all(['div', 'section'], 
                                       class_=lambda x: x and any(word in x.lower() 
                                       for word in ['pricing', 'plan', 'tier', 'package']))
        
        for element in pricing_elements[:5]:  # Limit to 5 tiers
            tier_data = {
                'name': self._extract_tier_name(element),
                'price': self._extract_price(element),
                'features': self._extract_tier_features(element),
                'popular': self._is_popular_tier(element)
            }
            
            if tier_data['name'] or tier_data['price']:
                pricing_tiers.append(tier_data)
        
        return pricing_tiers
    
    def _extract_tier_name(self, element) -> str:
        """Extract pricing tier name"""
        for tag in ['h2', 'h3', 'h4']:
            heading = element.find(tag)
            if heading:
                text = heading.get_text().strip()
                if len(text) < 50:  # Reasonable name length
                    return text
        
        # Look for class names containing tier/plan info
        name_element = element.find(class_=lambda x: x and any(word in x.lower() 
                                   for word in ['name', 'title', 'tier', 'plan']))
        if name_element:
            return name_element.get_text().strip()
        
        return ''
    
    def _extract_price(self, element) -> str:
        """Extract price from pricing element"""
        import re
        text = element.get_text()
        
        # Price patterns
        patterns = [
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # $99, $1,000.00
            r'€\d+(?:,\d{3})*(?:\.\d{2})?',   # €99
            r'£\d+(?:,\d{3})*(?:\.\d{2})?',   # £99
            r'\d+(?:,\d{3})*\s*(?:USD|EUR|GBP)'  # 99 USD
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return ''
    
    def _extract_tier_features(self, element) -> List[str]:
        """Extract features from pricing tier"""
        features = []
        
        # Look for feature lists
        lists = element.find_all(['ul', 'ol'])
        for ul in lists:
            items = ul.find_all('li')[:10]  # Limit features
            for item in items:
                feature = item.get_text().strip()
                if feature and len(feature) < 200:
                    features.append(feature)
        
        return features
    
    def _is_popular_tier(self, element) -> bool:
        """Check if tier is marked as popular"""
        text = element.get_text().lower()
        classes = ' '.join(element.get('class', [])).lower()
        
        popular_indicators = ['popular', 'recommended', 'best', 'most popular', 'featured']
        return any(indicator in text or indicator in classes for indicator in popular_indicators)
    
    async def _extract_product_features(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract product features"""
        features = []
        
        # Look for feature elements
        feature_elements = soup.find_all(['div', 'section'], 
                                       class_=lambda x: x and 'feature' in x.lower())
        
        for element in feature_elements[:15]:  # Limit features
            title_elem = element.find(['h3', 'h4', 'h5'])
            if title_elem:
                title = title_elem.get_text().strip()
                
                # Look for description
                desc_elem = element.find(['p', 'div'], 
                                       class_=lambda x: x and any(word in x.lower() 
                                       for word in ['desc', 'description', 'summary']))
                description = desc_elem.get_text().strip() if desc_elem else ''
                
                if title:
                    features.append({
                        'title': title,
                        'description': description[:300]  # Limit description
                    })
        
        return features
    
    async def _detect_technology_stack(self, website_data: WebsiteData) -> List[str]:
        """Detect technology stack from website content"""
        tech_stack = set()
        
        # Analyze all content
        all_text = ""
        for page_data in website_data.key_pages.values():
            if isinstance(page_data, dict):
                all_text += " ".join(str(v) for v in page_data.values() if isinstance(v, str))
        
        all_text = all_text.lower()
        
        # Technology indicators
        tech_patterns = {
            'React': r'\breact\b',
            'Vue.js': r'\bvue\.?js\b',
            'Angular': r'\bangular\b',
            'Node.js': r'\bnode\.?js\b',
            'Python': r'\bpython\b',
            'Java': r'\bjava\b(?!\s*script)',
            'JavaScript': r'\bjavascript\b|\bjs\b',
            'TypeScript': r'\btypescript\b|\bts\b',
            'AWS': r'\baws\b|amazon web services',
            'Google Cloud': r'google cloud|gcp',
            'Azure': r'\bazure\b',
            'Docker': r'\bdocker\b',
            'Kubernetes': r'\bkubernetes\b|\bk8s\b',
            'Elasticsearch': r'\belasticsearch\b',
            'MongoDB': r'\bmongodb\b',
            'PostgreSQL': r'\bpostgresql\b|\bpostgres\b',
            'Redis': r'\bredis\b',
            'GraphQL': r'\bgraphql\b',
            'REST API': r'\brest\s+api\b|\brestful\b'
        }
        
        for tech, pattern in tech_patterns.items():
            import re
            if re.search(pattern, all_text, re.IGNORECASE):
                tech_stack.add(tech)
        
        return list(tech_stack)
    
    async def _extract_content_themes(self, website_data: WebsiteData) -> List[str]:
        """Extract content themes"""
        themes = set()
        
        # Collect all text
        all_text = ""
        for page_data in website_data.key_pages.values():
            if isinstance(page_data, dict):
                headings = page_data.get('headings', [])
                content = page_data.get('key_content', [])
                all_text += " ".join(headings + content)
        
        all_text = all_text.lower()
        
        # Theme keywords
        theme_mapping = {
            'Search': ['search', 'find', 'discovery', 'query'],
            'AI/ML': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'algorithm'],
            'Personalization': ['personalization', 'personalized', 'custom', 'tailored'],
            'Analytics': ['analytics', 'insights', 'data', 'metrics', 'reporting'],
            'Performance': ['performance', 'speed', 'fast', 'optimization'],
            'Enterprise': ['enterprise', 'scalable', 'scale', 'business'],
            'Integration': ['api', 'integration', 'connect', 'sync'],
            'Ecommerce': ['ecommerce', 'e-commerce', 'retail', 'shopping', 'store'],
            'Security': ['security', 'secure', 'privacy', 'compliance'],
            'Cloud': ['cloud', 'saas', 'hosted', 'managed']
        }
        
        for theme, keywords in theme_mapping.items():
            if any(keyword in all_text for keyword in keywords):
                themes.add(theme)
        
        return list(themes)
    
    def _get_empty_result(self) -> WebsiteData:
        """Return empty website data"""
        return WebsiteData()