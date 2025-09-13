# src/competitor/scraper.py
"""
Web scraping functionality for competitor analysis
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse, quote
import re
import logging
from datetime import datetime

from .utils.web_utils import WebUtils
from .models import WebsiteData

logger = logging.getLogger(__name__)

class CompetitorScraper:
    """Enhanced web scraper for competitor websites"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('scraping', {})
        self.session = None
        self.web_utils = WebUtils()
        
        # Default headers
        self.headers = {
            'User-Agent': self.config.get('user_agent', 'CompetitorAnalysis Bot 1.0'),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.config.get('concurrent_requests', 3))
        timeout = aiohttp.ClientTimeout(total=self.config.get('timeout', 30))
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def scrape_competitor_website(self, competitor: Dict[str, Any], target_pages: List[Dict[str, str]]) -> WebsiteData:
        """Scrape comprehensive data from competitor website"""
        website = competitor['website']
        name = competitor['name']
        
        logger.info(f"Scraping {name} website: {website}")
        
        website_data = WebsiteData(
            pages_analyzed=[],
            key_pages={},
            technology_stack=[],
            seo_metrics={},
            content_themes=[],
            last_analyzed=datetime.now().isoformat()
        )
        
        try:
            # Process target pages based on priority
            high_priority_pages = [p for p in target_pages if p.get('priority') == 'high']
            medium_priority_pages = [p for p in target_pages if p.get('priority') == 'medium']
            
            # Always scrape high priority pages
            for page_info in high_priority_pages:
                await self._scrape_page(website, page_info, website_data)
                await asyncio.sleep(self.config.get('rate_limit', 1.0))
            
            # Scrape medium priority pages based on analysis depth
            depth = self.config.get('depth_level', 'standard')
            if depth in ['standard', 'comprehensive']:
                for page_info in medium_priority_pages:
                    await self._scrape_page(website, page_info, website_data)
                    await asyncio.sleep(self.config.get('rate_limit', 1.0))
            
            # Additional comprehensive analysis
            if depth == 'comprehensive':
                await self._deep_content_analysis(website_data)
            
            # Extract derived insights
            website_data.technology_stack = await self._detect_technology_stack(website_data)
            website_data.content_themes = await self._extract_content_themes(website_data)
            website_data.seo_metrics = await self._analyze_seo_metrics(website_data)
            
        except Exception as e:
            logger.error(f"Error scraping {name}: {e}")
        
        return website_data
    
    async def _scrape_page(self, base_url: str, page_info: Dict[str, str], website_data: WebsiteData) -> None:
        """Scrape a single page and extract information"""
        path = page_info['path']
        page_name = page_info['name']
        
        url = urljoin(base_url, path)
        
        try:
            html_content = await self._fetch_page_content(url)
            if not html_content:
                return
            
            soup = BeautifulSoup(html_content, 'html.parser')
            page_data = await self._analyze_page_content(soup, url, page_name)
            
            website_data.key_pages[page_name] = page_data
            website_data.pages_analyzed.append(url)
            
            logger.debug(f"Successfully scraped {url}")
            
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
    
    async def _fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch HTML content from a URL"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None
    
    async def _analyze_page_content(self, soup: BeautifulSoup, url: str, page_type: str) -> Dict[str, Any]:
        """Extract structured data from a page"""
        page_data = {
            'url': url,
            'page_type': page_type,
            'title': self._extract_title(soup),
            'meta_description': self._extract_meta_description(soup),
            'headings': self._extract_headings(soup),
            'key_content': self._extract_key_content(soup, page_type),
            'links': self._extract_links(soup, url),
            'images': self._extract_images(soup),
            'forms': self._extract_forms(soup),
            'structured_data': self._extract_structured_data(soup),
            'word_count': len(soup.get_text().split()),
            'last_scraped': datetime.now().isoformat()
        }
        
        # Page-specific extractions
        if page_type == 'pricing':
            page_data['pricing_info'] = await self._extract_pricing_information(soup)
        elif page_type == 'products':
            page_data['features'] = await self._extract_product_features(soup)
        elif page_type == 'customers' or page_type == 'case_studies':
            page_data['case_studies'] = await self._extract_case_studies(soup)
        elif page_type == 'careers':
            page_data['job_info'] = await self._extract_job_information(soup)
        elif page_type == 'about':
            page_data['company_info'] = await self._extract_company_information(soup)
        
        return page_data
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else ''
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description"""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if not meta_desc:
            meta_desc = soup.find('meta', attrs={'property': 'og:description'})
        return meta_desc.get('content', '').strip() if meta_desc else ''
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract headings hierarchy"""
        headings = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            headings.append({
                'level': tag.name,
                'text': tag.get_text().strip()
            })
        return headings[:20]  # Limit to prevent overwhelming data
    
    def _extract_key_content(self, soup: BeautifulSoup, page_type: str) -> List[str]:
        """Extract key content based on page type"""
        content_selectors = {
            'homepage': ['main', '.hero', '.intro', '.value-prop'],
            'products': ['.features', '.capabilities', '.product-info'],
            'pricing': ['.pricing', '.plans', '.tiers'],
            'about': ['.about', '.company-info', '.mission'],
            'customers': ['.testimonials', '.case-studies', '.customers'],
        }
        
        selectors = content_selectors.get(page_type, ['main', 'article', '.content'])
        key_content = []
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements[:3]:  # Limit per selector
                text = element.get_text().strip()
                if len(text) > 50 and len(text) < 1000:  # Reasonable content length
                    key_content.append(text)
        
        return key_content
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> Dict[str, List[str]]:
        """Extract and categorize links"""
        links = {'internal': [], 'external': [], 'social': []}
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            if self.web_utils.is_social_link(href):
                links['social'].append(href)
            elif self.web_utils.is_external_link(href, base_url):
                links['external'].append(href)
            else:
                links['internal'].append(href)
        
        return {k: list(set(v))[:10] for k, v in links.items()}  # Dedupe and limit
    
    def _extract_images(self, soup: BeautifulSoup) -> Dict[str, int]:
        """Extract image statistics"""
        images = soup.find_all('img')
        return {
            'total_images': len(images),
            'images_with_alt': len([img for img in images if img.get('alt')]),
            'logo_images': len([img for img in images if 'logo' in str(img.get('src', '')).lower()])
        }
    
    def _extract_forms(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract form information"""
        forms = []
        for form in soup.find_all('form'):
            form_data = {
                'action': form.get('action', ''),
                'method': form.get('method', 'GET').upper(),
                'fields': len(form.find_all(['input', 'select', 'textarea'])),
                'has_email_field': bool(form.find('input', {'type': 'email'})),
                'has_submit_button': bool(form.find(['input', 'button'], {'type': 'submit'}))
            }
            forms.append(form_data)
        
        return forms
    
    def _extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract structured data (JSON-LD, microdata)"""
        structured_data = {'json_ld': [], 'microdata': []}
        
        # JSON-LD
        for script in soup.find_all('script', {'type': 'application/ld+json'}):
            try:
                import json
                data = json.loads(script.string)
                structured_data['json_ld'].append(data)
            except:
                pass
        
        # Basic microdata
        microdata_items = soup.find_all(attrs={'itemtype': True})
        for item in microdata_items:
            structured_data['microdata'].append({
                'type': item.get('itemtype'),
                'properties': len(item.find_all(attrs={'itemprop': True}))
            })
        
        return structured_data
    
    async def _extract_pricing_information(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract pricing tiers and information"""
        pricing_data = {'tiers': [], 'currency': None, 'billing_periods': []}
        
        # Look for pricing containers
        pricing_selectors = [
            '.pricing-tier', '.price-card', '.plan', '.package',
            '[class*="pricing"]', '[class*="plan"]', '[class*="tier"]'
        ]
        
        pricing_elements = []
        for selector in pricing_selectors:
            pricing_elements.extend(soup.select(selector))
        
        # Extract tier information
        for element in pricing_elements[:5]:  # Limit to 5 tiers
            tier_data = {
                'name': self._extract_tier_name(element),
                'price': self._extract_price(element),
                'features': self._extract_tier_features(element),
                'popular': self._is_popular_tier(element)
            }
            
            if tier_data['name'] or tier_data['price']:
                pricing_data['tiers'].append(tier_data)
        
        # Extract currency and billing info
        pricing_data['currency'] = self._detect_currency(soup)
        pricing_data['billing_periods'] = self._detect_billing_periods(soup)
        
        return pricing_data
    
    def _extract_tier_name(self, element) -> str:
        """Extract pricing tier name"""
        name_selectors = ['h2', 'h3', 'h4', '.tier-name', '.plan-name', '.name']
        
        for selector in name_selectors:
            name_elem = element.select_one(selector)
            if name_elem:
                text = name_elem.get_text().strip()
                if text and len(text) < 50:  # Reasonable name length
                    return text
        
        return ''
    
    def _extract_price(self, element) -> str:
        """Extract price from pricing tier"""
        # Common price patterns
        price_patterns = [
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # $99, $1,000, $99.99
            r'€\d+(?:,\d{3})*(?:\.\d{2})?',   # €99
            r'£\d+(?:,\d{3})*(?:\.\d{2})?',   # £99
            r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)',  # 99 USD
        ]
        
        text = element.get_text()
        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return ''
    
    def _extract_tier_features(self, element) -> List[str]:
        """Extract features from pricing tier"""
        features = []
        
        # Look for feature lists
        feature_lists = element.find_all(['ul', 'ol'])
        for ul in feature_lists:
            items = ul.find_all('li')
            for item in items:
                feature_text = item.get_text().strip()
                if feature_text and len(feature_text) < 200:
                    features.append(feature_text)
        
        return features[:10]  # Limit features per tier
    
    def _is_popular_tier(self, element) -> bool:
        """Check if tier is marked as popular/recommended"""
        popular_indicators = ['popular', 'recommended', 'best', 'most popular']
        element_text = element.get_text().lower()
        element_classes = ' '.join(element.get('class', []))
        
        return any(indicator in element_text or indicator in element_classes 
                  for indicator in popular_indicators)
    
    def _detect_currency(self, soup: BeautifulSoup) -> Optional[str]:
        """Detect currency used in pricing"""
        text = soup.get_text()
        if '$' in text:
            return 'USD'
        elif '€' in text:
            return 'EUR'
        elif '£' in text:
            return 'GBP'
        return None
    
    def _detect_billing_periods(self, soup: BeautifulSoup) -> List[str]:
        """Detect billing periods mentioned"""
        billing_patterns = [
            r'\b(monthly|month|per month|/month)\b',
            r'\b(yearly|annually|annual|year|per year|/year)\b',
            r'\b(quarterly|quarter|per quarter|/quarter)\b',
            r'\b(weekly|week|per week|/week)\b'
        ]
        
        text = soup.get_text().lower()
        periods = []
        
        for pattern in billing_patterns:
            if re.search(pattern, text):
                periods.append(re.search(pattern, text).group(1))
        
        return list(set(periods))
    
    async def _extract_product_features(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract product features from product pages"""
        features = []
        
        # Feature extraction selectors
        feature_selectors = [
            '.feature', '.capability', '.benefit',
            '[class*="feature"]', '[class*="capability"]'
        ]
        
        for selector in feature_selectors:
            elements = soup.select(selector)
            for element in elements[:15]:  # Limit features
                feature_title = self._extract_feature_title(element)
                feature_description = self._extract_feature_description(element)
                
                if feature_title:
                    features.append({
                        'title': feature_title,
                        'description': feature_description
                    })
        
        return features
    
    def _extract_feature_title(self, element) -> str:
        """Extract feature title"""
        title_selectors = ['h3', 'h4', 'h5', '.title', '.name']
        
        for selector in title_selectors:
            title_elem = element.select_one(selector)
            if title_elem:
                return title_elem.get_text().strip()
        
        # Fallback to first text content
        text = element.get_text().strip()
        if text:
            # Take first sentence or first 100 chars
            first_sentence = text.split('.')[0]
            return first_sentence[:100] if len(first_sentence) < 100 else first_sentence
        
        return ''
    
    def _extract_feature_description(self, element) -> str:
        """Extract feature description"""
        # Try to find description elements
        desc_selectors = ['.description', '.desc', 'p']
        
        for selector in desc_selectors:
            desc_elem = element.select_one(selector)
            if desc_elem:
                desc_text = desc_elem.get_text().strip()
                if len(desc_text) > 20:  # Must be substantial
                    return desc_text[:500]  # Limit length
        
        return ''
    
    async def _extract_case_studies(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract customer case studies"""
        case_studies = []
        
        case_study_selectors = [
            '.case-study', '.customer-story', '.testimonial',
            '[class*="case"]', '[class*="story"]', '[class*="testimonial"]'
        ]
        
        for selector in case_study_selectors:
            elements = soup.select(selector)
            for element in elements[:10]:  # Limit case studies
                case_study = {
                    'title': self._extract_case_study_title(element),
                    'customer': self._extract_customer_name(element),
                    'industry': self._extract_industry(element),
                    'summary': self._extract_case_study_summary(element),
                    'results': self._extract_case_study_results(element)
                }
                
                if case_study['title'] or case_study['customer']:
                    case_studies.append(case_study)
        
        return case_studies
    
    def _extract_case_study_title(self, element) -> str:
        """Extract case study title"""
        title_selectors = ['h2', 'h3', 'h4', '.title']
        
        for selector in title_selectors:
            title_elem = element.select_one(selector)
            if title_elem:
                return title_elem.get_text().strip()[:200]
        
        return ''
    
    def _extract_customer_name(self, element) -> str:
        """Extract customer/company name"""
        # Look for company names in various formats
        text = element.get_text()
        
        # Common patterns for company names
        company_patterns = [
            r'Customer:\s*([^\n]+)',
            r'Company:\s*([^\n]+)',
            r'Client:\s*([^\n]+)',
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Look for quoted company names or names in specific elements
        company_elem = element.select_one('.company, .customer-name, .client')
        if company_elem:
            return company_elem.get_text().strip()
        
        return ''
    
    def _extract_industry(self, element) -> str:
        """Extract industry from case study"""
        text = element.get_text().lower()
        
        # Common industry keywords
        industries = [
            'retail', 'ecommerce', 'e-commerce', 'fashion', 'automotive',
            'healthcare', 'fintech', 'banking', 'insurance', 'travel',
            'media', 'publishing', 'saas', 'technology', 'manufacturing'
        ]
        
        for industry in industries:
            if industry in text:
                return industry.title()
        
        return ''
    
    def _extract_case_study_summary(self, element) -> str:
        """Extract case study summary"""
        # Look for summary or description
        summary_elem = element.select_one('.summary, .description, .overview')
        if summary_elem:
            return summary_elem.get_text().strip()[:500]
        
        # Fallback to first paragraph
        p_elem = element.select_one('p')
        if p_elem:
            return p_elem.get_text().strip()[:500]
        
        return ''
    
    def _extract_case_study_results(self, element) -> List[str]:
        """Extract quantitative results from case study"""
        text = element.get_text()
        results = []
        
        # Look for percentage improvements
        percentage_pattern = r'(\d+(?:\.\d+)?%\s+(?:increase|improvement|growth|boost|uplift|more))'
        percentages = re.findall(percentage_pattern, text, re.IGNORECASE)
        results.extend(percentages[:3])
        
        # Look for multiplier results
        multiplier_pattern = r'(\d+x\s+(?:increase|improvement|growth|boost|more|faster))'
        multipliers = re.findall(multiplier_pattern, text, re.IGNORECASE)
        results.extend(multipliers[:3])
        
        return results
    
    async def _extract_job_information(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract job posting information from careers pages"""
        job_info = {
            'total_openings': 0,
            'departments': [],
            'locations': [],
            'recent_postings': []
        }
        
        # Count job postings
        job_elements = soup.select('.job, .position, .opening, [class*="job"]')
        job_info['total_openings'] = len(job_elements)
        
        # Extract departments and locations
        departments = set()
        locations = set()
        
        for job_elem in job_elements[:20]:  # Limit processing
            # Extract department
            dept_elem = job_elem.select_one('.department, .team, .category')
            if dept_elem:
                departments.add(dept_elem.get_text().strip())
            
            # Extract location  
            location_elem = job_elem.select_one('.location, .office, .city')
            if location_elem:
                locations.add(location_elem.get_text().strip())
        
        job_info['departments'] = list(departments)
        job_info['locations'] = list(locations)
        
        return job_info
    
    async def _extract_company_information(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract company information from about pages"""
        company_info = {}
        
        # Extract founding year
        text = soup.get_text()
        year_pattern = r'(?:founded|established|started).*?(\d{4})'
        year_match = re.search(year_pattern, text, re.IGNORECASE)
        if year_match:
            company_info['founded'] = year_match.group(1)
        
        # Extract team size
        size_patterns = [
            r'(\d+(?:,\d{3})*)\s+(?:employees|people|team members)',
            r'team of (\d+(?:,\d{3})*)',
            r'(\d+(?:,\d{3})*)\+ (?:employees|people)'
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                company_info['employees'] = match.group(1)
                break
        
        # Extract headquarters
        hq_patterns = [
            r'(?:headquarters|headquartered|based|located).*?in\s+([^.\n]+)',
            r'(?:hq|office).*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in hq_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                company_info['headquarters'] = match.group(1).strip()
                break
        
        return company_info
    
    async def _deep_content_analysis(self, website_data: WebsiteData) -> None:
        """Perform deep content analysis for comprehensive mode"""
        # This would include more sophisticated analysis
        # - Content sentiment analysis
        # - Keyword density analysis  
        # - Competitive messaging analysis
        # - Technical SEO analysis
        pass
    
    async def _detect_technology_stack(self, website_data: WebsiteData) -> List[str]:
        """Detect technology stack from website analysis"""
        tech_stack = set()
        
        # Analyze all page content for technology indicators
        all_text = ""
        for page_data in website_data.key_pages.values():
            if isinstance(page_data, dict):
                all_text += " ".join(str(v) for v in page_data.values())
        
        all_text = all_text.lower()
        
        # Technology indicators
        tech_indicators = {
            'React': r'\breact\b',
            'Vue.js': r'\bvue\.js\b|\bvuejs\b',
            'Angular': r'\bangular\b',
            'Node.js': r'\bnode\.js\b|\bnodejs\b',
            'Python': r'\bpython\b',
            'Java': r'\bjava\b',
            'Ruby': r'\bruby\b',
            'PHP': r'\bphp\b',
            'AWS': r'\baws\b|amazon web services',
            'Google Cloud': r'google cloud|gcp',
            'Azure': r'microsoft azure|\bazure\b',
            'Docker': r'\bdocker\b',
            'Kubernetes': r'\bkubernetes\b|\bk8s\b',
            'Elasticsearch': r'\belasticsearch\b',
            'MongoDB': r'\bmongodb\b',
            'PostgreSQL': r'\bpostgresql\b|\bpostgres\b',
            'Redis': r'\bredis\b',
            'GraphQL': r'\bgraphql\b',
            'REST API': r'\brest api\b|\brestful\b',
            'Microservices': r'\bmicroservices\b'
        }
        
        for tech, pattern in tech_indicators.items():
            if re.search(pattern, all_text):
                tech_stack.add(tech)
        
        return list(tech_stack)
    
    async def _extract_content_themes(self, website_data: WebsiteData) -> List[str]:
        """Extract main content themes and topics"""
        # This would use more sophisticated NLP
        # For now, simple keyword extraction
        
        themes = set()
        
        # Common ecommerce/search themes
        theme_keywords = {
            'Search': ['search', 'find', 'discovery'],
            'AI/ML': ['artificial intelligence', 'machine learning', 'ai', 'ml'],
            'Personalization': ['personalization', 'personalized', 'custom'],
            'Analytics': ['analytics', 'insights', 'data'],
            'Performance': ['performance', 'speed', 'fast'],
            'Enterprise': ['enterprise', 'scalable', 'scale'],
            'API': ['api', 'integration', 'developer'],
            'Ecommerce': ['ecommerce', 'e-commerce', 'retail', 'shopping']
        }
        
        # Collect all text content
        all_text = ""
        for page_data in website_data.key_pages.values():
            if isinstance(page_data, dict):
                all_text += " ".join(str(v) for v in page_data.values())
        
        all_text = all_text.lower()
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                themes.add(theme)
        
        return list(themes)
    
    async def _analyze_seo_metrics(self, website_data: WebsiteData) -> Dict[str, Any]:
        """Analyze basic SEO metrics"""
        seo_metrics = {}
        
        for page_name, page_data in website_data.key_pages.items():
            if not isinstance(page_data, dict):
                continue
                
            # Title analysis
            title = page_data.get('title', '')
            seo_metrics[f'{page_name}_title_length'] = len(title)
            
            # Meta description analysis
            meta_desc = page_data.get('meta_description', '')
            seo_metrics[f'{page_name}_meta_desc_length'] = len(meta_desc)
            
            # Heading structure analysis
            headings = page_data.get('headings', [])
            h1_count = len([h for h in headings if h.get('level') == 'h1'])
            seo_metrics[f'{page_name}_h1_count'] = h1_count
            
            # Image alt analysis
            images = page_data.get('images', {})
            if images:
                total_images = images.get('total_images', 0)
                images_with_alt = images.get('images_with_alt', 0)
                alt_ratio = images_with_alt / total_images if total_images > 0 else 0
                seo_metrics[f'{page_name}_image_alt_ratio'] = round(alt_ratio, 2)
        
        return seo_metrics