# src/competitor/utils/web_utils.py
"""
Web scraping utilities and helper functions
"""

import re
from urllib.parse import urlparse, urljoin
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)

class WebUtils:
    """Utility functions for web scraping and analysis"""
    
    def __init__(self):
        # Social media domains
        self.social_domains = {
            'twitter.com', 'x.com', 'linkedin.com', 'facebook.com', 'instagram.com',
            'youtube.com', 'github.com', 'medium.com', 'discord.gg', 'reddit.com',
            'tiktok.com', 'snapchat.com', 'pinterest.com', 'telegram.org'
        }
        
        # Known technology indicators
        self.technology_indicators = {
            'React': [
                r'react\.js', r'reactjs', r'react\s+framework',
                r'data-reactroot', r'__react', r'react-dom'
            ],
            'Vue.js': [
                r'vue\.js', r'vuejs', r'vue\s+framework',
                r'data-v-', r'__vue__', r'vue-router'
            ],
            'Angular': [
                r'angular\.js', r'angularjs', r'angular\s+framework',
                r'ng-', r'angular-', r'@angular'
            ],
            'Next.js': [
                r'next\.js', r'nextjs', r'_next/',
                r'__NEXT_DATA__', r'next/head'
            ],
            'Nuxt.js': [
                r'nuxt\.js', r'nuxtjs', r'_nuxt/',
                r'__NUXT__', r'nuxt-link'
            ],
            'Gatsby': [
                r'gatsby\.js', r'gatsbyjs', r'gatsby-',
                r'___gatsby', r'public/static/'
            ],
            'WordPress': [
                r'wp-content', r'wp-includes', r'wordpress',
                r'/wp-json/', r'wp_enqueue_script'
            ],
            'Shopify': [
                r'shopify', r'cdn\.shopify\.com', r'myshopify\.com',
                r'Shopify\.theme', r'shopify_pay'
            ],
            'Salesforce': [
                r'salesforce', r'force\.com', r'visualforce',
                r'sfdc', r'lightning'
            ]
        }
    
    def is_external_link(self, href: str, base_url: str) -> bool:
        """Check if a link is external to the base domain"""
        try:
            if not href or href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:'):
                return False
            
            if href.startswith('javascript:') or href.startswith('data:'):
                return False
            
            # Parse URLs
            base_domain = urlparse(base_url).netloc.lower()
            
            if href.startswith('http'):
                link_domain = urlparse(href).netloc.lower()
            else:
                # Relative link
                return False
            
            # Remove www. for comparison
            base_domain = base_domain.replace('www.', '')
            link_domain = link_domain.replace('www.', '')
            
            return link_domain != base_domain and link_domain != ''
            
        except Exception as e:
            logger.debug(f"Error checking external link {href}: {e}")
            return False
    
    def is_social_link(self, href: str) -> bool:
        """Check if a link is to a social media platform"""
        try:
            if not href or not href.startswith('http'):
                return False
            
            domain = urlparse(href).netloc.lower()
            domain = domain.replace('www.', '')
            
            return any(social_domain in domain for social_domain in self.social_domains)
            
        except Exception:
            return False
    
    def extract_social_handles(self, links: List[str]) -> Dict[str, str]:
        """Extract social media handles from links"""
        handles = {}
        
        for link in links:
            if not self.is_social_link(link):
                continue
            
            try:
                parsed = urlparse(link)
                domain = parsed.netloc.lower().replace('www.', '')
                path = parsed.path.strip('/')
                
                if 'twitter.com' in domain or 'x.com' in domain:
                    if path and not path.startswith('intent/'):
                        handles['twitter'] = f"@{path.split('/')[0]}"
                
                elif 'linkedin.com' in domain:
                    if '/company/' in path:
                        company_name = path.split('/company/')[-1].split('/')[0]
                        handles['linkedin'] = company_name
                    elif '/in/' in path:
                        handles['linkedin'] = path.split('/in/')[-1].split('/')[0]
                
                elif 'github.com' in domain:
                    if path and len(path.split('/')) >= 1:
                        handles['github'] = path.split('/')[0]
                
                elif 'youtube.com' in domain:
                    if '/channel/' in path:
                        handles['youtube'] = path.split('/channel/')[-1].split('/')[0]
                    elif '/c/' in path:
                        handles['youtube'] = path.split('/c/')[-1].split('/')[0]
                    elif '/user/' in path:
                        handles['youtube'] = path.split('/user/')[-1].split('/')[0]
                
                elif 'facebook.com' in domain:
                    if path and not path.startswith('sharer/'):
                        handles['facebook'] = path.split('/')[0]
                
                elif 'instagram.com' in domain:
                    if path:
                        handles['instagram'] = f"@{path.split('/')[0]}"
                
            except Exception as e:
                logger.debug(f"Error extracting social handle from {link}: {e}")
                continue
        
        return handles
    
    def detect_technologies(self, html_content: str, url: str = '') -> Set[str]:
        """Detect technologies used on a website"""
        technologies = set()
        
        if not html_content:
            return technologies
        
        content_lower = html_content.lower()
        
        # Check each technology
        for tech, patterns in self.technology_indicators.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    technologies.add(tech)
                    break  # Found this tech, move to next
        
        # Additional checks
        
        # CDN detection
        if 'cdn.jsdelivr.net' in content_lower or 'unpkg.com' in content_lower:
            technologies.add('CDN')
        
        # Analytics detection
        if 'google-analytics' in content_lower or 'gtag(' in content_lower:
            technologies.add('Google Analytics')
        
        if 'googletagmanager' in content_lower:
            technologies.add('Google Tag Manager')
        
        # Payment systems
        if 'stripe' in content_lower:
            technologies.add('Stripe')
        
        if 'paypal' in content_lower:
            technologies.add('PayPal')
        
        # Cloud platforms
        if 'amazonaws.com' in content_lower:
            technologies.add('AWS')
        
        if 'googleapis.com' in content_lower:
            technologies.add('Google Cloud')
        
        if 'azure' in content_lower and 'microsoft' in content_lower:
            technologies.add('Microsoft Azure')
        
        # Search engines
        if 'algolia' in content_lower:
            technologies.add('Algolia')
        
        if 'elasticsearch' in content_lower:
            technologies.add('Elasticsearch')
        
        # A/B testing
        if 'optimizely' in content_lower:
            technologies.add('Optimizely')
        
        if 'google optimize' in content_lower:
            technologies.add('Google Optimize')
        
        return technologies
    
    def extract_contact_info(self, html_content: str) -> Dict[str, str]:
        """Extract contact information from HTML content"""
        contact_info = {}
        
        if not html_content:
            return contact_info
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, html_content)
        
        # Filter out common false positives
        valid_emails = []
        for email in emails:
            if not any(skip in email.lower() for skip in ['example.com', 'test.com', 'sample.com']):
                valid_emails.append(email)
        
        if valid_emails:
            contact_info['emails'] = list(set(valid_emails))[:5]  # Limit to 5 unique emails
        
        # Phone pattern (basic US/international)
        phone_patterns = [
            r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',  # US format
            r'\+?([0-9]{1,4})[-.\s]?([0-9]{3,4})[-.\s]?([0-9]{3,4})[-.\s]?([0-9]{3,4})'  # International
        ]
        
        phones = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, html_content)
            for match in matches:
                if isinstance(match, tuple):
                    phone = ''.join(match)
                else:
                    phone = match
                
                # Basic validation
                if len(phone.replace('-', '').replace('.', '').replace(' ', '')) >= 10:
                    phones.append(phone)
        
        if phones:
            contact_info['phones'] = list(set(phones))[:3]  # Limit to 3 unique phones
        
        return contact_info
    
    def extract_structured_data(self, html_content: str) -> Dict[str, List]:
        """Extract structured data (JSON-LD, microdata) from HTML"""
        structured_data = {
            'json_ld': [],
            'microdata': [],
            'open_graph': {},
            'twitter_cards': {}
        }
        
        if not html_content:
            return structured_data
        
        # JSON-LD extraction
        json_ld_pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
        json_ld_matches = re.findall(json_ld_pattern, html_content, re.DOTALL | re.IGNORECASE)
        
        for match in json_ld_matches:
            try:
                import json
                data = json.loads(match.strip())
                structured_data['json_ld'].append(data)
            except json.JSONDecodeError:
                continue
        
        # Open Graph extraction
        og_pattern = r'<meta[^>]*property=["\']og:([^"\']+)["\'][^>]*content=["\']([^"\']*)["\'][^>]*>'
        og_matches = re.findall(og_pattern, html_content, re.IGNORECASE)
        
        for prop, content in og_matches:
            structured_data['open_graph'][prop] = content
        
        # Twitter Cards extraction
        twitter_pattern = r'<meta[^>]*name=["\']twitter:([^"\']+)["\'][^>]*content=["\']([^"\']*)["\'][^>]*>'
        twitter_matches = re.findall(twitter_pattern, html_content, re.IGNORECASE)
        
        for prop, content in twitter_matches:
            structured_data['twitter_cards'][prop] = content
        
        # Basic microdata extraction
        microdata_pattern = r'<[^>]*itemtype=["\']([^"\']+)["\'][^>]*>'
        microdata_matches = re.findall(microdata_pattern, html_content, re.IGNORECASE)
        
        for itemtype in set(microdata_matches):
            structured_data['microdata'].append({'itemtype': itemtype})
        
        return structured_data
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ''
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_keywords(self, text: str, min_length: int = 3, max_keywords: int = 20) -> List[str]:
        """Extract keywords from text content"""
        if not text:
            return []
        
        # Convert to lowercase and split into words
        words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + ',}\b', text.lower())
        
        # Common stop words to exclude
        stop_words = {
            'the', 'and', 'are', 'for', 'with', 'you', 'your', 'our', 'can', 'will',
            'that', 'this', 'from', 'they', 'have', 'been', 'were', 'was', 'but',
            'not', 'all', 'more', 'new', 'use', 'get', 'may', 'how', 'now', 'than',
            'about', 'what', 'when', 'where', 'who', 'why', 'which', 'would', 'could',
            'should', 'website', 'page', 'site', 'company', 'business', 'service',
            'services', 'product', 'products', 'solution', 'solutions'
        }
        
        # Filter out stop words
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count word frequency
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, count in sorted_words[:max_keywords]]
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL for consistent comparison"""
        if not url:
            return ''
        
        # Remove fragment and query parameters
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        # Remove trailing slash
        if normalized.endswith('/') and len(normalized) > 1:
            normalized = normalized[:-1]
        
        return normalized.lower()
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible"""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
    
    def extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain
        except Exception:
            return None
    
    def get_robots_txt_url(self, base_url: str) -> str:
        """Get robots.txt URL for a domain"""
        try:
            parsed = urlparse(base_url)
            return f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        except Exception:
            return ''
    
    def parse_robots_txt(self, robots_content: str) -> Dict[str, List[str]]:
        """Parse robots.txt content"""
        rules = {
            'disallow': [],
            'allow': [],
            'crawl_delay': None,
            'sitemap': []
        }
        
        if not robots_content:
            return rules
        
        current_user_agent = None
        applies_to_all = False
        
        for line in robots_content.split('\n'):
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            if ':' not in line:
                continue
            
            directive, value = line.split(':', 1)
            directive = directive.strip().lower()
            value = value.strip()
            
            if directive == 'user-agent':
                current_user_agent = value.lower()
                applies_to_all = (value == '*')
            
            elif applies_to_all or current_user_agent == '*':
                if directive == 'disallow':
                    rules['disallow'].append(value)
                elif directive == 'allow':
                    rules['allow'].append(value)
                elif directive == 'crawl-delay':
                    try:
                        rules['crawl_delay'] = float(value)
                    except ValueError:
                        pass
                elif directive == 'sitemap':
                    rules['sitemap'].append(value)
        
        return rules
    
    def should_respect_robots(self, url: str, robots_rules: Dict[str, List[str]]) -> bool:
        """Check if URL should be crawled based on robots.txt rules"""
        if not robots_rules:
            return True
        
        try:
            parsed = urlparse(url)
            path = parsed.path
            
            # Check disallow rules
            for disallow_path in robots_rules.get('disallow', []):
                if disallow_path and path.startswith(disallow_path):
                    # Check if explicitly allowed
                    for allow_path in robots_rules.get('allow', []):
                        if allow_path and path.startswith(allow_path):
                            return True
                    return False
            
            return True
            
        except Exception:
            return True  # Default to allowing if parsing fails