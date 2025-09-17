# src/competitor/scraper.py
"""
Advanced web scraping engine for competitor analysis.
Features rate limiting, caching, robots.txt compliance, and robust error handling.
"""

import asyncio
import aiohttp
import logging
import time
import hashlib
import json
from urllib.parse import urljoin, urlparse, robots
from urllib.robotparser import RobotFileParser
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re

# Third-party imports
from bs4 import BeautifulSoup
import aiofiles
from aiohttp import ClientSession, ClientTimeout, ClientError
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ScrapingResult:
    """Result of a web scraping operation."""
    url: str
    content: Optional[str] = None
    title: Optional[str] = None
    meta_description: Optional[str] = None
    meta_keywords: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    status_code: Optional[int] = None
    load_time: Optional[float] = None
    technologies: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    scripts: List[str] = field(default_factory=list)
    stylesheets: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    scraped_at: datetime = field(default_factory=datetime.now)
    page_type: Optional[str] = None
    word_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "content": self.content,
            "title": self.title,
            "meta_description": self.meta_description,
            "meta_keywords": self.meta_keywords,
            "headers": dict(self.headers),
            "status_code": self.status_code,
            "load_time": self.load_time,
            "technologies": self.technologies,
            "links": self.links[:50],  # Limit for serialization
            "images": self.images[:20],
            "scripts": self.scripts[:10],
            "stylesheets": self.stylesheets[:10],
            "success": self.success,
            "error": self.error,
            "scraped_at": self.scraped_at.isoformat(),
            "page_type": self.page_type,
            "word_count": self.word_count
        }


@dataclass
class RateLimiter:
    """Rate limiting for web requests."""
    requests_per_second: float = 1.0
    burst_limit: int = 5
    _request_times: List[float] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    async def clear_cache(self):
        """Clear expired cache entries."""
        await self.cache.clear_expired()
    
    async def health_check(self, url: str) -> Dict[str, Any]:
        """
        Perform a health check on a website.
        
        Args:
            url: URL to check
            
        Returns:
            Health check results
        """
        start_time = time.time()
        
        try:
            if not self.session:
                await self._create_session()
            
            async with self.session.get(url) as response:
                load_time = time.time() - start_time
                
                return {
                    "url": url,
                    "status_code": response.status,
                    "load_time": load_time,
                    "available": response.status < 400,
                    "headers": dict(response.headers),
                    "checked_at": datetime.now().isoformat()
                }
        
        except Exception as e:
            return {
                "url": url,
                "status_code": None,
                "load_time": time.time() - start_time,
                "available": False,
                "error": str(e),
                "checked_at": datetime.now().isoformat()
            }


# Utility functions
async def quick_scrape(url: str, config=None) -> ScrapingResult:
    """Quick scrape of a single URL."""
    async with WebScraper(config) as scraper:
        return await scraper.scrape_url(url)


async def batch_scrape(urls: List[str], config=None) -> List[ScrapingResult]:
    """Batch scrape multiple URLs."""
    async with WebScraper(config) as scraper:
        tasks = [scraper.scrape_url(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs are from the same domain."""
    return extract_domain(url1) == extract_domain(url2)


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common unwanted characters
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    return text


# Example usage and testing
if __name__ == "__main__":
    import sys
    import argparse
    
    async def main():
        parser = argparse.ArgumentParser(description="Web Scraper for Competitor Analysis")
        parser.add_argument("url", help="URL to scrape")
        parser.add_argument("--output", "-o", help="Output file for results")
        parser.add_argument("--pages", "-p", nargs="+", 
                          help="Additional pages to scrape (relative paths)")
        parser.add_argument("--cache-clear", action="store_true",
                          help="Clear expired cache before scraping")
        parser.add_argument("--health-check", action="store_true",
                          help="Perform health check only")
        parser.add_argument("--verbose", "-v", action="store_true",
                          help="Verbose logging")
        
        args = parser.parse_args()
        
        # Set up logging
        level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        async with WebScraper() as scraper:
            if args.cache_clear:
                await scraper.clear_cache()
                print("Cache cleared")
            
            if args.health_check:
                # Health check only
                health = await scraper.health_check(args.url)
                print(f"Health Check Results:")
                print(f"  URL: {health['url']}")
                print(f"  Status: {health['status_code']}")
                print(f"  Available: {health['available']}")
                print(f"  Load Time: {health['load_time']:.2f}s")
                return
            
            # Determine pages to scrape
            target_pages = [{"path": "/", "name": "homepage", "priority": "high"}]
            
            if args.pages:
                for page in args.pages:
                    target_pages.append({
                        "path": page,
                        "name": page.strip('/').replace('/', '_') or "page",
                        "priority": "medium"
                    })
            
            # Scrape website
            print(f"Scraping competitor website: {args.url}")
            results = await scraper.scrape_competitor_website(args.url, target_pages)
            
            # Display results
            print(f"\nScraping Results:")
            print(f"Total pages: {len(results)}")
            successful = [r for r in results if r.success]
            print(f"Successful: {len(successful)}")
            print(f"Failed: {len(results) - len(successful)}")
            
            print(f"\nPage Details:")
            for result in results:
                status = "✓" if result.success else "✗"
                load_time = f"{result.load_time:.2f}s" if result.load_time else "N/A"
                print(f"  {status} {result.page_type}: {result.url} ({load_time})")
                if result.title:
                    print(f"    Title: {result.title[:80]}...")
                if result.error:
                    print(f"    Error: {result.error}")
            
            # Display statistics
            stats = scraper.get_stats()
            print(f"\nStatistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            
            # Save results if requested
            if args.output:
                output_data = {
                    "url": args.url,
                    "scraped_at": datetime.now().isoformat(),
                    "results": [result.to_dict() for result in results],
                    "stats": stats
                }
                
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f"\nResults saved to: {args.output}")
    
    # Run the scraper
    if sys.version_info >= (3, 7):
        asyncio.run(main())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main()) acquire(self):
        """Acquire permission to make a request."""
        async with self._lock:
            now = time.time()
            
            # Remove old request times (older than 1 second)
            self._request_times = [t for t in self._request_times if now - t < 1.0]
            
            # Check burst limit
            if len(self._request_times) >= self.burst_limit:
                sleep_time = 1.0 - (now - self._request_times[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
            
            # Check rate limit
            if self._request_times and self.requests_per_second > 0:
                time_since_last = now - self._request_times[-1]
                min_interval = 1.0 / self.requests_per_second
                
                if time_since_last < min_interval:
                    sleep_time = min_interval - time_since_last
                    logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
            
            self._request_times.append(time.time())


class CacheManager:
    """Manages web scraping cache."""
    
    def __init__(self, cache_dir: Union[str, Path] = "cache/web", ttl_hours: int = 24):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files
            ttl_hours: Time to live for cached content in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
    
    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.json"
    
    async def get(self, url: str) -> Optional[ScrapingResult]:
        """Get cached result for a URL."""
        cache_path = self._get_cache_path(url)
        
        if not cache_path.exists():
            return None
        
        try:
            async with aiofiles.open(cache_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
            
            # Check if cache is still valid
            scraped_at = datetime.fromisoformat(data["scraped_at"])
            if datetime.now() - scraped_at > timedelta(hours=self.ttl_hours):
                logger.debug(f"Cache expired for {url}")
                return None
            
            # Reconstruct ScrapingResult
            result = ScrapingResult(url=url)
            for key, value in data.items():
                if hasattr(result, key) and key != "scraped_at":
                    setattr(result, key, value)
            result.scraped_at = scraped_at
            
            logger.debug(f"Cache hit for {url}")
            return result
            
        except Exception as e:
            logger.warning(f"Error reading cache for {url}: {e}")
            return None
    
    async def set(self, url: str, result: ScrapingResult):
        """Cache a scraping result."""
        cache_path = self._get_cache_path(url)
        
        try:
            async with aiofiles.open(cache_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(result.to_dict(), indent=2))
            
            logger.debug(f"Cached result for {url}")
        except Exception as e:
            logger.warning(f"Error writing cache for {url}: {e}")
    
    async def clear_expired(self):
        """Remove expired cache files."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        data = json.loads(content)
                    
                    scraped_at = datetime.fromisoformat(data["scraped_at"])
                    if datetime.now() - scraped_at > timedelta(hours=self.ttl_hours):
                        cache_file.unlink()
                        logger.debug(f"Removed expired cache: {cache_file}")
                
                except Exception as e:
                    logger.warning(f"Error checking cache file {cache_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")


class RobotsChecker:
    """Checks robots.txt compliance."""
    
    def __init__(self):
        self._robots_cache: Dict[str, RobotFileParser] = {}
        self._cache_timeout = 3600  # 1 hour
        self._last_check: Dict[str, float] = {}
    
    async def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """
        Check if URL can be fetched according to robots.txt.
        
        Args:
            url: URL to check
            user_agent: User agent string
            
        Returns:
            True if URL can be fetched, False otherwise
        """
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            robots_url = urljoin(base_url, "/robots.txt")
            
            # Check cache
            now = time.time()
            if (robots_url in self._robots_cache and 
                robots_url in self._last_check and
                now - self._last_check[robots_url] < self._cache_timeout):
                
                rp = self._robots_cache[robots_url]
                return rp.can_fetch(user_agent, url)
            
            # Fetch robots.txt
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(robots_url, timeout=10) as response:
                        if response.status == 200:
                            robots_content = await response.text()
                            rp = RobotFileParser()
                            rp.set_url(robots_url)
                            rp.read_raw(robots_content)
                            rp.modified()
                        else:
                            # No robots.txt or error - assume allowed
                            rp = RobotFileParser()
                            rp.set_url(robots_url)
                            rp.read_raw("")
                            rp.modified()
                
                except Exception:
                    # Error fetching robots.txt - assume allowed
                    rp = RobotFileParser()
                    rp.set_url(robots_url)
                    rp.read_raw("")
                    rp.modified()
            
            self._robots_cache[robots_url] = rp
            self._last_check[robots_url] = now
            
            return rp.can_fetch(user_agent, url)
            
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}")
            return True  # Assume allowed if error


class TechnologyDetector:
    """Detects technologies used on websites."""
    
    # Technology signatures
    TECH_SIGNATURES = {
        # JavaScript Frameworks
        "react": [r"react", r"_react", r"reactdom"],
        "vue": [r"vue\.js", r"_vue", r"vuejs"],
        "angular": [r"angular", r"ng-", r"angularjs"],
        "jquery": [r"jquery", r"\$\("],
        "bootstrap": [r"bootstrap", r"btn-", r"col-"],
        
        # Analytics
        "google-analytics": [r"google-analytics", r"gtag\(", r"ga\("],
        "gtm": [r"googletagmanager"],
        "mixpanel": [r"mixpanel"],
        "segment": [r"analytics\.js", r"segment"],
        
        # CMS/Platforms
        "wordpress": [r"wp-content", r"wordpress"],
        "shopify": [r"shopify", r"\.myshopify"],
        "magento": [r"magento"],
        "drupal": [r"drupal"],
        
        # CDNs
        "cloudflare": [r"cloudflare", r"cf-ray"],
        "fastly": [r"fastly"],
        "amazon-cloudfront": [r"cloudfront"],
        
        # Other
        "stripe": [r"js\.stripe\.com", r"stripe"],
        "intercom": [r"intercom"],
        "hotjar": [r"hotjar"],
        "zendesk": [r"zendesk"]
    }
    
    def detect_from_html(self, html: str, headers: Dict[str, str]) -> List[str]:
        """
        Detect technologies from HTML content and headers.
        
        Args:
            html: HTML content
            headers: HTTP headers
            
        Returns:
            List of detected technologies
        """
        detected = set()
        html_lower = html.lower()
        
        # Check HTML content
        for tech, patterns in self.TECH_SIGNATURES.items():
            for pattern in patterns:
                if re.search(pattern, html_lower, re.IGNORECASE):
                    detected.add(tech)
                    break
        
        # Check headers
        for header, value in headers.items():
            header_lower = header.lower()
            value_lower = value.lower()
            
            if "server" in header_lower:
                if "nginx" in value_lower:
                    detected.add("nginx")
                elif "apache" in value_lower:
                    detected.add("apache")
                elif "cloudflare" in value_lower:
                    detected.add("cloudflare")
            
            if "x-powered-by" in header_lower:
                if "express" in value_lower:
                    detected.add("express")
                elif "php" in value_lower:
                    detected.add("php")
        
        return list(detected)


class ContentExtractor:
    """Extracts and analyzes content from HTML."""
    
    def __init__(self):
        self.tech_detector = TechnologyDetector()
    
    def extract_content(self, html: str, url: str, headers: Dict[str, str]) -> ScrapingResult:
        """
        Extract structured content from HTML.
        
        Args:
            html: HTML content
            url: Source URL
            headers: HTTP headers
            
        Returns:
            ScrapingResult with extracted content
        """
        result = ScrapingResult(url=url, headers=headers)
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract basic metadata
            result.title = self._extract_title(soup)
            result.meta_description = self._extract_meta_description(soup)
            result.meta_keywords = self._extract_meta_keywords(soup)
            
            # Extract main content
            result.content = self._extract_main_content(soup)
            result.word_count = len(result.content.split()) if result.content else 0
            
            # Extract links and resources
            result.links = self._extract_links(soup, url)
            result.images = self._extract_images(soup, url)
            result.scripts = self._extract_scripts(soup, url)
            result.stylesheets = self._extract_stylesheets(soup, url)
            
            # Detect technologies
            result.technologies = self.tech_detector.detect_from_html(html, headers)
            
            # Determine page type
            result.page_type = self._determine_page_type(url, result.title, result.content)
            
            result.success = True
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            result.error = str(e)
        
        return result
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        # Fallback to h1
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text(strip=True)
        
        return None
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta description."""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            return meta_desc.get('content', '').strip()
        
        # Fallback to Open Graph description
        og_desc = soup.find('meta', attrs={'property': 'og:description'})
        if og_desc:
            return og_desc.get('content', '').strip()
        
        return None
    
    def _extract_meta_keywords(self, soup: BeautifulSoup) -> List[str]:
        """Extract meta keywords."""
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            keywords = meta_keywords.get('content', '')
            return [kw.strip() for kw in keywords.split(',') if kw.strip()]
        
        return []
    
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract main page content."""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Try to find main content area
        main_selectors = [
            'main',
            '[role="main"]',
            '.main-content',
            '#main-content',
            '.content',
            '#content',
            'article',
            '.post-content'
        ]
        
        for selector in main_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                return main_element.get_text(separator=' ', strip=True)
        
        # Fallback to body content
        body = soup.find('body')
        if body:
            return body.get_text(separator=' ', strip=True)
        
        # Last resort - entire document
        return soup.get_text(separator=' ', strip=True)
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from the page."""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith(('http://', 'https://')):
                links.append(href)
            elif href.startswith('/'):
                links.append(urljoin(base_url, href))
        
        return list(set(links))  # Remove duplicates
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract image URLs."""
        images = []
        for img in soup.find_all('img', src=True):
            src = img['src']
            if src.startswith(('http://', 'https://')):
                images.append(src)
            elif src.startswith('/'):
                images.append(urljoin(base_url, src))
        
        return images
    
    def _extract_scripts(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract script URLs."""
        scripts = []
        for script in soup.find_all('script', src=True):
            src = script['src']
            if src.startswith(('http://', 'https://')):
                scripts.append(src)
            elif src.startswith('/'):
                scripts.append(urljoin(base_url, src))
        
        return scripts
    
    def _extract_stylesheets(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract stylesheet URLs."""
        stylesheets = []
        for link in soup.find_all('link', {'rel': 'stylesheet', 'href': True}):
            href = link['href']
            if href.startswith(('http://', 'https://')):
                stylesheets.append(href)
            elif href.startswith('/'):
                stylesheets.append(urljoin(base_url, href))
        
        return stylesheets
    
    def _determine_page_type(self, url: str, title: Optional[str], content: Optional[str]) -> str:
        """Determine the type of page based on URL and content."""
        url_lower = url.lower()
        title_lower = (title or "").lower()
        content_lower = (content or "").lower()
        
        # Check URL patterns
        if any(pattern in url_lower for pattern in ['/pricing', '/price', '/plans']):
            return 'pricing'
        elif any(pattern in url_lower for pattern in ['/about', '/company', '/team']):
            return 'about'
        elif any(pattern in url_lower for pattern in ['/product', '/features', '/solution']):
            return 'product'
        elif any(pattern in url_lower for pattern in ['/contact', '/support', '/help']):
            return 'contact'
        elif any(pattern in url_lower for pattern in ['/blog', '/news', '/article']):
            return 'blog'
        elif url_lower.count('/') <= 3:  # Likely homepage
            return 'homepage'
        
        # Check content patterns
        if any(word in title_lower for word in ['pricing', 'plans', 'cost']):
            return 'pricing'
        elif any(word in content_lower[:500] for word in ['about us', 'our team', 'founded']):
            return 'about'
        
        return 'general'


class WebScraper:
    """Advanced web scraper with rate limiting and caching."""
    
    def __init__(self, config=None):
        """
        Initialize web scraper.
        
        Args:
            config: Configuration object with scraping settings
        """
        self.config = config
        
        # Set up configuration with defaults
        if config and hasattr(config, 'scraping'):
            scraping_config = config.scraping
            self.rate_limiter = RateLimiter(
                requests_per_second=scraping_config.rate_limit,
                burst_limit=scraping_config.concurrent_requests
            )
            self.user_agent = scraping_config.user_agent
            self.timeout = scraping_config.timeout
            self.max_pages = scraping_config.max_pages_per_site
            self.delay = scraping_config.delay_between_requests
            self.respect_robots = scraping_config.respect_robots_txt
        else:
            # Default configuration
            self.rate_limiter = RateLimiter(requests_per_second=1.0, burst_limit=3)
            self.user_agent = "CompetitorAnalysis Bot 1.0"
            self.timeout = 30
            self.max_pages = 50
            self.delay = 2.0
            self.respect_robots = True
        
        # Initialize components
        self.cache = CacheManager()
        self.robots_checker = RobotsChecker()
        self.content_extractor = ContentExtractor()
        self.session: Optional[ClientSession] = None
        
        # Statistics
        self.stats = {
            "requests_made": 0,
            "cache_hits": 0,
            "errors": 0,
            "robots_blocked": 0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_session()
    
    async def _create_session(self):
        """Create HTTP session."""
        timeout = ClientTimeout(total=self.timeout)
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
        }
        
        self.session = ClientSession(
            timeout=timeout,
            headers=headers,
            connector=aiohttp.TCPConnector(
                limit=10,
                limit_per_host=3,
                keepalive_timeout=30
            )
        )
    
    async def _close_session(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
    
    async def scrape_url(self, url: str, force_refresh: bool = False) -> ScrapingResult:
        """
        Scrape a single URL.
        
        Args:
            url: URL to scrape
            force_refresh: Skip cache and force fresh scrape
            
        Returns:
            ScrapingResult with extracted content
        """
        # Check cache first (unless forced refresh)
        if not force_refresh:
            cached_result = await self.cache.get(url)
            if cached_result:
                self.stats["cache_hits"] += 1
                return cached_result
        
        # Check robots.txt
        if self.respect_robots:
            if not await self.robots_checker.can_fetch(url, self.user_agent):
                self.stats["robots_blocked"] += 1
                logger.warning(f"Robots.txt disallows scraping: {url}")
                return ScrapingResult(
                    url=url,
                    success=False,
                    error="Blocked by robots.txt"
                )
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        if not self.session:
            await self._create_session()
        
        start_time = time.time()
        
        try:
            self.stats["requests_made"] += 1
            
            async with self.session.get(url) as response:
                load_time = time.time() - start_time
                
                # Check response status
                if response.status >= 400:
                    self.stats["errors"] += 1
                    return ScrapingResult(
                        url=url,
                        status_code=response.status,
                        load_time=load_time,
                        success=False,
                        error=f"HTTP {response.status}"
                    )
                
                # Get content
                html = await response.text()
                headers = dict(response.headers)
                
                # Extract content
                result = self.content_extractor.extract_content(html, url, headers)
                result.status_code = response.status
                result.load_time = load_time
                
                # Cache result
                await self.cache.set(url, result)
                
                logger.info(f"Successfully scraped: {url} ({load_time:.2f}s)")
                return result
        
        except asyncio.TimeoutError:
            self.stats["errors"] += 1
            logger.error(f"Timeout scraping {url}")
            return ScrapingResult(
                url=url,
                success=False,
                error="Request timeout",
                load_time=time.time() - start_time
            )
        
        except ClientError as e:
            self.stats["errors"] += 1
            logger.error(f"Client error scraping {url}: {e}")
            return ScrapingResult(
                url=url,
                success=False,
                error=f"Client error: {str(e)}",
                load_time=time.time() - start_time
            )
        
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Unexpected error scraping {url}: {e}")
            return ScrapingResult(
                url=url,
                success=False,
                error=f"Unexpected error: {str(e)}",
                load_time=time.time() - start_time
            )
    
    async def scrape_competitor_website(self, 
                                      base_url: str, 
                                      target_pages: Optional[List[Dict[str, str]]] = None) -> List[ScrapingResult]:
        """
        Scrape multiple pages from a competitor's website.
        
        Args:
            base_url: Base URL of the website
            target_pages: List of pages to scrape with metadata
            
        Returns:
            List of ScrapingResult objects
        """
        if not target_pages:
            # Default pages to scrape
            target_pages = [
                {"path": "/", "name": "homepage", "priority": "high"},
                {"path": "/pricing", "name": "pricing", "priority": "high"},
                {"path": "/products", "name": "products", "priority": "medium"},
                {"path": "/features", "name": "features", "priority": "medium"},
                {"path": "/about", "name": "about", "priority": "low"},
                {"path": "/company", "name": "company", "priority": "low"}
            ]
        
        results = []
        scraped_count = 0
        
        for page_info in target_pages:
            if scraped_count >= self.max_pages:
                logger.warning(f"Reached max pages limit ({self.max_pages}) for {base_url}")
                break
            
            # Construct full URL
            path = page_info.get("path", "/")
            if path.startswith("http"):
                url = path
            else:
                url = urljoin(base_url.rstrip('/') + '/', path.lstrip('/'))
            
            logger.info(f"Scraping {page_info.get('name', 'page')}: {url}")
            
            # Scrape the page
            result = await self.scrape_url(url)
            result.page_type = page_info.get("name", "unknown")
            results.append(result)
            
            scraped_count += 1
            
            # Add delay between requests
            if self.delay > 0 and scraped_count < len(target_pages):
                await asyncio.sleep(self.delay)
        
        successful_scrapes = sum(1 for r in results if r.success)
        logger.info(f"Completed scraping {base_url}: {successful_scrapes}/{len(results)} successful")
        
        return results
    
    async def scrape_multiple_competitors(self, 
                                        competitor_urls: List[str],
                                        target_pages: Optional[List[Dict[str, str]]] = None) -> Dict[str, List[ScrapingResult]]:
        """
        Scrape multiple competitor websites.
        
        Args:
            competitor_urls: List of competitor base URLs
            target_pages: Pages to scrape from each site
            
        Returns:
            Dictionary mapping URLs to scraping results
        """
        results = {}
        
        for url in competitor_urls:
            logger.info(f"Starting scrape of competitor: {url}")
            try:
                competitor_results = await self.scrape_competitor_website(url, target_pages)
                results[url] = competitor_results
            except Exception as e:
                logger.error(f"Failed to scrape competitor {url}: {e}")
                results[url] = [ScrapingResult(url=url, success=False, error=str(e))]
            
            # Delay between competitors
            if self.delay > 0:
                await asyncio.sleep(self.delay * 2)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scraping statistics."""
        return {
            **self.stats,
            "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["requests_made"]) * 100,
            "error_rate": self.stats["errors"] / max(1, self.stats["requests_made"]) * 100
        }
    
    async def