# src/competitor/collectors/base.py
"""
Base classes for data collectors
"""

import asyncio
import aiohttp
import json
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RateLimitedSession:
    """HTTP session with rate limiting and retry logic"""
    
    def __init__(self, rate_limit: float = 1.0, max_retries: int = 3):
        self.rate_limit = rate_limit  # Requests per second
        self.max_retries = max_retries
        self.last_request_time = 0
        self.session = None
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=10)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'CompetitorAnalysis/1.0 (Professional Research Tool)'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _wait_for_rate_limit(self):
        """Wait to respect rate limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def get(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make GET request with rate limiting and retry logic"""
        await self._wait_for_rate_limit()
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url, **kwargs) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '')
                        if 'application/json' in content_type:
                            return await response.json()
                        else:
                            return await response.text()
                    elif response.status == 429:  # Rate limited
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Rate limited on {url}, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
            except Exception as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
        
        return None
    
    async def post(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make POST request with rate limiting and retry logic"""
        await self._wait_for_rate_limit()
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(url, **kwargs) as response:
                    if response.status in [200, 201]:
                        content_type = response.headers.get('content-type', '')
                        if 'application/json' in content_type:
                            return await response.json()
                        else:
                            return await response.text()
                    elif response.status == 429:
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited on {url}, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
            except Exception as e:
                logger.warning(f"POST request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
        
        return None

class BaseCollector(ABC):
    """Base class for all data collectors"""
    
    def __init__(self, config: Dict[str, Any], collector_type: str):
        self.config = config
        self.collector_type = collector_type
        self.enabled = config.get('enabled', True)
        self.rate_limit = config.get('rate_limit', 1.0)
        
    @abstractmethod
    async def _collect_data(self, competitor_name: str, **kwargs) -> Union[Dict, List, None]:
        """Collect data for a specific competitor"""
        pass
    
    @abstractmethod
    def _get_empty_result(self) -> Union[Dict, List, None]:
        """Return empty result when collection fails"""
        pass
    
    async def collect(self, competitor_name: str, **kwargs) -> Union[Dict, List, None]:
        """Main collection method with error handling"""
        if not self.enabled:
            logger.debug(f"{self.collector_type} collector disabled")
            return self._get_empty_result()
        
        try:
            logger.info(f"Collecting {self.collector_type} data for {competitor_name}")
            result = await self._collect_data(competitor_name, **kwargs)
            return result if result is not None else self._get_empty_result()
        except Exception as e:
            logger.error(f"{self.collector_type} collection failed for {competitor_name}: {e}")
            return self._get_empty_result()

class CachedCollector(BaseCollector):
    """Base collector with caching capabilities"""
    
    def __init__(self, config: Dict[str, Any], collector_type: str):
        super().__init__(config, collector_type)
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache_ttl = config.get('cache_ttl_hours', 24) * 3600  # Convert to seconds
        self.cache_dir = Path(config.get('cache_dir', 'cache')) / collector_type
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, competitor_name: str, **kwargs) -> str:
        """Generate cache key for the request"""
        key_data = f"{competitor_name}_{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.json"
    
    async def _load_from_cache(self, cache_key: str) -> Optional[Union[Dict, List]]:
        """Load data from cache if valid"""
        if not self.cache_enabled:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            if cache_path.exists():
                cache_stat = cache_path.stat()
                cache_age = time.time() - cache_stat.st_mtime
                
                if cache_age < self.cache_ttl:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    logger.debug(f"Loaded {self.collector_type} data from cache")
                    return cached_data
                else:
                    # Cache expired, remove it
                    cache_path.unlink()
                    logger.debug(f"Cache expired for {self.collector_type}")
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
        
        return None
    
    async def _save_to_cache(self, cache_key: str, data: Union[Dict, List]) -> None:
        """Save data to cache"""
        if not self.cache_enabled or data is None:
            return
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved {self.collector_type} data to cache")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    async def collect(self, competitor_name: str, **kwargs) -> Union[Dict, List, None]:
        """Collect with caching support"""
        if not self.enabled:
            return self._get_empty_result()
        
        # Try cache first
        cache_key = self._get_cache_key(competitor_name, **kwargs)
        cached_result = await self._load_from_cache(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        # Collect fresh data
        try:
            logger.info(f"Collecting {self.collector_type} data for {competitor_name}")
            result = await self._collect_data(competitor_name, **kwargs)
            
            if result is not None:
                await self._save_to_cache(cache_key, result)
                return result
            else:
                return self._get_empty_result()
                
        except Exception as e:
            logger.error(f"{self.collector_type} collection failed for {competitor_name}: {e}")
            return self._get_empty_result()

class CollectorManager:
    """Manages all data collectors"""
    
    def __init__(self, config, llm_provider):
        self.config = config
        self.llm_provider = llm_provider
        
        # Initialize collectors
        from .website import WebsiteCollector
        from .funding import FundingCollector
        from .jobs import JobCollector
        from .news import NewsCollector
        from .social import SocialCollector
        from .github_activity import GitHubCollector
        from .patents import PatentCollector
        
        self.collectors = {
            'website': WebsiteCollector(config.get_data_source_config('company_websites')),
            'funding': FundingCollector(config.get_data_source_config('funding_data')),
            'jobs': JobCollector(config.get_data_source_config('job_boards')),
            'news': NewsCollector(config.get_data_source_config('news_sources')),
            'social': SocialCollector(config.get_data_source_config('social_media')),
            'github': GitHubCollector(config.get_data_source_config('github_repos')),
            'patents': PatentCollector(config.get_data_source_config('patent_databases'))
        }
    
    async def collect_funding_data(self, competitor_name: str):
        """Collect funding data"""
        return await self.collectors['funding'].collect(competitor_name)
    
    async def collect_job_postings(self, competitor_name: str):
        """Collect job posting data"""
        return await self.collectors['jobs'].collect(competitor_name)
    
    async def collect_news_mentions(self, competitor_name: str, days_back: int = 90):
        """Collect news mentions"""
        return await self.collectors['news'].collect(competitor_name, days_back=days_back)
    
    async def collect_social_presence(self, competitor_name: str):
        """Collect social media presence"""
        return await self.collectors['social'].collect(competitor_name)
    
    async def collect_github_activity(self, competitor_name: str):
        """Collect GitHub activity"""
        return await self.collectors['github'].collect(competitor_name)
    
    async def collect_patent_data(self, competitor_name: str):
        """Collect patent data"""
        return await self.collectors['patents'].collect(competitor_name)
    
    async def cleanup(self):
        """Clean up resources"""
        # Close any persistent connections
        for collector in self.collectors.values():
            if hasattr(collector, 'cleanup'):
                try:
                    await collector.cleanup()
                except Exception as e:
                    logger.warning(f"Collector cleanup warning: {e}")