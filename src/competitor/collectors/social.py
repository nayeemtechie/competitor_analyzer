# src/competitor/collectors/social.py
"""
Social media presence collector for competitive intelligence
"""

import asyncio
import os
from typing import Dict, List, Any, Optional
from urllib.parse import quote
import re
import logging

from .base import CachedCollector, RateLimitedSession
from ..models import SocialPresence

logger = logging.getLogger(__name__)

class SocialCollector(CachedCollector):
    """Collects social media presence and activity data"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "social")
        self.platforms = config.get('platforms', ['linkedin', 'twitter', 'github', 'youtube'])
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.linkedin_access_token = os.getenv('LINKEDIN_ACCESS_TOKEN')
        
    async def _collect_data(self, competitor_name: str, **kwargs) -> List[SocialPresence]:
        """Collect social media presence data"""
        social_presence = []
        
        # Collect from each enabled platform
        collection_tasks = []
        
        if 'linkedin' in self.platforms:
            collection_tasks.append(self._collect_linkedin_presence(competitor_name))
        
        if 'twitter' in self.platforms:
            collection_tasks.append(self._collect_twitter_presence(competitor_name))
        
        if 'github' in self.platforms:
            collection_tasks.append(self._collect_github_presence(competitor_name))
        
        if 'youtube' in self.platforms:
            collection_tasks.append(self._collect_youtube_presence(competitor_name))
        
        # Execute all collection tasks
        results = await asyncio.gather(*collection_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, SocialPresence):
                social_presence.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Social media collection task failed: {result}")
        
        return social_presence
    
    async def _collect_linkedin_presence(self, competitor_name: str) -> Optional[SocialPresence]:
        """Collect LinkedIn company page data"""
        try:
            # Try to find company LinkedIn page
            company_handle = await self._find_linkedin_company_handle(competitor_name)
            
            if company_handle:
                # Get company page data
                if self.linkedin_access_token:
                    return await self._get_linkedin_company_data(company_handle)
                else:
                    # Fallback to web scraping approach (limited data)
                    return await self._scrape_linkedin_company_page(company_handle)
            
        except Exception as e:
            logger.warning(f"LinkedIn  collection  failed: {e}")