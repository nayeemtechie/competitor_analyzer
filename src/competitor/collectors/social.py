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
            logger.warning(f"LinkedIn collection failed: {e}")
        
        return None
    
    async def _find_linkedin_company_handle(self, competitor_name: str) -> Optional[str]:
        """Find LinkedIn company handle"""
        # Known company mappings for common competitors
        known_handles = {
            'algolia': 'algolia',
            'constructor.io': 'constructor-io',
            'bloomreach': 'bloomreach',
            'elasticsearch': 'elastic',
            'coveo': 'coveo',
            'unbxd': 'unbxd',
            'klevu': 'klevu'
        }
        
        company_key = competitor_name.lower().replace(' ', '').replace('.', '')
        if company_key in known_handles:
            return known_handles[company_key]
        
        # Try to search via web scraping
        return await self._search_linkedin_company(competitor_name)
    
    async def _search_linkedin_company(self, competitor_name: str) -> Optional[str]:
        """Search for LinkedIn company page"""
        try:
            search_url = f"https://www.linkedin.com/search/results/companies/"
            params = {'keywords': competitor_name}
            
            async with RateLimitedSession(rate_limit=0.5) as session:
                # LinkedIn has strong anti-scraping measures
                # This is a simplified approach - real implementation would need headers, delays, etc.
                response = await session.get(search_url, params=params)
                
                if response and isinstance(response, str):
                    # Extract company handle from search results (simplified)
                    handle_pattern = r'/company/([^/\s"]+)'
                    matches = re.findall(handle_pattern, response)
                    
                    for match in matches:
                        if competitor_name.lower() in match.lower():
                            return match
                
        except Exception as e:
            logger.debug(f"LinkedIn search failed: {e}")
        
        return None
    
    async def _get_linkedin_company_data(self, company_handle: str) -> Optional[SocialPresence]:
        """Get LinkedIn company data via API"""
        if not self.linkedin_access_token:
            return None
        
        try:
            url = f"https://api.linkedin.com/v2/organizations/{company_handle}"
            headers = {
                'Authorization': f'Bearer {self.linkedin_access_token}',
                'X-Restli-Protocol-Version': '2.0.0'
            }
            
            async with RateLimitedSession(rate_limit=1.0) as session:
                response = await session.get(url, headers=headers)
                
                if response:
                    followers_count = response.get('followersCount', 0)
                    
                    return SocialPresence(
                        platform='linkedin',
                        handle=company_handle,
                        url=f"https://www.linkedin.com/company/{company_handle}",
                        followers=followers_count,
                        activity_level=self._assess_activity_level(followers_count, 'linkedin'),
                        last_post_date=None  # Would need additional API call
                    )
                
        except Exception as e:
            logger.warning(f"LinkedIn API error: {e}")
        
        return None
    
    async def _scrape_linkedin_company_page(self, company_handle: str) -> Optional[SocialPresence]:
        """Scrape LinkedIn company page (fallback method)"""
        try:
            url = f"https://www.linkedin.com/company/{company_handle}"
            
            async with RateLimitedSession(rate_limit=0.5) as session:
                response = await session.get(url)
                
                if response and isinstance(response, str):
                    # Extract follower count (simplified pattern matching)
                    follower_patterns = [
                        r'(\d+(?:,\d{3})*)\s+followers',
                        r'(\d+(?:,\d{3})*)\s+follower',
                        r'"followerCount":(\d+)'
                    ]
                    
                    followers = 0
                    for pattern in follower_patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            followers = int(match.group(1).replace(',', ''))
                            break
                    
                    return SocialPresence(
                        platform='linkedin',
                        handle=company_handle,
                        url=url,
                        followers=followers,
                        activity_level=self._assess_activity_level(followers, 'linkedin')
                    )
                
        except Exception as e:
            logger.warning(f"LinkedIn scraping failed: {e}")
        
        return None
    
    async def _collect_twitter_presence(self, competitor_name: str) -> Optional[SocialPresence]:
        """Collect Twitter/X presence data"""
        try:
            # Find Twitter handle
            twitter_handle = await self._find_twitter_handle(competitor_name)
            
            if twitter_handle:
                if self.twitter_bearer_token:
                    return await self._get_twitter_data_api(twitter_handle)
                else:
                    return await self._scrape_twitter_profile(twitter_handle)
            
        except Exception as e:
            logger.warning(f"Twitter collection failed: {e}")
        
        return None
    
    async def _find_twitter_handle(self, competitor_name: str) -> Optional[str]:
        """Find Twitter handle for company"""
        # Known handles for common competitors
        known_handles = {
            'algolia': 'algolia',
            'constructor.io': 'constructorio',
            'bloomreach': 'bloomreach',
            'elasticsearch': 'elastic',
            'coveo': 'coveo',
            'unbxd': 'unbxd',
            'klevu': 'klevu'
        }
        
        company_key = competitor_name.lower().replace(' ', '').replace('.', '')
        if company_key in known_handles:
            return known_handles[company_key]
        
        # Simple heuristic: try common variations
        variations = [
            competitor_name.lower().replace(' ', ''),
            competitor_name.lower().replace(' ', '_'),
            competitor_name.lower().replace('.', ''),
            competitor_name.split()[0].lower()  # First word only
        ]
        
        # Return first variation as guess (would need verification in real implementation)
        return variations[0] if variations else None
    
    async def _get_twitter_data_api(self, handle: str) -> Optional[SocialPresence]:
        """Get Twitter data via API"""
        try:
            url = f"https://api.twitter.com/2/users/by/username/{handle}"
            params = {'user.fields': 'public_metrics,created_at'}
            headers = {'Authorization': f'Bearer {self.twitter_bearer_token}'}
            
            async with RateLimitedSession(rate_limit=1.0) as session:
                response = await session.get(url, headers=headers, params=params)
                
                if response and 'data' in response:
                    user_data = response['data']
                    metrics = user_data.get('public_metrics', {})
                    
                    return SocialPresence(
                        platform='twitter',
                        handle=handle,
                        url=f"https://twitter.com/{handle}",
                        followers=metrics.get('followers_count', 0),
                        activity_level=self._assess_activity_level(metrics.get('tweet_count', 0), 'twitter'),
                        last_post_date=None  # Would need timeline API
                    )
                
        except Exception as e:
            logger.warning(f"Twitter API error: {e}")
        
        return None
    
    async def _scrape_twitter_profile(self, handle: str) -> Optional[SocialPresence]:
        """Scrape Twitter profile (fallback)"""
        try:
            url = f"https://twitter.com/{handle}"
            
            # Twitter has strong anti-scraping measures
            # This would require sophisticated approaches in real implementation
            logger.debug(f"Twitter scraping for {handle} (placeholder)")
            
            # Return mock data for demonstration
            return SocialPresence(
                platform='twitter',
                handle=handle,
                url=url,
                followers=None,  # Unable to scrape
                activity_level='unknown'
            )
            
        except Exception as e:
            logger.warning(f"Twitter scraping failed: {e}")
        
        return None
    
    async def _collect_github_presence(self, competitor_name: str) -> Optional[SocialPresence]:
        """Collect GitHub organization presence"""
        try:
            # Find GitHub organization
            github_org = await self._find_github_organization(competitor_name)
            
            if github_org:
                return await self._get_github_org_data(github_org)
            
        except Exception as e:
            logger.warning(f"GitHub collection failed: {e}")
        
        return None
    
    async def _find_github_organization(self, competitor_name: str) -> Optional[str]:
        """Find GitHub organization name"""
        # Known organizations
        known_orgs = {
            'algolia': 'algolia',
            'constructor.io': 'Constructor-io',
            'bloomreach': 'bloomreach',
            'elasticsearch': 'elastic',
            'coveo': 'coveo'
        }
        
        company_key = competitor_name.lower()
        return known_orgs.get(company_key)
    
    async def _get_github_org_data(self, org_name: str) -> Optional[SocialPresence]:
        """Get GitHub organization data"""
        try:
            url = f"https://api.github.com/orgs/{org_name}"
            headers = {}
            
            github_token = os.getenv('GITHUB_TOKEN')
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            async with RateLimitedSession(rate_limit=1.0) as session:
                response = await session.get(url, headers=headers)
                
                if response:
                    return SocialPresence(
                        platform='github',
                        handle=org_name,
                        url=response.get('html_url', f"https://github.com/{org_name}"),
                        followers=response.get('followers', 0),
                        activity_level=self._assess_activity_level(response.get('public_repos', 0), 'github')
                    )
                
        except Exception as e:
            logger.warning(f"GitHub API error: {e}")
        
        return None
    
    async def _collect_youtube_presence(self, competitor_name: str) -> Optional[SocialPresence]:
        """Collect YouTube channel presence"""
        try:
            # YouTube channel discovery would require YouTube Data API
            # For now, return placeholder
            logger.debug(f"YouTube collection for {competitor_name} (placeholder)")
            
            # This would involve:
            # 1. Searching for company channels
            # 2. Getting channel statistics
            # 3. Analyzing recent video activity
            
            return None
            
        except Exception as e:
            logger.warning(f"YouTube collection failed: {e}")
        
        return None
    
    def _assess_activity_level(self, metric_value: int, platform: str) -> str:
        """Assess activity level based on platform metrics"""
        if platform == 'linkedin':
            # Based on follower count
            if metric_value > 50000:
                return 'high'
            elif metric_value > 10000:
                return 'medium'
            elif metric_value > 1000:
                return 'low'
            else:
                return 'minimal'
        
        elif platform == 'twitter':
            # Based on tweet count or followers
            if metric_value > 10000:
                return 'high'
            elif metric_value > 1000:
                return 'medium'
            elif metric_value > 100:
                return 'low'
            else:
                return 'minimal'
        
        elif platform == 'github':
            # Based on public repositories
            if metric_value > 50:
                return 'high'
            elif metric_value > 20:
                return 'medium'
            elif metric_value > 5:
                return 'low'
            else:
                return 'minimal'
        
        else:
            return 'unknown'
    
    def _get_empty_result(self) -> List[SocialPresence]:
        """Return empty social presence list"""
        return []

class SocialMediaAnalyzer:
    """Analyzes social media presence for competitive intelligence"""
    
    def analyze_social_presence(self, social_data: List[SocialPresence]) -> Dict[str, Any]:
        """Analyze social media presence across platforms"""
        if not social_data:
            return {'analysis': 'No social media data available'}
        
        analysis = {
            'platform_coverage': self._analyze_platform_coverage(social_data),
            'audience_reach': self._analyze_audience_reach(social_data),
            'engagement_assessment': self._assess_engagement_levels(social_data),
            'brand_presence_strength': self._assess_brand_presence(social_data),
            'social_media_strategy': self._infer_social_strategy(social_data)
        }
        
        return analysis
    
    def _analyze_platform_coverage(self, social_data: List[SocialPresence]) -> Dict[str, Any]:
        """Analyze which platforms the company is active on"""
        platforms = [presence.platform for presence in social_data]
        platform_coverage = {
            'active_platforms': platforms,
            'platform_count': len(platforms),
            'coverage_assessment': self._assess_platform_coverage(platforms)
        }
        
        return platform_coverage
    
    def _assess_platform_coverage(self, platforms: List[str]) -> str:
        """Assess platform coverage strategy"""
        platform_count = len(platforms)
        
        if platform_count >= 4:
            return "Comprehensive - Active across multiple platforms"
        elif platform_count >= 2:
            return "Moderate - Active on key platforms"
        elif platform_count == 1:
            return "Focused - Single platform strategy"
        else:
            return "Minimal - Limited social presence"
    
    def _analyze_audience_reach(self, social_data: List[SocialPresence]) -> Dict[str, Any]:
        """Analyze total audience reach"""
        total_followers = 0
        platform_breakdown = {}
        
        for presence in social_data:
            if presence.followers:
                total_followers += presence.followers
                platform_breakdown[presence.platform] = presence.followers
        
        return {
            'total_followers': total_followers,
            'platform_breakdown': platform_breakdown,
            'reach_assessment': self._assess_reach_level(total_followers)
        }
    
    def _assess_reach_level(self, total_followers: int) -> str:
        """Assess overall reach level"""
        if total_followers > 100000:
            return "High reach - Strong social media following"
        elif total_followers > 25000:
            return "Medium reach - Decent social media presence"
        elif total_followers > 5000:
            return "Low reach - Growing social media presence"
        else:
            return "Minimal reach - Limited social media following"
    
    def _assess_engagement_levels(self, social_data: List[SocialPresence]) -> Dict[str, str]:
        """Assess engagement levels by platform"""
        engagement = {}
        
        for presence in social_data:
            activity_level = presence.activity_level or 'unknown'
            engagement[presence.platform] = f"{activity_level.title()} activity level"
        
        return engagement
    
    def _assess_brand_presence(self, social_data: List[SocialPresence]) -> str:
        """Assess overall brand presence strength"""
        if not social_data:
            return "No social media presence detected"
        
        # Calculate presence score
        score = 0
        
        # Platform diversity bonus
        score += len(social_data) * 2
        
        # Follower count bonus
        total_followers = sum(p.followers for p in social_data if p.followers)
        if total_followers > 50000:
            score += 5
        elif total_followers > 10000:
            score += 3
        elif total_followers > 1000:
            score += 1
        
        # Activity level bonus
        high_activity_count = len([p for p in social_data if p.activity_level == 'high'])
        score += high_activity_count * 2
        
        # Assessment based on score
        if score >= 15:
            return "Strong - Comprehensive social media presence"
        elif score >= 10:
            return "Good - Active social media presence"
        elif score >= 5:
            return "Moderate - Basic social media presence"
        else:
            return "Weak - Limited social media presence"
    
    def _infer_social_strategy(self, social_data: List[SocialPresence]) -> Dict[str, str]:
        """Infer social media strategy"""
        strategy = {}
        
        platforms = [p.platform for p in social_data]
        
        # Platform strategy
        if 'linkedin' in platforms and 'twitter' in platforms:
            strategy['target_audience'] = "B2B focused - Professional audience"
        elif 'linkedin' in platforms:
            strategy['target_audience'] = "Business professionals"
        elif 'twitter' in platforms:
            strategy['target_audience'] = "Tech community and broader audience"
        else:
            strategy['target_audience'] = "General audience"
        
        # Content strategy inference
        github_present = 'github' in platforms
        if github_present:
            strategy['content_focus'] = "Developer-oriented with open source emphasis"
        else:
            strategy['content_focus'] = "Business and marketing focused"
        
        # Engagement strategy
        high_activity_platforms = [p.platform for p in social_data if p.activity_level == 'high']
        if high_activity_platforms:
            strategy['engagement_approach'] = f"Active engagement on {', '.join(high_activity_platforms)}"
        else:
            strategy['engagement_approach'] = "Moderate engagement across platforms"
        
        return strategy