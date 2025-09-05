# src/competitor/collectors/news.py
"""
News and media mention collector for competitive intelligence
"""

import asyncio
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import quote
import re
import logging

from .base import CachedCollector, RateLimitedSession
from ..models import NewsItem

logger = logging.getLogger(__name__)

class NewsCollector(CachedCollector):
    """Collects news mentions and media coverage"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "news")
        self.sources = config.get('sources', ['techcrunch', 'venturebeat', 'searchengineland'])
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.google_cx = os.getenv('GOOGLE_SEARCH_CX')
        
    async def _collect_data(self, competitor_name: str, days_back: int = 90, **kwargs) -> List[NewsItem]:
        """Collect news mentions from various sources"""
        all_news = []
        
        # Collect from different sources
        collection_tasks = []
        
        # News API
        if self.news_api_key:
            collection_tasks.append(self._collect_from_news_api(competitor_name, days_back))
        
        # Google News via Custom Search
        if self.google_api_key and self.google_cx:
            collection_tasks.append(self._collect_from_google_news(competitor_name, days_back))
        
        # Specific tech publications
        collection_tasks.extend([
            self._collect_from_techcrunch(competitor_name, days_back),
            self._collect_from_venturebeat(competitor_name, days_back),
            self._collect_from_searchengineland(competitor_name, days_back)
        ])
        
        # Execute all collection tasks
        results = await asyncio.gather(*collection_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_news.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"News collection task failed: {result}")
        
        # Deduplicate and filter
        filtered_news = self._filter_and_deduplicate_news(all_news, days_back)
        
        return filtered_news[:50]  # Limit to 50 most relevant articles
    
    async def _collect_from_news_api(self, competitor_name: str, days_back: int) -> List[NewsItem]:
        """Collect from News API"""
        news_items = []
        
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{competitor_name}"',
                'from': from_date,
                'sortBy': 'publishedAt',
                'apiKey': self.news_api_key,
                'language': 'en',
                'pageSize': 20
            }
            
            async with RateLimitedSession(rate_limit=1.0) as session:
                response = await session.get(url, params=params)
                
                if response and 'articles' in response:
                    for article in response['articles']:
                        news_item = NewsItem(
                            title=article.get('title', ''),
                            source=article.get('source', {}).get('name', 'News API'),
                            date=article.get('publishedAt', ''),
                            url=article.get('url', ''),
                            summary=article.get('description', '')
                        )
                        
                        if self._is_relevant_article(news_item, competitor_name):
                            news_items.append(news_item)
                
        except Exception as e:
            logger.warning(f"News API collection failed for {competitor_name}: {e}")
        
        return news_items
    
    async def _collect_from_google_news(self, competitor_name: str, days_back: int) -> List[NewsItem]:
        """Collect from Google News via Custom Search API"""
        news_items = []
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_cx,
                'q': f'"{competitor_name}" news',
                'sort': 'date',
                'dateRestrict': f'd{days_back}',
                'num': 10
            }
            
            async with RateLimitedSession(rate_limit=1.0) as session:
                response = await session.get(url, params=params)
                
                if response and 'items' in response:
                    for item in response['items']:
                        news_item = NewsItem(
                            title=item.get('title', ''),
                            source='Google News',
                            date=item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time', ''),
                            url=item.get('link', ''),
                            summary=item.get('snippet', '')
                        )
                        
                        if self._is_relevant_article(news_item, competitor_name):
                            news_items.append(news_item)
                
        except Exception as e:
            logger.warning(f"Google News collection failed for {competitor_name}: {e}")
        
        return news_items
    
    async def _collect_from_techcrunch(self, competitor_name: str, days_back: int) -> List[NewsItem]:
        """Collect from TechCrunch"""
        news_items = []
        
        try:
            # TechCrunch search
            search_url = f"https://search.techcrunch.com/search"
            params = {
                'query': competitor_name,
                'time': 'recent'  # Recent articles
            }
            
            async with RateLimitedSession(rate_limit=2.0) as session:
                # This would require proper web scraping of TechCrunch search results
                # For demonstration, creating mock data
                logger.debug(f"TechCrunch search for {competitor_name} (placeholder)")
                
                # Mock news items for demonstration
                if competitor_name.lower() in ['algolia', 'constructor', 'bloomreach', 'coveo']:
                    news_items.append(NewsItem(
                        title=f"{competitor_name} Announces New AI-Powered Search Features",
                        source="TechCrunch",
                        date="2024-01-15T10:00:00Z",
                        url=f"https://techcrunch.com/2024/01/15/{competitor_name.lower()}-ai-search",
                        summary=f"{competitor_name} unveils advanced AI capabilities to enhance ecommerce search relevance",
                        sentiment="positive"
                    ))
                
        except Exception as e:
            logger.warning(f"TechCrunch collection failed for {competitor_name}: {e}")
        
        return news_items
    
    async def _collect_from_venturebeat(self, competitor_name: str, days_back: int) -> List[NewsItem]:
        """Collect from VentureBeat"""
        news_items = []
        
        try:
            # VentureBeat has an API, but for demonstration using mock data
            logger.debug(f"VentureBeat search for {competitor_name} (placeholder)")
            
            if len(competitor_name) > 6:  # Simple condition for mock data
                news_items.append(NewsItem(
                    title=f"{competitor_name} Raises Series C Funding",
                    source="VentureBeat",
                    date="2024-01-10T14:00:00Z",
                    url=f"https://venturebeat.com/2024/01/10/{competitor_name.lower()}-funding",
                    summary=f"{competitor_name} secures major funding round to accelerate growth",
                    sentiment="positive"
                ))
                
        except Exception as e:
            logger.warning(f"VentureBeat collection failed for {competitor_name}: {e}")
        
        return news_items
    
    async def _collect_from_searchengineland(self, competitor_name: str, days_back: int) -> List[NewsItem]:
        """Collect from Search Engine Land"""
        news_items = []
        
        try:
            # Search Engine Land is particularly relevant for search companies
            logger.debug(f"Search Engine Land search for {competitor_name} (placeholder)")
            
            # Mock industry-specific news
            if 'search' in competitor_name.lower() or competitor_name.lower() in ['algolia', 'elasticsearch', 'coveo']:
                news_items.append(NewsItem(
                    title=f"Industry Analysis: {competitor_name}'s Search Technology Evolution",
                    source="Search Engine Land",
                    date="2024-01-12T09:00:00Z",
                    url=f"https://searchengineland.com/analysis-{competitor_name.lower()}-search-tech",
                    summary=f"Deep dive into {competitor_name}'s latest search technology improvements",
                    sentiment="neutral"
                ))
                
        except Exception as e:
            logger.warning(f"Search Engine Land collection failed for {competitor_name}: {e}")
        
        return news_items
    
    def _is_relevant_article(self, news_item: NewsItem, competitor_name: str) -> bool:
        """Check if article is relevant to the competitor"""
        if not news_item.title and not news_item.summary:
            return False
        
        # Check for company name in title or summary
        content = f"{news_item.title} {news_item.summary}".lower()
        company_lower = competitor_name.lower()
        
        # Direct company name match
        if company_lower in content:
            return True
        
        # Check for variations (e.g., "Algolia" vs "Algolia Search")
        company_parts = company_lower.split()
        if len(company_parts) > 1:
            # Check if main part of company name appears
            main_part = max(company_parts, key=len)
            if len(main_part) > 3 and main_part in content:
                return True
        
        return False
    
    def _filter_and_deduplicate_news(self, news_items: List[NewsItem], days_back: int) -> List[NewsItem]:
        """Filter news by date and remove duplicates"""
        filtered_items = []
        seen_urls = set()
        seen_titles = set()
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for item in news_items:
            # Skip if URL or title already seen
            if item.url in seen_urls:
                continue
            
            # Check for very similar titles (potential duplicates)
            title_words = set(item.title.lower().split())
            is_duplicate = False
            
            for seen_title in seen_titles:
                seen_words = set(seen_title.lower().split())
                # If >80% of words overlap, consider duplicate
                if title_words and len(title_words.intersection(seen_words)) / len(title_words.union(seen_words)) > 0.8:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            # Check date filter
            if item.date and self._is_within_date_range(item.date, cutoff_date):
                filtered_items.append(item)
                seen_urls.add(item.url)
                seen_titles.add(item.title)
            elif not item.date:
                # Include undated items (they might be recent)
                filtered_items.append(item)
                seen_urls.add(item.url)
                seen_titles.add(item.title)
        
        # Sort by date (most recent first)
        filtered_items.sort(key=lambda x: x.date or '', reverse=True)
        
        return filtered_items
    
    def _is_within_date_range(self, date_str: str, cutoff_date: datetime) -> bool:
        """Check if date string is within range"""
        try:
            # Parse various date formats
            from dateutil import parser
            article_date = parser.parse(date_str)
            return article_date >= cutoff_date
        except Exception:
            return True  # Include if we can't parse the date
    
    def _detect_sentiment(self, news_item: NewsItem) -> str:
        """Simple sentiment detection"""
        content = f"{news_item.title} {news_item.summary}".lower()
        
        positive_words = [
            'announces', 'launches', 'raises', 'funding', 'growth', 'expansion',
            'partnership', 'award', 'recognition', 'breakthrough', 'success',
            'innovative', 'leading', 'improved', 'enhanced'
        ]
        
        negative_words = [
            'lawsuit', 'controversy', 'decline', 'loss', 'layoffs', 'criticism',
            'problems', 'issues', 'challenges', 'struggling', 'competition',
            'threat', 'concern', 'warning'
        ]
        
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _get_empty_result(self) -> List[NewsItem]:
        """Return empty news list"""
        return []

class NewsAnalyzer:
    """Analyzes news patterns for competitive intelligence"""
    
    def analyze_news_trends(self, news_items: List[NewsItem]) -> Dict[str, Any]:
        """Analyze trends in news coverage"""
        if not news_items:
            return {'analysis': 'No news data available'}
        
        analysis = {
            'total_mentions': len(news_items),
            'sentiment_distribution': self._analyze_sentiment_distribution(news_items),
            'source_breakdown': self._analyze_source_distribution(news_items),
            'temporal_trends': self._analyze_temporal_trends(news_items),
            'key_themes': self._extract_key_themes(news_items),
            'media_momentum': self._calculate_media_momentum(news_items)
        }
        
        return analysis
    
    def _analyze_sentiment_distribution(self, news_items: List[NewsItem]) -> Dict[str, int]:
        """Analyze sentiment distribution"""
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for item in news_items:
            sentiment = item.sentiment or 'neutral'
            sentiment_counts[sentiment] += 1
        
        return sentiment_counts
    
    def _analyze_source_distribution(self, news_items: List[NewsItem]) -> Dict[str, int]:
        """Analyze which sources mention the company most"""
        source_counts = {}
        
        for item in news_items:
            source = item.source or 'Unknown'
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return dict(sorted(source_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_temporal_trends(self, news_items: List[NewsItem]) -> Dict[str, Any]:
        """Analyze temporal patterns in news coverage"""
        from collections import defaultdict
        from datetime import datetime
        
        monthly_counts = defaultdict(int)
        
        for item in news_items:
            if item.date:
                try:
                    from dateutil import parser
                    date = parser.parse(item.date)
                    month_key = date.strftime('%Y-%m')
                    monthly_counts[month_key] += 1
                except:
                    continue
        
        return {
            'monthly_distribution': dict(monthly_counts),
            'peak_month': max(monthly_counts.items(), key=lambda x: x[1])[0] if monthly_counts else None,
            'coverage_trend': self._determine_coverage_trend(monthly_counts)
        }
    
    def _extract_key_themes(self, news_items: List[NewsItem]) -> List[str]:
        """Extract key themes from news coverage"""
        all_text = " ".join([f"{item.title} {item.summary}" for item in news_items]).lower()
        
        # Business themes
        business_themes = [
            'funding', 'investment', 'series', 'valuation', 'ipo', 'acquisition',
            'partnership', 'expansion', 'growth', 'revenue', 'customers'
        ]
        
        # Technology themes  
        tech_themes = [
            'ai', 'artificial intelligence', 'machine learning', 'search',
            'analytics', 'personalization', 'algorithm', 'api', 'platform'
        ]
        
        # Market themes
        market_themes = [
            'ecommerce', 'retail', 'competition', 'market share', 'enterprise',
            'saas', 'cloud', 'digital transformation'
        ]
        
        found_themes = []
        for theme_list in [business_themes, tech_themes, market_themes]:
            for theme in theme_list:
                if theme in all_text:
                    found_themes.append(theme)
        
        return found_themes[:10]  # Top 10 themes
    
    def _calculate_media_momentum(self, news_items: List[NewsItem]) -> str:
        """Calculate overall media momentum"""
        recent_items = [item for item in news_items if self._is_recent(item.date, days=30)]
        
        if len(recent_items) > 10:
            return "High - Very active media coverage"
        elif len(recent_items) > 5:
            return "Medium - Steady media presence"
        elif len(recent_items) > 1:
            return "Low - Limited recent coverage"
        else:
            return "Minimal - Little media attention"
    
    def _determine_coverage_trend(self, monthly_counts) -> str:
        """Determine if coverage is increasing, decreasing, or stable"""
        if len(monthly_counts) < 2:
            return "insufficient_data"
        
        months = sorted(monthly_counts.keys())
        recent_months = months[-3:]  # Last 3 months
        earlier_months = months[:-3] if len(months) > 3 else []
        
        if not earlier_months:
            return "insufficient_data"
        
        recent_avg = sum(monthly_counts[m] for m in recent_months) / len(recent_months)
        earlier_avg = sum(monthly_counts[m] for m in earlier_months) / len(earlier_months)
        
        if recent_avg > earlier_avg * 1.2:
            return "increasing"
        elif recent_avg < earlier_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _is_recent(self, date_str: str, days: int = 30) -> bool:
        """Check if date is recent"""
        if not date_str:
            return False
        
        try:
            from dateutil import parser
            date = parser.parse(date_str)
            cutoff = datetime.now() - timedelta(days=days)
            return date >= cutoff
        except:
            return False