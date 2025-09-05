# src/competitor/collectors/patents.py
"""
Patent data collector for R&D competitive intelligence
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re
import logging

from .base import CachedCollector, RateLimitedSession
from ..models import PatentData

logger = logging.getLogger(__name__)

class PatentCollector(CachedCollector):
    """Collects patent data for R&D intelligence"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "patents")
        self.sources = config.get('sources', ['google_patents'])
        
    async def _collect_data(self, competitor_name: str, **kwargs) -> Optional[PatentData]:
        """Collect patent data from available sources"""
        try:
            # For demonstration, using mock patent data
            # Real implementation would use Google Patents API, USPTO API, etc.
            
            patent_data = PatentData()
            
            # Simulate patent search
            patents = await self._search_patents(competitor_name)
            
            if patents:
                patent_data.total_patents = len(patents)
                patent_data.recent_patents = patents[:10]  # Most recent 10
                patent_data.technology_areas = self._extract_technology_areas(patents)
                patent_data.filing_trend = self._analyze_filing_trend(patents)
            
            return patent_data if patent_data.total_patents > 0 else None
            
        except Exception as e:
            logger.warning(f"Patent collection failed for {competitor_name}: {e}")
            return None
    
    async def _search_patents(self, competitor_name: str) -> List[Dict[str, str]]:
        """Search for patents by company name"""
        patents = []
        
        # Mock patent data for demonstration
        # Real implementation would query patent databases
        
        if competitor_name.lower() in ['algolia', 'elasticsearch', 'coveo']:
            # Generate mock patent data for search companies
            mock_patents = [
                {
                    'title': 'Method and System for Real-time Search Relevance Optimization',
                    'publication_number': 'US20240000001A1',
                    'publication_date': '2024-01-01',
                    'inventors': ['John Smith', 'Jane Doe'],
                    'abstract': 'A system for optimizing search relevance using machine learning algorithms...',
                    'technology_area': 'Information Retrieval'
                },
                {
                    'title': 'Distributed Search Index Management System',
                    'publication_number': 'US20240000002A1', 
                    'publication_date': '2024-02-15',
                    'inventors': ['Alice Johnson'],
                    'abstract': 'A distributed system for managing search indices across multiple nodes...',
                    'technology_area': 'Distributed Systems'
                },
                {
                    'title': 'AI-Powered Query Understanding and Intent Recognition',
                    'publication_number': 'US20240000003A1',
                    'publication_date': '2024-03-10',
                    'inventors': ['Bob Wilson', 'Carol Brown'],
                    'abstract': 'Methods for understanding user queries using natural language processing...',
                    'technology_area': 'Natural Language Processing'
                }
            ]
            
            patents = mock_patents
        
        return patents
    
    def _extract_technology_areas(self, patents: List[Dict[str, str]]) -> List[str]:
        """Extract technology areas from patent data"""
        areas = set()
        
        for patent in patents:
            # Extract from explicit technology area
            if 'technology_area' in patent:
                areas.add(patent['technology_area'])
            
            # Extract from title and abstract using keywords
            text = f"{patent.get('title', '')} {patent.get('abstract', '')}".lower()
            
            # Technology area mapping
            area_keywords = {
                'Machine Learning': ['machine learning', 'neural network', 'ai', 'artificial intelligence'],
                'Information Retrieval': ['search', 'indexing', 'relevance', 'ranking', 'query'],
                'Natural Language Processing': ['nlp', 'natural language', 'text processing', 'language model'],
                'Distributed Systems': ['distributed', 'cluster', 'scalability', 'load balancing'],
                'Data Processing': ['data processing', 'analytics', 'big data', 'streaming'],
                'User Interface': ['ui', 'user interface', 'visualization', 'dashboard'],
                'Security': ['security', 'encryption', 'authentication', 'privacy'],
                'Cloud Computing': ['cloud', 'serverless', 'microservices', 'container']
            }
            
            for area, keywords in area_keywords.items():
                if any(keyword in text for keyword in keywords):
                    areas.add(area)
        
        return list(areas)
    
    def _analyze_filing_trend(self, patents: List[Dict[str, str]]) -> str:
        """Analyze patent filing trend over time"""
        if not patents:
            return "insufficient_data"
        
        # Group patents by year
        yearly_counts = {}
        current_year = datetime.now().year
        
        for patent in patents:
            pub_date = patent.get('publication_date', '')
            try:
                year = int(pub_date.split('-')[0])
                yearly_counts[year] = yearly_counts.get(year, 0) + 1
            except (ValueError, IndexError):
                continue
        
        if len(yearly_counts) < 2:
            return "insufficient_data"
        
        # Calculate trend
        years = sorted(yearly_counts.keys())
        recent_years = [y for y in years if y >= current_year - 3]
        earlier_years = [y for y in years if y < current_year - 3]
        
        if not earlier_years or not recent_years:
            return "insufficient_data"
        
        recent_avg = sum(yearly_counts[y] for y in recent_years) / len(recent_years)
        earlier_avg = sum(yearly_counts[y] for y in earlier_years) / len(earlier_years)
        
        if recent_avg > earlier_avg * 1.2:
            return "increasing"
        elif recent_avg < earlier_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _get_empty_result(self) -> Optional[PatentData]:
        """Return empty patent data"""
        return None

class PatentAnalyzer:
    """Analyzes patent data for competitive R&D intelligence"""
    
    def analyze_patent_portfolio(self, patent_data: PatentData) -> Dict[str, Any]:
        """Analyze patent portfolio for competitive insights"""
        if not patent_data or patent_data.total_patents == 0:
            return {'analysis': 'No patent data available'}
        
        analysis = {
            'portfolio_size': self._assess_portfolio_size(patent_data),
            'innovation_focus': self._analyze_innovation_focus(patent_data),
            'r_and_d_activity': self._assess_rd_activity(patent_data),
            'technology_strategy': self._analyze_technology_strategy(patent_data),
            'competitive_positioning': self._assess_competitive_positioning(patent_data)
        }
        
        return analysis
    
    def _assess_portfolio_size(self, patent_data: PatentData) -> str:
        """Assess patent portfolio size"""
        total = patent_data.total_patents
        
        if total > 100:
            return f"Large portfolio ({total} patents) - Significant IP investment"
        elif total > 50:
            return f"Medium portfolio ({total} patents) - Moderate IP protection"
        elif total > 20:
            return f"Small portfolio ({total} patents) - Selective IP strategy"
        else:
            return f"Minimal portfolio ({total} patents) - Limited IP protection"
    
    def _analyze_innovation_focus(self, patent_data: PatentData) -> Dict[str, Any]:
        """Analyze areas of innovation focus"""
        if not patent_data.technology_areas:
            return {'primary_focus': 'Unknown', 'diversity_score': 0}
        
        # Count patents by technology area
        area_counts = {}
        for patent in patent_data.recent_patents:
            area = patent.get('technology_area', 'Other')
            area_counts[area] = area_counts.get(area, 0) + 1
        
        # Sort by frequency
        sorted_areas = sorted(area_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'primary_focus': sorted_areas[0][0] if sorted_areas else 'Unknown',
            'technology_distribution': dict(sorted_areas),
            'diversity_score': len(patent_data.technology_areas),
            'focus_assessment': self._assess_focus_strategy(len(patent_data.technology_areas))
        }
    
    def _assess_focus_strategy(self, num_areas: int) -> str:
        """Assess technology focus strategy"""
        if num_areas > 8:
            return "Broad focus - Diversified innovation across multiple areas"
        elif num_areas > 5:
            return "Moderate focus - Innovation in several key areas"
        elif num_areas > 2:
            return "Focused approach - Concentrated innovation efforts"
        else:
            return "Highly focused - Specialized innovation in few areas"
    
    def _assess_rd_activity(self, patent_data: PatentData) -> Dict[str, str]:
        """Assess R&D activity level"""
        activity = {
            'filing_trend': patent_data.filing_trend,
            'recent_activity': self._assess_recent_activity(patent_data),
            'innovation_velocity': self._calculate_innovation_velocity(patent_data)
        }
        
        return activity
    
    def _assess_recent_activity(self, patent_data: PatentData) -> str:
        """Assess recent patent activity"""
        if not patent_data.recent_patents:
            return "No recent activity"
        
        # Count patents from last 2 years
        current_year = datetime.now().year
        recent_count = 0
        
        for patent in patent_data.recent_patents:
            pub_date = patent.get('publication_date', '')
            try:
                year = int(pub_date.split('-')[0])
                if year >= current_year - 2:
                    recent_count += 1
            except (ValueError, IndexError):
                continue
        
        if recent_count > 10:
            return "High - Very active patent filing"
        elif recent_count > 5:
            return "Medium - Steady patent activity"
        elif recent_count > 1:
            return "Low - Limited recent patents"
        else:
            return "Minimal - Very few recent patents"
    
    def _calculate_innovation_velocity(self, patent_data: PatentData) -> str:
        """Calculate innovation velocity"""
        total = patent_data.total_patents
        trend = patent_data.filing_trend
        
        if total > 50 and trend == "increasing":
            return "High velocity - Accelerating innovation"
        elif total > 20 and trend in ["increasing", "stable"]:
            return "Medium velocity - Consistent innovation"
        elif total > 10:
            return "Low velocity - Slow innovation pace"
        else:
            return "Minimal velocity - Very limited innovation"
    
    def _analyze_technology_strategy(self, patent_data: PatentData) -> Dict[str, str]:
        """Analyze technology strategy from patent patterns"""
        strategy = {}
        
        areas = patent_data.technology_areas
        if not areas:
            return {'overall_strategy': 'Cannot determine from available data'}
        
        # Defensive vs offensive strategy
        if len(areas) > 5:
            strategy['patent_strategy'] = "Broad defensive - Wide IP coverage"
        else:
            strategy['patent_strategy'] = "Focused offensive - Targeted IP protection"
        
        # Technology positioning
        ai_related = any(area in ['Machine Learning', 'Natural Language Processing', 'AI'] 
                        for area in areas)
        infrastructure_related = any(area in ['Distributed Systems', 'Cloud Computing', 'Data Processing'] 
                                   for area in areas)
        
        if ai_related and infrastructure_related:
            strategy['technology_positioning'] = "Full-stack innovation - AI + Infrastructure"
        elif ai_related:
            strategy['technology_positioning'] = "AI-focused innovation"
        elif infrastructure_related:
            strategy['technology_positioning'] = "Infrastructure-focused innovation"
        else:
            strategy['technology_positioning'] = "Domain-specific innovation"
        
        return strategy
    
    def _assess_competitive_positioning(self, patent_data: PatentData) -> str:
        """Assess competitive positioning based on patent portfolio"""
        total = patent_data.total_patents
        trend = patent_data.filing_trend
        diversity = len(patent_data.technology_areas)
        
        # Calculate competitive score
        score = 0
        
        # Portfolio size contribution
        if total > 100:
            score += 4
        elif total > 50:
            score += 3
        elif total > 20:
            score += 2
        elif total > 5:
            score += 1
        
        # Trend contribution
        if trend == "increasing":
            score += 2
        elif trend == "stable":
            score += 1
        
        # Diversity contribution
        if diversity > 6:
            score += 2
        elif diversity > 3:
            score += 1
        
        # Assessment based on total score
        if score >= 7:
            return "Strong - Well-positioned with comprehensive IP portfolio"
        elif score >= 5:
            return "Moderate - Decent IP position with room for growth"
        elif score >= 3:
            return "Weak - Limited IP protection, vulnerable to competition"
        else:
            return "Minimal - Very limited IP assets, high competitive risk"