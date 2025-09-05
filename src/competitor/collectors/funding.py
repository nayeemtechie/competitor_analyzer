# src/competitor/collectors/funding.py
"""
Funding data collector using Crunchbase and other sources
"""

import os
import json
from typing import Dict, List, Any, Optional
import logging

from .base import CachedCollector, RateLimitedSession
from ..models import FundingInfo

logger = logging.getLogger(__name__)

class FundingCollector(CachedCollector):
    """Collects funding and financial data"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "funding")
        self.crunchbase_api_key = os.getenv('CRUNCHBASE_API_KEY')
        self.pitchbook_api_key = os.getenv('PITCHBOOK_API_KEY')
        
    async def _collect_data(self, competitor_name: str, **kwargs) -> Optional[FundingInfo]:
        """Collect funding data from available sources"""
        funding_info = FundingInfo()
        
        # Try Crunchbase first
        if self.crunchbase_api_key:
            crunchbase_data = await self._collect_from_crunchbase(competitor_name)
            if crunchbase_data:
                funding_info = self._merge_crunchbase_data(funding_info, crunchbase_data)
        
        # Try other sources as fallback
        if not funding_info.total_funding:
            # Try web scraping public sources
            public_data = await self._collect_from_public_sources(competitor_name)
            if public_data:
                funding_info = self._merge_public_data(funding_info, public_data)
        
        return funding_info if funding_info.total_funding else None
    
    async def _collect_from_crunchbase(self, competitor_name: str) -> Optional[Dict[str, Any]]:
        """Collect from Crunchbase API"""
        if not self.crunchbase_api_key:
            logger.debug("Crunchbase API key not available")
            return None
        
        # Search for organization
        search_url = f"https://api.crunchbase.com/api/v4/searches/organizations"
        
        headers = {
            'X-cb-user-key': self.crunchbase_api_key,
            'Content-Type': 'application/json'
        }
        
        search_payload = {
            "field_ids": [
                "name",
                "short_description", 
                "funding_total",
                "last_funding_at",
                "last_funding_type",
                "valuation"
            ],
            "query": [
                {
                    "type": "predicate",
                    "field_id": "name",
                    "operator_id": "contains",
                    "values": [competitor_name]
                }
            ]
        }
        
        try:
            async with RateLimitedSession(rate_limit=2.0) as session:
                # Search for company
                response = await session.post(
                    search_url,
                    headers=headers,
                    json=search_payload
                )
                
                if response and 'entities' in response:
                    entities = response['entities']
                    if entities:
                        # Take the first match
                        company = entities[0]
                        
                        # Get detailed information
                        company_id = company.get('uuid')
                        if company_id:
                            detail_data = await self._get_crunchbase_details(company_id, headers, session)
                            return detail_data
                        
                        return company.get('properties', {})
                
        except Exception as e:
            logger.warning(f"Crunchbase API error for {competitor_name}: {e}")
        
        return None
    
    async def _get_crunchbase_details(self, company_id: str, headers: Dict, session) -> Optional[Dict]:
        """Get detailed company information from Crunchbase"""
        detail_url = f"https://api.crunchbase.com/api/v4/entities/organizations/{company_id}"
        
        params = {
            'field_ids': [
                'funding_total',
                'last_funding_at',
                'last_funding_type', 
                'num_funding_rounds',
                'valuation',
                'investors',
                'funding_rounds'
            ]
        }
        
        try:
            response = await session.get(detail_url, headers=headers, params=params)
            if response and 'properties' in response:
                return response['properties']
        except Exception as e:
            logger.warning(f"Crunchbase details error: {e}")
        
        return None
    
    def _merge_crunchbase_data(self, funding_info: FundingInfo, data: Dict[str, Any]) -> FundingInfo:
        """Merge Crunchbase data into funding info"""
        funding_info.total_funding = self._format_currency(data.get('funding_total'))
        funding_info.last_round_type = data.get('last_funding_type')
        funding_info.last_round_date = data.get('last_funding_at')
        funding_info.valuation = self._format_currency(data.get('valuation'))
        
        # Extract investors
        if 'investors' in data and data['investors']:
            funding_info.investors = [inv.get('name', '') for inv in data['investors'][:10]]
        
        # Determine funding trend (simplified)
        num_rounds = data.get('num_funding_rounds', 0)
        if num_rounds > 5:
            funding_info.funding_trend = "active"
        elif num_rounds > 2:
            funding_info.funding_trend = "moderate"
        else:
            funding_info.funding_trend = "limited"
        
        return funding_info
    
    async def _collect_from_public_sources(self, competitor_name: str) -> Optional[Dict[str, Any]]:
        """Collect funding data from public web sources"""
        public_data = {}
        
        # Try searching for funding announcements
        search_queries = [
            f'"{competitor_name}" funding round series',
            f'"{competitor_name}" raises million',
            f'"{competitor_name}" investment valuation'
        ]
        
        # This would integrate with search engines or news APIs
        # For now, return None (placeholder for actual implementation)
        return None
    
    def _merge_public_data(self, funding_info: FundingInfo, data: Dict[str, Any]) -> FundingInfo:
        """Merge public source data"""
        # Placeholder for public data parsing
        return funding_info
    
    def _format_currency(self, amount: Any) -> Optional[str]:
        """Format currency amount consistently"""
        if not amount:
            return None
        
        try:
            if isinstance(amount, str):
                # Try to extract numeric value
                import re
                numeric_match = re.search(r'[\d,]+\.?\d*', amount)
                if numeric_match:
                    numeric_value = float(numeric_match.group().replace(',', ''))
                else:
                    return amount
            else:
                numeric_value = float(amount)
            
            # Format based on magnitude
            if numeric_value >= 1_000_000_000:
                return f"${numeric_value / 1_000_000_000:.1f}B"
            elif numeric_value >= 1_000_000:
                return f"${numeric_value / 1_000_000:.1f}M"
            elif numeric_value >= 1_000:
                return f"${numeric_value / 1_000:.1f}K"
            else:
                return f"${numeric_value:.0f}"
                
        except (ValueError, TypeError):
            return str(amount) if amount else None
    
    def _get_empty_result(self) -> Optional[FundingInfo]:
        """Return empty funding info"""
        return None