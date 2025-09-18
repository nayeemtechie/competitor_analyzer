# src/competitor/analyzer.py
"""
Main competitor analysis orchestrator - FIXED VERSION
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import CompetitorConfig
from .models import CompetitorProfile, CompetitorIntelligence, ThreatLevel
from .scraper import CompetitorScraper
from .collectors import CollectorManager
from .analysis import AnalysisEngine
from .reports import ReportGenerator

# Fixed import path
try:
    from ..llm.provider import LLMProvider
except ImportError:
    # Fallback for different import structures
    try:
        from src.llm.provider import LLMProvider
    except ImportError:
        from llm.provider import LLMProvider

logger = logging.getLogger(__name__)

class CompetitorAnalyzer:
    """Main orchestrator for competitor analysis"""
    
    def __init__(self, config_path: str = "competitor_config.yaml"):
        self.config = CompetitorConfig(config_path)
        
        # Initialize LLM provider with better error handling
        try:
            self.llm_provider = LLMProvider()
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            raise RuntimeError(f"LLM provider initialization failed: {e}")
        
        self.collector_manager = CollectorManager(self.config, self.llm_provider)
        self.analysis_engine = AnalysisEngine(self.config, self.llm_provider)
        self.report_generator = ReportGenerator(self.config, self.llm_provider)
        
        # Ensure output directory exists
        output_dir = Path(self.config.output_config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
    
    async def analyze_all_competitors(self, competitor_names: Optional[List[str]] = None) -> CompetitorIntelligence:
        """Analyze all configured competitors or specific ones"""
        logger.info("Starting comprehensive competitor analysis")
        
        # Filter competitors if specific names provided
        competitors_to_analyze = self.config.competitors
        if competitor_names:
            competitors_to_analyze = [
                comp for comp in self.config.competitors
                if comp['name'].lower() in [name.lower() for name in competitor_names]
            ]
        
        if not competitors_to_analyze:
            logger.warning("No competitors found to analyze")
            return CompetitorIntelligence()
        
        logger.info(f"Analyzing {len(competitors_to_analyze)} competitors: {[c['name'] for c in competitors_to_analyze]}")
        
        # Collect data for all competitors
        profiles = []
        start_time = datetime.now()
        
        for competitor_config in competitors_to_analyze:
            try:
                profile = await self.analyze_single_competitor(competitor_config)
                if profile:
                    profiles.append(profile)
                    await self._save_individual_profile(profile)
            except Exception as e:
                logger.error(f"Failed to analyze {competitor_config['name']}: {e}")
                continue
        
        # Calculate analysis duration
        analysis_duration = (datetime.now() - start_time).total_seconds() / 60
        
        # Create intelligence container
        intelligence = CompetitorIntelligence(
            profiles=profiles,
            config=self.config.get_analysis_config_object(),
            metadata={
                'total_competitors_analyzed': len(profiles),
                'analysis_duration_minutes': round(analysis_duration, 2),
                'data_sources_used': self._get_active_data_sources(),
                'llm_models_used': self._get_llm_models_used(),
                'analysis_timestamp': datetime.now().isoformat()
            }
        )
        
        # Perform cross-competitor analysis
        try:
            intelligence = await self.analysis_engine.perform_cross_competitor_analysis(intelligence)
        except Exception as e:
            logger.error(f"Cross-competitor analysis failed: {e}")
            # Continue without cross-analysis
        
        logger.info(f"Analysis completed successfully for {len(profiles)} competitors")
        return intelligence
    
    async def analyze_single_competitor(self, competitor_config: Dict[str, Any]) -> Optional[CompetitorProfile]:
        """Analyze a single competitor comprehensively"""
        competitor_name = competitor_config['name']
        competitor_website = competitor_config['website']
        
        logger.info(f"Starting analysis of {competitor_name}")
        
        try:
            # Initialize competitor profile
            profile = CompetitorProfile(
                name=competitor_name,
                website=competitor_website,
                target_markets=competitor_config.get('market_segment', []),
                last_analyzed=datetime.now(),
                competitive_threat=ThreatLevel(competitor_config.get('competitive_threat', 'medium'))
            )
            
            # Collect data from various sources with better error handling
            await self._collect_competitor_data(profile, competitor_config)
            
            # Perform AI-powered analysis
            try:
                profile = await self.analysis_engine.analyze_competitor_profile(profile)
            except Exception as e:
                logger.warning(f"Analysis engine failed for {competitor_name}: {e}")
                # Continue with basic profile
            
            # Update config with analysis date
            self.config.update_competitor_analysis_date(competitor_name, profile.last_analyzed)
            
            logger.info(f"Completed analysis of {competitor_name}")
            return profile
            
        except Exception as e:
            logger.error(f"Error analyzing {competitor_name}: {e}")
            return None
    
    async def _collect_competitor_data(self, profile: CompetitorProfile, config: Dict[str, Any]) -> None:
        """Collect data from all enabled sources with improved error handling"""
        collection_tasks = []
        
        # Website scraping
        if self.config.is_data_source_enabled('company_websites'):
            collection_tasks.append(self._collect_website_data(profile, config))
        
        # Funding data
        if self.config.is_data_source_enabled('funding_data'):
            collection_tasks.append(self._collect_funding_data(profile))
        
        # Job postings
        if self.config.is_data_source_enabled('job_boards'):
            collection_tasks.append(self._collect_job_data(profile))
        
        # News mentions
        if self.config.is_data_source_enabled('news_sources'):
            collection_tasks.append(self._collect_news_data(profile))
        
        # Social media
        if self.config.is_data_source_enabled('social_media'):
            collection_tasks.append(self._collect_social_data(profile))
        
        # GitHub activity
        if self.config.is_data_source_enabled('github_repos'):
            collection_tasks.append(self._collect_github_data(profile))
        
        # Patent data (if enabled)
        if self.config.is_data_source_enabled('patent_databases'):
            collection_tasks.append(self._collect_patent_data(profile))
        
        # Execute all collection tasks concurrently with error handling
        if collection_tasks:
            results = await asyncio.gather(*collection_tasks, return_exceptions=True)
            
            # Log any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    task_name = ['website', 'funding', 'jobs', 'news', 'social', 'github', 'patents'][i]
                    logger.warning(f"Data collection failed for {task_name}: {result}")
    
    async def _collect_website_data(self, profile: CompetitorProfile, config: Dict[str, Any]) -> None:
        """Collect website data using scraper"""
        try:
            target_pages = self.config.get_target_pages()
            
            async with CompetitorScraper(self.config.analysis_config) as scraper:
                website_data = await scraper.scrape_competitor_website(config, target_pages)
                profile.website_data = website_data
                
                # Extract derived information safely
                if website_data and website_data.key_pages:
                    profile.key_features = self._extract_features_from_website(website_data)
                    profile.technology_stack = website_data.technology_stack or []
                    profile.case_studies = self._extract_case_studies_from_website(website_data)
                    
        except Exception as e:
            logger.warning(f"Website data collection failed for {profile.name}: {e}")
    
    async def _collect_funding_data(self, profile: CompetitorProfile) -> None:
        """Collect funding and financial data"""
        try:
            funding_data = await self.collector_manager.collect_funding_data(profile.name)
            profile.funding_info = funding_data
        except Exception as e:
            logger.warning(f"Funding data collection failed for {profile.name}: {e}")
    
    async def _collect_job_data(self, profile: CompetitorProfile) -> None:
        """Collect job posting data"""
        try:
            job_data = await self.collector_manager.collect_job_postings(profile.name)
            profile.job_postings = job_data or []
        except Exception as e:
            logger.warning(f"Job data collection failed for {profile.name}: {e}")
    
    async def _collect_news_data(self, profile: CompetitorProfile) -> None:
        """Collect news and media mentions"""
        try:
            news_config = self.config.get_data_source_config('news_sources')
            days_back = news_config.get('days_back', 90)
            news_data = await self.collector_manager.collect_news_mentions(profile.name, days_back)
            profile.recent_news = news_data or []
        except Exception as e:
            logger.warning(f"News data collection failed for {profile.name}: {e}")
    
    async def _collect_social_data(self, profile: CompetitorProfile) -> None:
        """Collect social media presence data"""
        try:
            social_data = await self.collector_manager.collect_social_presence(profile.name)
            profile.social_presence = social_data or []
        except Exception as e:
            logger.warning(f"Social data collection failed for {profile.name}: {e}")
    
    async def _collect_github_data(self, profile: CompetitorProfile) -> None:
        """Collect GitHub activity data"""
        try:
            github_data = await self.collector_manager.collect_github_activity(profile.name)
            profile.github_activity = github_data
        except Exception as e:
            logger.warning(f"GitHub data collection failed for {profile.name}: {e}")
    
    async def _collect_patent_data(self, profile: CompetitorProfile) -> None:
        """Collect patent data"""
        try:
            patent_data = await self.collector_manager.collect_patent_data(profile.name)
            profile.patent_data = patent_data
        except Exception as e:
            logger.warning(f"Patent data collection failed for {profile.name}: {e}")
    
    def _extract_features_from_website(self, website_data) -> List[str]:
        """Extract key features from website scraping data with error handling"""
        features = []
        
        try:
            # Extract from products page
            if website_data.key_pages and 'products' in website_data.key_pages:
                product_data = website_data.key_pages['products']
                if isinstance(product_data, dict) and 'features' in product_data:
                    for feature in product_data['features']:
                        if isinstance(feature, dict) and 'title' in feature:
                            features.append(feature.get('title', ''))
                        elif isinstance(feature, str):
                            features.append(feature)
            
            # Extract from homepage
            if website_data.key_pages and 'homepage' in website_data.key_pages:
                homepage_data = website_data.key_pages['homepage']
                if isinstance(homepage_data, dict) and 'key_content' in homepage_data:
                    # Extract feature-like content from homepage
                    for content in homepage_data['key_content']:
                        if isinstance(content, str) and ('feature' in content.lower() or 'capability' in content.lower()):
                            features.append(content[:100])  # Limit length
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
        
        return list(set(features))[:20]  # Dedupe and limit
    
    def _extract_case_studies_from_website(self, website_data) -> List[Dict[str, str]]:
        """Extract case studies from website data with error handling"""
        case_studies = []
        
        try:
            # Check customers and case studies pages
            for page_name in ['customers', 'case_studies']:
                if website_data.key_pages and page_name in website_data.key_pages:
                    page_data = website_data.key_pages[page_name]
                    if isinstance(page_data, dict) and 'case_studies' in page_data:
                        case_studies.extend(page_data['case_studies'])
        except Exception as e:
            logger.warning(f"Case study extraction failed: {e}")
        
        return case_studies[:10]  # Limit case studies
    
    async def _save_individual_profile(self, profile: CompetitorProfile) -> None:
        """Save individual competitor profile with better error handling"""
        try:
            output_dir = Path(self.config.output_config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{profile.name.lower().replace(' ', '_').replace('.', '_')}_profile.json"
            filepath = output_dir / filename
            
            import json
            from dataclasses import asdict
            
            # Convert to dict for JSON serialization
            profile_dict = asdict(profile)
            
            # Convert enums to their values for JSON serialization
            def enum_serializer(obj):
                if hasattr(obj, 'value'):
                    return obj.value
                elif hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                return str(obj)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(profile_dict, f, indent=2, default=enum_serializer, ensure_ascii=False)
            
            logger.debug(f"Saved individual profile: {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save profile for {profile.name}: {e}")
    
    async def generate_reports(self, intelligence: CompetitorIntelligence) -> List[str]:
        """Generate all configured report formats"""
        if not intelligence.profiles:
            logger.warning("No competitor profiles to generate reports from")
            return []
        
        logger.info("Generating competitor analysis reports")
        
        generated_files = []
        output_formats = self.config.output_config.get('formats', ['pdf'])
        
        for format_type in output_formats:
            try:
                if format_type == 'pdf':
                    pdf_file = await self.report_generator.generate_pdf_report(intelligence)
                    generated_files.append(pdf_file)
                
                elif format_type == 'docx':
                    docx_file = await self.report_generator.generate_docx_report(intelligence)
                    generated_files.append(docx_file)
                
                elif format_type == 'json':
                    json_file = await self.report_generator.generate_json_report(intelligence)
                    generated_files.append(json_file)
                
                else:
                    logger.warning(f"Unknown report format: {format_type}")
                    
            except Exception as e:
                logger.error(f"Failed to generate {format_type} report: {e}")
        
        logger.info(f"Generated {len(generated_files)} report files")
        return generated_files
    
    def _get_active_data_sources(self) -> List[str]:
        """Get list of active data sources"""
        active_sources = []
        
        try:
            if self.config.is_data_source_enabled('company_websites'):
                active_sources.append('websites')
            if self.config.is_data_source_enabled('funding_data'):
                active_sources.append('funding')
            if self.config.is_data_source_enabled('job_boards'):
                active_sources.append('jobs')
            if self.config.is_data_source_enabled('news_sources'):
                active_sources.append('news')
            if self.config.is_data_source_enabled('social_media'):
                active_sources.append('social')
            if self.config.is_data_source_enabled('github_repos'):
                active_sources.append('github')
            if self.config.is_data_source_enabled('patent_databases'):
                active_sources.append('patents')
        except Exception as e:
            logger.warning(f"Error determining active data sources: {e}")
        
        return active_sources
    
    def _get_llm_models_used(self) -> Dict[str, str]:
        """Get LLM models configured for different tasks"""
        try:
            return {
                'analysis': self.config.get_llm_model('analysis'),
                'summary': self.config.get_llm_model('summary'),
                'comparison': self.config.get_llm_model('comparison')
            }
        except Exception as e:
            logger.warning(f"Error getting LLM models: {e}")
            return {}
    
    async def update_competitor_threat_levels(self, intelligence: CompetitorIntelligence) -> None:
        """Update threat levels based on analysis"""
        for profile in intelligence.profiles:
            try:
                # Calculate threat level based on various factors
                threat_score = self._calculate_threat_score(profile)
                
                if threat_score >= 0.8:
                    profile.threat_level = ThreatLevel.CRITICAL
                elif threat_score >= 0.6:
                    profile.threat_level = ThreatLevel.HIGH
                elif threat_score >= 0.4:
                    profile.threat_level = ThreatLevel.MEDIUM
                else:
                    profile.threat_level = ThreatLevel.LOW
                
                profile.overall_threat_score = threat_score
            except Exception as e:
                logger.warning(f"Threat level calculation failed for {profile.name}: {e}")
    
    def _calculate_threat_score(self, profile: CompetitorProfile) -> float:
        """Calculate numerical threat score (0.0 to 1.0) with better error handling"""
        score = 0.0

        try:
            # Funding momentum (recent large rounds = higher threat)
            if profile.funding_info and profile.funding_info.last_round_amount:
                try:
                    amount = self._normalize_funding_amount(profile.funding_info.last_round_amount)
                    if amount > 100:  # $100M+ round
                        score += 0.3
                    elif amount > 50:  # $50M+ round
                        score += 0.2
                    elif amount > 10:  # $10M+ round
                        score += 0.1
                except Exception:
                    pass
            
            # Market segment overlap
            if profile.target_markets:
                # This would be more sophisticated with actual overlap calculation
                if 'enterprise' in profile.target_markets:
                    score += 0.2
                if 'mid-market' in profile.target_markets:
                    score += 0.1
            
            # Technology advancement indicators
            if profile.technology_stack:
                advanced_tech = ['AI', 'Machine Learning', 'GraphQL', 'Kubernetes', 'Microservices']
                tech_score = len([tech for tech in profile.technology_stack if tech in advanced_tech])
                score += min(tech_score * 0.05, 0.2)
            
            # Recent activity (news, jobs)
            if profile.recent_news and len(profile.recent_news) > 10:
                score += 0.1
            
            if profile.job_postings and len(profile.job_postings) > 20:
                score += 0.1

        except Exception as e:
            logger.warning(f"Error calculating threat score: {e}")

        return min(score, 1.0)  # Cap at 1.0

    @staticmethod
    def _normalize_funding_amount(amount_str: str) -> float:
        """Convert funding strings like '1.2B' or '300M' to float millions"""
        if not amount_str:
            return 0.0

        try:
            # Clean the string
            cleaned = amount_str.replace(',', '').replace('$', '').strip().upper()
            multiplier = 1.0
            
            if cleaned.endswith('B'):
                multiplier = 1000.0
                cleaned = cleaned[:-1]
            elif cleaned.endswith('M'):
                multiplier = 1.0
                cleaned = cleaned[:-1]
            
            # Remove other currency symbols
            cleaned = cleaned.replace('€', '').replace('£', '')
            
            return float(cleaned) * multiplier
        except (ValueError, AttributeError):
            return 0.0
    
    async def generate_executive_summary(self, intelligence: CompetitorIntelligence) -> str:
        """Generate executive summary of competitive landscape"""
        if not intelligence.profiles:
            return "No competitor data available for analysis."
        
        try:
            return await self.analysis_engine.generate_executive_summary(intelligence)
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            return "Executive summary generation failed."
    
    async def get_competitive_threats_by_level(self, intelligence: CompetitorIntelligence) -> Dict[str, List[str]]:
        """Get competitors grouped by threat level"""
        threats = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
        
        for profile in intelligence.profiles:
            threat_level = profile.threat_level.value if profile.threat_level else 'medium'
            threats[threat_level].append(profile.name)
        
        return threats
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            await self.collector_manager.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
        
        logger.info("Competitor analyzer cleanup completed")

# Utility function for CLI usage
async def run_competitor_analysis(
    config_path: str = "competitor_config.yaml",
    competitor_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    formats: Optional[List[str]] = None
) -> List[str]:
    """
    Convenience function to run complete competitor analysis
    
    Args:
        config_path: Path to configuration file
        competitor_names: Specific competitors to analyze (None for all)
        output_dir: Output directory override
        formats: Report formats override
    
    Returns:
        List of generated report file paths
    """
    analyzer = None
    try:
        analyzer = CompetitorAnalyzer(config_path)
        
        # Override configuration if needed
        if output_dir:
            analyzer.config._config_data['output']['output_dir'] = output_dir
        if formats:
            analyzer.config._config_data['output']['formats'] = formats
        
        # Run analysis
        intelligence = await analyzer.analyze_all_competitors(competitor_names)
        
        # Update threat levels
        await analyzer.update_competitor_threat_levels(intelligence)
        
        # Generate reports
        report_files = await analyzer.generate_reports(intelligence)
        
        return report_files
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return []
        
    finally:
        if analyzer:
            await analyzer.cleanup()