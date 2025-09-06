# src/competitor/analysis/innovation.py
"""
Innovation trajectory analysis for competitive intelligence
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class InnovationAnalyzer:
    """Analyzes innovation trends and R&D activities"""
    
    def __init__(self, config, llm_provider):
        self.config = config
        self.llm_provider = llm_provider
    
    async def analyze_innovation_trajectory(self, profile) -> Dict[str, Any]:
        """Analyze innovation trajectory for a single competitor"""
        analysis = {
            'innovation_score': 0.0,
            'innovation_indicators': {},
            'technology_trends': [],
            'rd_investment_signals': {},
            'innovation_velocity': 'unknown',
            'future_innovation_areas': []
        }
        
        # Calculate innovation indicators
        analysis['innovation_indicators'] = self._calculate_innovation_indicators(profile)
        
        # Calculate overall innovation score
        analysis['innovation_score'] = self._calculate_innovation_score(analysis['innovation_indicators'])
        
        # Analyze technology trends
        analysis['technology_trends'] = self._analyze_technology_trends(profile)
        
        # Assess R&D investment signals
        analysis['rd_investment_signals'] = self._analyze_rd_investment(profile)
        
        # Determine innovation velocity
        analysis['innovation_velocity'] = self._assess_innovation_velocity(profile)
        
        # Predict future innovation areas
        analysis['future_innovation_areas'] = await self._predict_innovation_areas(profile)
        
        return analysis
    
    def _calculate_innovation_indicators(self, profile) -> Dict[str, Any]:
        """Calculate various innovation indicators"""
        indicators = {
            'patent_activity': 0,
            'github_innovation': 0,
            'technology_adoption': 0,
            'feature_innovation': 0,
            'rd_hiring': 0,
            'research_publications': 0
        }
        
        # Patent activity indicator
        if profile.patent_data:
            patent_count = profile.patent_data.total_patents
            if patent_count > 50:
                indicators['patent_activity'] = 10
            elif patent_count > 20:
                indicators['patent_activity'] = 8
            elif patent_count > 10:
                indicators['patent_activity'] = 6
            elif patent_count > 5:
                indicators['patent_activity'] = 4
            elif patent_count > 0:
                indicators['patent_activity'] = 2
            
            # Bonus for increasing trend
            if profile.patent_data.filing_trend == 'increasing':
                indicators['patent_activity'] += 2
        
        # GitHub innovation indicator
        if profile.github_activity:
            github_score = 0
            
            # Language diversity
            lang_count = len(profile.github_activity.languages)
            if lang_count > 10:
                github_score += 4
            elif lang_count > 5:
                github_score += 2
            elif lang_count > 2:
                github_score += 1
            
            # Activity level
            activity = profile.github_activity.activity_score or 0
            if activity > 80:
                github_score += 4
            elif activity > 60:
                github_score += 3
            elif activity > 40:
                github_score += 2
            elif activity > 20:
                github_score += 1
            
            # Contributors (community innovation)
            contributors = profile.github_activity.contributors
            if contributors > 100:
                github_score += 2
            elif contributors > 50:
                github_score += 1
            
            indicators['github_innovation'] = min(github_score, 10)
        
        # Technology adoption indicator
        if profile.technology_stack:
            tech_score = 0
            
            # Modern/cutting-edge technologies
            cutting_edge_tech = ['AI', 'Machine Learning', 'GraphQL', 'Kubernetes', 
                                'Microservices', 'Serverless', 'Edge Computing']
            
            for tech in profile.technology_stack:
                if tech in cutting_edge_tech:
                    tech_score += 1
            
            indicators['technology_adoption'] = min(tech_score * 1.5, 10)
        
        # Feature innovation indicator
        if profile.key_features:
            feature_score = 0
            
            # Innovation keywords in features
            innovation_keywords = ['ai', 'machine learning', 'neural', 'semantic', 'natural language',
                                 'computer vision', 'real-time', 'streaming', 'federated', 'vector']
            
            for feature in profile.key_features:
                feature_lower = feature.lower()
                keyword_count = sum(1 for keyword in innovation_keywords if keyword in feature_lower)
                feature_score += keyword_count
            
            indicators['feature_innovation'] = min(feature_score * 0.5, 10)
        
        # R&D hiring indicator
        if profile.job_postings:
            rd_score = 0
            
            # R&D related job postings
            rd_departments = ['engineering', 'research', 'data science', 'ai', 'ml', 'product']
            rd_titles = ['research', 'scientist', 'ai', 'machine learning', 'data scientist', 
                        'principal engineer', 'staff engineer', 'architect']
            
            rd_jobs = []
            for job in profile.job_postings:
                if any(dept in job.department.lower() for dept in rd_departments):
                    rd_jobs.append(job)
                elif any(title in job.title.lower() for title in rd_titles):
                    rd_jobs.append(job)
            
            rd_ratio = len(rd_jobs) / len(profile.job_postings) if profile.job_postings else 0
            
            if rd_ratio > 0.6:
                rd_score = 10
            elif rd_ratio > 0.4:
                rd_score = 8
            elif rd_ratio > 0.3:
                rd_score = 6
            elif rd_ratio > 0.2:
                rd_score = 4
            elif rd_ratio > 0.1:
                rd_score = 2
            
            indicators['rd_hiring'] = rd_score
        
        return indicators
    
    def _calculate_innovation_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate overall innovation score (0.0 to 10.0)"""
        if not indicators:
            return 0.0
        
        # Weighted average of indicators
        weights = {
            'patent_activity': 0.25,
            'github_innovation': 0.20,
            'technology_adoption': 0.20,
            'feature_innovation': 0.15,
            'rd_hiring': 0.15,
            'research_publications': 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for indicator, score in indicators.items():
            if indicator in weights and score > 0:
                weighted_score += score * weights[indicator]
                total_weight += weights[indicator]
        
        if total_weight > 0:
            return round(weighted_score / total_weight * 10, 1)
        else:
            return 0.0
    
    def _analyze_technology_trends(self, profile) -> List[Dict[str, str]]:
        """Analyze technology adoption trends"""
        trends = []
        
        # Analyze technology stack for trends
        if profile.technology_stack:
            # AI/ML trend
            ai_technologies = [tech for tech in profile.technology_stack 
                             if tech in ['AI', 'Machine Learning', 'Neural Networks', 'TensorFlow', 'PyTorch']]
            if ai_technologies:
                trends.append({
                    'trend': 'AI/ML Adoption',
                    'technologies': ', '.join(ai_technologies),
                    'significance': 'High - Core competitive differentiator'
                })
            
            # Cloud-native trend
            cloud_technologies = [tech for tech in profile.technology_stack 
                                if tech in ['Kubernetes', 'Docker', 'Microservices', 'Serverless']]
            if cloud_technologies:
                trends.append({
                    'trend': 'Cloud-Native Architecture',
                    'technologies': ', '.join(cloud_technologies),
                    'significance': 'Medium - Scalability and operational efficiency'
                })
            
            # Modern API trend
            api_technologies = [tech for tech in profile.technology_stack 
                              if tech in ['GraphQL', 'REST API', 'gRPC']]
            if api_technologies:
                trends.append({
                    'trend': 'Modern API Development',
                    'technologies': ', '.join(api_technologies),
                    'significance': 'Medium - Developer experience and integration'
                })
        
        # Analyze features for emerging trends
        if profile.key_features:
            feature_text = ' '.join(profile.key_features).lower()
            
            # Real-time/streaming trend
            if any(keyword in feature_text for keyword in ['real-time', 'streaming', 'live', 'instant']):
                trends.append({
                    'trend': 'Real-time Processing',
                    'technologies': 'Real-time indexing, streaming analytics',
                    'significance': 'High - User experience and competitive advantage'
                })
            
            # Personalization trend
            if any(keyword in feature_text for keyword in ['personalization', 'recommendation', 'custom']):
                trends.append({
                    'trend': 'Advanced Personalization',
                    'technologies': 'ML-driven recommendations, behavioral analysis',
                    'significance': 'High - Customer engagement and conversion'
                })
            
            # Voice/conversational trend
            if any(keyword in feature_text for keyword in ['voice', 'conversational', 'nlp', 'natural language']):
                trends.append({
                    'trend': 'Conversational Interfaces',
                    'technologies': 'NLP, voice recognition, chatbots',
                    'significance': 'Medium - Next-generation user interfaces'
                })
        
        return trends
    
    def _analyze_rd_investment(self, profile) -> Dict[str, Any]:
        """Analyze R&D investment signals"""
        rd_signals = {
            'hiring_intensity': 'low',
            'research_focus_areas': [],
            'innovation_partnerships': [],
            'funding_allocation': 'unknown'
        }
        
        # Analyze hiring for R&D intensity
        if profile.job_postings:
            total_jobs = len(profile.job_postings)
            
            # R&D related hiring
            rd_jobs = [job for job in profile.job_postings 
                      if any(keyword in job.title.lower() 
                      for keyword in ['research', 'scientist', 'engineer', 'architect', 'ai', 'ml'])]
            
            rd_ratio = len(rd_jobs) / total_jobs if total_jobs > 0 else 0
            
            if rd_ratio > 0.5:
                rd_signals['hiring_intensity'] = 'very high'
            elif rd_ratio > 0.3:
                rd_signals['hiring_intensity'] = 'high'
            elif rd_ratio > 0.2:
                rd_signals['hiring_intensity'] = 'medium'
            elif rd_ratio > 0.1:
                rd_signals['hiring_intensity'] = 'low'
            else:
                rd_signals['hiring_intensity'] = 'minimal'
            
            # Identify research focus areas from job titles and requirements
            focus_areas = set()
            for job in rd_jobs:
                title_lower = job.title.lower()
                if 'ai' in title_lower or 'machine learning' in title_lower:
                    focus_areas.add('Artificial Intelligence/ML')
                if 'data' in title_lower:
                    focus_areas.add('Data Science/Analytics')
                if 'cloud' in title_lower or 'infrastructure' in title_lower:
                    focus_areas.add('Cloud Infrastructure')
                if 'security' in title_lower:
                    focus_areas.add('Security/Privacy')
                if 'mobile' in title_lower or 'frontend' in title_lower:
                    focus_areas.add('User Experience/Mobile')
            
            rd_signals['research_focus_areas'] = list(focus_areas)
        
        # Analyze news for partnership signals
        if profile.recent_news:
            partnership_news = [news for news in profile.recent_news 
                              if any(keyword in news.title.lower() 
                              for keyword in ['partnership', 'collaboration', 'acquisition', 'invest'])]
            
            for news in partnership_news:
                if any(keyword in news.title.lower() 
                      for keyword in ['university', 'research', 'lab', 'ai', 'innovation']):
                    rd_signals['innovation_partnerships'].append(news.title)
        
        return rd_signals
    
    def _assess_innovation_velocity(self, profile) -> str:
        """Assess the pace of innovation"""
        velocity_factors = []
        
        # GitHub activity velocity
        if profile.github_activity:
            activity_score = profile.github_activity.activity_score or 0
            if activity_score > 80:
                velocity_factors.append('high_github_activity')
            elif activity_score > 50:
                velocity_factors.append('medium_github_activity')
        
        # Patent filing velocity
        if profile.patent_data:
            if profile.patent_data.filing_trend == 'increasing':
                velocity_factors.append('increasing_patents')
            elif profile.patent_data.total_patents > 20:
                velocity_factors.append('active_patenting')
        
        # Feature development velocity
        if profile.key_features and len(profile.key_features) > 20:
            velocity_factors.append('comprehensive_features')
        
        # Hiring velocity for R&D
        if profile.job_postings:
            rd_jobs = [job for job in profile.job_postings 
                      if 'engineering' in job.department.lower() or 'research' in job.title.lower()]
            if len(rd_jobs) > 10:
                velocity_factors.append('aggressive_rd_hiring')
            elif len(rd_jobs) > 5:
                velocity_factors.append('steady_rd_hiring')
        
        # Recent funding for innovation
        if profile.funding_info and profile.funding_info.last_round_date:
            try:
                last_round = datetime.fromisoformat(profile.funding_info.last_round_date.replace('Z', '+00:00'))
                if (datetime.now() - last_round).days < 365:
                    velocity_factors.append('recent_funding')
            except:
                pass
        
        # Assess overall velocity
        high_velocity_count = len([f for f in velocity_factors 
                                 if f in ['high_github_activity', 'increasing_patents', 
                                         'aggressive_rd_hiring', 'recent_funding']])
        
        if high_velocity_count >= 3:
            return 'very high - Rapid innovation across multiple dimensions'
        elif high_velocity_count >= 2:
            return 'high - Strong innovation momentum'
        elif len(velocity_factors) >= 3:
            return 'medium - Steady innovation pace'
        elif len(velocity_factors) >= 1:
            return 'low - Limited innovation signals'
        else:
            return 'minimal - Little evidence of active innovation'
    
    async def _predict_innovation_areas(self, profile) -> List[str]:
        """Predict future innovation areas based on current trends"""
        future_areas = []
        
        # Analyze hiring trends for future focus
        if profile.job_postings:
            recent_jobs = profile.job_postings[-10:]  # Last 10 job postings
            
            # AI/ML hiring surge
            ai_jobs = [job for job in recent_jobs 
                      if any(keyword in job.title.lower() 
                      for keyword in ['ai', 'machine learning', 'data scientist', 'ml engineer'])]
            if len(ai_jobs) > 2:
                future_areas.append('Advanced AI/ML capabilities - Significant hiring in AI roles')
            
            # Infrastructure/platform hiring
            infra_jobs = [job for job in recent_jobs 
                         if any(keyword in job.title.lower() 
                         for keyword in ['devops', 'platform', 'infrastructure', 'cloud'])]
            if len(infra_jobs) > 1:
                future_areas.append('Platform scalability - Infrastructure team expansion')
            
            # Product/UX hiring
            product_jobs = [job for job in recent_jobs 
                           if any(keyword in job.title.lower() 
                           for keyword in ['product', 'ux', 'ui', 'design'])]
            if len(product_jobs) > 1:
                future_areas.append('User experience innovation - Product team growth')
        
        # Analyze technology stack for future directions
        if profile.technology_stack:
            # Edge computing trend
            if 'Kubernetes' in profile.technology_stack:
                future_areas.append('Edge computing deployment - Kubernetes infrastructure suggests edge readiness')
            
            # API-first strategy
            if 'GraphQL' in profile.technology_stack:
                future_areas.append('API ecosystem expansion - Modern API technologies indicate platform strategy')
        
        # Analyze patent areas for research direction
        if profile.patent_data and profile.patent_data.technology_areas:
            emerging_areas = profile.patent_data.technology_areas
            
            if 'Natural Language Processing' in emerging_areas:
                future_areas.append('Conversational interfaces - NLP patents suggest voice/chat capabilities')
            
            if 'Machine Learning' in emerging_areas:
                future_areas.append('Automated optimization - ML patents indicate self-improving systems')
        
        # Analyze recent news for strategic direction
        if profile.recent_news:
            recent_news_text = ' '.join([news.title + ' ' + (news.summary or '') 
                                       for news in profile.recent_news[-5:]])
            
            if any(keyword in recent_news_text.lower() 
                  for keyword in ['acquisition', 'acquire', 'partnership']):
                future_areas.append('Strategic acquisitions - Recent M&A activity suggests capability expansion')
            
            if any(keyword in recent_news_text.lower() 
                  for keyword in ['international', 'global', 'europe', 'asia']):
                future_areas.append('Global expansion - International focus indicates market scaling')
        
        return future_areas[:5]  # Limit to top 5 predicted areas
    
    async def compare_innovation_across_competitors(self, profiles: List) -> Dict[str, Any]:
        """Compare innovation capabilities across multiple competitors"""
        if len(profiles) < 2:
            return {'error': 'Need at least 2 competitors for innovation comparison'}
        
        comparison = {
            'innovation_leaders': [],
            'innovation_categories': {},
            'technology_adoption_trends': {},
            'rd_investment_comparison': {},
            'innovation_gaps': []
        }
        
        # Analyze each competitor's innovation
        competitor_innovations = []
        for profile in profiles:
            innovation_analysis = await self.analyze_innovation_trajectory(profile)
            competitor_innovations.append({
                'name': profile.name,
                'innovation_score': innovation_analysis['innovation_score'],
                'indicators': innovation_analysis['innovation_indicators'],
                'velocity': innovation_analysis['innovation_velocity'],
                'trends': innovation_analysis['technology_trends']
            })
        
        # Rank by innovation score
        comparison['innovation_leaders'] = sorted(
            competitor_innovations,
            key=lambda x: x['innovation_score'],
            reverse=True
        )
        
        # Analyze innovation categories
        comparison['innovation_categories'] = self._analyze_innovation_categories(competitor_innovations)
        
        # Technology adoption trends
        comparison['technology_adoption_trends'] = self._analyze_technology_adoption_trends(profiles)
        
        # R&D investment comparison
        comparison['rd_investment_comparison'] = self._compare_rd_investments(profiles)
        
        # Identify innovation gaps
        comparison['innovation_gaps'] = self._identify_innovation_gaps(profiles)
        
        return comparison
    
    def _analyze_innovation_categories(self, competitor_innovations: List[Dict]) -> Dict[str, Any]:
        """Analyze innovation across different categories"""
        categories = {
            'patent_leaders': [],
            'github_leaders': [],
            'technology_leaders': [],
            'feature_innovators': [],
            'rd_hiring_leaders': []
        }
        
        for competitor in competitor_innovations:
            indicators = competitor['indicators']
            name = competitor['name']
            
            # Top performers in each category
            if indicators.get('patent_activity', 0) >= 8:
                categories['patent_leaders'].append(name)
            
            if indicators.get('github_innovation', 0) >= 7:
                categories['github_leaders'].append(name)
            
            if indicators.get('technology_adoption', 0) >= 7:
                categories['technology_leaders'].append(name)
            
            if indicators.get('feature_innovation', 0) >= 6:
                categories['feature_innovators'].append(name)
            
            if indicators.get('rd_hiring', 0) >= 6:
                categories['rd_hiring_leaders'].append(name)
        
        return categories
    
    def _analyze_technology_adoption_trends(self, profiles: List) -> Dict[str, Any]:
        """Analyze technology adoption trends across competitors"""
        trends = {
            'ai_ml_adoption': 0,
            'cloud_native_adoption': 0,
            'modern_api_adoption': 0,
            'emerging_technologies': []
        }
        
        total_competitors = len(profiles)
        technology_frequency = {}
        
        for profile in profiles:
            if not profile.technology_stack:
                continue
            
            # Track specific technology adoptions
            if any(tech in profile.technology_stack for tech in ['AI', 'Machine Learning']):
                trends['ai_ml_adoption'] += 1
            
            if any(tech in profile.technology_stack for tech in ['Kubernetes', 'Microservices', 'Docker']):
                trends['cloud_native_adoption'] += 1
            
            if any(tech in profile.technology_stack for tech in ['GraphQL', 'gRPC']):
                trends['modern_api_adoption'] += 1
            
            # Count all technologies
            for tech in profile.technology_stack:
                technology_frequency[tech] = technology_frequency.get(tech, 0) + 1
        
        # Convert to percentages
        trends['ai_ml_adoption'] = round(trends['ai_ml_adoption'] / total_competitors, 2)
        trends['cloud_native_adoption'] = round(trends['cloud_native_adoption'] / total_competitors, 2)
        trends['modern_api_adoption'] = round(trends['modern_api_adoption'] / total_competitors, 2)
        
        # Identify emerging technologies (used by few but growing)
        emerging_threshold = max(1, total_competitors // 4)  # Used by at least 25% or 1 competitor
        trends['emerging_technologies'] = [
            {'technology': tech, 'adoption_count': count}
            for tech, count in sorted(technology_frequency.items(), key=lambda x: x[1], reverse=True)
            if count >= emerging_threshold
        ][:10]
        
        return trends
    
    def _compare_rd_investments(self, profiles: List) -> Dict[str, Any]:
        """Compare R&D investment signals across competitors"""
        comparison = {
            'hiring_intensity_distribution': {},
            'focus_areas_popularity': {},
            'average_rd_ratio': 0.0,
            'top_rd_investors': []
        }
        
        rd_ratios = []
        all_focus_areas = []
        hiring_intensities = []
        
        for profile in profiles:
            rd_signals = self._analyze_rd_investment(profile)
            
            # Collect hiring intensity
            intensity = rd_signals['hiring_intensity']
            hiring_intensities.append(intensity)
            
            # Collect focus areas
            all_focus_areas.extend(rd_signals['research_focus_areas'])
            
            # Calculate R&D ratio
            if profile.job_postings:
                rd_jobs = [job for job in profile.job_postings 
                          if any(keyword in job.title.lower() 
                          for keyword in ['research', 'scientist', 'engineer', 'ai', 'ml'])]
                rd_ratio = len(rd_jobs) / len(profile.job_postings)
                rd_ratios.append((profile.name, rd_ratio))
        
        # Analyze distributions
        intensity_counts = {}
        for intensity in hiring_intensities:
            intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1
        
        comparison['hiring_intensity_distribution'] = intensity_counts
        
        # Focus areas popularity
        focus_area_counts = {}
        for area in all_focus_areas:
            focus_area_counts[area] = focus_area_counts.get(area, 0) + 1
        
        comparison['focus_areas_popularity'] = dict(sorted(focus_area_counts.items(), 
                                                         key=lambda x: x[1], reverse=True))
        
        # Average R&D ratio
        if rd_ratios:
            comparison['average_rd_ratio'] = round(sum(ratio for _, ratio in rd_ratios) / len(rd_ratios), 2)
            comparison['top_rd_investors'] = sorted(rd_ratios, key=lambda x: x[1], reverse=True)[:5]
        
        return comparison
    
    def _identify_innovation_gaps(self, profiles: List) -> List[str]:
        """Identify innovation gaps and opportunities"""
        gaps = []
        
        # Technology gaps
        all_technologies = set()
        for profile in profiles:
            if profile.technology_stack:
                all_technologies.update(profile.technology_stack)
        
        # Check for missing emerging technologies
        emerging_tech_gaps = [
            'Edge Computing', 'Quantum Computing', 'Blockchain', 'AR/VR',
            'IoT', 'Serverless', '5G', 'WebAssembly'
        ]
        
        for tech in emerging_tech_gaps:
            if tech not in all_technologies:
                gaps.append(f"Technology gap: {tech} not adopted by any competitor")
        
        # Patent gaps
        all_patent_areas = set()
        for profile in profiles:
            if profile.patent_data and profile.patent_data.technology_areas:
                all_patent_areas.update(profile.patent_data.technology_areas)
        
        important_patent_areas = [
            'Computer Vision', 'Natural Language Processing', 'Distributed Systems',
            'Security', 'Data Compression', 'User Interface'
        ]
        
        for area in important_patent_areas:
            if area not in all_patent_areas:
                gaps.append(f"Patent gap: {area} not covered in IP portfolios")
        
        # Feature innovation gaps
        all_features = []
        for profile in profiles:
            if profile.key_features:
                all_features.extend([f.lower() for f in profile.key_features])
        
        feature_text = ' '.join(all_features)
        
        if 'voice search' not in feature_text:
            gaps.append("Innovation gap: Voice search capabilities underrepresented")
        
        if 'visual search' not in feature_text:
            gaps.append("Innovation gap: Visual search functionality missing")
        
        if 'federated' not in feature_text:
            gaps.append("Innovation gap: Federated search capabilities limited")
        
        return gaps[:7]  # Return top 7 gaps