# src/competitor/collectors/github_activity.py
"""
GitHub activity collector for technical competitive intelligence
"""

import os
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from .base import CachedCollector, RateLimitedSession
from ..models import GitHubActivity

logger = logging.getLogger(__name__)

class GitHubCollector(CachedCollector):
    """Collects GitHub activity and repository data"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "github")
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.analyze_repos = config.get('analyze_public_repos', True)
        self.track_contributions = config.get('track_contributions', True)
        self.language_analysis = config.get('language_analysis', True)
        
    async def _collect_data(self, competitor_name: str, **kwargs) -> Optional[GitHubActivity]:
        """Collect GitHub activity data"""
        try:
            # Find GitHub organization
            org_name = await self._find_github_organization(competitor_name)
            
            if not org_name:
                logger.debug(f"No GitHub organization found for {competitor_name}")
                return None
            
            # Collect organization data
            org_data = await self._get_organization_data(org_name)
            
            if not org_data:
                return None
            
            # Collect repository data if enabled
            repos_data = []
            if self.analyze_repos:
                repos_data = await self._get_repositories_data(org_name)
            
            # Analyze activity patterns
            activity_metrics = await self._analyze_activity_patterns(org_name, repos_data)
            
            # Create GitHub activity object
            github_activity = GitHubActivity(
                organization=org_name,
                public_repos=org_data.get('public_repos', 0),
                contributors=await self._count_unique_contributors(repos_data),
                total_commits=activity_metrics.get('total_commits', 0),
                languages=activity_metrics.get('top_languages', []),
                activity_score=activity_metrics.get('activity_score', 0.0),
                last_updated=datetime.now().isoformat()
            )
            
            return github_activity
            
        except Exception as e:
            logger.warning(f"GitHub collection failed for {competitor_name}: {e}")
            return None
    
    async def _find_github_organization(self, competitor_name: str) -> Optional[str]:
        """Find GitHub organization name"""
        # Known mappings for demonstration
        known_orgs = {
            'algolia': 'algolia',
            'constructor.io': 'Constructor-io',
            'bloomreach': 'bloomreach', 
            'elasticsearch': 'elastic',
            'coveo': 'coveo',
            'unbxd': 'unbxd',
            'klevu': 'klevu'
        }
        
        if competitor_name.lower() in known_orgs:
            return known_orgs[competitor_name.lower()]
        
        # Try to search for organization
        search_result = await self._search_github_organizations(competitor_name)
        return search_result
    
    async def _search_github_organizations(self, competitor_name: str) -> Optional[str]:
        """Search for GitHub organizations"""
        try:
            url = "https://api.github.com/search/users"
            params = {
                'q': f"{competitor_name} type:org",
                'per_page': 5
            }
            
            headers = self._get_headers()
            
            async with RateLimitedSession(rate_limit=1.0) as session:
                response = await session.get(url, headers=headers, params=params)
                
                if response and 'items' in response:
                    for org in response['items']:
                        # Simple name matching
                        if competitor_name.lower() in org.get('login', '').lower():
                            return org['login']
                
        except Exception as e:
            logger.warning(f"GitHub organization search failed: {e}")
        
        return None
    
    async def _get_organization_data(self, org_name: str) -> Optional[Dict[str, Any]]:
        """Get GitHub organization data"""
        try:
            url = f"https://api.github.com/orgs/{org_name}"
            headers = self._get_headers()
            
            async with RateLimitedSession(rate_limit=1.0) as session:
                response = await session.get(url, headers=headers)
                return response
                
        except Exception as e:
            logger.warning(f"GitHub org data collection failed: {e}")
            return None
    
    async def _get_repositories_data(self, org_name: str, max_repos: int = 100) -> List[Dict[str, Any]]:
        """Get repository data for the organization"""
        repos = []
        
        try:
            url = f"https://api.github.com/orgs/{org_name}/repos"
            params = {
                'type': 'public',
                'sort': 'updated',
                'per_page': min(max_repos, 100)
            }
            
            headers = self._get_headers()
            
            async with RateLimitedSession(rate_limit=1.0) as session:
                response = await session.get(url, headers=headers, params=params)
                
                if response:
                    repos = response if isinstance(response, list) else []
                    
                    # Get additional data for top repositories
                    top_repos = sorted(repos, key=lambda r: r.get('stargazers_count', 0), reverse=True)[:20]
                    
                    # Enrich with additional data
                    for repo in top_repos[:10]:  # Limit to avoid rate limits
                        repo_name = repo['name']
                        
                        # Get languages
                        languages = await self._get_repository_languages(org_name, repo_name)
                        repo['languages_data'] = languages
                        
                        # Get recent activity
                        activity = await self._get_repository_activity(org_name, repo_name)
                        repo['activity_data'] = activity
                        
                        await asyncio.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            logger.warning(f"Repository data collection failed: {e}")
        
        return repos
    
    async def _get_repository_languages(self, org_name: str, repo_name: str) -> Dict[str, int]:
        """Get programming languages used in a repository"""
        try:
            url = f"https://api.github.com/repos/{org_name}/{repo_name}/languages"
            headers = self._get_headers()
            
            async with RateLimitedSession(rate_limit=1.0) as session:
                response = await session.get(url, headers=headers)
                return response if response else {}
                
        except Exception as e:
            logger.debug(f"Languages collection failed for {repo_name}: {e}")
            return {}
    
    async def _get_repository_activity(self, org_name: str, repo_name: str) -> Dict[str, Any]:
        """Get recent activity data for a repository"""
        try:
            # Get recent commits
            commits_url = f"https://api.github.com/repos/{org_name}/{repo_name}/commits"
            params = {
                'since': (datetime.now() - timedelta(days=90)).isoformat(),
                'per_page': 100
            }
            
            headers = self._get_headers()
            
            async with RateLimitedSession(rate_limit=1.0) as session:
                commits_response = await session.get(commits_url, headers=headers, params=params)
                
                activity_data = {
                    'recent_commits': len(commits_response) if commits_response else 0,
                    'last_commit_date': None,
                    'active_contributors': set()
                }
                
                if commits_response and isinstance(commits_response, list):
                    # Extract last commit date
                    if commits_response:
                        activity_data['last_commit_date'] = commits_response[0].get('commit', {}).get('author', {}).get('date')
                    
                    # Count unique contributors
                    for commit in commits_response:
                        author = commit.get('author', {})
                        if author and author.get('login'):
                            activity_data['active_contributors'].add(author['login'])
                    
                    activity_data['active_contributors'] = len(activity_data['active_contributors'])
                
                return activity_data
                
        except Exception as e:
            logger.debug(f"Activity collection failed for {repo_name}: {e}")
            return {'recent_commits': 0, 'active_contributors': 0}
    
    async def _count_unique_contributors(self, repos_data: List[Dict[str, Any]]) -> int:
        """Count unique contributors across all repositories"""
        contributors = set()
        
        for repo in repos_data:
            activity = repo.get('activity_data', {})
            repo_contributors = activity.get('active_contributors', 0)
            
            # This is a simplified approach - in reality, you'd need to collect actual contributor data
            if isinstance(repo_contributors, int):
                contributors.add(f"contributor_{repo['name']}_{i}" for i in range(repo_contributors))
        
        return len(contributors)
    
    async def _analyze_activity_patterns(self, org_name: str, repos_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze activity patterns across repositories"""
        analysis = {
            'total_commits': 0,
            'top_languages': [],
            'activity_score': 0.0,
            'most_active_repo': None,
            'language_diversity': 0
        }
        
        if not repos_data:
            return analysis
        
        # Aggregate languages across all repos
        language_totals = {}
        total_commits = 0
        most_active_repo = None
        max_commits = 0
        
        for repo in repos_data:
            # Count commits
            activity = repo.get('activity_data', {})
            repo_commits = activity.get('recent_commits', 0)
            total_commits += repo_commits
            
            # Track most active repository
            if repo_commits > max_commits:
                max_commits = repo_commits
                most_active_repo = repo.get('name')
            
            # Aggregate languages
            languages = repo.get('languages_data', {})
            for lang, bytes_count in languages.items():
                language_totals[lang] = language_totals.get(lang, 0) + bytes_count
        
        # Calculate top languages
        top_languages = sorted(language_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        analysis['top_languages'] = [lang for lang, _ in top_languages]
        
        # Calculate activity score
        analysis['total_commits'] = total_commits
        analysis['most_active_repo'] = most_active_repo
        analysis['language_diversity'] = len(language_totals)
        
        # Activity score based on commits, repos, and languages
        repo_count = len(repos_data)
        activity_score = min((total_commits * 0.1) + (repo_count * 2) + (len(language_totals) * 5), 100.0)
        analysis['activity_score'] = round(activity_score, 1)
        
        return analysis
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for GitHub API requests"""
        headers = {
            'Accept': 'application/vnd.github+json',
            'User-Agent': 'CompetitorAnalysis/1.0'
        }
        
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        
        return headers
    
    def _get_empty_result(self) -> Optional[GitHubActivity]:
        """Return empty GitHub activity"""
        return None

class GitHubAnalyzer:
    """Analyzes GitHub activity for competitive intelligence"""
    
    def analyze_technical_profile(self, github_activity: GitHubActivity) -> Dict[str, Any]:
        """Analyze technical profile from GitHub activity"""
        if not github_activity:
            return {'analysis': 'No GitHub data available'}
        
        analysis = {
            'technical_maturity': self._assess_technical_maturity(github_activity),
            'open_source_strategy': self._analyze_open_source_strategy(github_activity),
            'technology_focus': self._analyze_technology_focus(github_activity),
            'development_velocity': self._assess_development_velocity(github_activity),
            'community_engagement': self._assess_community_engagement(github_activity),
            'innovation_indicators': self._identify_innovation_indicators(github_activity)
        }
        
        return analysis
    
    def _assess_technical_maturity(self, github_activity: GitHubActivity) -> str:
        """Assess technical maturity based on GitHub metrics"""
    def _analyze_open_source_strategy(self, github_activity: GitHubActivity) -> Dict[str, str]:
        """Analyze open source strategy"""
        strategy = {}
        
        repo_count = github_activity.public_repos
        activity_score = github_activity.activity_score or 0
        
        # Repository strategy
        if repo_count > 50:
            strategy['repository_strategy'] = "Extensive - Many public repositories indicating strong open source commitment"
        elif repo_count > 20:
            strategy['repository_strategy'] = "Active - Moderate number of public repositories"
        elif repo_count > 5:
            strategy['repository_strategy'] = "Selective - Few but potentially high-quality repositories"
        else:
            strategy['repository_strategy'] = "Minimal - Limited open source presence"
        
        # Activity strategy
        if activity_score > 80:
            strategy['activity_strategy'] = "Highly active - Regular contributions and updates"
        elif activity_score > 50:
            strategy['activity_strategy'] = "Moderately active - Steady development activity"
        elif activity_score > 20:
            strategy['activity_strategy'] = "Low activity - Infrequent updates"
        else:
            strategy['activity_strategy'] = "Inactive - Minimal recent activity"
        
        return strategy
    
    def _analyze_technology_focus(self, github_activity: GitHubActivity) -> Dict[str, Any]:
        """Analyze technology focus from languages"""
        tech_analysis = {
            'primary_languages': github_activity.languages[:3],
            'language_diversity_score': len(github_activity.languages),
            'tech_stack_assessment': self._assess_tech_stack(github_activity.languages)
        }
        
        return tech_analysis
    
    def _assess_tech_stack(self, languages: List[str]) -> str:
        """Assess technology stack sophistication"""
        if not languages:
            return "Unknown - No language data available"
        
        modern_languages = {'TypeScript', 'Go', 'Rust', 'Kotlin', 'Swift', 'Python'}
        web_languages = {'JavaScript', 'TypeScript', 'HTML', 'CSS', 'React', 'Vue'}
        backend_languages = {'Python', 'Java', 'Go', 'Rust', 'C++', 'Scala'}
        
        lang_set = set(languages)
        
        modern_count = len(lang_set.intersection(modern_languages))
        web_count = len(lang_set.intersection(web_languages))
        backend_count = len(lang_set.intersection(backend_languages))
        
        if modern_count >= 2 and (web_count >= 2 or backend_count >= 2):
            return "Modern - Contemporary technology stack with diverse capabilities"
        elif modern_count >= 1 or web_count >= 2 or backend_count >= 2:
            return "Standard - Solid technology foundation"
        else:
            return "Traditional - Conventional technology choices"
    
    def _assess_development_velocity(self, github_activity: GitHubActivity) -> str:
        """Assess development velocity"""
        commits = github_activity.total_commits
        contributors = github_activity.contributors
        
        # Calculate velocity score
        velocity_score = (commits / 100) + (contributors * 2)
        
        if velocity_score > 50:
            return "High - Very active development with many contributors"
        elif velocity_score > 20:
            return "Medium - Steady development activity"
        elif velocity_score > 5:
            return "Low - Limited development activity"
        else:
            return "Minimal - Very low development velocity"
    
    def _assess_community_engagement(self, github_activity: GitHubActivity) -> str:
        """Assess community engagement level"""
        contributors = github_activity.contributors
        repos = github_activity.public_repos
        
        if contributors > 100 and repos > 20:
            return "High - Large community with many contributors"
        elif contributors > 50 or repos > 10:
            return "Medium - Growing community engagement"
        elif contributors > 10 or repos > 5:
            return "Low - Small but active community"
        else:
            return "Minimal - Limited community involvement"
    
    def _identify_innovation_indicators(self, github_activity: GitHubActivity) -> List[str]:
        """Identify innovation indicators from GitHub activity"""
        indicators = []
        
        # Language diversity as innovation indicator
        if len(github_activity.languages) > 8:
            indicators.append("High language diversity suggests experimental approach")
        
        # Activity level
        if github_activity.activity_score and github_activity.activity_score > 80:
            indicators.append("High activity score indicates rapid iteration")
        
        # Modern languages
        modern_langs = {'Rust', 'Go', 'TypeScript', 'Kotlin', 'Swift'}
        if any(lang in github_activity.languages for lang in modern_langs):
            indicators.append("Adoption of modern programming languages")
        
        # Repository count
        if github_activity.public_repos > 50:
            indicators.append("Large number of repositories suggests active experimentation")
        
        # Contributor engagement
        if github_activity.contributors > 50:
            indicators.append("High contributor count indicates strong developer community")
        
        return indicators[:5]  # Limit to top 5 indicators_repos
        contributor_count = github_activity.contributors
        language_diversity = len(github_activity.languages)
        
        maturity_score = (
            min(repo_count / 20, 5) +  # Repository count (max 5 points)
            min(contributor_count / 50, 5) +  # Contributor count (max 5 points)
            min(language_diversity / 5, 3)  # Language diversity (max 3 points)
        )
        
        if maturity_score >= 10:
            return "High - Mature open source practice"
        elif maturity_score >= 6:
            return "Medium - Developing open source presence"
        elif maturity_score >= 3:
            return "Low - Limited open source activity"
        else:
            return "Minimal - Very limited GitHub presence"
    
    def _analyze_open_source_strategy(self, github_activity: GitHubActivity) -> Dict[str, str]:
        """Analyze open source strategy"""
        strategy = {}
        
        repo_count = github_activity.public