# src/competitor/collectors/jobs.py
"""
Job posting collector for competitor hiring intelligence
"""

import asyncio
from typing import Dict, List, Any
from urllib.parse import quote
import re
import logging

from .base import CachedCollector, RateLimitedSession
from ..models import JobPosting

logger = logging.getLogger(__name__)

class JobCollector(CachedCollector):
    """Collects job posting data for hiring intelligence"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "jobs")
        self.sources = config.get('sources', ['linkedin', 'glassdoor', 'indeed'])
        self.max_jobs = config.get('max_jobs_per_company', 20)
        self.focus_departments = config.get('focus_departments', 
                                          ['engineering', 'product', 'sales', 'marketing'])
        
    async def _collect_data(self, competitor_name: str, **kwargs) -> List[JobPosting]:
        """Collect job postings from various sources"""
        all_jobs = []
        
        # Collect from each enabled source
        collection_tasks = []
        
        if 'linkedin' in self.sources:
            collection_tasks.append(self._collect_linkedin_jobs(competitor_name))
        
        if 'glassdoor' in self.sources:
            collection_tasks.append(self._collect_glassdoor_jobs(competitor_name))
        
        if 'indeed' in self.sources:
            collection_tasks.append(self._collect_indeed_jobs(competitor_name))
        
        # Execute all collection tasks
        results = await asyncio.gather(*collection_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_jobs.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Job collection task failed: {result}")
        
        # Deduplicate and filter
        filtered_jobs = self._filter_and_deduplicate_jobs(all_jobs)
        
        return filtered_jobs[:self.max_jobs]
    
    async def _collect_linkedin_jobs(self, competitor_name: str) -> List[JobPosting]:
        """Collect jobs from LinkedIn (via web scraping)"""
        jobs = []
        
        try:
            # This would require LinkedIn scraping or API access
            # For demonstration, returning mock data structure
            
            # In a real implementation, this would:
            # 1. Search LinkedIn jobs for the company
            # 2. Parse job listings
            # 3. Extract job details
            
            search_url = f"https://www.linkedin.com/jobs/search/?keywords={quote(competitor_name)}"
            
            async with RateLimitedSession(rate_limit=3.0) as session:
                # LinkedIn has strong anti-scraping measures
                # This would need proper implementation with headers, delays, etc.
                logger.debug(f"LinkedIn job search for {competitor_name} (placeholder)")
                
                # Mock job data for demonstration
                if competitor_name.lower() in ['algolia', 'constructor', 'bloomreach']:
                    jobs.extend([
                        JobPosting(
                            title="Senior Software Engineer",
                            department="engineering",
                            location="San Francisco, CA",
                            posted_date="2024-01-15",
                            url=f"{search_url}#mock",
                            requirements=["5+ years experience", "Python", "React"]
                        ),
                        JobPosting(
                            title="Product Manager",
                            department="product",
                            location="Remote",
                            posted_date="2024-01-10",
                            url=f"{search_url}#mock2",
                            requirements=["Product management", "Analytics", "B2B SaaS"]
                        )
                    ])
                
        except Exception as e:
            logger.warning(f"LinkedIn jobs collection failed for {competitor_name}: {e}")
        
        return jobs
    
    async def _collect_glassdoor_jobs(self, competitor_name: str) -> List[JobPosting]:
        """Collect jobs from Glassdoor"""
        jobs = []
        
        try:
            # Similar to LinkedIn, this would require proper implementation
            search_query = quote(f"{competitor_name} jobs")
            
            async with RateLimitedSession(rate_limit=2.0) as session:
                logger.debug(f"Glassdoor job search for {competitor_name} (placeholder)")
                
                # Mock data for demonstration
                if len(competitor_name) > 5:  # Simple check to vary results
                    jobs.append(JobPosting(
                        title="Sales Engineer",
                        department="sales",
                        location="New York, NY",
                        posted_date="2024-01-12",
                        requirements=["Technical sales", "SaaS experience"]
                    ))
                
        except Exception as e:
            logger.warning(f"Glassdoor jobs collection failed for {competitor_name}: {e}")
        
        return jobs
    
    async def _collect_indeed_jobs(self, competitor_name: str) -> List[JobPosting]:
        """Collect jobs from Indeed"""
        jobs = []
        
        try:
            search_params = {
                'q': competitor_name,
                'l': '',  # All locations
                'sort': 'date'
            }
            
            async with RateLimitedSession(rate_limit=2.0) as session:
                logger.debug(f"Indeed job search for {competitor_name} (placeholder)")
                
                # Mock data
                jobs.append(JobPosting(
                    title="DevOps Engineer", 
                    department="engineering",
                    location="Austin, TX",
                    posted_date="2024-01-08",
                    requirements=["Docker", "Kubernetes", "AWS"]
                ))
                
        except Exception as e:
            logger.warning(f"Indeed jobs collection failed for {competitor_name}: {e}")
        
        return jobs
    
    def _filter_and_deduplicate_jobs(self, jobs: List[JobPosting]) -> List[JobPosting]:
        """Filter jobs by focus departments and remove duplicates"""
        filtered_jobs = []
        seen_jobs = set()
        
        for job in jobs:
            # Filter by department if specified
            if self.focus_departments:
                if not any(dept.lower() in job.department.lower() 
                          for dept in self.focus_departments):
                    continue
            
            # Create unique key for deduplication
            job_key = (job.title.lower(), job.department.lower(), job.location)
            
            if job_key not in seen_jobs:
                seen_jobs.add(job_key)
                filtered_jobs.append(job)
        
        # Sort by posted date (most recent first)
        filtered_jobs.sort(key=lambda x: x.posted_date or '', reverse=True)
        
        return filtered_jobs
    
    def _extract_department_from_title(self, job_title: str) -> str:
        """Extract department from job title"""
        title_lower = job_title.lower()
        
        # Department mapping based on common job titles
        department_keywords = {
            'engineering': ['engineer', 'developer', 'architect', 'devops', 'sre', 'backend', 'frontend', 'fullstack'],
            'product': ['product manager', 'product owner', 'pm', 'product designer', 'ux', 'ui'],
            'sales': ['sales', 'account executive', 'business development', 'sales engineer'],
            'marketing': ['marketing', 'content', 'demand generation', 'growth', 'brand'],
            'data': ['data scientist', 'data engineer', 'analyst', 'machine learning'],
            'customer success': ['customer success', 'support', 'technical account manager'],
            'hr': ['people', 'human resources', 'recruiter', 'talent'],
            'finance': ['finance', 'accounting', 'controller', 'financial analyst'],
            'legal': ['legal', 'counsel', 'compliance', 'privacy']
        }
        
        for department, keywords in department_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                return department
        
        return 'other'
    
    def _extract_requirements_from_description(self, description: str) -> List[str]:
        """Extract key requirements from job description"""
        if not description:
            return []
        
        requirements = []
        desc_lower = description.lower()
        
        # Technical skills
        tech_skills = [
            'python', 'javascript', 'java', 'go', 'rust', 'react', 'vue', 'angular',
            'node.js', 'django', 'flask', 'spring', 'kubernetes', 'docker', 'aws',
            'gcp', 'azure', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'machine learning', 'ai', 'data science', 'analytics'
        ]
        
        for skill in tech_skills:
            if skill in desc_lower:
                requirements.append(skill)
        
        # Experience requirements
        import re
        experience_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?experience', desc_lower)
        if experience_match:
            requirements.append(f"{experience_match.group(1)}+ years experience")
        
        # Education requirements
        if 'bachelor' in desc_lower or 'bs' in desc_lower:
            requirements.append("Bachelor's degree")
        if 'master' in desc_lower or 'ms' in desc_lower:
            requirements.append("Master's degree")
        if 'phd' in desc_lower:
            requirements.append("PhD")
        
        return requirements[:10]  # Limit to top 10 requirements
    
    def _get_empty_result(self) -> List[JobPosting]:
        """Return empty job list"""
        return []

class CompanyJobAnalyzer:
    """Analyzes job posting patterns for competitive intelligence"""
    
    def __init__(self):
        self.department_growth_indicators = {
            'engineering': ['scale', 'growth', 'expanding team', 'rapid growth'],
            'sales': ['new markets', 'expansion', 'quota carrying', 'enterprise sales'],
            'product': ['product-market fit', 'new features', 'roadmap', 'user experience'],
            'data': ['ai', 'machine learning', 'data-driven', 'analytics platform']
        }
    
    def analyze_hiring_trends(self, jobs: List[JobPosting]) -> Dict[str, Any]:
        """Analyze hiring trends from job postings"""
        if not jobs:
            return {'analysis': 'No job data available'}
        
        analysis = {
            'total_openings': len(jobs),
            'department_breakdown': self._analyze_department_distribution(jobs),
            'location_analysis': self._analyze_location_trends(jobs),
            'growth_signals': self._detect_growth_signals(jobs),
            'skill_demands': self._analyze_skill_demands(jobs),
            'hiring_velocity': self._estimate_hiring_velocity(jobs)
        }
        
        return analysis
    
    def _analyze_department_distribution(self, jobs: List[JobPosting]) -> Dict[str, int]:
        """Analyze distribution of jobs by department"""
        department_counts = {}
        
        for job in jobs:
            dept = job.department.lower()
            department_counts[dept] = department_counts.get(dept, 0) + 1
        
        return dict(sorted(department_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_location_trends(self, jobs: List[JobPosting]) -> Dict[str, Any]:
        """Analyze location distribution and remote work trends"""
        locations = {}
        remote_count = 0
        
        for job in jobs:
            if not job.location:
                continue
                
            location = job.location.lower()
            if 'remote' in location or 'anywhere' in location:
                remote_count += 1
            else:
                # Extract city/state
                clean_location = location.split(',')[0].strip()
                locations[clean_location] = locations.get(clean_location, 0) + 1
        
        return {
            'top_locations': dict(sorted(locations.items(), key=lambda x: x[1], reverse=True)[:5]),
            'remote_percentage': round(remote_count / len(jobs) * 100, 1) if jobs else 0
        }
    
    def _detect_growth_signals(self, jobs: List[JobPosting]) -> List[str]:
        """Detect signals of company growth from job postings"""
        growth_signals = []
        
        # High engineering hiring
        eng_jobs = [j for j in jobs if 'engineering' in j.department.lower()]
        if len(eng_jobs) > len(jobs) * 0.4:  # >40% engineering jobs
            growth_signals.append("Heavy engineering hiring suggests product scaling")
        
        # Sales team expansion
        sales_jobs = [j for j in jobs if 'sales' in j.department.lower()]
        if len(sales_jobs) > 3:
            growth_signals.append("Sales team expansion indicates revenue growth focus")
        
        # Senior level hiring
        senior_jobs = [j for j in jobs if any(level in j.title.lower() 
                      for level in ['senior', 'lead', 'principal', 'staff', 'director'])]
        if len(senior_jobs) > len(jobs) * 0.5:
            growth_signals.append("High proportion of senior roles suggests rapid scaling")
        
        # International expansion
        international_jobs = [j for j in jobs if j.location and 
                            any(country in j.location.lower() 
                                for country in ['uk', 'europe', 'canada', 'australia', 'singapore'])]
        if international_jobs:
            growth_signals.append("International hiring indicates global expansion")
        
        return growth_signals
    
    def _analyze_skill_demands(self, jobs: List[JobPosting]) -> Dict[str, int]:
        """Analyze most demanded skills across job postings"""
        skill_counts = {}
        
        for job in jobs:
            for requirement in job.requirements:
                req_lower = requirement.lower()
                skill_counts[req_lower] = skill_counts.get(req_lower, 0) + 1
        
        # Return top 10 skills
        return dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _estimate_hiring_velocity(self, jobs: List[JobPosting]) -> str:
        """Estimate hiring velocity based on posting frequency"""
        if len(jobs) > 15:
            return "High - Very active hiring"
        elif len(jobs) > 8:
            return "Medium - Steady hiring"
        elif len(jobs) > 3:
            return "Low - Selective hiring"
        else:
            return "Minimal - Limited hiring activity"