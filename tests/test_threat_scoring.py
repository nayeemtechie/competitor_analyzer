from competitor.analyzer import CompetitorAnalyzer
from competitor.models import (
    CompetitorProfile,
    FundingInfo,
    NewsItem,
    JobPosting,
)


def make_profile():
    return CompetitorProfile(
        name='TestCo',
        website='https://example.com',
        funding_info=FundingInfo(last_round_amount='150M'),
        target_markets=['enterprise', 'mid-market'],
        technology_stack=['AI', 'Kubernetes', 'GraphQL', 'Machine Learning'],
        recent_news=[NewsItem(title='', source='', sentiment='positive')] * 11,
        job_postings=[JobPosting(title='', department='eng')] * 21,
    )


def test_calculate_threat_score():
    analyzer = CompetitorAnalyzer.__new__(CompetitorAnalyzer)
    profile = make_profile()
    score = analyzer._calculate_threat_score(profile)
    assert score == 1.0
