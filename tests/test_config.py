import yaml

from competitor.config import CompetitorConfig


def test_loads_existing_config(tmp_path):
    config_path = tmp_path / 'config.yaml'
    data = {
        'llm': {'provider': 'test-provider'},
        'competitors': [{'name': 'TestCo', 'website': 'https://example.com'}],
        'analysis': {},
        'data_sources': {},
        'output': {}
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)

    cfg = CompetitorConfig(config_path=str(config_path))

    assert cfg.competitors[0]['name'] == 'TestCo'
    assert cfg.llm_config['provider'] == 'test-provider'


def test_is_data_source_enabled_normalises_aliases(tmp_path):
    """All keys used by CompetitorAnalyzer should resolve to the same setting."""
    config_path = tmp_path / 'config.yaml'
    data = {
        'llm': {'provider': 'test-provider'},
        'competitors': [],
        'analysis': {},
        'output': {},
        'data_sources': {
            'company_websites': True,
            'funding_data': {'enabled': True},
            'job_boards': {'enabled': False},
            'news_sources': {'enabled': True},
            'social_media': {'enabled': False},
            'github_repos': {'enabled': True},
            'patent_databases': {'enabled': False},
        },
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)

    cfg = CompetitorConfig(config_path=str(config_path))

    status_by_attribute = {
        'company_websites': True,
        'funding_data': True,
        'job_boards': False,
        'news_sources': True,
        'social_media': False,
        'github_repos': True,
        'patent_databases': False,
    }

    analyzer_aliases = {
        'company_websites': ['websites'],
        'funding_data': ['funding'],
        'job_boards': ['jobs'],
        'news_sources': ['news'],
        'social_media': ['social'],
        'github_repos': ['github'],
        'patent_databases': ['patents'],
    }

    for attr_name, expected in status_by_attribute.items():
        assert cfg.is_data_source_enabled(attr_name) == expected
        for alias in analyzer_aliases[attr_name]:
            assert cfg.is_data_source_enabled(alias) == expected
