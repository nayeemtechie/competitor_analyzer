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
