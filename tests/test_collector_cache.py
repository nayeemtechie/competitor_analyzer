import asyncio
import importlib.util
from pathlib import Path

# Load CachedCollector directly without importing the entire package
BASE_PATH = Path(__file__).resolve().parents[1] / 'src/competitor/collectors/base.py'
spec = importlib.util.spec_from_file_location('collector_base', BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
CachedCollector = base.CachedCollector


class DummyCollector(CachedCollector):
    def __init__(self, config):
        super().__init__(config, 'dummy')
        self.call_count = 0

    async def _collect_data(self, competitor_name, **kwargs):
        self.call_count += 1
        return {'name': competitor_name, 'calls': self.call_count}

    def _get_empty_result(self):
        return {}


def test_cached_collector_uses_cache(tmp_path):
    config = {
        'enabled': True,
        'cache_enabled': True,
        'cache_ttl_hours': 1,
        'cache_dir': str(tmp_path)
    }
    collector = DummyCollector(config)

    first = asyncio.run(collector.collect('Acme'))
    assert collector.call_count == 1

    second = asyncio.run(collector.collect('Acme'))
    assert collector.call_count == 1
    assert second == first
