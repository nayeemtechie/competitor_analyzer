import sys
import json
import types
from pathlib import Path

# Ensure the src directory is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

# Provide a minimal YAML implementation if PyYAML isn't available
if 'yaml' not in sys.modules:
    yaml_stub = types.ModuleType('yaml')
    yaml_stub.safe_load = lambda stream: json.load(stream)
    yaml_stub.dump = lambda data, stream, **kwargs: json.dump(data, stream)
    sys.modules['yaml'] = yaml_stub

# Stub aiohttp for environments without the real library
if 'aiohttp' not in sys.modules:
    aiohttp_stub = types.ModuleType('aiohttp')

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    aiohttp_stub.ClientSession = _Dummy
    aiohttp_stub.TCPConnector = _Dummy
    aiohttp_stub.ClientTimeout = _Dummy
    sys.modules['aiohttp'] = aiohttp_stub

# Stub scraper module to avoid syntax errors during import
scraper_stub = types.ModuleType('competitor.scraper')
class _Scraper:
    pass
scraper_stub.CompetitorScraper = _Scraper
sys.modules['competitor.scraper'] = scraper_stub

# Minimal collectors package to satisfy imports
collectors_stub = types.ModuleType('competitor.collectors')
class _CollectorManager:
    def __init__(self, *args, **kwargs):
        pass
collectors_stub.CollectorManager = _CollectorManager
sys.modules['competitor.collectors'] = collectors_stub

# Stub analysis and reports modules used by CompetitorAnalyzer
analysis_stub = types.ModuleType('competitor.analysis')
class _AnalysisEngine:
    def __init__(self, *args, **kwargs):
        pass
    async def perform_cross_competitor_analysis(self, intelligence):
        return intelligence
analysis_stub.AnalysisEngine = _AnalysisEngine
sys.modules['competitor.analysis'] = analysis_stub

reports_stub = types.ModuleType('competitor.reports')
class _ReportGenerator:
    def __init__(self, *args, **kwargs):
        pass
    async def generate_reports(self, intelligence):
        return []
reports_stub.ReportGenerator = _ReportGenerator
sys.modules['competitor.reports'] = reports_stub
