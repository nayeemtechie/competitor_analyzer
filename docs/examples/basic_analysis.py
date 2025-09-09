"""Basic analysis example.

Replace the placeholder values with real competitor names
and configuration before running.
"""

import asyncio
from competitor.analyzer import run_competitor_analysis

if __name__ == "__main__":
    # TODO: update competitor list
    asyncio.run(run_competitor_analysis(["ExampleCo"]))
