"""Custom collector placeholder example."""

class CustomCollector:
    """Stub collector to demonstrate extension points."""

    def collect(self, competitor_name: str) -> dict:
        # TODO: implement actual collection logic
        return {"competitor": competitor_name, "data": "placeholder"}
