import asyncio
import json
import pickle
from types import SimpleNamespace

import pytest

import src.llm.evidence_analysis as evidence_module
from src.llm.evidence_analysis import (
    EvidenceRecord,
    OpenAIResponsesProvider,
    RecommendationAnalysisLayer,
    RecommendationEvidenceBuilder,
    StructuredAnalysisProvider,
    StructuredProviderResponse,
    UngroundedAnalysisError,
)


def crawler_output():
    return {
        "url": "https://shop.example/products/shoe",
        "page_type": "product",
        "recommendation_placements": [
            {
                "page_url": "https://shop.example/products/shoe",
                "page_type": "product",
                "placement_title": "Similar Products",
                "placement_position": 1,
                "product_count": 4,
                "detected_type": "Similar Products",
                "confidence": 0.94,
                "supporting_dom_evidence": [
                    'recommendation heading: "Similar Products"',
                    "carousel/slider attribute: related-products swiper",
                ],
            }
        ],
    }


def grounded_response(evidence_id="crawler:placement:1"):
    return {
        "recommendation_strategy": [
            {
                "interpretation": "The collected evidence suggests a product-similarity strategy.",
                "evidence_ids": [evidence_id],
                "confidence": "high",
            }
        ],
        "personalization_maturity": {
            "level": "Foundational",
            "rationale": "The evidence suggests one contextual recommendation placement.",
            "evidence_ids": [evidence_id],
            "confidence": "medium",
        },
        "presales_recommendations": [
            {
                "recommendation": "Test a broader set of recommendation placements.",
                "rationale": "The evidence indicates only one observed placement in this crawl.",
                "evidence_ids": [evidence_id],
                "priority": "high",
            }
        ],
        "opportunity_summary": {
            "summary": "The evidence suggests an opportunity to test additional contexts.",
            "evidence_ids": [evidence_id],
        },
        "limitations": ["The crawl is a point-in-time observation."],
    }


class StubProvider(StructuredAnalysisProvider):
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def generate_structured(self, *, instructions, evidence, schema):
        self.calls.append(
            {"instructions": instructions, "evidence": evidence, "schema": schema}
        )
        return StructuredProviderResponse(
            data=self.response,
            provider="stub",
            model="stub-model",
            tokens_used=42,
        )


def test_analyzes_only_normalized_crawler_and_audit_evidence():
    provider = StubProvider(grounded_response())
    layer = RecommendationAnalysisLayer(provider)
    audit = {
        "findings": [
            {
                "page_url": "https://shop.example/products/shoe",
                "check": "recommendation_diversity",
                "result": "one placement type observed",
                "api_key": "must-not-leak",
            }
        ]
    }

    result = asyncio.run(layer.analyze(crawler_output(), audit))

    assert result.provider == "stub"
    assert result.model == "stub-model"
    assert result.personalization_maturity["level"] == "Foundational"
    assert [item["source"] for item in result.evidence_catalog] == [
        "crawler",
        "crawler",
        "audit",
    ]
    serialized_call = json.dumps(provider.calls[0])
    assert "must-not-leak" not in serialized_call
    assert "api_key" not in serialized_call
    assert "outside knowledge" in provider.calls[0]["instructions"]


def test_rejects_provider_citations_not_present_in_evidence():
    provider = StubProvider(grounded_response("crawler:placement:999"))
    layer = RecommendationAnalysisLayer(provider)

    with pytest.raises(UngroundedAnalysisError, match="not supplied"):
        asyncio.run(layer.analyze(crawler_output()))


def test_rejects_unqualified_claims_about_site_capabilities():
    response = grounded_response()
    response["recommendation_strategy"][0]["interpretation"] = (
        "The site uses collaborative filtering."
    )
    layer = RecommendationAnalysisLayer(StubProvider(response))

    with pytest.raises(UngroundedAnalysisError, match="unqualified claim"):
        asyncio.run(layer.analyze(crawler_output()))


def test_provider_neutral_validation_rejects_incomplete_items():
    response = grounded_response()
    del response["presales_recommendations"][0]["rationale"]
    layer = RecommendationAnalysisLayer(StubProvider(response))

    with pytest.raises(UngroundedAnalysisError, match="rationale"):
        asyncio.run(layer.analyze(crawler_output()))


def test_empty_evidence_returns_unknown_without_calling_provider():
    provider = StubProvider(grounded_response())
    result = asyncio.run(RecommendationAnalysisLayer(provider).analyze({}))

    assert result.personalization_maturity["level"] == "Unknown"
    assert result.provider == "none"
    assert provider.calls == []


def test_builder_deduplicates_placements_copied_into_summary_and_raw_page():
    output = {
        "recommendation_placements": crawler_output()["recommendation_placements"],
        "raw_pages": [crawler_output()],
    }

    evidence = RecommendationEvidenceBuilder().build(output)

    placements = [item for item in evidence if item.category == "recommendation_placement"]
    assert len(placements) == 1
    assert placements[0].evidence_id == "crawler:placement:1"


def test_explicit_evidence_rejects_non_crawler_or_audit_sources():
    with pytest.raises(ValueError, match="crawler.*audit"):
        EvidenceRecord(
            evidence_id="web:1",
            source="web",
            category="search_result",
            data={"result": "not collected by the crawler"},
        )


class FakeResponsesAPI:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output_text=json.dumps(grounded_response()),
            usage=SimpleNamespace(total_tokens=17),
        )


def test_openai_provider_disables_storage_and_uses_strict_schema():
    responses = FakeResponsesAPI()
    client = SimpleNamespace(responses=responses)
    provider = OpenAIResponsesProvider(client=client, model="test-model")

    result = asyncio.run(
        provider.generate_structured(
            instructions="Use only evidence.",
            evidence=[{"evidence_id": "crawler:placement:1"}],
            schema=RecommendationAnalysisLayer.OUTPUT_SCHEMA,
        )
    )

    assert result.provider == "openai"
    assert result.tokens_used == 17
    assert responses.kwargs["store"] is False
    assert responses.kwargs["text"]["format"]["type"] == "json_schema"
    assert responses.kwargs["text"]["format"]["strict"] is True
    assert "tools" not in responses.kwargs


def test_openai_api_key_is_not_retained_or_serializable(monkeypatch):
    client = SimpleNamespace(responses=FakeResponsesAPI())
    seen = []

    def fake_openai_client(*, api_key):
        seen.append(api_key)
        return client

    monkeypatch.setattr(evidence_module, "AsyncOpenAI", fake_openai_client)
    provider = OpenAIResponsesProvider(api_key="ephemeral-test-key")

    assert seen == ["ephemeral-test-key"]
    assert "api_key" not in provider.__dict__
    assert "ephemeral-test-key" not in repr(provider)
    with pytest.raises(TypeError, match="may not be serialized"):
        pickle.dumps(provider)
