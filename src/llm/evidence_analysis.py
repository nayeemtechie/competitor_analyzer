"""Provider-neutral, evidence-bounded recommendation analysis.

This module is intentionally separate from collection and deterministic audit
logic.  Providers receive only normalized crawler/audit evidence, and every
interpretation returned to callers must cite evidence identifiers that were
present in that input.
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

try:  # pragma: no cover - exercised with an injected client in unit tests
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - optional provider dependency
    AsyncOpenAI = None


_SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:api[_-]?key|(?:access|refresh|bearer)?[_-]?token|"
    r"auth(?:orization)?|password|secret)(?:$|_)",
    re.IGNORECASE,
)
_UNQUALIFIED_SITE_CLAIM = re.compile(
    r"\b(?:the\s+)?(?:site|website|brand|retailer|company)\s+"
    r"(?:currently\s+)?(?:has|have|uses|offers|contains|includes|features|"
    r"deploys|runs|supports|provides|personalizes|is\s+using)\b",
    re.IGNORECASE,
)
_INFERENCE_LANGUAGE = re.compile(
    r"\b(?:evidence|suggests?|indicates?|may|might|could|likely|appears?|"
    r"potentially|interpretation|hypothesis)\b",
    re.IGNORECASE,
)


class EvidenceValidationError(ValueError):
    """Raised when evidence is malformed or includes an unsupported source."""


class UngroundedAnalysisError(ValueError):
    """Raised when provider output is not traceable to supplied evidence."""


@dataclass(frozen=True)
class EvidenceRecord:
    """One immutable unit of crawler or audit evidence."""

    evidence_id: str
    source: str
    category: str
    data: Mapping[str, Any]
    page_url: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise EvidenceValidationError("Evidence records require an evidence_id")
        if self.source not in {"crawler", "audit"}:
            raise EvidenceValidationError("Evidence source must be 'crawler' or 'audit'")
        if not self.category.strip():
            raise EvidenceValidationError("Evidence records require a category")

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "category": self.category,
            "page_url": self.page_url,
            "data": _json_safe(self.data),
        }


@dataclass
class StructuredProviderResponse:
    """Provider-neutral structured response and non-secret request metadata."""

    data: Dict[str, Any]
    provider: str
    model: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None


class StructuredAnalysisProvider(ABC):
    """Small provider contract used by evidence analysis."""

    @abstractmethod
    async def generate_structured(
        self,
        *,
        instructions: str,
        evidence: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
    ) -> StructuredProviderResponse:
        """Generate JSON matching ``schema`` using only ``evidence``."""


class OpenAIResponsesProvider(StructuredAnalysisProvider):
    """OpenAI Responses API implementation of the structured provider contract.

    Keys are accepted only as a constructor argument or environment value,
    passed directly to the SDK, and never retained as a field or serialized.
    ``store=False`` is set for every response.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        client: Optional[Any] = None,
        max_output_tokens: int = 2000,
    ) -> None:
        if client is None:
            if AsyncOpenAI is None:
                raise ImportError("OpenAI library not installed. Run: pip install openai")
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not resolved_key:
                raise RuntimeError("OPENAI_API_KEY is required for the OpenAI provider")
            client = AsyncOpenAI(api_key=resolved_key)

        self._client = client
        self._model = model
        self._max_output_tokens = max_output_tokens

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def __repr__(self) -> str:
        return "OpenAIResponsesProvider(model={!r})".format(self._model)

    def __getstate__(self) -> Dict[str, Any]:
        raise TypeError("LLM provider instances may not be serialized")

    async def generate_structured(
        self,
        *,
        instructions: str,
        evidence: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
    ) -> StructuredProviderResponse:
        started = time.monotonic()
        response = await self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=json.dumps({"evidence": list(evidence)}, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "grounded_recommendation_analysis",
                    "strict": True,
                    "schema": dict(schema),
                }
            },
            max_output_tokens=self._max_output_tokens,
            store=False,
        )
        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise RuntimeError("OpenAI returned no structured output text")

        try:
            data = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenAI returned invalid structured JSON") from exc

        usage = getattr(response, "usage", None)
        tokens_used = getattr(usage, "total_tokens", None) if usage else None
        return StructuredProviderResponse(
            data=data,
            provider=self.provider_name,
            model=self._model,
            tokens_used=tokens_used,
            latency_ms=round((time.monotonic() - started) * 1000, 2),
        )


class RecommendationEvidenceBuilder:
    """Normalize crawler placements and audit findings into evidence records."""

    _PLACEMENT_FIELDS = (
        "page_url",
        "page_type",
        "placement_title",
        "placement_position",
        "product_count",
        "detected_type",
        "confidence",
        "supporting_dom_evidence",
    )

    def build(
        self,
        crawler_output: Any,
        audit_output: Optional[Any] = None,
    ) -> List[EvidenceRecord]:
        records: List[EvidenceRecord] = []
        safe_crawler = _json_safe(crawler_output)
        placements = self._collect_key(safe_crawler, "recommendation_placements")

        seen_placements: Set[str] = set()
        placement_number = 0
        for placement in placements:
            if not isinstance(placement, Mapping):
                continue
            payload = {
                key: _json_safe(placement.get(key))
                for key in self._PLACEMENT_FIELDS
                if key in placement
            }
            fingerprint = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            if fingerprint in seen_placements:
                continue
            seen_placements.add(fingerprint)
            placement_number += 1
            records.append(
                EvidenceRecord(
                    evidence_id="crawler:placement:{}".format(placement_number),
                    source="crawler",
                    category="recommendation_placement",
                    page_url=_optional_string(payload.get("page_url")),
                    data=payload,
                )
            )

        page_urls = sorted(self._collect_page_urls(safe_crawler))
        if page_urls:
            records.append(
                EvidenceRecord(
                    evidence_id="crawler:coverage:1",
                    source="crawler",
                    category="crawl_coverage",
                    data={
                        "pages_analyzed": page_urls,
                        "detected_recommendation_placement_count": placement_number,
                    },
                )
            )

        for number, finding in enumerate(self._audit_findings(audit_output), start=1):
            safe_finding = _json_safe(finding)
            payload = (
                safe_finding
                if isinstance(safe_finding, Mapping)
                else {"finding": safe_finding}
            )
            records.append(
                EvidenceRecord(
                    evidence_id="audit:finding:{}".format(number),
                    source="audit",
                    category="audit_finding",
                    page_url=self._finding_page_url(payload),
                    data=payload,
                )
            )

        return records

    def _collect_key(self, value: Any, target_key: str) -> List[Any]:
        matches: List[Any] = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key == target_key and isinstance(child, list):
                    matches.extend(child)
                else:
                    matches.extend(self._collect_key(child, target_key))
        elif isinstance(value, list):
            for child in value:
                matches.extend(self._collect_key(child, target_key))
        return matches

    def _collect_page_urls(self, value: Any) -> Set[str]:
        urls: Set[str] = set()
        if isinstance(value, Mapping):
            for key, child in value.items():
                if (
                    key in {"url", "page_url"}
                    and isinstance(child, str)
                    and child.startswith(("http://", "https://"))
                ):
                    urls.add(child)
                elif key == "pages_analyzed" and isinstance(child, list):
                    urls.update(
                        item for item in child
                        if isinstance(item, str) and item.startswith(("http://", "https://"))
                    )
                else:
                    urls.update(self._collect_page_urls(child))
        elif isinstance(value, list):
            for child in value:
                urls.update(self._collect_page_urls(child))
        return urls

    def _audit_findings(self, audit_output: Any) -> List[Any]:
        if audit_output is None:
            return []
        safe_audit = _json_safe(audit_output)
        if isinstance(safe_audit, list):
            return safe_audit
        if isinstance(safe_audit, Mapping):
            for key in ("findings", "results", "checks"):
                value = safe_audit.get(key)
                if isinstance(value, list):
                    return value
            return [safe_audit]
        return [{"finding": safe_audit}]

    @staticmethod
    def _finding_page_url(finding: Mapping[str, Any]) -> Optional[str]:
        for key in ("page_url", "url"):
            value = finding.get(key)
            if isinstance(value, str):
                return value
        return None


@dataclass
class RecommendationAnalysisResult:
    """Validated analysis plus the exact evidence catalog used to produce it."""

    recommendation_strategy: List[Dict[str, Any]]
    personalization_maturity: Dict[str, Any]
    presales_recommendations: List[Dict[str, Any]]
    opportunity_summary: Dict[str, Any]
    limitations: List[str]
    evidence_catalog: List[Dict[str, Any]]
    provider: str
    model: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_strategy": list(self.recommendation_strategy),
            "personalization_maturity": dict(self.personalization_maturity),
            "presales_recommendations": list(self.presales_recommendations),
            "opportunity_summary": dict(self.opportunity_summary),
            "limitations": list(self.limitations),
            "evidence_catalog": list(self.evidence_catalog),
            "provider": self.provider,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
        }


class RecommendationAnalysisLayer:
    """Turn deterministic findings into grounded recommendation strategy analysis."""

    SYSTEM_INSTRUCTIONS = """You are an evidence-constrained ecommerce presales analyst.

Analyze only the records inside the supplied evidence array. Do not use outside knowledge,
browse, or infer that the site contains a feature, technology, behavior, or placement that
is not explicitly present in those records.

Rules:
1. Every strategy interpretation, maturity rationale, recommendation rationale, and
   opportunity summary must cite one or more supplied evidence_id values.
2. Treat crawler and audit findings as observations; do not strengthen their confidence.
3. Phrase strategy and maturity conclusions as interpretations (for example, "the evidence
   suggests"), never as newly observed facts about the site.
4. Recommendations are proposed future actions, not claims about current capabilities.
5. Absence of detected evidence is not proof that a capability is absent. Put coverage gaps
   and uncertainty in limitations.
6. If evidence is insufficient, use maturity level "Unknown" and say why.
7. Do not mention any fact, vendor, algorithm, customer behavior, business result, or
   implementation detail unless it is present in the cited evidence.
8. Return only the requested JSON structure."""

    OUTPUT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "recommendation_strategy": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "interpretation": {"type": "string"},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": ["interpretation", "evidence_ids", "confidence"],
                },
            },
            "personalization_maturity": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["Unknown", "Foundational", "Developing", "Advanced"],
                    },
                    "rationale": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["level", "rationale", "evidence_ids", "confidence"],
            },
            "presales_recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "recommendation": {"type": "string"},
                        "rationale": {"type": "string"},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": ["recommendation", "rationale", "evidence_ids", "priority"],
                },
            },
            "opportunity_summary": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "evidence_ids"],
            },
            "limitations": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "recommendation_strategy",
            "personalization_maturity",
            "presales_recommendations",
            "opportunity_summary",
            "limitations",
        ],
    }

    def __init__(
        self,
        provider: StructuredAnalysisProvider,
        *,
        evidence_builder: Optional[RecommendationEvidenceBuilder] = None,
    ) -> None:
        self.provider = provider
        self.evidence_builder = evidence_builder or RecommendationEvidenceBuilder()

    @classmethod
    def openai(
        cls,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        client: Optional[Any] = None,
    ) -> "RecommendationAnalysisLayer":
        """Create the first supported provider without retaining its API key."""

        return cls(
            OpenAIResponsesProvider(api_key=api_key, model=model, client=client)
        )

    async def analyze(
        self,
        crawler_output: Any,
        audit_output: Optional[Any] = None,
    ) -> RecommendationAnalysisResult:
        evidence = self.evidence_builder.build(crawler_output, audit_output)
        return await self.analyze_evidence(evidence)

    async def analyze_evidence(
        self,
        evidence: Iterable[EvidenceRecord],
    ) -> RecommendationAnalysisResult:
        records = list(evidence)
        self._validate_evidence(records)
        catalog = [record.to_prompt_dict() for record in records]

        if not records:
            return RecommendationAnalysisResult(
                recommendation_strategy=[],
                personalization_maturity={
                    "level": "Unknown",
                    "rationale": "No crawler or audit evidence was supplied.",
                    "evidence_ids": [],
                    "confidence": "low",
                },
                presales_recommendations=[],
                opportunity_summary={
                    "summary": "Insufficient evidence to summarize opportunities.",
                    "evidence_ids": [],
                },
                limitations=["No crawler or audit evidence was supplied; no LLM call was made."],
                evidence_catalog=[],
                provider="none",
                model="none",
            )

        response = await self.provider.generate_structured(
            instructions=self.SYSTEM_INSTRUCTIONS,
            evidence=catalog,
            schema=self.OUTPUT_SCHEMA,
        )
        self._validate_analysis(response.data, {record.evidence_id for record in records})

        return RecommendationAnalysisResult(
            recommendation_strategy=list(response.data["recommendation_strategy"]),
            personalization_maturity=dict(response.data["personalization_maturity"]),
            presales_recommendations=list(response.data["presales_recommendations"]),
            opportunity_summary=dict(response.data["opportunity_summary"]),
            limitations=list(response.data["limitations"]),
            evidence_catalog=catalog,
            provider=response.provider,
            model=response.model,
            tokens_used=response.tokens_used,
            latency_ms=response.latency_ms,
        )

    def _validate_evidence(self, records: Sequence[EvidenceRecord]) -> None:
        ids = [record.evidence_id for record in records]
        if len(ids) != len(set(ids)):
            raise EvidenceValidationError("Evidence identifiers must be unique")

    def _validate_analysis(self, data: Mapping[str, Any], allowed_ids: Set[str]) -> None:
        required = set(self.OUTPUT_SCHEMA["required"])
        if not isinstance(data, Mapping) or not required.issubset(data):
            raise UngroundedAnalysisError("Provider output is missing required analysis fields")

        maturity = data.get("personalization_maturity")
        if not isinstance(maturity, Mapping) or maturity.get("level") not in {
            "Unknown", "Foundational", "Developing", "Advanced"
        }:
            raise UngroundedAnalysisError("Provider returned an invalid maturity assessment")

        cited_items: List[Mapping[str, Any]] = []
        for key in ("recommendation_strategy", "presales_recommendations"):
            value = data.get(key)
            if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
                raise UngroundedAnalysisError("{} must be a list of objects".format(key))
            cited_items.extend(value)

        for item in data["recommendation_strategy"]:
            self._require_text(item, "interpretation")
            self._require_enum(item, "confidence", {"low", "medium", "high"})
        self._require_text(maturity, "rationale")
        self._require_enum(maturity, "confidence", {"low", "medium", "high"})
        for item in data["presales_recommendations"]:
            self._require_text(item, "recommendation")
            self._require_text(item, "rationale")
            self._require_enum(item, "priority", {"low", "medium", "high"})

        opportunity = data.get("opportunity_summary")
        if not isinstance(opportunity, Mapping):
            raise UngroundedAnalysisError("opportunity_summary must be an object")
        self._require_text(opportunity, "summary")

        cited_items.extend([maturity, opportunity])

        for item in cited_items:
            if not isinstance(item, Mapping):
                raise UngroundedAnalysisError("Analysis items must be objects")
            evidence_ids = item.get("evidence_ids")
            if not isinstance(evidence_ids, list) or not all(
                isinstance(evidence_id, str) for evidence_id in evidence_ids
            ):
                raise UngroundedAnalysisError("Every analysis item must contain evidence_ids")
            if not evidence_ids and not (
                item is maturity and maturity.get("level") == "Unknown"
            ):
                raise UngroundedAnalysisError("Every analysis item must cite collected evidence")
            unknown_ids = set(evidence_ids) - allowed_ids
            if unknown_ids:
                raise UngroundedAnalysisError(
                    "Provider cited evidence that was not supplied: {}".format(
                        ", ".join(sorted(unknown_ids))
                    )
                )
            self._reject_unqualified_site_claims(item)

        limitations = data.get("limitations")
        if not isinstance(limitations, list) or not all(
            isinstance(item, str) for item in limitations
        ):
            raise UngroundedAnalysisError("limitations must be a list of strings")
        for limitation in limitations:
            self._reject_unqualified_site_claims({"limitation": limitation})

    @staticmethod
    def _require_text(item: Mapping[str, Any], key: str) -> None:
        if not isinstance(item.get(key), str) or not item[key].strip():
            raise UngroundedAnalysisError("{} must be a non-empty string".format(key))

    @staticmethod
    def _require_enum(item: Mapping[str, Any], key: str, allowed: Set[str]) -> None:
        if item.get(key) not in allowed:
            raise UngroundedAnalysisError("{} contains an unsupported value".format(key))

    def _reject_unqualified_site_claims(self, item: Mapping[str, Any]) -> None:
        for key, value in item.items():
            if key == "evidence_ids" or not isinstance(value, str):
                continue
            if _UNQUALIFIED_SITE_CLAIM.search(value) and not _INFERENCE_LANGUAGE.search(value):
                raise UngroundedAnalysisError(
                    "Provider output made an unqualified claim about the site"
                )


def _optional_string(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _json_safe(value: Any) -> Any:
    """Return JSON-safe evidence while dropping credential-shaped fields."""

    if is_dataclass(value):
        value = asdict(value)
    elif hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(child)
            for key, child in value.items()
            if not _SENSITIVE_KEY.search(str(key))
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "EvidenceRecord",
    "EvidenceValidationError",
    "UngroundedAnalysisError",
    "StructuredAnalysisProvider",
    "StructuredProviderResponse",
    "OpenAIResponsesProvider",
    "RecommendationEvidenceBuilder",
    "RecommendationAnalysisResult",
    "RecommendationAnalysisLayer",
]
