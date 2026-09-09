"""Deterministic detection of ecommerce recommendation placements.

The detector deliberately uses only DOM signals.  It is suitable for running
as part of the crawler before any LLM-backed analysis and keeps a short audit
trail explaining why every placement was emitted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from bs4 import BeautifulSoup
from bs4.element import Tag


UNKNOWN = "Unknown"


@dataclass
class RecommendationPlacement:
    """A recommendation widget found on a crawled page.

    ``placement_position`` is the one-based DOM order among detected
    placements on the page, rather than a brittle pixel coordinate.
    """

    page_url: str
    page_type: Optional[str]
    placement_title: Optional[str]
    placement_position: int
    product_count: int
    detected_type: str
    confidence: float
    supporting_dom_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "page_url": self.page_url,
            "page_type": self.page_type,
            "placement_title": self.placement_title,
            "placement_position": self.placement_position,
            "product_count": self.product_count,
            "detected_type": self.detected_type,
            "confidence": self.confidence,
            "supporting_dom_evidence": list(self.supporting_dom_evidence),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "RecommendationPlacement":
        return cls(
            page_url=str(data.get("page_url") or ""),
            page_type=data.get("page_type") if isinstance(data.get("page_type"), str) else None,
            placement_title=(
                data.get("placement_title")
                if isinstance(data.get("placement_title"), str)
                else None
            ),
            placement_position=int(data.get("placement_position") or 0),
            product_count=int(data.get("product_count") or 0),
            detected_type=str(data.get("detected_type") or UNKNOWN),
            confidence=float(data.get("confidence") or 0.0),
            supporting_dom_evidence=[
                str(item) for item in data.get("supporting_dom_evidence", [])
            ],
        )


@dataclass
class _Candidate:
    container: Tag
    title: Optional[str]
    cards: List[Tag]
    detected_type: str
    evidence: List[str]
    strength: int


class RecommendationPlacementDetector:
    """Find recommendation placements from headings and product structures."""

    # Specific phrases come before broader phrases so, for example,
    # "customers also bought" is not swallowed by a generic "also bought" rule.
    LABEL_PATTERNS: Sequence[Tuple[str, Sequence[str]]] = (
        (
            "Frequently Bought Together",
            (
                r"\bfrequently\s+(?:bought|purchased)\s+together\b",
                r"\bcommonly\s+bought\s+together\b",
                r"\boften\s+bought\s+together\b",
                r"\bbuy\s+(?:these|it)\s+together\b",
                r"\bcomplete\s+the\s+(?:set|look)\b",
            ),
        ),
        (
            "Customers Also Bought",
            (
                r"\b(?:customers?|people|others|shoppers?)\s+also\s+(?:bought|purchased|viewed)\b",
                r"\b(?:customers?|people|shoppers?)\s+who\s+(?:bought|viewed).+also\s+(?:bought|viewed)\b",
                r"\bwhat\s+(?:customers?|others)\s+(?:bought|purchased)\b",
            ),
        ),
        (
            "Recently Viewed",
            (
                r"\brecently\s+viewed\b",
                r"\bviewed\s+recently\b",
                r"\b(?:your\s+)?browsing\s+history\b",
            ),
        ),
        (
            "You May Also Like",
            (
                r"\byou\s+(?:may|might|could)(?:\s+also)?\s+like\b",
                r"\bwe\s+think\s+you(?:'|’)ll\s+like\b",
                r"\byou(?:'|’)ll\s+also\s+love\b",
            ),
        ),
        (
            "Similar Products",
            (
                r"\bsimilar\s+(?:products?|items?|styles?)\b",
                r"\bmore\s+like\s+this\b",
                r"\brelated\s+(?:products?|items?)\b",
            ),
        ),
        (
            "Trending",
            (
                r"\btrending(?:\s+(?:now|products?|items?))?\b",
                r"\bwhat(?:'|’)s\s+hot\b",
                r"\bhot\s+right\s+now\b",
            ),
        ),
        (
            "Popular",
            (
                r"\bmost\s+popular\b",
                r"^popular$",
                r"\bpopular\s+(?:products?|items?|picks?|choices?)\b",
                r"\bbest[\s-]*sellers?\b",
                r"\btop\s+(?:selling|rated)\b",
            ),
        ),
        (
            "Personalized",
            (
                r"\bpersonal(?:ized|ised)\b",
                r"\brecommended\s+for\s+you\b",
                r"\b(?:picked|chosen|curated)\s+(?:just\s+)?for\s+you\b",
                r"\bjust\s+for\s+you\b",
                r"\bbased\s+on\s+your\b",
                r"\btop\s+picks\s+for\s+you\b",
                r"^for\s+you$",
                r"\binspired\s+by\s+your\b",
                r"\bbecause\s+you\b",
            ),
        ),
    )

    _RECOMMENDATION_MARKER = re.compile(
        r"recommend|related|similar|also[-_\s]?(?:like|bought)|"
        r"frequently[-_\s]?bought|recently[-_\s]?viewed|popular|trending|"
        r"best[-_\s]?seller|cross[-_\s]?sell|up[-_\s]?sell|personal",
        re.IGNORECASE,
    )
    _CAROUSEL_MARKER = re.compile(
        r"carousel|slider|swiper|slick|splide|glide|flickity|scroll-snap",
        re.IGNORECASE,
    )
    _PRODUCT_CARD_MARKER = re.compile(
        r"(?:^|[-_\s])(?:product|sku|merchandise|recommendation)"
        r"[-_\s]?(?:card|tile|item)(?:$|[-_\s])",
        re.IGNORECASE,
    )
    _PRICE_PATTERN = re.compile(
        r"(?:[$€£¥₹]\s?\d)|(?:\d[\d,.]*\s?(?:USD|EUR|GBP|INR)\b)",
        re.IGNORECASE,
    )
    _GENERIC_LIST_TITLES = re.compile(
        r"^(?:all\s+)?(?:products?|items?|results?|shop|catalog|collection)$",
        re.IGNORECASE,
    )

    def detect(
        self,
        html: Union[str, BeautifulSoup, Tag],
        page_url: str,
        page_type: Optional[str] = None,
    ) -> List[RecommendationPlacement]:
        """Return recommendation widgets in their one-based DOM order."""

        if isinstance(html, str):
            soup = BeautifulSoup(html, "html.parser")
        else:
            soup = html

        if not soup:
            return []

        candidates: List[_Candidate] = []

        # Explicit recommendation headings are the strongest human-readable
        # evidence and work even when sites use generated CSS class names.
        for heading in soup.find_all(re.compile(r"^h[1-6]$")):
            title = self._clean_text(heading.get_text(" ", strip=True))
            detected_type = self.classify_title(title)
            if detected_type == UNKNOWN:
                continue
            container = self._find_heading_container(heading)
            cards = self._find_product_cards(container)
            if cards:
                self._add_candidate(
                    candidates,
                    container,
                    title,
                    cards,
                    detected_type,
                    [
                        'recommendation heading: "{}"'.format(self._short(title)),
                        "container: {}".format(self._element_signature(container)),
                    ],
                    strength=5,
                )

        # Recommendation and carousel attributes capture accessible labels,
        # vendor widgets, and JS component names.
        for element in soup.find_all(["section", "aside", "article", "div", "ul", "ol"]):
            if self._inside_ignored_region(element) or self._looks_like_product_card(element):
                continue
            attrs = self._attribute_text(element)
            recommendation_marker = bool(self._RECOMMENDATION_MARKER.search(attrs))
            carousel_marker = bool(self._CAROUSEL_MARKER.search(attrs))
            if not recommendation_marker and not carousel_marker:
                continue

            cards = self._find_product_cards(element)
            if len(cards) < 2 and not recommendation_marker:
                continue
            if not cards:
                continue

            title = self._find_widget_title(element)
            detected_type = self.classify_title("{} {}".format(title or "", attrs))
            evidence = ["container: {}".format(self._element_signature(element))]
            if recommendation_marker:
                evidence.append(
                    'recommendation-related attribute: "{}"'.format(self._short(attrs))
                )
            if carousel_marker:
                evidence.append('carousel/slider attribute: "{}"'.format(self._short(attrs)))
            if title:
                evidence.append('widget title: "{}"'.format(self._short(title)))
            self._add_candidate(
                candidates,
                element,
                title,
                cards,
                detected_type,
                evidence,
                strength=4 if recommendation_marker else 3,
            )

        # Finally find semantic sections containing repeated product cards.
        # This intentionally produces Unknown when no recommendation label is
        # available; downstream analysis can distinguish evidence from guesses.
        for element in soup.find_all(["section", "aside", "article"]):
            if self._inside_ignored_region(element):
                continue
            cards = self._find_product_cards(element)
            if not 2 <= len(cards) <= 24:
                continue
            title = self._find_widget_title(element)
            if title and self._GENERIC_LIST_TITLES.match(title):
                continue
            detected_type = self.classify_title(title or "")
            self._add_candidate(
                candidates,
                element,
                title,
                cards,
                detected_type,
                [
                    "repeated product structure: {} cards".format(len(cards)),
                    "container: {}".format(self._element_signature(element)),
                ],
                strength=2 if title else 1,
            )

        candidates = self._deduplicate(candidates)
        dom_order = {id(tag): index for index, tag in enumerate(soup.find_all(True))}
        candidates.sort(key=lambda item: dom_order.get(id(item.container), 0))

        placements: List[RecommendationPlacement] = []
        for position, candidate in enumerate(candidates, start=1):
            evidence = list(candidate.evidence)
            evidence.append("product cards: {}".format(len(candidate.cards)))
            placements.append(
                RecommendationPlacement(
                    page_url=page_url,
                    page_type=page_type,
                    placement_title=candidate.title,
                    placement_position=position,
                    product_count=len(candidate.cards),
                    detected_type=candidate.detected_type,
                    confidence=self._confidence(candidate),
                    supporting_dom_evidence=self._unique(evidence),
                )
            )
        return placements

    @classmethod
    def classify_title(cls, text: str) -> str:
        """Map a title or DOM label onto the supported taxonomy."""

        normalised = cls._normalise_for_matching(text)
        for label, patterns in cls.LABEL_PATTERNS:
            if any(re.search(pattern, normalised, re.IGNORECASE) for pattern in patterns):
                return label
        return UNKNOWN

    def _find_heading_container(self, heading: Tag) -> Tag:
        fallback = heading
        for ancestor in heading.parents:
            if not isinstance(ancestor, Tag) or ancestor.name in {"body", "html", "[document]"}:
                break
            cards = self._find_product_cards(ancestor)
            if cards and fallback is heading:
                fallback = ancestor
            if len(cards) >= 2 and (
                ancestor.name in {"section", "aside", "article"}
                or self._has_component_marker(ancestor)
            ):
                return ancestor
        return fallback

    def _find_widget_title(self, container: Tag) -> Optional[str]:
        for heading in container.find_all(re.compile(r"^h[1-6]$")):
            card_ancestor = heading.find_parent(self._looks_like_product_card)
            if card_ancestor is not None and card_ancestor is not container:
                continue
            title = self._clean_text(heading.get_text(" ", strip=True))
            if title:
                return title[:200]

        for attr in ("aria-label", "data-title", "data-heading"):
            value = container.get(attr)
            if isinstance(value, str) and value.strip():
                return self._clean_text(value)[:200]
        return None

    def _find_product_cards(self, container: Tag) -> List[Tag]:
        strong = [
            element
            for element in container.find_all(True)
            if self._looks_like_product_card(element)
        ]
        strong = self._remove_nested_cards(strong)
        if len(strong) >= 2:
            return self._deduplicate_products(strong)

        best_group: List[Tag] = []
        parents: Iterable[Tag] = [container]
        parents = list(parents) + list(container.find_all(["div", "ul", "ol"]))
        for parent in parents:
            groups: Dict[Tuple[str, Tuple[str, ...]], List[Tag]] = {}
            for child in parent.find_all(recursive=False):
                if not isinstance(child, Tag) or self._product_score(child) < 2:
                    continue
                signature = self._repeated_signature(child)
                groups.setdefault(signature, []).append(child)
            for group in groups.values():
                if len(group) > len(best_group):
                    best_group = group

        if len(best_group) >= 2:
            return self._deduplicate_products(best_group)
        return self._deduplicate_products(strong)

    def _looks_like_product_card(self, element: Tag) -> bool:
        attrs = self._attribute_text(element)
        if self._PRODUCT_CARD_MARKER.search(attrs):
            return True
        classes = element.get("class") or []
        if isinstance(classes, str):
            classes = classes.split()
        if element.name in {"li", "article"} and "product" in {
            str(item).lower() for item in classes
        }:
            return True
        itemtype = str(element.get("itemtype") or "")
        if "schema.org/product" in itemtype.lower():
            return True
        return any(
            element.has_attr(attr)
            for attr in ("data-product-id", "data-productid", "data-sku", "data-item-id")
        )

    def _product_score(self, element: Tag) -> int:
        score = 0
        if self._looks_like_product_card(element):
            score += 3
        if element.find("img"):
            score += 1
        if element.find("a", href=re.compile(r"/(?:products?|p|dp|item)/", re.IGNORECASE)):
            score += 1
        if self._PRICE_PATTERN.search(element.get_text(" ", strip=True)):
            score += 1
        if element.find(attrs={"itemprop": re.compile(r"^(?:name|price|offers)$", re.IGNORECASE)}):
            score += 1
        button = element.find(["button", "a"], string=re.compile(r"add|buy|shop", re.IGNORECASE))
        if button:
            score += 1
        return score

    def _remove_nested_cards(self, cards: List[Tag]) -> List[Tag]:
        card_ids: Set[int] = {id(card) for card in cards}
        result: List[Tag] = []
        for card in cards:
            if any(id(parent) in card_ids for parent in card.parents if isinstance(parent, Tag)):
                continue
            result.append(card)
        return result

    def _deduplicate_products(self, cards: List[Tag]) -> List[Tag]:
        unique_cards: List[Tag] = []
        identities: Set[str] = set()
        for card in cards:
            identity = self._product_identity(card)
            if identity and identity in identities:
                continue
            if identity:
                identities.add(identity)
            unique_cards.append(card)
        return unique_cards

    def _product_identity(self, card: Tag) -> str:
        for attr in ("data-product-id", "data-productid", "data-sku", "data-item-id"):
            if card.get(attr):
                return "{}:{}".format(attr, card.get(attr))
        link = card.find("a", href=True)
        if link and link.get("href"):
            href = str(link.get("href")).split("?")[0]
            if href not in {"#", "/"} and not href.lower().startswith("javascript:"):
                return "href:{}".format(href)
        label = card.get("aria-label")
        if label:
            return "label:{}".format(self._clean_text(str(label)).lower())
        text = self._clean_text(card.get_text(" ", strip=True))[:120].lower()
        return "text:{}".format(text) if text else ""

    def _repeated_signature(self, element: Tag) -> Tuple[str, Tuple[str, ...]]:
        classes = element.get("class") or []
        if isinstance(classes, str):
            classes = classes.split()
        ignored = {"active", "first", "last", "selected", "visible", "hidden"}
        normalised = tuple(sorted(str(item).lower() for item in classes if item not in ignored))
        return element.name, normalised

    def _add_candidate(
        self,
        candidates: List[_Candidate],
        container: Tag,
        title: Optional[str],
        cards: List[Tag],
        detected_type: str,
        evidence: List[str],
        strength: int,
    ) -> None:
        cards = self._deduplicate_products(cards)
        if not cards:
            return
        if len(cards) < 2 and detected_type == UNKNOWN:
            return
        if len(cards) >= 2:
            evidence.append("repeated product cards detected")
        candidates.append(
            _Candidate(
                container=container,
                title=title,
                cards=cards,
                detected_type=detected_type,
                evidence=evidence,
                strength=strength,
            )
        )

    def _deduplicate(self, candidates: List[_Candidate]) -> List[_Candidate]:
        ranked = sorted(
            candidates,
            key=lambda item: (
                item.detected_type != UNKNOWN,
                item.strength,
                bool(item.title),
                -len(list(item.container.parents)),
            ),
            reverse=True,
        )
        kept: List[_Candidate] = []
        for candidate in ranked:
            candidate_cards = {id(card) for card in candidate.cards}
            duplicate = False
            for existing in kept:
                existing_cards = {id(card) for card in existing.cards}
                overlap = len(candidate_cards & existing_cards)
                smaller = min(len(candidate_cards), len(existing_cards))
                if candidate.container is existing.container or (smaller and overlap / smaller >= 0.8):
                    existing.evidence = self._unique(existing.evidence + candidate.evidence)
                    if not existing.title and candidate.title:
                        existing.title = candidate.title
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept

    def _confidence(self, candidate: _Candidate) -> float:
        if candidate.detected_type != UNKNOWN:
            score = 0.72
        else:
            score = 0.35
        attrs = self._attribute_text(candidate.container)
        if self._RECOMMENDATION_MARKER.search(attrs):
            score += 0.10
        if self._CAROUSEL_MARKER.search(attrs):
            score += 0.07
        if len(candidate.cards) >= 2:
            score += 0.08
        if len(candidate.cards) >= 4:
            score += 0.03
        if candidate.title:
            score += 0.04
        return round(min(score, 0.99), 2)

    def _has_component_marker(self, element: Tag) -> bool:
        attrs = self._attribute_text(element)
        return bool(
            self._RECOMMENDATION_MARKER.search(attrs)
            or self._CAROUSEL_MARKER.search(attrs)
        )

    def _inside_ignored_region(self, element: Tag) -> bool:
        return element.find_parent(["nav", "header", "footer"]) is not None

    @staticmethod
    def _attribute_text(element: Tag) -> str:
        parts: List[str] = []
        for name, value in element.attrs.items():
            if name not in {"id", "class", "role", "aria-label"} and not name.startswith("data-"):
                continue
            if isinstance(value, (list, tuple)):
                parts.extend(str(item) for item in value)
            else:
                parts.append(str(value))
        return RecommendationPlacementDetector._normalise_for_matching(" ".join(parts))

    @staticmethod
    def _normalise_for_matching(text: str) -> str:
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")
        text = re.sub(r"[_-]+", " ", text)
        return re.sub(r"\s+", " ", text).strip().lower()

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def _short(text: str, limit: int = 160) -> str:
        text = RecommendationPlacementDetector._clean_text(text)
        return text if len(text) <= limit else text[: limit - 1] + "…"

    @staticmethod
    def _element_signature(element: Tag) -> str:
        signature = element.name
        if element.get("id"):
            signature += "#{}".format(element.get("id"))
        classes = element.get("class") or []
        if isinstance(classes, str):
            classes = classes.split()
        if classes:
            signature += ".{}".format(".".join(str(item) for item in classes[:4]))
        return signature

    @staticmethod
    def _unique(values: Iterable[str]) -> List[str]:
        result: List[str] = []
        seen: Set[str] = set()
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result


def detect_recommendation_placements(
    html: Union[str, BeautifulSoup, Tag],
    page_url: str,
    page_type: Optional[str] = None,
) -> List[RecommendationPlacement]:
    """Convenience wrapper for one-off placement detection."""

    return RecommendationPlacementDetector().detect(html, page_url, page_type)


__all__ = [
    "RecommendationPlacement",
    "RecommendationPlacementDetector",
    "detect_recommendation_placements",
]
