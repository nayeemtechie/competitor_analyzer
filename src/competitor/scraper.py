"""competitor.scraper
=====================

Utilities for scraping competitor websites.

This module intentionally favours reliability over raw scraping
performance.  The previous version of this file was truncated which left
syntax errors and missing implementations that prevented the application
from even importing the module.  The implementation below restores a
coherent, fully type annotated scraper that can be imported safely while
retaining the public surface used throughout the project.

The design is centred around two layers:

``WebScraper``
    Handles low level HTTP fetching, caching, robots.txt verification and
    HTML parsing into :class:`ScrapingResult` objects.

``CompetitorScraper``
    Builds on ``WebScraper`` to provide higher level aggregates (page
    summaries, detected technologies, inferred content themes, etc.)
    that the rest of the application expects when analysing
    competitors.

Both layers expose async context managers so they can be used with
``async with`` blocks which guarantees that HTTP sessions are closed
properly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
from aiohttp import ClientError, ClientSession, ClientTimeout
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ScrapingResult:
    """Represents the outcome of scraping a single URL."""

    url: str
    content: Optional[str] = None
    title: Optional[str] = None
    meta_description: Optional[str] = None
    meta_keywords: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    status_code: Optional[int] = None
    load_time: Optional[float] = None
    technologies: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    scripts: List[str] = field(default_factory=list)
    stylesheets: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    scraped_at: datetime = field(default_factory=datetime.utcnow)
    page_type: Optional[str] = None
    word_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result for caching/JSON output."""

        return {
            "url": self.url,
            "content": self.content,
            "title": self.title,
            "meta_description": self.meta_description,
            "meta_keywords": list(self.meta_keywords),
            "headers": dict(self.headers),
            "status_code": self.status_code,
            "load_time": self.load_time,
            "technologies": list(self.technologies),
            "links": list(self.links),
            "images": list(self.images),
            "scripts": list(self.scripts),
            "stylesheets": list(self.stylesheets),
            "success": self.success,
            "error": self.error,
            "scraped_at": self.scraped_at.isoformat(),
            "page_type": self.page_type,
            "word_count": self.word_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScrapingResult":
        """Reconstruct a :class:`ScrapingResult` from cached JSON."""

        kwargs = dict(data)
        if "scraped_at" in kwargs and isinstance(kwargs["scraped_at"], str):
            kwargs["scraped_at"] = datetime.fromisoformat(kwargs["scraped_at"])
        return cls(**kwargs)


@dataclass
class WebsiteScrapeSummary:
    """Aggregated information about a competitor website."""

    competitor: str
    base_url: str
    scraped_at: datetime = field(default_factory=datetime.utcnow)
    key_pages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pages_analyzed: List[str] = field(default_factory=list)
    technology_stack: List[str] = field(default_factory=list)
    content_themes: List[str] = field(default_factory=list)
    case_studies: List[Dict[str, Any]] = field(default_factory=list)
    raw_pages: List[ScrapingResult] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON serialisable representation."""

        return {
            "competitor": self.competitor,
            "base_url": self.base_url,
            "scraped_at": self.scraped_at.isoformat(),
            "key_pages": self.key_pages,
            "pages_analyzed": self.pages_analyzed,
            "technology_stack": self.technology_stack,
            "content_themes": self.content_themes,
            "case_studies": self.case_studies,
            "stats": self.stats,
            "raw_pages": [page.to_dict() for page in self.raw_pages],
        }


# ---------------------------------------------------------------------------
# Support utilities
# ---------------------------------------------------------------------------


@dataclass
class RateLimiter:
    """Simple asynchronous rate limiter."""

    requests_per_second: float = 1.0
    burst_limit: int = 5
    _request_times: deque = field(default_factory=deque)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self) -> None:
        """Wait until a new request can be made."""

        async with self._lock:
            now = time.time()

            # Drop entries older than one second to enforce burst window
            while self._request_times and now - self._request_times[0] > 1.0:
                self._request_times.popleft()

            # Respect burst limits
            if self.burst_limit > 0:
                while len(self._request_times) >= self.burst_limit:
                    oldest = self._request_times[0]
                    wait_time = 1.0 - (now - oldest)
                    if wait_time <= 0:
                        self._request_times.popleft()
                        break
                    await asyncio.sleep(wait_time)
                    now = time.time()
                    while self._request_times and now - self._request_times[0] > 1.0:
                        self._request_times.popleft()

            # Respect per-second rate limit
            if self.requests_per_second > 0 and self._request_times:
                min_interval = 1.0 / self.requests_per_second
                elapsed = now - self._request_times[-1]
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)

            self._request_times.append(time.time())


class CacheManager:
    """Persist scraping results to disk with a simple TTL."""

    def __init__(self, cache_dir: Union[str, Path] = "cache/web", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def _path_for(self, url: str) -> Path:
        key = hashlib.md5(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    async def get(self, url: str) -> Optional[ScrapingResult]:
        path = self._path_for(url)
        if not path.exists():
            return None

        try:
            raw_payload = await asyncio.to_thread(path.read_text, encoding="utf-8")
            payload = json.loads(raw_payload)

            scraped_at = datetime.fromisoformat(payload.get("scraped_at"))
            if datetime.utcnow() - scraped_at > self.ttl:
                # Expired cache entry
                await asyncio.to_thread(path.unlink, missing_ok=True)
                return None

            return ScrapingResult.from_dict(payload)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Failed to load cache for %s: %s", url, exc)
            return None

    async def set(self, url: str, result: ScrapingResult) -> None:
        path = self._path_for(url)
        try:
            payload = json.dumps(result.to_dict(), indent=2)
            await asyncio.to_thread(path.write_text, payload, encoding="utf-8")
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Failed to write cache for %s: %s", url, exc)

    async def clear_expired(self) -> None:
        for item in self.cache_dir.glob("*.json"):
            try:
                raw_payload = await asyncio.to_thread(item.read_text, encoding="utf-8")
                payload = json.loads(raw_payload)
                scraped_at = datetime.fromisoformat(payload.get("scraped_at"))
                if datetime.utcnow() - scraped_at > self.ttl:
                    await asyncio.to_thread(item.unlink, missing_ok=True)
            except Exception:  # pragma: no cover - defensive guard
                await asyncio.to_thread(item.unlink, missing_ok=True)

    async def clear_all(self) -> None:
        for item in self.cache_dir.glob("*.json"):
            await asyncio.to_thread(item.unlink, missing_ok=True)


class RobotsChecker:
    """Caches robots.txt lookups for polite scraping."""

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[RobotFileParser, float]] = {}
        self._ttl = 3600  # seconds

    async def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        now = time.time()
        parser: Optional[RobotFileParser] = None
        cached = self._cache.get(robots_url)
        if cached and now - cached[1] < self._ttl:
            parser = cached[0]
        else:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(robots_url, timeout=10) as response:
                        if response.status == 200:
                            content = await response.text()
                            parser = RobotFileParser()
                            parser.parse(content.splitlines())
                        else:
                            parser = RobotFileParser()
                            parser.parse("")
            except Exception:
                parser = RobotFileParser()
                parser.parse("")

            self._cache[robots_url] = (parser, now)

        return parser.can_fetch(user_agent, url)


class TechnologyDetector:
    """Very lightweight technology detector based on HTML signatures."""

    TECH_SIGNATURES: Dict[str, Iterable[str]] = {
        "react": [r"react", r"data-reactroot"],
        "vue": [r"vue", r"v-\w+"],
        "angular": [r"angular", r"ng-[a-z]+"],
        "next.js": [r"next[\.-]js"],
        "nuxt": [r"nuxt"],
        "tailwind": [r"tailwind"],
        "bootstrap": [r"bootstrap"],
        "google analytics": [r"google-analytics", r"gtag\("],
        "segment": [r"segment\.(?:io|com)"],
        "hotjar": [r"hotjar"],
        "hubspot": [r"hubspot"],
    }

    def detect(self, html: str, headers: Dict[str, str]) -> List[str]:
        detected: set[str] = set()
        html_lower = html.lower()

        for tech, patterns in self.TECH_SIGNATURES.items():
            if any(re.search(pattern, html_lower, re.IGNORECASE) for pattern in patterns):
                detected.add(tech)

        server_header = headers.get("server", "").lower()
        if "nginx" in server_header:
            detected.add("nginx")
        if "apache" in server_header:
            detected.add("apache")
        if "cloudflare" in server_header:
            detected.add("cloudflare")

        powered = headers.get("x-powered-by", "").lower()
        if powered:
            detected.add(powered)

        return sorted(detected)


class ContentExtractor:
    """Transforms raw HTML into :class:`ScrapingResult` objects."""

    def __init__(self) -> None:
        self._tech_detector = TechnologyDetector()

    def extract(self, html: str, url: str, headers: Dict[str, str]) -> ScrapingResult:
        soup = BeautifulSoup(html, "html.parser")

        result = ScrapingResult(url=url, headers=headers)
        result.title = self._extract_title(soup)
        result.meta_description = self._extract_meta_description(soup)
        result.meta_keywords = self._extract_meta_keywords(soup)
        result.content = self._extract_main_content(soup)
        result.word_count = len(result.content.split()) if result.content else 0
        result.links = self._extract_links(soup, url)
        result.images = self._extract_images(soup, url)
        result.scripts = self._extract_scripts(soup, url)
        result.stylesheets = self._extract_stylesheets(soup, url)
        result.technologies = self._tech_detector.detect(html, headers)
        result.page_type = self._determine_page_type(url, result.title, result.content)
        result.success = True
        return result

    # --- helper extraction methods -------------------------------------------------

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        heading = soup.find("h1")
        return heading.get_text(strip=True) if heading else None

    def _extract_meta_description(self, soup: BeautifulSoup) -> Optional[str]:
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            return meta["content"].strip()
        og = soup.find("meta", attrs={"property": "og:description"})
        if og and og.get("content"):
            return og["content"].strip()
        return None

    def _extract_meta_keywords(self, soup: BeautifulSoup) -> List[str]:
        meta = soup.find("meta", attrs={"name": "keywords"})
        if meta and meta.get("content"):
            return [keyword.strip() for keyword in meta["content"].split(",") if keyword.strip()]
        return []

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        main = soup.find("main") or soup.find("article")
        if main:
            text = main.get_text(" ", strip=True)
        else:
            text = soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:15000]

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        links: List[str] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if href.startswith("mailto:") or href.startswith("tel:"):
                continue
            if href.startswith("http://") or href.startswith("https://"):
                links.append(href)
            elif href.startswith("/"):
                links.append(urljoin(base_url, href))
        return links[:100]

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        images: List[str] = []
        for img in soup.find_all("img", src=True):
            src = img["src"].strip()
            if src.startswith("http://") or src.startswith("https://"):
                images.append(src)
            elif src.startswith("/"):
                images.append(urljoin(base_url, src))
        return images[:50]

    def _extract_scripts(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        scripts: List[str] = []
        for script in soup.find_all("script", src=True):
            src = script["src"].strip()
            if src.startswith("http://") or src.startswith("https://"):
                scripts.append(src)
            elif src.startswith("/"):
                scripts.append(urljoin(base_url, src))
        return scripts[:50]

    def _extract_stylesheets(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        stylesheets: List[str] = []
        for link in soup.find_all("link", rel=lambda value: value and "stylesheet" in value, href=True):
            href = link["href"].strip()
            if href.startswith("http://") or href.startswith("https://"):
                stylesheets.append(href)
            elif href.startswith("/"):
                stylesheets.append(urljoin(base_url, href))
        return stylesheets[:50]

    def _determine_page_type(
        self, url: str, title: Optional[str], content: Optional[str]
    ) -> Optional[str]:
        url_lower = url.lower()
        if any(token in url_lower for token in ["/pricing", "price", "plans"]):
            return "pricing"
        if any(token in url_lower for token in ["/about", "company", "team"]):
            return "about"
        if any(token in url_lower for token in ["/product", "/features"]):
            return "product"
        if any(token in url_lower for token in ["/customers", "case"]):
            return "case_studies"
        if any(token in url_lower for token in ["/blog", "news"]):
            return "blog"
        if title and "pricing" in title.lower():
            return "pricing"
        if not content:
            return None
        snippet = content[:400].lower()
        if "contact" in snippet:
            return "contact"
        if "careers" in snippet or "jobs" in snippet:
            return "careers"
        return None


# ---------------------------------------------------------------------------
# Web scraper implementation
# ---------------------------------------------------------------------------


def _extract_scraping_settings(config: Any) -> Dict[str, Any]:
    defaults = {
        "delay_between_requests": 2.0,
        "max_pages_per_site": 50,
        "timeout": 30,
        "concurrent_requests": 3,
        "rate_limit": 1.0,
        "user_agent": "CompetitorAnalysis Bot 1.0",
        "respect_robots_txt": True,
        "cache_dir": "cache/web",
        "cache_ttl_hours": 24,
        "target_pages": [
            {"path": "/", "name": "homepage", "priority": "high"},
            {"path": "/pricing", "name": "pricing", "priority": "high"},
            {"path": "/products", "name": "products", "priority": "medium"},
        ],
    }

    if config is None:
        return defaults

    if isinstance(config, dict):
        data = config.get("scraping", config)
    else:
        data = getattr(config, "scraping", config)
        if not isinstance(data, dict):
            data = {
                key: getattr(data, key)
                for key in defaults
                if hasattr(data, key)
            }

    settings = defaults.copy()
    for key, value in data.items():
        if value is not None:
            settings[key] = value
    return settings


class WebScraper:
    """High level asynchronous web scraper used across the project."""

    def __init__(self, config: Any = None) -> None:
        settings = _extract_scraping_settings(config)

        self.delay = float(settings["delay_between_requests"])
        self.max_pages = int(settings["max_pages_per_site"])
        self.timeout = int(settings["timeout"])
        self.user_agent = settings["user_agent"]
        self.respect_robots = bool(settings["respect_robots_txt"])

        self.rate_limiter = RateLimiter(
            requests_per_second=float(settings["rate_limit"]),
            burst_limit=int(settings["concurrent_requests"]),
        )
        self.cache = CacheManager(
            cache_dir=settings["cache_dir"],
            ttl_hours=int(settings["cache_ttl_hours"]),
        )
        self.robots_checker = RobotsChecker()
        self.content_extractor = ContentExtractor()

        self._session: Optional[ClientSession] = None
        self._settings = settings
        self._closed = False

        self.stats: Dict[str, int] = {
            "requests_made": 0,
            "cache_hits": 0,
            "errors": 0,
            "robots_blocked": 0,
        }

    # -- context manager ---------------------------------------------------------

    async def __aenter__(self) -> "WebScraper":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _ensure_session(self) -> None:
        if self._session is not None:
            return

        timeout = ClientTimeout(total=self.timeout)
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }

        connector = aiohttp.TCPConnector(limit=self._settings["concurrent_requests"], limit_per_host=3)
        self._session = ClientSession(timeout=timeout, headers=headers, connector=connector)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._closed = True

    # -- cache helpers -----------------------------------------------------------

    async def clear_cache(self, *, expired_only: bool = True) -> None:
        if expired_only:
            await self.cache.clear_expired()
        else:
            await self.cache.clear_all()

    async def health_check(self, url: str) -> Dict[str, Any]:
        await self._ensure_session()
        start = time.time()
        try:
            async with self._session.get(url, timeout=self.timeout) as response:
                load_time = time.time() - start
                return {
                    "url": url,
                    "status_code": response.status,
                    "load_time": load_time,
                    "available": response.status < 500,
                    "headers": dict(response.headers),
                    "checked_at": datetime.utcnow().isoformat(),
                }
        except Exception as exc:
            return {
                "url": url,
                "status_code": None,
                "load_time": time.time() - start,
                "available": False,
                "error": str(exc),
                "checked_at": datetime.utcnow().isoformat(),
            }

    # -- core scraping -----------------------------------------------------------

    async def scrape_url(self, url: str, *, force_refresh: bool = False) -> ScrapingResult:
        if not force_refresh:
            cached = await self.cache.get(url)
            if cached:
                self.stats["cache_hits"] += 1
                return cached

        if self.respect_robots and not await self.robots_checker.can_fetch(url, self.user_agent):
            self.stats["robots_blocked"] += 1
            return ScrapingResult(url=url, success=False, error="Blocked by robots.txt")

        await self.rate_limiter.acquire()
        await self._ensure_session()

        start = time.time()
        try:
            async with self._session.get(url, timeout=self.timeout) as response:
                load_time = time.time() - start
                self.stats["requests_made"] += 1

                if response.status >= 400:
                    self.stats["errors"] += 1
                    return ScrapingResult(
                        url=url,
                        status_code=response.status,
                        load_time=load_time,
                        success=False,
                        error=f"HTTP {response.status}",
                    )

                html = await response.text()
                headers = dict(response.headers)
                result = self.content_extractor.extract(html, url, headers)
                result.status_code = response.status
                result.load_time = load_time

                await self.cache.set(url, result)
                return result
        except asyncio.TimeoutError:
            self.stats["errors"] += 1
            return ScrapingResult(url=url, success=False, error="Request timeout")
        except ClientError as exc:
            self.stats["errors"] += 1
            return ScrapingResult(url=url, success=False, error=f"Client error: {exc}")
        except Exception as exc:  # pragma: no cover - defensive guard
            self.stats["errors"] += 1
            logger.debug("Unexpected error scraping %s: %s", url, exc)
            return ScrapingResult(url=url, success=False, error=str(exc))

    async def scrape_competitor_website(
        self, base_url: str, target_pages: Optional[List[Dict[str, Any]]] = None
    ) -> List[ScrapingResult]:
        if not target_pages:
            target_pages = list(self._settings["target_pages"])

        results: List[ScrapingResult] = []
        pages_scraped = 0

        for page in target_pages:
            if pages_scraped >= self.max_pages:
                break

            path = page.get("path", "/")
            if path.startswith("http://") or path.startswith("https://"):
                url = path
            else:
                url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))

            result = await self.scrape_url(url)
            if result.page_type is None:
                result.page_type = page.get("name")
            results.append(result)
            pages_scraped += 1

            if self.delay and pages_scraped < len(target_pages):
                await asyncio.sleep(self.delay)

        return results

    async def scrape_multiple_competitors(
        self, competitor_urls: List[str], target_pages: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, List[ScrapingResult]]:
        output: Dict[str, List[ScrapingResult]] = {}
        for url in competitor_urls:
            try:
                output[url] = await self.scrape_competitor_website(url, target_pages)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.debug("Failed to scrape %s: %s", url, exc)
                output[url] = [ScrapingResult(url=url, success=False, error=str(exc))]
        return output

    def get_stats(self) -> Dict[str, Any]:
        requests = max(1, self.stats["requests_made"])
        return {
            **self.stats,
            "cache_hit_rate": self.stats["cache_hits"] / requests * 100,
            "error_rate": self.stats["errors"] / requests * 100,
        }


# ---------------------------------------------------------------------------
# Competitor focused wrapper
# ---------------------------------------------------------------------------


class CompetitorScraper:
    """High level helper tailored for the competitor analyser."""

    def __init__(self, analysis_config: Optional[Dict[str, Any]] = None) -> None:
        self._analysis_config = analysis_config or {}
        self._settings = _extract_scraping_settings(self._analysis_config)
        self._default_pages = list(self._settings.get("target_pages", []))
        self._scraper = WebScraper(self._analysis_config)

    async def __aenter__(self) -> "CompetitorScraper":
        await self._scraper.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._scraper.__aexit__(exc_type, exc, tb)

    async def scrape_competitor_website(
        self,
        competitor_config: Union[str, Dict[str, Any]],
        target_pages: Optional[List[Dict[str, Any]]] = None,
    ) -> WebsiteScrapeSummary:
        if isinstance(competitor_config, str):
            competitor_name = competitor_config
            base_url = competitor_config
        else:
            competitor_name = competitor_config.get("name", "unknown")
            base_url = competitor_config.get("website") or competitor_config.get("url")
            if not base_url:
                raise ValueError("competitor_config must contain a 'website' entry")

        pages = target_pages or self._default_pages
        scrape_results = await self._scraper.scrape_competitor_website(base_url, pages)

        summary = WebsiteScrapeSummary(competitor=competitor_name, base_url=base_url)
        summary.raw_pages = scrape_results
        summary.stats = self._scraper.get_stats()

        technologies: set[str] = set()
        page_map: Dict[str, Dict[str, Any]] = {}

        for config, result in zip(pages, scrape_results):
            page_name = config.get("name") or result.page_type or result.url
            page_summary = self._summarise_page(result)
            page_map[page_name] = page_summary
            if result.success:
                summary.pages_analyzed.append(page_name)
            technologies.update(result.technologies)

        summary.key_pages = page_map
        summary.technology_stack = sorted(technologies)
        summary.case_studies = self._extract_case_studies(page_map)
        summary.content_themes = self._infer_content_themes(page_map)
        return summary

    async def scrape_multiple(self, competitors: List[Dict[str, Any]]) -> Dict[str, WebsiteScrapeSummary]:
        results: Dict[str, WebsiteScrapeSummary] = {}
        for competitor in competitors:
            summary = await self.scrape_competitor_website(competitor)
            results[competitor.get("name", competitor.get("website", "unknown"))] = summary
        return results

    # -- helper analytics ------------------------------------------------------

    def _summarise_page(self, result: ScrapingResult) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "url": result.url,
            "status_code": result.status_code,
            "load_time": result.load_time,
            "title": result.title,
            "meta_description": result.meta_description,
            "word_count": result.word_count,
            "technologies": list(result.technologies),
            "links": result.links[:20],
            "success": result.success,
            "error": result.error,
        }
        if result.content:
            summary["content_snippet"] = result.content[:800]
        if result.page_type:
            summary["page_type"] = result.page_type
        return summary

    def _extract_case_studies(self, pages: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        case_studies: List[Dict[str, Any]] = []
        for name, page in pages.items():
            if "case" not in name.lower() and "customer" not in name.lower():
                continue
            links = page.get("links", [])
            for link in links:
                if any(token in link.lower() for token in ["case", "customer", "success"]):
                    case_studies.append({"title": name.title(), "url": link})
        return case_studies

    def _infer_content_themes(self, pages: Dict[str, Dict[str, Any]]) -> List[str]:
        keywords = Counter()
        theme_keywords = {
            "ai": ["ai", "artificial intelligence", "machine learning"],
            "personalisation": ["personalization", "personalisation", "custom"],
            "search": ["search", "discovery"],
            "analytics": ["analytics", "insights", "metrics"],
            "ecommerce": ["commerce", "retail", "shop"],
        }

        for page in pages.values():
            snippet = (page.get("content_snippet") or "").lower()
            for theme, patterns in theme_keywords.items():
                if any(keyword in snippet for keyword in patterns):
                    keywords[theme] += 1

        return [theme for theme, _ in keywords.most_common(10)]


# ---------------------------------------------------------------------------
# Convenience helpers used in other modules/tests
# ---------------------------------------------------------------------------


async def quick_scrape(url: str, config: Any = None) -> ScrapingResult:
    async with WebScraper(config) as scraper:
        return await scraper.scrape_url(url)


async def batch_scrape(urls: List[str], config: Any = None) -> List[ScrapingResult]:
    async with WebScraper(config) as scraper:
        tasks = [scraper.scrape_url(url) for url in urls]
        return await asyncio.gather(*tasks)


def extract_domain(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def is_same_domain(url1: str, url2: str) -> bool:
    return extract_domain(url1) == extract_domain(url2)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r'[^\w\s.,!?;:\-()"]', "", text)
    return text


__all__ = [
    "ScrapingResult",
    "WebsiteScrapeSummary",
    "RateLimiter",
    "CacheManager",
    "RobotsChecker",
    "TechnologyDetector",
    "ContentExtractor",
    "WebScraper",
    "CompetitorScraper",
    "quick_scrape",
    "batch_scrape",
    "extract_domain",
    "is_same_domain",
    "clean_text",
]
