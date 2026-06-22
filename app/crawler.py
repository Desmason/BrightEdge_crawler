from dataclasses import asdict, dataclass
from typing import Optional
from urllib.parse import urlparse

try:
    from .classifier import classify_page
    from .topic_extractor import extract_topics_tfidf
except ImportError:
    from classifier import classify_page
    from topic_extractor import extract_topics_tfidf

import httpx
from bs4 import BeautifulSoup


MAX_RESPONSE_SIZE = 5 * 1024 * 1024  # 5 MB
REQUEST_TIMEOUT = httpx.Timeout(10.0)
UNWANTED_TEXT_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "iframe",
}
DEFAULT_HEADERS = {
    "User-Agent": (
        "BrightEdgeCandidateCrawler/1.0 "
        "(Educational software engineering assignment)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class CrawlResult:
    original_url: str
    final_url: str
    status_code: int
    title: Optional[str]
    description: Optional[str]
    body: str
    content_type: Optional[str]
    page_type: str
    classification_method: str
    classification_confidence: float
    classification_scores: Optional[dict[str, float]]
    topics: list[dict]


class CrawlError(Exception):
    """Custom exception for crawler-related failures."""


def validate_url(url: str) -> None:
    """
    Verify that the input is a valid HTTP or HTTPS URL.
    """

    parsed_url = urlparse(url)

    if parsed_url.scheme not in {"http", "https"}:
        raise CrawlError("URL must begin with http:// or https://")

    if not parsed_url.netloc:
        raise CrawlError("URL must contain a valid domain name")


def extract_meta_description(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract the page description.

    First checks the normal HTML meta description.
    If that is missing, it checks Open Graph metadata.
    """

    description_tag = soup.find(
        "meta",
        attrs={"name": lambda value: value and value.lower() == "description"},
    )

    if description_tag and description_tag.get("content"):
        return description_tag["content"].strip()

    og_description = soup.find(
        "meta",
        attrs={"property": "og:description"},
    )

    if og_description and og_description.get("content"):
        return og_description["content"].strip()

    return None


def extract_body_text(soup: BeautifulSoup) -> str:
    """
    Return clean visible text without mutating the parsed document.
    """

    root = soup.body or soup
    visible_strings = []

    for text_node in root.find_all(string=True):
        if text_node.parent and text_node.parent.name in UNWANTED_TEXT_TAGS:
            continue

        text = text_node.strip()

        if text:
            visible_strings.append(text)

    # Replace repeated whitespace with a single space.
    return " ".join(" ".join(visible_strings).split())


async def fetch_html(
    client: httpx.AsyncClient,
    url: str,
) -> tuple[bytes, str, int, str]:
    """
    Download HTML while enforcing the response size limit during streaming.
    """

    try:
        async with client.stream("GET", url) as response:
            if response.status_code >= 400:
                raise CrawlError(
                    f"The website returned HTTP status {response.status_code}"
                )

            content_type = response.headers.get("content-type", "")

            if "text/html" not in content_type.lower():
                raise CrawlError(
                    f"Unsupported content type: {content_type or 'unknown'}"
                )

            content = bytearray()

            async for chunk in response.aiter_bytes():
                content.extend(chunk)

                if len(content) > MAX_RESPONSE_SIZE:
                    raise CrawlError("The webpage exceeded the 5 MB size limit")

            return (
                bytes(content),
                str(response.url),
                response.status_code,
                content_type,
            )

    except httpx.TimeoutException as exc:
        raise CrawlError("The website took too long to respond") from exc

    except httpx.RequestError as exc:
        raise CrawlError(f"Could not retrieve the URL: {exc}") from exc


async def crawl_url(
    url: str,
    client: httpx.AsyncClient,
) -> dict:
    """
    Download and parse one webpage.
    """

    validate_url(url)

    content, final_url, status_code, content_type = await fetch_html(
        client=client,
        url=url,
    )

    soup = BeautifulSoup(content, "lxml")

    title = None

    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    description = extract_meta_description(soup)
    body = extract_body_text(soup)

    classification = classify_page(
        soup=soup,
        url=final_url,
        title=title,
        description=description,
        body=body,
    )

    topics = extract_topics_tfidf(
        soup=soup,
        title=title,
        description=description,
        body=body,
    )

    result = CrawlResult(
        original_url=url,
        final_url=final_url,
        status_code=status_code,
        title=title,
        description=description,
        body=body,
        content_type=content_type,
        page_type=classification.page_type,
        classification_method=classification.method,
        classification_confidence=classification.confidence,
        classification_scores=classification.similarities,
        topics=topics,
    )

    return asdict(result)


if __name__ == "__main__":
    import argparse
    import asyncio
    import json

    parser = argparse.ArgumentParser(
        description="Crawl one URL and print extracted metadata as JSON.",
    )
    parser.add_argument(
        "url",
        help="HTTP or HTTPS URL to crawl.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output.",
    )
    args = parser.parse_args()

    async def run_cli() -> int:
        async with httpx.AsyncClient(
            headers=DEFAULT_HEADERS,
            follow_redirects=True,
            timeout=REQUEST_TIMEOUT,
        ) as cli_client:
            try:
                result = await crawl_url(
                    url=args.url,
                    client=cli_client,
                )
            except CrawlError as error:
                print(
                    json.dumps(
                        {
                            "error": str(error),
                        },
                        indent=2 if args.pretty else None,
                    )
                )
                return 1

        print(
            json.dumps(
                result,
                indent=2 if args.pretty else None,
            )
        )
        return 0

    raise SystemExit(asyncio.run(run_cli()))
