import json
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CATEGORY_DESCRIPTIONS = {
    "product": (
        "product shopping price brand model specifications features reviews "
        "buy purchase add to cart shipping availability customer ratings "
        "product details warranty"
    ),
    "news_article": (
        "breaking news report journalist reporter publication date headline "
        "current events politics business technology world news investigation "
        "news story"
    ),
    "blog_article": (
        "blog article guide tutorial tips advice how to educational content "
        "personal experience opinion recommendations informational article"
    ),
    "category_page": (
        "category collection product listing search results filters sorting "
        "brands departments multiple products shopping items browse products"
    ),
}


@dataclass
class ClassificationResult:
    page_type: str
    method: str
    confidence: float
    similarities: Optional[dict[str, float]]


def collect_json_ld_types(data, collected_types: set[str]) -> None:
    """
    Recursively collect all Schema.org @type values from JSON-LD data.
    """

    if isinstance(data, dict):
        json_type = data.get("@type")

        if isinstance(json_type, str):
            collected_types.add(json_type.lower())

        elif isinstance(json_type, list):
            for item in json_type:
                if isinstance(item, str):
                    collected_types.add(item.lower())

        for value in data.values():
            collect_json_ld_types(value, collected_types)

    elif isinstance(data, list):
        for item in data:
            collect_json_ld_types(item, collected_types)


def extract_json_ld_types(soup: BeautifulSoup) -> set[str]:
    """
    Extract Schema.org @type values from JSON-LD script elements.
    """

    collected_types: set[str] = set()

    json_ld_scripts = soup.find_all(
        "script",
        attrs={"type": "application/ld+json"},
    )

    for script in json_ld_scripts:
        raw_json = script.string or script.get_text(strip=True)

        if not raw_json:
            continue

        try:
            parsed_data = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            continue

        collect_json_ld_types(parsed_data, collected_types)

    return collected_types


def classify_with_structured_signals(
    soup: BeautifulSoup,
    url: str,
) -> ClassificationResult | None:
    """
    Use strong HTML and structured-data signals first.

    Returns None when there is no sufficiently strong rule-based signal.
    """

    json_ld_types = extract_json_ld_types(soup)

    product_types = {
        "product",
        "individualproduct",
        "productgroup",
    }

    news_types = {
        "newsarticle",
        "reportagenewsarticle",
        "analysisnewsarticle",
    }

    blog_types = {
        "blogposting",
        "techarticle",
        "howto",
    }

    article_types = {
        "article",
        "socialmediaposting",
    }

    if json_ld_types & product_types:
        return ClassificationResult(
            page_type="product",
            method="json_ld",
            confidence=1.0,
            similarities=None,
        )

    if json_ld_types & news_types:
        return ClassificationResult(
            page_type="news_article",
            method="json_ld",
            confidence=1.0,
            similarities=None,
        )

    if json_ld_types & blog_types:
        return ClassificationResult(
            page_type="blog_article",
            method="json_ld",
            confidence=1.0,
            similarities=None,
        )

    if json_ld_types & article_types:
        return ClassificationResult(
            page_type="blog_article",
            method="json_ld",
            confidence=0.95,
            similarities=None,
        )

    og_type_tag = soup.find(
        "meta",
        attrs={"property": "og:type"},
    )

    if og_type_tag and og_type_tag.get("content"):
        og_type = og_type_tag["content"].strip().lower()

        if "product" in og_type:
            return ClassificationResult(
                page_type="product",
                method="open_graph",
                confidence=0.95,
                similarities=None,
            )

        if "article" in og_type:
            return ClassificationResult(
                page_type="blog_article",
                method="open_graph",
                confidence=0.85,
                similarities=None,
            )

    parsed_url = urlparse(url)
    path = parsed_url.path.lower()
    normalized_path = path.strip("/")

    if normalized_path == "":
        return ClassificationResult(
            page_type="homepage",
            method="url_pattern",
            confidence=0.9,
            similarities=None,
        )

    product_path_patterns = [
        "/product/",
        "/products/",
        "/dp/",
        "/item/",
    ]

    if any(pattern in path for pattern in product_path_patterns):
        return ClassificationResult(
            page_type="product",
            method="url_pattern",
            confidence=0.85,
            similarities=None,
        )

    category_path_patterns = [
        "/category/",
        "/categories/",
        "/collections/",
        "/search/",
        "/shop/",
    ]

    if any(pattern in path for pattern in category_path_patterns):
        return ClassificationResult(
            page_type="category_page",
            method="url_pattern",
            confidence=0.8,
            similarities=None,
        )

    page_text = soup.get_text(" ", strip=True).lower()

    if "add to cart" in page_text or "buy now" in page_text:
        return ClassificationResult(
            page_type="product",
            method="html_signal",
            confidence=0.8,
            similarities=None,
        )

    return None


def extract_classification_text(
    soup: BeautifulSoup,
    title: str | None,
    description: str | None,
    body: str,
) -> str:
    """
    Combine webpage sections into one weighted classification document.
    """

    headings = " ".join(
        heading.get_text(" ", strip=True)
        for heading in soup.find_all(["h1", "h2", "h3"])
    )

    title_text = title or ""
    description_text = description or ""
    body_sample = body[:15000]

    return " ".join(
        [
            title_text,
            title_text,
            title_text,
            description_text,
            description_text,
            headings,
            headings,
            body_sample,
        ]
    )


def classify_with_cosine_similarity(
    soup: BeautifulSoup,
    title: str | None,
    description: str | None,
    body: str,
    minimum_confidence: float = 0.05,
) -> ClassificationResult:
    """
    Use TF-IDF and cosine similarity when strong structured signals
    are unavailable.
    """

    page_text = extract_classification_text(
        soup=soup,
        title=title,
        description=description,
        body=body,
    )

    if not page_text.strip():
        return ClassificationResult(
            page_type="other",
            method="tfidf_cosine_similarity",
            confidence=0.0,
            similarities=None,
        )

    category_names = list(CATEGORY_DESCRIPTIONS.keys())
    category_documents = list(CATEGORY_DESCRIPTIONS.values())

    documents = [page_text] + category_documents

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )

    try:
        document_vectors = vectorizer.fit_transform(documents)
    except ValueError:
        return ClassificationResult(
            page_type="other",
            method="tfidf_cosine_similarity",
            confidence=0.0,
            similarities=None,
        )

    page_vector = document_vectors[0]
    category_vectors = document_vectors[1:]

    raw_scores = cosine_similarity(
        page_vector,
        category_vectors,
    )[0]

    similarities = {
        category_name: round(float(score), 4)
        for category_name, score in zip(category_names, raw_scores)
    }

    best_index = int(raw_scores.argmax())
    best_category = category_names[best_index]
    best_score = float(raw_scores[best_index])

    if best_score < minimum_confidence:
        best_category = "other"

    return ClassificationResult(
        page_type=best_category,
        method="tfidf_cosine_similarity",
        confidence=round(best_score, 4),
        similarities=similarities,
    )


def classify_page(
    soup: BeautifulSoup,
    url: str,
    title: str | None,
    description: str | None,
    body: str,
) -> ClassificationResult:
    """
    Hybrid classifier.

    First uses strong structured signals. If none are available,
    falls back to TF-IDF cosine similarity.
    """

    structured_result = classify_with_structured_signals(
        soup=soup,
        url=url,
    )

    if structured_result is not None:
        return structured_result

    return classify_with_cosine_similarity(
        soup=soup,
        title=title,
        description=description,
        body=body,
    )
