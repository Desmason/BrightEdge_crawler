import re

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer


CUSTOM_STOP_WORDS = ENGLISH_STOP_WORDS | {
    "account",
    "according",
    "advertisement",
    "app",
    "bag",
    "cart",
    "cnn",
    "com",
    "copyright",
    "download",
    "feedback",
    "finds",
    "footer",
    "help",
    "homepage",
    "https",
    "including",
    "just",
    "like",
    "login",
    "menu",
    "official",
    "page",
    "privacy",
    "search",
    "said",
    "says",
    "shipping",
    "sign",
    "site",
    "shop",
    "store",
    "terms",
    "www",
    "feature",
    "features",
    "video",
    "watch",
    "work",
    "use",
    "used",
    "using",
    "values",
}
SHORT_USEFUL_TOPICS = {
    "ai",
    "ml",
    "seo",
}
LOW_SIGNAL_PHRASE_WORDS = {
    "according",
    "including",
    "just",
    "like",
    "said",
    "says",
    "use",
    "used",
    "using",
}


def clean_text(text: str) -> str:
    """
    Normalize text before topic extraction.
    """

    text = text.lower()
    text = re.sub(r"\b(men|women|kid|child|children)'s\b", r"\1s", text)
    text = re.sub(r"\b[a-z0-9.-]+\.(?:com|org|net|edu|gov|io|co)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_heading_text(soup: BeautifulSoup) -> str:
    """
    Extract h1, h2, and h3 text.
    """

    headings = soup.find_all(["h1", "h2", "h3"])

    return " ".join(
        heading.get_text(" ", strip=True)
        for heading in headings
    )


def is_redundant_topic(
    candidate: str,
    selected_topics: list[str],
) -> bool:
    """
    Avoid returning many nearly identical topics.

    Example:
    toaster
    compact toaster
    cuisinart compact toaster
    """

    candidate_words = set(candidate.split())

    for selected in selected_topics:
        selected_words = set(selected.split())

        if candidate == selected:
            return True

        if candidate_words == selected_words:
            return True

        if candidate_words.issubset(selected_words):
            return True

        if (
            selected_words.issubset(candidate_words)
            and len(selected_words) >= 2
        ):
            return True

    return False


def is_useful_topic(topic: str) -> bool:
    """
    Remove navigation, domain, and low-information topics.
    """

    words = topic.split()

    if not words:
        return False

    if all(word in CUSTOM_STOP_WORDS for word in words):
        return False

    if any(word in {"com", "www", "http", "https"} for word in words):
        return False

    if any(word.isdigit() for word in words):
        return False

    if len(words) > 1 and any(
        word in LOW_SIGNAL_PHRASE_WORDS
        for word in words
    ):
        return False

    if (
        len(words) == 1
        and len(words[0]) < 4
        and words[0] not in SHORT_USEFUL_TOPICS
    ):
        return False

    return True


def appears_as_phrase(
    topic: str,
    documents: list[str],
) -> bool:
    """
    Verify that multi-word topics appeared next to each other in the source.
    """

    pattern = re.compile(rf"\b{re.escape(topic)}\b")

    return any(
        pattern.search(document)
        for document in documents
    )


def extract_topics_tfidf(
    soup: BeautifulSoup,
    title: str | None,
    description: str | None,
    body: str,
    limit: int = 10,
) -> list[dict]:
    """
    Extract relevant topics using weighted TF-IDF scores.

    Title, description, headings, and body are treated as separate
    documents and receive different importance weights.
    """

    heading_text = extract_heading_text(soup)

    sections = [
        (clean_text(title or ""), 4.0),
        (clean_text(description or ""), 3.0),
        (clean_text(heading_text), 3.5),
        (clean_text(body[:20000]), 1.0),
    ]

    sections = [
        (text, weight)
        for text, weight in sections
        if text
    ]

    if not sections:
        return []

    documents = [
        text
        for text, _weight in sections
    ]

    section_weights = [
        weight
        for _text, weight in sections
    ]

    vectorizer = TfidfVectorizer(
        stop_words=list(CUSTOM_STOP_WORDS),
        lowercase=True,
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
        sublinear_tf=True,
        token_pattern=r"(?u)\b(?:ai|ml|seo|[a-zA-Z][a-zA-Z0-9\-]{2,}|[0-9][a-zA-Z0-9\-]*)\b",
    )

    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()

    weighted_scores: dict[str, float] = {}

    for row_index, section_weight in enumerate(section_weights):
        row = matrix.getrow(row_index)

        for column_index, tfidf_score in zip(
            row.indices,
            row.data,
        ):
            topic = feature_names[column_index]

            weighted_scores[topic] = (
                weighted_scores.get(topic, 0.0)
                + float(tfidf_score) * section_weight
            )

    ranked_topics = sorted(
        weighted_scores.items(),
        key=lambda item: (
            item[1] * (1 + 0.2 * (len(item[0].split()) - 1)),
            len(item[0].split()),
        ),
        reverse=True,
    )

    selected_topic_names: list[str] = []
    results: list[dict] = []

    for topic, score in ranked_topics:
        word_count = len(topic.split())

        if not is_useful_topic(topic):
            continue

        if word_count > 1 and not appears_as_phrase(
            topic=topic,
            documents=documents,
        ):
            continue

        if word_count == 1 and score < 1.0:
            continue

        if is_redundant_topic(
            candidate=topic,
            selected_topics=selected_topic_names,
        ):
            continue

        selected_topic_names.append(topic)

        results.append(
            {
                "topic": topic,
                "score": round(score, 4),
            }
        )

        if len(results) >= limit:
            break

    return results
