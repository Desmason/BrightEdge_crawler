import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUTPUT_ROOT = Path("crawl_output")


def normalize_year_month(year_month: str | None) -> str:
    """
    Use the provided crawl month or default to the current UTC month.
    """

    if year_month:
        return year_month

    return datetime.now(timezone.utc).strftime("%Y-%m")


def write_batch_export(
    results: list[dict[str, Any]],
    year_month: str | None = None,
) -> dict[str, str | int]:
    """
    Write batch crawl results as JSONL metadata plus separate body text files.

    JSONL is easy for future database loaders to process because each line is
    one independent record.
    """

    normalized_year_month = normalize_year_month(year_month)
    output_dir = OUTPUT_ROOT / normalized_year_month
    content_dir = output_dir / "content"
    metadata_path = output_dir / "metadata.jsonl"

    content_dir.mkdir(parents=True, exist_ok=True)

    records_written = 0

    with metadata_path.open("a", encoding="utf-8") as metadata_file:
        for item in results:
            record = build_metadata_record(
                item=item,
                year_month=normalized_year_month,
                content_dir=content_dir,
            )

            metadata_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            metadata_file.write("\n")
            records_written += 1

    return {
        "year_month": normalized_year_month,
        "metadata_path": str(metadata_path),
        "content_dir": str(content_dir),
        "records_written": records_written,
    }


def build_metadata_record(
    item: dict[str, Any],
    year_month: str,
    content_dir: Path,
) -> dict[str, Any]:
    exported_at = datetime.now(timezone.utc).isoformat()

    if not item.get("ok"):
        return {
            "year_month": year_month,
            "exported_at": exported_at,
            "original_url": item.get("url"),
            "crawl_status": "failed",
            "error_message": item.get("error"),
        }

    result = item["result"]
    body = result.get("body") or ""
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    body_path = content_dir / f"{body_hash}.txt"
    body_path.write_text(body, encoding="utf-8")

    return {
        "year_month": year_month,
        "exported_at": exported_at,
        "original_url": result.get("original_url"),
        "final_url": result.get("final_url"),
        "domain": extract_domain(result.get("final_url")),
        "crawl_status": "succeeded",
        "status_code": result.get("status_code"),
        "content_type": result.get("content_type"),
        "title": result.get("title"),
        "description": result.get("description"),
        "page_type": result.get("page_type"),
        "classification_method": result.get("classification_method"),
        "classification_confidence": result.get("classification_confidence"),
        "classification_scores": result.get("classification_scores"),
        "topics": result.get("topics"),
        "body_text_path": str(body_path),
        "body_length": len(body),
        "body_hash": body_hash,
    }


def extract_domain(url: str | None) -> str | None:
    if not url:
        return None

    from urllib.parse import urlparse

    return urlparse(url).netloc.lower() or None
