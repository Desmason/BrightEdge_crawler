import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

try:
    from .crawler import DEFAULT_HEADERS, REQUEST_TIMEOUT, CrawlError, crawl_url
    from .storage import write_batch_export
except ImportError:
    from crawler import DEFAULT_HEADERS, REQUEST_TIMEOUT, CrawlError, crawl_url
    from storage import write_batch_export


def read_urls(path: Path) -> list[str]:
    urls = []

    for line in path.read_text(encoding="utf-8").splitlines():
        url = line.strip()

        if not url or url.startswith("#"):
            continue

        urls.append(url)

    return urls


async def crawl_urls_from_file(
    urls: list[str],
    year_month: str,
    max_concurrency: int,
) -> dict:
    started_at = datetime.now(timezone.utc)
    semaphore = asyncio.Semaphore(max_concurrency)

    async with httpx.AsyncClient(
        headers=DEFAULT_HEADERS,
        follow_redirects=True,
        timeout=REQUEST_TIMEOUT,
    ) as client:

        async def crawl_one(url: str) -> dict:
            async with semaphore:
                try:
                    result = await crawl_url(
                        url=url,
                        client=client,
                    )
                    return {
                        "url": url,
                        "ok": True,
                        "result": result,
                    }

                except CrawlError as error:
                    return {
                        "url": url,
                        "ok": False,
                        "error": str(error),
                    }

        results = await asyncio.gather(
            *(
                crawl_one(url)
                for url in urls
            )
        )

    export = write_batch_export(
        results=results,
        year_month=year_month,
    )
    finished_at = datetime.now(timezone.utc)
    succeeded = sum(1 for item in results if item["ok"])
    failed = sum(1 for item in results if not item["ok"])
    input_count = len(urls)

    return {
        "year_month": year_month,
        "status": "completed",
        "input_count": input_count,
        "succeeded": succeeded,
        "failed": failed,
        "completion_percentage": round(
            ((succeeded + failed) / input_count) * 100,
            2,
        ),
        "max_concurrency": max_concurrency,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round(
            (finished_at - started_at).total_seconds(),
            3,
        ),
        "export": export,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl URLs from a text file and export JSONL metadata.",
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Text file containing one URL per line.",
    )
    parser.add_argument(
        "--year-month",
        required=True,
        help="Crawl partition in YYYY-MM format, such as 2026-07.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Number of URLs to crawl at the same time.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON summary.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be at least 1")

    urls = read_urls(args.input_file)

    if not urls:
        raise SystemExit("Input file did not contain any URLs")

    summary = asyncio.run(
        crawl_urls_from_file(
            urls=urls,
            year_month=args.year_month,
            max_concurrency=args.max_concurrency,
        )
    )

    print(
        json.dumps(
            summary,
            indent=2 if args.pretty else None,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
