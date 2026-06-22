import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, HttpUrl

try:
    from .crawler import DEFAULT_HEADERS, REQUEST_TIMEOUT, CrawlError, crawl_url
    from .storage import write_batch_export
except ImportError:
    from crawler import DEFAULT_HEADERS, REQUEST_TIMEOUT, CrawlError, crawl_url
    from storage import write_batch_export


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.http_client = httpx.AsyncClient(
        headers=DEFAULT_HEADERS,
        follow_redirects=True,
        timeout=REQUEST_TIMEOUT,
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        ),
    )

    try:
        yield
    finally:
        await app.state.http_client.aclose()


app = FastAPI(
    title="BrightEdge Metadata Crawler",
    description="Crawls a webpage and extracts its HTML metadata and body text.",
    version="1.0.0",
    lifespan=lifespan,
)


class CrawlRequest(BaseModel):
    url: HttpUrl


class BatchCrawlRequest(BaseModel):
    urls: list[HttpUrl] = Field(
        min_length=1,
        max_length=50,
    )
    max_concurrency: int = Field(
        default=5,
        ge=1,
        le=10,
    )
    save_results: bool = False
    year_month: str | None = Field(
        default=None,
        pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
    )


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "healthy",
    }


@app.post("/crawl")
async def crawl_page(
    request: Request,
    crawl_request: CrawlRequest,
) -> dict:
    try:
        return await crawl_url(
            url=str(crawl_request.url),
            client=request.app.state.http_client,
        )

    except CrawlError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.post("/crawl/batch")
async def crawl_batch(
    request: Request,
    batch_request: BatchCrawlRequest,
) -> dict:
    semaphore = asyncio.Semaphore(batch_request.max_concurrency)

    async def crawl_one(url: HttpUrl) -> dict:
        async with semaphore:
            try:
                result = await crawl_url(
                    url=str(url),
                    client=request.app.state.http_client,
                )
                return {
                    "url": str(url),
                    "ok": True,
                    "result": result,
                }

            except CrawlError as error:
                return {
                    "url": str(url),
                    "ok": False,
                    "error": str(error),
                }

    results = await asyncio.gather(
        *(
            crawl_one(url)
            for url in batch_request.urls
        )
    )

    response = {
        "count": len(results),
        "max_concurrency": batch_request.max_concurrency,
        "results": results,
    }

    if batch_request.save_results:
        response["export"] = write_batch_export(
            results=results,
            year_month=batch_request.year_month,
        )

    return response
