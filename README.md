# BrightEdge Metadata Crawler

Python crawler that extracts page metadata, body text, page type, and topics. It can run as a FastAPI service, a single-URL CLI, or a text-file ingestion job.

## Features

- URL validation, redirects, HTML-only filtering
- Streaming 5 MB response limit
- Title, description, body, page type, topic extraction
- Async batch crawling with `max_concurrency`
- JSONL metadata export plus separate body text files
- Text-file ingestion for monthly URL inputs

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## FastAPI Demo

```bash
uvicorn app.main:app --reload
```

Docs:

```text
http://127.0.0.1:8000/docs
```

Single crawl:

```bash
curl -s -X POST "http://127.0.0.1:8000/crawl" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.fanatics.com/soccer-national-teams/south-korea-national-team/jerseys/-south-korea-national-team-nike-2026-home-match-authentic-jersey-red/o-2523+t-81731371+d-31771267+f-421220235+z-9-2532509534?sku=211158183"}' | jq
```

Batch crawl with sample URLs:

```bash
curl -s -X POST "http://127.0.0.1:8000/crawl/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://www.fanatics.com/soccer-national-teams/south-korea-national-team/jerseys/-south-korea-national-team-nike-2026-home-match-authentic-jersey-red/o-2523+t-81731371+d-31771267+f-421220235+z-9-2532509534?sku=211158183",
      "https://www.bbc.com/news/articles/clye4ky2yzpo",
      "https://www.cnn.com/2025/09/23/tech/google-study-90-percent-tech-jobs-ai",
      "https://www.nike.com/t/usmnt-2026-stadium-home-mens-dri-fit-soccer-replica-jersey-udZjfuxa/IB5339-133"
    ],
    "max_concurrency": 2
  }' | jq
```

Batch crawl and save output:

```bash
curl -s -X POST "http://127.0.0.1:8000/crawl/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://www.bbc.com/news/articles/clye4ky2yzpo",
      "https://www.nike.com/t/usmnt-2026-stadium-home-mens-dri-fit-soccer-replica-jersey-udZjfuxa/IB5339-133"
    ],
    "max_concurrency": 2,
    "save_results": true,
    "year_month": "2026-07"
  }' | jq
```

## CLI Demo

Single URL:

```bash
python -m app.crawler "https://www.bbc.com/news/articles/clye4ky2yzpo" --pretty
```

Text-file ingestion:

```bash
python -m app.ingest_file sample_inputs/urls_2026_07.txt \
  --year-month 2026-07 \
  --max-concurrency 5 \
  --pretty
```

Output:

```text
crawl_output/2026-07/metadata.jsonl
crawl_output/2026-07/content/*.txt
```

## Output Schema

API result fields:

- `original_url`, `final_url`, `status_code`, `content_type`
- `title`, `description`, `body`
- `page_type`, `classification_method`, `classification_confidence`
- `classification_scores`, `topics`

Saved JSONL fields:

- `year_month`, `original_url`, `final_url`, `domain`, `crawl_status`
- `status_code`, `content_type`, `title`, `description`, `page_type`, `topics`
- `body_text_path`, `body_length`, `body_hash`, `error_message`

## Assignment Docs

- `scale_design.txt`: Part 2 scale/storage/SLO/monitoring design
- `poc_release_plan.txt`: Part 3 POC plan, blockers, estimates, release criteria

## Notes

- `max_concurrency` controls how many URLs run at the same time.
- `/crawl/batch` is capped at 50 URLs for local demo safety.
- Some sites return `403 Forbidden`; the crawler records this as a per-URL failure.
