# Installation

library-query needs three things: a Python 3.11+ environment, a Qdrant
container, and an embeddings endpoint. Setup takes about five minutes.

## 1. Clone and sync

```bash
git clone https://github.com/marcosconci1/library-query
cd library-query
uv sync
```

## 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | What it does |
|---|---|
| `LIBRARY_ROOT` | Absolute path to the folder your documents live in |
| `QDRANT_URL` | Where Qdrant listens (default `http://127.0.0.1:6333`) |
| `EMBEDDING_BASE_URL` | OpenAI-compatible embeddings endpoint |
| `EMBEDDING_MODEL` | Model id (default `openai/text-embedding-3-small`) |
| `EMBEDDING_DIM` | Vector size (default `1536`) |
| `OPENROUTER_API_KEY` | Key for the embeddings endpoint |

Any OpenAI-compatible endpoint works: OpenRouter, a local vLLM or Ollama
server, or a vendor directly. The tool never talks to anything except the
endpoint you configure and the Qdrant port you point at. For OpenRouter,
create an API key at [openrouter.ai/keys](https://openrouter.ai/keys).

## 3. Start Qdrant

```bash
docker compose -f docker/docker-compose.yml up -d
```

The container listens on loopback only, so the vector store is reachable
from your machine and from nothing else.

## 4. Index and ask

```bash
uv run library-query index
uv run library-query query "your question"
```

The first index pass extracts and embeds every supported file. Later
passes read a content-hash ledger and touch only files that changed, so
re-indexing after adding a few PDFs takes seconds, not minutes.

## Commands

### Locator

| Command | Effect |
|---|---|
| `index` | Sync the index with the folder (incremental) |
| `query "<question>"` | Ask; prints ranked places |
| `find "<text>"` | Literal string search: exact, not semantic |
| `status` | Ledger and index health |
| `plan-layout` | Print the proposed shelf layout for a folder |
| `browse` | Print or regenerate navigation files |

### Evidence API

| Command | Effect |
|---|---|
| `research` | Multi-question evidence bundle (JSON request in, JSON out) |
| `inspect --item` | Bibliographic details for one item |
| `source --document` | Bounded indexed chunks in source order |
| `passage --evidence-id` | Verify and read one passage with neighboring context |
| `describe` | Machine-readable tool schema |

### Housekeeping

| Command | Effect |
|---|---|
| `cache status` / `cache clear` | Inspect or reset the local retrieval cache |
| `readiness` | Audit index metadata for the research API (`--repair` to fix) |
| `sync-zotero-notes` | Materialize Zotero annotation markdown (opt-in write) |
| `doctor` | Check environment, folder, Qdrant, ledger (exit code 1 on failure) |
| `diagnose` | Print a redacted diagnostic report for bug reports |

## Reading results

A query prints each hit as one place: the source document, the page or
section anchor, and the matching chunk text. Results are ranked by
semantic distance, closest first. The tool returns places, not answers:
read the cited page to form your own conclusion.

Human `query` and `find` output label PDF hits `PDF p.<n>`. The number is
the legacy `page` value, a 1-based PDF page index. It can differ from the
number printed on the page, and it can be one page later than where the
passage starts (see the citation rule below).

`research`, `source` and `passage` print JSON. For PDF passages:

- `location.page_start`, `location.page_end`, `location.printed_start`,
  `location.printed_end` sit inside the `location` object of `research`
  results, `source` chunks and `passage` chunks. The printed values are
  `null` when the PDF has no real page-label scheme.
- `section` is a sibling of `location`: `{path, start_page, start_printed}`,
  the outline path of the section that contains `page_start`. It is `null`
  when the PDF has no outline. It is absent on points indexed before
  0.2.0.

To cite a PDF passage, take the PDF page from `location.page_start` and the
printed page from `location.printed_start`. They describe the same page.
Write `printed p. X (PDF p. Y)`, with X from `printed_start` and Y from
`page_start`. When `page_end` differs from `page_start`, give the range, for
example `printed pp. 277–278 (PDF pp. 299–300)`, with the printed range from
`printed_start` and `printed_end`. When `printed_start` is `null`, cite
`PDF p. Y` alone.

`page` and `location.value` keep their 0.1.0 meaning. When a chunk begins
with text carried over from the previous page, they can be one page later
than `location.page_start`, so do not cite the page from them.

### Upgrading from 0.1.0

0.2.0 changes the extractor version, which is part of the index
fingerprint. The next `library-query index` sees the mismatch, re-extracts
and re-embeds every document in the ledger, and publishes a new index
revision. Cached retrieval candidates are tied to the index revision, so
queries after reindexing use candidates for the new revision. Until then,
points indexed by 0.1.0 keep working and return results without the new fields.

Repeated queries with the same phrasing are served from a local retrieval
cache (`retrieval-cache.sqlite3` in your cache directory). The cache is
disposable: delete it at any time and the next query rebuilds it.

## Requirements

- Python 3.11 or newer
- Docker (for Qdrant)
- An embeddings endpoint and its API key
- Documents in `pdf`, `epub`, `md`, or `txt` format
