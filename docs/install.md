# Installation

library-query needs three things: a Python 3.11+ environment, a Qdrant
container, and an embeddings endpoint. Setup takes about five minutes.

## One-command install

Run these commands from a terminal:

```bash
git clone https://github.com/marcosconci1/library-query
cd library-query
./install.sh
```

The installer checks for `uv`, Docker Compose, and Hermes, then syncs the
engine and links `library-query` into `~/.local/bin` or the folder you pass
with `--bin-dir`. It warns when that folder is not on your PATH. Setup creates
`.env` and starts Qdrant when Docker Compose is available. In a terminal, the
installer asks for an empty API key with a hidden prompt. It then checks the
engine and installs the Hermes plugin when Hermes is available.

Options:

- `--check` reports planned installer work. The setup `--check` command changes no configuration and starts nothing; it may append a line to the local diagnostics journal. Setup checks run only when the engine is already installed.
- `--profile NAME` installs the plugin for a Hermes profile. Repeat it for more profiles.
- `--no-hermes` skips plugin installation.
- `--no-docker` skips starting Qdrant.
- `--library PATH` sets `LIBRARY_ROOT` only when it is empty or still the template value. Otherwise setup prints `library_root: not changed, the env file already sets it`. Relative paths resolve against the folder where you run the command.
- `--bin-dir DIR` chooses where the `library-query` command link goes.
- `--help` prints the installer options.

The installer creates a link to the repository launcher. Running it again is
safe: it syncs the engine, keeps your `.env` values, and reinstalls the plugin
when Hermes is available.
It supports Linux and macOS 12.3 or newer, requires `uv`, and works without
Docker if you start Qdrant yourself.

The installer creates `.env` from `.env.example`. Set `LIBRARY_ROOT` to your
document folder, or pass `--library <folder>` to the installer or to
`library-query setup`. Set `OPENROUTER_API_KEY` in `.env`, then run
`library-query index` to build the first index.

## Hermes plugin

The plugin adds five tools and the `library-consult` skill. The tools search
the library, browse works, inspect an item, list a document's chunks, and read
a passage.

When Hermes is installed, `install.sh` installs and enables the plugin from
this checkout, then runs its doctor check. To install it manually, run:

```bash
hermes plugins install "file://$(pwd)#integrations/hermes/library-query" --force --enable --yes-deps
hermes plugins doctor library-query
hermes plugins enable library-query
hermes gateway restart
```

Update it with `hermes plugins update library-query`. Remove it with
`hermes plugins remove library-query`. Jev filtering is optional. When Jev is
installed and configured, each returned passage passes a relevance check for
the question. Without Jev, the plugin returns passages in retrieval order
without filtering.

The plugin finds the engine through `LIBRARY_QUERY_BIN`, then the plugin setting
`bin`, then the `library-query.sh` launcher of a full checkout, then `PATH`. If
the Hermes gateway does not have `~/.local/bin` (or your `--bin-dir`) on its
`PATH`, set `plugins.entries.library-query.settings.bin` to the full path of the
launcher.

The plugin and engine versions stay equal. Reinstall the plugin after `git pull`.

## Manual install

### 1. Clone and sync

```bash
git clone https://github.com/marcosconci1/library-query
cd library-query
uv sync
```

### 2. Configure

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

### 3. Start Qdrant

```bash
docker compose -f docker/docker-compose.yml up -d
```

The container listens on loopback only, so the vector store is reachable
from your machine and from nothing else.

### 4. Index and ask

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
