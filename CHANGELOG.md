# Changelog

## 0.5.2 — 2026-11-22

### Fixed

- Diagnostic journal retention and appends now share an inter-process `flock` on a sidecar lock file so concurrent CLI processes cannot erase newly appended events during trim.
- Timed-out Jev relevance evaluations in the Hermes plugin are now bounded by a fixed in-flight worker slot pool (`JEV_MAX_IN_FLIGHT = 4`) whose slots are released only when the handler exits, preventing unbounded daemon thread accumulation.

### Changed

- New Qdrant collections created by `library-query index` now enable `int8` scalar quantization (`quantile: 0.99`, `always_ram: true`) by default to prevent cold page-cache search timeouts on shared hosts.

## 0.5.1 — 2026-11-21

### Fixed

- Qdrant search timeouts now surface as `research_timeout` instead of `qdrant_unreachable`, and the server-side cap no longer exceeds the client deadline.

## 0.5.0 — 2026-11-19

### Changed

- **Behavior change:** Differentiated `library_query` weight budgets. The `light`
  weight tier now resolves to a lower retrieval budget (3 sources, 1 passage per
  source, 1,200 max excerpt characters) instead of sharing the full budget.
  `meaningful` and `important` continue to use the full budget (6 sources, 2 passages
  per source, 1,600 max excerpt characters). Exact targeted requests continue to
  resolve the full budget regardless of weight.
- **Behavior change:** Passage selection in the Hermes plugin now rotates across
  distinct sources so responses provide connectable references from different works,
  and reports a `diversity_demoted` receipt counter when rotation displaces passages
  that pure relevance ranking would have selected.
- **Additive receipt and passage fields:** The research engine now emits
  `full_chars` (uncut chunk character length) on each passage and advertises it
  in the response schema, and the Hermes plugin receipt now includes `omitted`
  (up to 6 identity-only stubs for approved passages dropped for budget or source
  diversity, in relevance order), `corpus_matches` (forwarded from the engine's
  `query_results` `eligible_source_count`), and a deterministic `next_step` hint
  (`expand` or `target_omitted`).

## 0.4.0 — 2026-11-11

### Changed

- **Compatibility note:** When `library_inspect` receives a non-canonical `item`
  argument (anything other than `item-<32 hex>`, such as `null`, a title, a file
  path, or a `doc_<64 hex>` document ID), the Hermes plugin now rejects the input
  before invoking the CLI and reports `receipt.filtering_reason` as
  `invalid_inspect_item` (previously `invalid_research_request` for invalid
  non-empty strings and `retrieval_or_request_unavailable` for missing or
  non-string values), along with guidance to resolve legacy works via
  `library_source(path=...)` and `target.document_ids`.

### Fixed

- Guided discovery of legacy and unmapped library books in the Hermes plugin
  tool schemas (`library_browse`, `library_inspect`, `library_source`,
  `library_query`) and the `library-consult` skill: when `library_browse`
  returns `item_id: null` with `sources` paths for an unambiguous work, agents
  are directed to pass the path from `sources` to `library_source(path=...)` to
  obtain `document_id` (`doc_<64 hex>`) and query `library_query` with
  `target.document_ids` instead of calling `library_inspect` or stopping.
- Clarified guidance for ambiguous titles in the `library_browse` tool description
  and `library-consult` skill: agents should resolve ambiguity before selecting
  a work. This is guidance, not an enforced selection guard.

### Upgrade

Run `./install.sh` to sync the engine and reinstall the Hermes plugin, or
reinstall the plugin manually:

```bash
hermes plugins install "file://$(pwd)#integrations/hermes/library-query" --force --enable --yes-deps
```

Then restart the Hermes gateway so the updated plugin and skill are loaded:

```bash
hermes gateway restart
```

The plugin and engine versions stay equal (`0.4.0`). Nothing needs reindexing
for this release because `EXTRACTOR_VERSION` is unchanged (`3`).

## 0.3.0 — 2026-11-11

### Added

- `library-query --version` prints the installed version.
- `library-query setup` creates `.env` from the template with mode `0600`, sets
  `LIBRARY_ROOT` only when empty or still the template value, and supports the
  hidden `--ask-key` API key prompt. Relative `--library` paths resolve against
  the folder where you run the command. It starts Qdrant with Docker Compose
  when needed, runs the doctor, and lists remaining setup steps. `--check`
  changes no configuration and starts nothing; it may append a line to the
  local diagnostics journal.
- `install.sh` syncs the engine, links `library-query` onto PATH, runs setup,
  and installs and checks the Hermes plugin for each selected profile. It
  supports `--check`, `--profile`, `--no-hermes`, `--no-docker`, `--library`,
  and `--bin-dir`.
- The Hermes plugin in `integrations/hermes/library-query` provides the
  `library_query`, `library_browse`, `library_inspect`, `library_source`, and
  `library_passage` tools in the `library` toolset, plus the `library-consult`
  skill. It resolves its engine from `LIBRARY_QUERY_BIN`, the plugin `bin`
  setting, the repository launcher, then PATH.

### Changed

- Jev relevance filtering is optional. Without Jev, tools return evidence in
  retrieval order and the receipt reports `filtering_status` as `unfiltered`.
  With Jev registered but unconfigured, or when Jev fails, the plugin returns
  no evidence.
- The launcher follows a symlink to find the repository.

### Fixed

- Human `find` output labels EPUB hits `section N` instead of `PDF p.N`, matching `query`.
- A blank value in .env no longer overrides a default, so the state folder stays out of the working directory.
- setup refuses library folder names that contain line separators.
- setup resolves a relative --library against the folder you ran it from.
- setup writes .env with mode 0600 even when the file already existed.

### Upgrade

Run `./install.sh` to install the engine and plugin. To install the plugin
manually, run:

```bash
hermes plugins install "file://$(pwd)#integrations/hermes/library-query" --force --enable --yes-deps
```

The plugin and engine versions stay equal, so reinstall the plugin after
`git pull`. Nothing needs reindexing for this release because `EXTRACTOR_VERSION`
is unchanged.

## 0.2.0 — 2026-11-10

### Added

- PDF passages carry a page range (`page_start`, `page_end`), printed page
  labels (`printed_start`, `printed_end`) and the outline section path.
  `research`, `source` and `passage` return them. Cite the PDF page from
  `location.page_start`; `page` keeps its meaning and can be one page later.

### Changed

- Human `query` and `find` output labels PDF hits `PDF p.<n>`.
- `EXTRACTOR_VERSION` is now 3. The next `library-query index` re-extracts
  and re-embeds every document in the ledger.
- The launcher ignores a foreign `PYTHONPATH` and `PYTHONHOME`.
- The cache worktree check no longer treats a `.git` above the home
  directory as a project worktree.

### Upgrade

Run `library-query index` after updating. Until it finishes, points
indexed by 0.1.0 keep working without the new fields. The new index revision
makes cached retrieval candidates from the prior revision ineligible.

## 0.1.0 — Initial public release

First public version of library-query: a semantic locator for a local
document library. `library-query index` extracts and embeds the
supported files under `LIBRARY_ROOT` into a local Qdrant index, and
`library-query query "<question>"` returns ranked places: the document,
the page, and the matching passage. Reading results never writes to the
library folder; the only writers are the opt-in `browse --write` and
`sync-zotero-notes` commands.
