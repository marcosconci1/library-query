# Changelog

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
