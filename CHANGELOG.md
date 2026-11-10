# Changelog

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
