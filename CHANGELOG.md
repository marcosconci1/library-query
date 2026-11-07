# Changelog

## Initial public release

First public version of library-query: a semantic locator for a local
document library. `library-query index` extracts and embeds the
supported files under `LIBRARY_ROOT` into a local Qdrant index, and
`library-query query "<question>"` returns ranked places: the document,
the page, and the matching passage. Reading results never writes to the
library folder; the only writers are the opt-in `browse --write` and
`sync-zotero-notes` commands.
