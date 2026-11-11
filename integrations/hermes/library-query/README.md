# library-query plugin for Hermes

This plugin gives Hermes agents tools to find where an answer lives in your document library: the document, the page and the passage. It calls the library-query command line program and keeps no evidence or secrets of its own.

## Tools

All tools belong to the `library` toolset.

- `library_query` retrieves bounded evidence for a focused question. It takes a weight (light, meaningful or important) and an optional exact target.
- `library_browse` searches the catalogue for works by title or author.
- `library_inspect` shows the metadata of one item.
- `library_source` lists chunk identities and locations of a document.
- `library_passage` reads the chunks before or after a returned passage.

The bundled skill `library-consult` teaches an agent when to consult the library, how to dig deeper in up to three rounds, and how to cite PDF pages from `location.page_start`.

## Limits

A response holds at most three excerpts, 4,800 text characters (more when whole passages are shown) and 12,000 serialized characters. Each call has a 30 second subprocess limit and the full tool has a 45 second deadline. Failures return no evidence and a short error category.

## Finding the engine

The plugin looks for the library-query program in this order:

1. The environment variable `LIBRARY_QUERY_BIN`.
2. The plugin setting `bin`, under `plugins.entries.library-query.settings`.
3. The `library-query.sh` launcher at the root of the repository, when the plugin runs from a full checkout.
4. A `library-query` program on `PATH`.

If the Hermes gateway PATH does not include `~/.local/bin` or the folder you
selected with `--bin-dir`, set `plugins.entries.library-query.settings.bin` to
the full path of the launcher.

If none exists, the tools return the error `engine_not_found`.

## Jev filter

With Jev installed and configured, every returned passage was judged relevant for the question. Filtering failures return no evidence.

Without Jev, evidence is returned unfiltered in retrieval order and the receipt says `unfiltered`.

## Tests

Run the plugin tests from the repository root:

    uv run pytest -q tests/test-hermes-filter.py tests/test-hermes-filter-expand.py tests/test_plugin_engine_path.py

The tests need no network and no Hermes install.
