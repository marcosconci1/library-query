# Contributing

Thanks for your interest in library-query. This page explains how to set
up the project and what a good contribution looks like.

## Setup

```bash
git clone https://github.com/marcosconci1/library-query
cd library-query
uv sync --locked
uv run --locked python -m pytest tests -q -k "not live"
```

Tests marked `live` exercise a running Qdrant index and the embedding
endpoint. They skip automatically when neither is reachable, and they run
only against a library you point them at yourself.

## How to work

1. Open an issue first for anything that changes behavior.
2. Branch from `main`, named by the work: `feat/short-name`,
   `fix/short-name`, or `docs/short-name`.
3. Commit messages follow
   [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
4. Run the test suite before you open a pull request. A change that
   breaks tests will not merge.
5. Keep pull requests small. One behavior per pull request.

## Boundaries the code must keep

* Read commands must never write to the library folder. Writes live in
  the two opt-in commands, and every written file must carry a
  generated-file marker.
* No telemetry, no hosted component, no network call beyond the
  configured embeddings endpoint.
* Document changes in `docs/` in the same pull request as the code
  change they describe.
