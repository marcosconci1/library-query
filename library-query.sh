#!/usr/bin/env bash
set -euo pipefail

module_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LIBRARY_QUERY_ENV_FILE="$module_dir/.env"
exec uv run --directory "$module_dir" --frozen library-query "$@"
