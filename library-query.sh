#!/usr/bin/env bash
set -euo pipefail

# Follow a symlink to the real script so the repo is found when launched from a PATH link.
script_path="$(readlink -f -- "${BASH_SOURCE[0]}" 2>/dev/null || true)"
[ -n "$script_path" ] || script_path="${BASH_SOURCE[0]}"
module_dir="$(CDPATH='' cd -- "$(dirname -- "$script_path")" && pwd)"
# Inherit nothing from a foreign Python: the module venv must resolve its own imports.
unset PYTHONPATH PYTHONHOME
export LIBRARY_QUERY_ENV_FILE="$module_dir/.env"
# uv runs from the repository; keep the caller's folder so a relative --library resolves there.
export LIBRARY_QUERY_CALLER_CWD="$PWD"
exec uv run --directory "$module_dir" --frozen library-query "$@"
