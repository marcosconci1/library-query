#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --check             Report actions without changing anything
  --profile NAME      Install the Hermes plugin for NAME (repeatable)
  --no-hermes         Skip Hermes plugin installation
  --no-docker         Do not start Qdrant
  --library PATH      Set the document library directory
  --bin-dir DIR       Install the command link in DIR (default: $HOME/.local/bin)
  -h, --help          Show this help
EOF
}

fail_usage() {
    usage >&2
    exit 2
}

resolve_script() {
    local source resolved source_dir link
    source=$1
    if resolved=$(readlink -f -- "$source" 2>/dev/null) && [ -n "$resolved" ]; then
        printf '%s\n' "$resolved"
        return
    fi
    case "$source" in
        /*) ;;
        *) source="$(CDPATH= cd -- "$(dirname -- "$source")" && pwd -P)/${source##*/}" ;;
    esac
    while [ -L "$source" ]; do
        source_dir=$(CDPATH= cd -- "$(dirname -- "$source")" && pwd -P)
        link=$(readlink "$source") || return 1
        case "$link" in
            /*) source=$link ;;
            *) source="$source_dir/$link" ;;
        esac
    done
    source_dir=$(CDPATH= cd -- "$(dirname -- "$source")" && pwd -P)
    printf '%s/%s\n' "$source_dir" "${source##*/}"
}

profiles=()
check_only=false
no_hermes=false
no_docker=false
library_path=
bin_dir="${HOME:?HOME must be set}/.local/bin"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --check) check_only=true; shift ;;
        --profile)
            [ "$#" -ge 2 ] && [ -n "$2" ] || fail_usage
            profiles+=("$2"); shift 2 ;;
        --no-hermes) no_hermes=true; shift ;;
        --no-docker) no_docker=true; shift ;;
        --library)
            [ "$#" -ge 2 ] && [ -n "$2" ] || fail_usage
            library_path=$2; shift 2 ;;
        --bin-dir)
            [ "$#" -ge 2 ] && [ -n "$2" ] || fail_usage
            bin_dir=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail_usage ;;
    esac
done

script_path="$(resolve_script "${BASH_SOURCE[0]}")"
repo_dir="$(CDPATH= cd -- "$(dirname -- "$script_path")" && pwd)"
launcher="$repo_dir/library-query.sh"
target="$bin_dir/library-query"
install_failed=false
plugin_installed=false

prerequisites() {
    if ! command -v uv >/dev/null 2>&1; then
        printf '%s\n' '[fail] uv is required; install uv: https://docs.astral.sh/uv/' >&2
        exit 1
    fi
    printf '%s\n' '[ok] uv found'

    if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
        printf '%s\n' '[ok] Docker Compose found'
    else
        printf '%s\n' '[warn] Docker Compose unavailable; start Qdrant by hand'
    fi

    if command -v hermes >/dev/null 2>&1; then
        printf '%s\n' '[ok] Hermes found'
    else
        printf '%s\n' '[skip] Hermes not found; plugin installation skipped'
    fi
}

sync_engine() {
    if [ "$check_only" = true ]; then
        printf '%s\n' '[skip] uv sync --locked (check only)'
    elif (CDPATH= cd -- "$repo_dir" && uv sync --locked); then
        printf '%s\n' '[ok] engine synced'
    else
        printf '%s\n' '[fail] uv sync --locked failed' >&2
        install_failed=true
    fi
}

install_command_link() {
    if [ "$check_only" = true ]; then
        printf '[skip] command link would be %s -> %s\n' "$target" "$launcher"
    else
        if [ -e "$target" ] && [ ! -L "$target" ]; then
            printf '[fail] refusing to replace non-symlink %s\n' "$target" >&2
            install_failed=true
            return
        fi
        if [ -L "$target" ] && [ "$(readlink "$target")" = "$launcher" ]; then
            printf '[ok] command link already ready: %s\n' "$target"
        elif [ -L "$target" ]; then
            if ! mkdir -p -- "$bin_dir" || ! ln -sfn -- "$launcher" "$target"; then
                printf '[fail] could not replace command link %s\n' "$target" >&2
                install_failed=true
                return
            fi
            printf '[ok] command link updated: %s\n' "$target"
        else
            if ! mkdir -p -- "$bin_dir" || ! ln -sfn -- "$launcher" "$target"; then
                printf '[fail] could not create command link %s\n' "$target" >&2
                install_failed=true
                return
            fi
            printf '[ok] command link ready: %s\n' "$target"
        fi
    fi

    case ":${PATH:-}:" in
        *":$bin_dir:"*) ;;
        *) printf '[warn] add this line to your shell profile: export PATH="%s:$PATH"\n' "$bin_dir" ;;
    esac
}

setup_engine() {
    local args=(setup)
    if [ "$no_docker" = true ]; then args+=(--no-docker); fi
    if [ -n "$library_path" ]; then args+=(--library "$library_path"); fi
    if [ "$check_only" = true ]; then
        if [ ! -d "$repo_dir/.venv" ]; then
            printf '%s\n' '[skip] setup check needs the installed engine; run ./install.sh without --check first'
            return
        fi
        args+=(--check)
    elif [ -t 0 ] && [ -t 1 ]; then
        args+=(--ask-key)
    fi
    if "$launcher" "${args[@]}"; then
        printf '%s\n' '[ok] engine setup checks passed'
    else
        printf '%s\n' '[warn] setup reported open items'
    fi
}

install_hermes() {
    local profile label
    if [ "$no_hermes" = true ]; then
        printf '%s\n' '[skip] Hermes plugin installation disabled (--no-hermes)'
        return
    fi
    if ! command -v hermes >/dev/null 2>&1; then
        return
    fi
    if [ "$check_only" = true ]; then
        local plugin_uri="file://$repo_dir#integrations/hermes/library-query"
        if [ "${#profiles[@]}" -eq 0 ]; then
            printf '[skip] check only; would run: hermes plugins install "%s" --force --enable --yes-deps\n' "$plugin_uri"
            printf '%s\n' '[skip] check only; would run: hermes plugins doctor library-query'
        else
            for profile in "${profiles[@]}"; do
                printf '[skip] check only; would run: hermes -p %s plugins install "%s" --force --enable --yes-deps\n' "$profile" "$plugin_uri"
                printf '[skip] check only; would run: hermes -p %s plugins doctor library-query\n' "$profile"
            done
        fi
        return
    fi

    if [ "${#profiles[@]}" -eq 0 ]; then
        profiles=("")
    fi
    for profile in "${profiles[@]}"; do
        label=${profile:-active profile}
        local prefix=()
        if [ -n "$profile" ]; then prefix=(-p "$profile"); fi
        # The "+" form keeps an empty array safe under set -u on bash 3.2 (macOS).
        if hermes ${prefix[@]+"${prefix[@]}"} plugins install "file://$repo_dir#integrations/hermes/library-query" --force --enable --yes-deps \
            && hermes ${prefix[@]+"${prefix[@]}"} plugins doctor library-query; then
            printf '[ok] Hermes plugin installed for %s\n' "$label"
            plugin_installed=true
        else
            printf '[fail] Hermes plugin failed for %s\n' "$label" >&2
            install_failed=true
        fi
    done
}

print_next_steps() {
    local key_present=false library_present=false line library_value
    if [ -f "$repo_dir/.env" ]; then
        while IFS= read -r line || [ -n "$line" ]; do
            case "$line" in
                OPENROUTER_API_KEY=*)
                    [ -n "${line#OPENROUTER_API_KEY=}" ] && key_present=true
                    ;;
                LIBRARY_ROOT=*)
                    library_value=${line#LIBRARY_ROOT=}
                    library_value="${library_value#"${library_value%%[![:space:]]*}"}"
                    library_value="${library_value%"${library_value##*[![:space:]]}"}"
                    case "$library_value" in
                        ""|/absolute/path/to/library) ;;
                        *) library_present=true ;;
                    esac
                    ;;
            esac
        done < "$repo_dir/.env"
    fi
    printf '%s\n' 'Next:'
    if [ "$library_present" = false ]; then
        printf '  Set LIBRARY_ROOT in %s/.env or rerun ./install.sh --library <folder>\n' "$repo_dir"
    fi
    if [ "$key_present" = false ]; then
        printf '  Set OPENROUTER_API_KEY in %s/.env\n' "$repo_dir"
    fi
    printf '%s\n' '  Run library-query index'
    if [ "$plugin_installed" = true ]; then
        printf '%s\n' '  Restart the Hermes gateway with: hermes gateway restart'
    fi
}

prerequisites
sync_engine
install_command_link
setup_engine
install_hermes

if [ "$check_only" != true ] || [ -d "$repo_dir/.venv" ]; then
    if version=$("$launcher" --version 2>/dev/null); then
        version=${version#library-query }
        printf '[ok] engine version %s\n' "$version"
    fi
fi
print_next_steps

if [ "$install_failed" = true ]; then
    exit 1
fi
