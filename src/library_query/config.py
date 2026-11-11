"""Shared configuration for library-query, loaded from environment / .env."""
import os
from pathlib import Path

def repo_root() -> Path:
    """Checkout root: the directory that holds .env.example and docker/."""
    return Path(__file__).resolve().parents[2]


def env_file_path() -> Path:
    """The env file in use: LIBRARY_QUERY_ENV_FILE if set, else .env at the repo root."""
    configured_path = os.environ.get("LIBRARY_QUERY_ENV_FILE")
    return Path(configured_path) if configured_path else repo_root() / ".env"


def load_env() -> None:
    """Populate os.environ from a project .env if present (no hard dep on python-dotenv)."""
    env_path = env_file_path()
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if not value:
            # A blank template line must not mask the default: os.environ.get(name, default)
            # returns "" for a set-but-empty name.
            continue
        if not os.environ.get(key.strip()):
            os.environ[key.strip()] = value

load_env()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
# Seconds to wait on a Qdrant call. The client default of 5s is too tight for bulk
# upserts and for queries running alongside them.
QDRANT_TIMEOUT = int(os.environ.get("QDRANT_TIMEOUT", "60"))
# Serialized request-body ceiling for one point-upsert batch. Qdrant rejects any
# request over 32 MiB; one document's whole point set can exceed that, so upserts
# are packed by wire size, not chunk count. Half the server limit leaves margin
# for encoder differences (non-ASCII escaping, framing) between measurer and client.
QDRANT_MAX_BATCH_BYTES = int(os.environ.get("LIBRARY_QUERY_QDRANT_MAX_BATCH_BYTES", str(16 * 1024 * 1024)))
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "openai/text-embedding-3-small")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Optional local Better BibTeX JSON-RPC endpoint. It is used only by the
# explicit `sync-zotero-notes` command; indexing and querying never require it.
ZOTERO_BBT_URL = os.environ.get("ZOTERO_BBT_URL", "http://127.0.0.1:23119/better-bibtex/json-rpc")
ZOTERO_BBT_TIMEOUT = int(os.environ.get("ZOTERO_BBT_TIMEOUT", "20"))

# Total service-work deadline (seconds) for one `research` request: readiness
# inspection, embedding, and every grouped search share this budget. Each service
# wait is capped by the remaining time rather than the indexing embedding timeout.
RESEARCH_TIMEOUT = float(os.environ.get("LIBRARY_QUERY_RESEARCH_TIMEOUT", "25"))

# The library folder this tool indexes.
LIBRARY_ROOT = Path(os.environ.get("LIBRARY_ROOT", str(Path.home() / "library")))

# Where the processed-ledger and other local state live (never inside the library repo).
STATE_DIR = Path(os.environ.get("LIBRARY_QUERY_STATE", str(Path.home() / ".local/share/library-query")))
LEDGER_PATH = STATE_DIR / "ledger.json"

# Vector collection name. Overridable so disposable smoke/evaluation collections
# never touch the live corpus; all store paths resolve through this value.
COLLECTION = os.environ.get("LIBRARY_QUERY_COLLECTION", "library")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
# Embedding request bounds: providers reject requests over both a total-token
# budget and a per-request input count; a whole book must be split.
EMBED_BATCH_SIZE = 64
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".epub"}
