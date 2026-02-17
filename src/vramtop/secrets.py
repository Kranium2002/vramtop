"""Secrets resolution: env vars -> config file -> None.

No keyring dependency -- fails on headless GPU servers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib  # type: ignore[import-not-found]
else:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import tomli as tomllib

_PREFIX = "VRAMTOP_"
_CONFIG_DIR = Path("~/.config/vramtop").expanduser()
_SECRETS_FILE = _CONFIG_DIR / "secrets.toml"


def _load_secrets_file() -> dict[str, str]:
    """Load secrets.toml, refusing if group/other permissions are set."""
    if not _SECRETS_FILE.is_file():
        return {}
    st = os.stat(_SECRETS_FILE)
    if st.st_mode & 0o077:
        return {}
    try:
        with open(_SECRETS_FILE, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return {k: str(v) for k, v in data.items() if isinstance(v, str)}


def get_secret(key: str) -> str | None:
    """Resolve a secret by key.

    Resolution order:
    1. Environment variable VRAMTOP_<KEY>
    2. ~/.config/vramtop/secrets.toml (refused if group/other perms)
    3. None
    """
    env_val = os.environ.get(f"{_PREFIX}{key.upper()}")
    if env_val is not None:
        return env_val
    secrets = _load_secrets_file()
    return secrets.get(key)
