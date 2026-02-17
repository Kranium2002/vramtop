"""Model file detection via /proc/<pid>/fd/ readlinks."""

from __future__ import annotations

import os

from vramtop.enrichment import ModelFileInfo
from vramtop.permissions import is_same_user

_MODEL_EXTENSIONS = frozenset({
    ".safetensors", ".gguf", ".pt", ".bin", ".onnx", ".pth", ".h5", ".tflite",
})

_MAX_FDS = 4096


def scan_model_files(pid: int) -> list[ModelFileInfo]:
    """Scan /proc/<pid>/fd/ for open model files.

    Deduplicates by inode. Scans at most _MAX_FDS entries.
    Requires same-UID.
    """
    if not is_same_user(pid):
        return []

    fd_dir = f"/proc/{pid}/fd"
    try:
        entries = os.listdir(fd_dir)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []

    seen_inodes: set[int] = set()
    results: list[ModelFileInfo] = []

    for i, entry in enumerate(entries):
        if i >= _MAX_FDS:
            break

        link_path = os.path.join(fd_dir, entry)
        try:
            target = os.readlink(link_path)
        except (FileNotFoundError, OSError):
            continue

        ext = _get_model_extension(target)
        if ext is None:
            continue

        try:
            st = os.stat(target)
        except (FileNotFoundError, OSError):
            continue

        if st.st_ino in seen_inodes:
            continue
        seen_inodes.add(st.st_ino)

        results.append(ModelFileInfo(
            path=target,
            size_bytes=st.st_size,
            extension=ext,
        ))

    return results


def _get_model_extension(path: str) -> str | None:
    """Return the model extension if the path matches, else None."""
    _, ext = os.path.splitext(path)
    ext_lower = ext.lower()
    if ext_lower in _MODEL_EXTENSIONS:
        return ext_lower
    return None
