"""Shared checkout-relative paths for the server and asset preparation."""

import os
from pathlib import Path


def repository_root():
    explicit = os.getenv("FASTDA3_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "da3_streaming/configs/base_config.yaml").is_file():
        return checkout
    return Path.cwd().resolve()


def resolve_path(value, root=None):
    path = Path(value).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else ((root or repository_root()) / path).resolve()
    )
