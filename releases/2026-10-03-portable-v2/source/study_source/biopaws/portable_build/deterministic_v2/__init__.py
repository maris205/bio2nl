"""New deterministic raw-build integration, separate from frozen version 1."""
from .patches import patch_pair_source, patch_remote_source

__all__ = ["patch_pair_source", "patch_remote_source"]
