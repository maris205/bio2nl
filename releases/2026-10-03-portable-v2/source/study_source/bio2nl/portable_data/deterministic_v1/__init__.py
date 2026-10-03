"""Deterministic serialization patches for a new, explicitly versioned snapshot."""

from .patches import patch_source, write_patched_snapshot
from .remote_provenance import historical_remote_erratum, verify_current_remote_manifest

__all__ = ['patch_source', 'write_patched_snapshot', 'historical_remote_erratum',
           'verify_current_remote_manifest']
