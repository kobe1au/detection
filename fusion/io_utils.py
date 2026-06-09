"""Safe I/O utilities for AEG payload loading."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from fusion.payload_contract import validate_aeg_payload


LOGGER = logging.getLogger(__name__)


def load_aeg_payload(
    path: Path | str,
    *,
    validate: bool = False,
    expected_node_feature_dim: int | None = None,
    weights_only: bool = True,
) -> dict[str, Any]:
    """Load AEG PT file with defense-in-depth.
    
    This is the canonical safe loading function for all AEG payload files.
    Use this instead of torch.load() directly to ensure security and consistency.
    
    Args:
        path: Path to .pt file
        validate: Whether to validate payload contract after load
        expected_node_feature_dim: Optional node_x width expected by the caller
        weights_only: Use weights_only=True for pickle safety (PyTorch 2.0+)
    
    Returns:
        AEG payload dictionary
        
    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If validation fails
        
    Security:
        - Primary: weights_only=True restricts pickle deserialization (PyTorch 2.0+)
        - Secondary: validate_aeg_payload checks schema/structure post-load
        - Fallback: Graceful degradation for older PyTorch versions with warning
        
    Performance:
        - Uses mmap=True for memory-mapped loading when available
        - Reduces memory footprint for large PT files
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PT file not found: {path}")
    # torch.load(..., mmap=True) requires a string filename on current PyTorch.
    # Passing a Path works without mmap but fails before the fallback otherwise.
    path_str = str(path)
    
    if not weights_only:
        # Legacy mode - only for explicitly trusted files
        LOGGER.warning(
            "Loading %s without weights_only protection. "
            "Only use this for trusted PT files.",
            path.name,
        )
        payload = torch.load(path_str, map_location="cpu")
    else:
        # Try safe loading with progressive fallbacks
        try:
            # PyTorch 2.0+: safe deserialization + memory-mapped loading
            payload = torch.load(
                path_str,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except (TypeError, ValueError):
            try:
                # Fallback 1: weights_only without mmap
                payload = torch.load(
                    path_str,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                # Fallback 2: legacy mode (no pickle safety)
                LOGGER.warning(
                    "PyTorch version does not support weights_only/mmap. "
                    "Loading %s with legacy torch.load; only use trusted PT files.",
                    path.name,
                )
                payload = torch.load(path_str, map_location="cpu")
    
    # Secondary validation (post-load structure check)
    if validate:
        validate_aeg_payload(payload, expected_node_feature_dim=expected_node_feature_dim)
    
    return payload
