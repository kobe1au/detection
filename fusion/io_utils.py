"""Safe I/O utilities for AEG payload and checkpoint loading."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from fusion.payload_contract import validate_aeg_payload


LOGGER = logging.getLogger(__name__)


def _compat_error_message(exc: BaseException) -> str:
    return str(exc).lower()


def _arg_unsupported(exc: BaseException, arg_name: str) -> bool:
    message = _compat_error_message(exc)
    return arg_name in message and (
        "unexpected keyword" in message
        or "invalid keyword" in message
        or "not supported" in message
        or "got an unexpected keyword argument" in message
    )


def _supports_retry_without_mmap(exc: BaseException) -> bool:
    return _arg_unsupported(exc, "mmap")


def _weights_only_unsupported(exc: BaseException) -> bool:
    return _arg_unsupported(exc, "weights_only")


def _weights_only_rejected(exc: BaseException) -> bool:
    if _weights_only_unsupported(exc):
        return False
    message = _compat_error_message(exc)
    return (
        "weights only load failed" in message
        or "unsupported global" in message
        or "safe_globals" in message
    )


def load_aeg_payload(
    path: Path | str,
    *,
    validate: bool = False,
    expected_node_feature_dim: int | None = None,
    weights_only: bool = True,
) -> dict[str, Any]:
    """Load an AEG PT payload with fail-closed safe-loading semantics."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PT file not found: {path}")
    path_str = str(path)

    if not weights_only:
        LOGGER.warning(
            "Loading %s without weights_only protection. Only use this for trusted PT files.",
            path.name,
        )
        payload = torch.load(path_str, map_location="cpu")
    else:
        try:
            payload = torch.load(
                path_str,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except Exception as exc:
            if _weights_only_unsupported(exc):
                raise RuntimeError(
                    "This PyTorch version does not support weights_only safe loading for AEG payloads. "
                    "Upgrade PyTorch or explicitly opt into weights_only=False only for trusted files."
                ) from exc
            if not _supports_retry_without_mmap(exc):
                raise
            try:
                payload = torch.load(
                    path_str,
                    map_location="cpu",
                    weights_only=True,
                )
            except Exception as retry_exc:
                if _weights_only_unsupported(retry_exc):
                    raise RuntimeError(
                        "This PyTorch version does not support weights_only safe loading for AEG payloads. "
                        "Upgrade PyTorch or explicitly opt into weights_only=False only for trusted files."
                    ) from retry_exc
                raise

    if validate:
        validate_aeg_payload(payload, expected_node_feature_dim=expected_node_feature_dim)

    return payload


def load_checkpoint(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = True,
) -> dict[str, Any]:
    """Load a training checkpoint with best-effort safe loading and explicit fallback warnings."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    path_str = str(path)

    if not weights_only:
        LOGGER.warning(
            "Loading checkpoint %s without weights_only protection. Only use this for trusted files.",
            path.name,
        )
        checkpoint = torch.load(path_str, map_location=map_location)
    else:
        try:
            checkpoint = torch.load(
                path_str,
                map_location=map_location,
                weights_only=True,
                mmap=True,
            )
        except Exception as exc:
            if _supports_retry_without_mmap(exc):
                try:
                    checkpoint = torch.load(
                        path_str,
                        map_location=map_location,
                        weights_only=True,
                    )
                except Exception as retry_exc:
                    if _weights_only_unsupported(retry_exc) or _weights_only_rejected(retry_exc):
                        LOGGER.warning(
                            "Falling back to legacy torch.load for checkpoint %s due to safe-load incompatibility: %s",
                            path.name,
                            retry_exc,
                        )
                        checkpoint = torch.load(path_str, map_location=map_location)
                    else:
                        raise
            elif _weights_only_unsupported(exc):
                LOGGER.warning(
                    "Falling back to legacy torch.load for checkpoint %s due to safe-load incompatibility: %s",
                    path.name,
                    exc,
                )
                checkpoint = torch.load(path_str, map_location=map_location)
            elif _weights_only_rejected(exc):
                LOGGER.warning(
                    "Falling back to legacy torch.load for checkpoint %s due to safe-load rejection: %s",
                    path.name,
                    exc,
                )
                checkpoint = torch.load(path_str, map_location=map_location)
            else:
                raise

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}")
    return checkpoint
