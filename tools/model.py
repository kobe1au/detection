"""Compatibility wrapper for the current API + graph model.

The old experimental vision model was removed from the training stack. Import
from this module only if an older notebook/script still expects `tools.model`.
"""

from fusion.model import ApiSequenceEncoder, MalwareModelWithXAttn

__all__ = ["ApiSequenceEncoder", "MalwareModelWithXAttn"]
