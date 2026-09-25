"""Recover nonpolar parent candidates and polarization branches from polar endpoints."""

from .config import RecoveryConfig
from .parent_recovery import ParentRecoveryResult, recover_parent_branches
from .version import __version__

__all__ = [
    "MACEFieldModel",
    "ParentRecoveryResult",
    "RecoveryConfig",
    "__version__",
    "recover_parent_branches",
]


def __getattr__(name):
    """Load the MACE adapter only when a caller requests it."""

    if name == "MACEFieldModel":
        from .model import MACEFieldModel

        return MACEFieldModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
