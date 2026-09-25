"""Small settings object for endpoint-only parent recovery."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryConfig:
    """Numerical settings used by the inverse-distortion reconstruction."""

    symprec: float = 0.01
    polarization_distance_tolerance: float = 0.08
    polarization_tolerance: float = 0.001
    strain_delta: float = 5.0e-4
    atomic_metric: float = 1.0
    strain_metric: float = 25.0

    def validate(self) -> None:
        """Reject invalid tolerances before loading a model checkpoint."""

        if self.symprec <= 0:
            raise ValueError("symprec must be positive")
        if self.polarization_distance_tolerance <= 0:
            raise ValueError("polarization distance tolerance must be positive")
        if self.polarization_tolerance <= 0:
            raise ValueError("polarization tolerance must be positive")
        if self.strain_delta <= 0:
            raise ValueError("strain finite-difference step must be positive")
        if self.atomic_metric <= 0 or self.strain_metric <= 0:
            raise ValueError("atomic and strain metric weights must be positive")
