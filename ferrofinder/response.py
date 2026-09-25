"""Atomic and lattice response Jacobians for inverse-polarization updates."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .polarization import branch_match


def atomic_reduced_jacobian(becs, cell) -> np.ndarray:
    """Return d(reduced polarization)/d(Cartesian positions) from BECs."""

    matrix = np.asarray(cell, dtype=float).reshape(3, 3)
    values = np.asarray(becs, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError("BECs must have shape (n_atoms, 3, 3)")
    return np.einsum("ia,nab->inb", np.linalg.inv(matrix).T, values).reshape(3, -1)


def strain_basis(index: int) -> np.ndarray:
    """Return a symmetric unit tensor in Voigt order xx, yy, zz, yz, xz, xy."""

    if not 0 <= index < 6:
        raise IndexError(index)
    tensor = np.zeros((3, 3), dtype=float)
    if index < 3:
        tensor[index, index] = 1.0
    else:
        first, second = ((1, 2), (0, 2), (0, 1))[index - 3]
        tensor[first, second] = tensor[second, first] = 1.0
    return tensor


def strained_cell(cell, strain) -> np.ndarray:
    """Apply symmetric logarithmic strain to an ASE row-vector cell."""

    return np.asarray(cell, dtype=float).reshape(3, 3) @ expm(
        np.asarray(strain, dtype=float).reshape(3, 3)
    )


def lattice_reduced_jacobian(
    model,
    atoms,
    *,
    delta: float = 5.0e-4,
    reference_reduced_polarization=None,
) -> np.ndarray:
    """Finite-difference reduced polarization versus six log-strain modes.

    Strains keep fractional positions fixed, so this returns the clamped-ion
    strain response, including the piezoelectric contribution in reduced
    polarization coordinates.
    """

    if delta <= 0:
        raise ValueError("strain finite-difference delta must be positive")
    fractions = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    if reference_reduced_polarization is None:
        reference = model.evaluate(
            atoms,
            include_response=False,
        ).reduced_polarization
    else:
        reference = np.asarray(reference_reduced_polarization, dtype=float).reshape(3)
    jacobian = np.empty((3, 6), dtype=float)
    for index in range(6):
        displaced = []
        for sign in (1.0, -1.0):
            image = atoms.copy()
            image.calc = None
            image.set_cell(strained_cell(cell, sign * delta * strain_basis(index)), scale_atoms=False)
            image.set_scaled_positions(fractions)
            value = model.evaluate(
                image,
                include_response=False,
            ).reduced_polarization
            displaced.append(branch_match(reference, value))
        jacobian[:, index] = (displaced[0] - displaced[1]) / (2.0 * delta)
    return jacobian


def generalized_jacobian(atomic, lattice) -> np.ndarray:
    """Join the atomic (3 × 3N) and lattice (3 × 6) response blocks."""

    atomic, lattice = np.asarray(atomic, dtype=float), np.asarray(lattice, dtype=float)
    if atomic.ndim != 2 or atomic.shape[0] != 3 or lattice.shape != (3, 6):
        raise ValueError("expected atomic shape (3, 3N) and lattice shape (3, 6)")
    return np.concatenate((atomic, lattice), axis=1)


def minimum_metric_displacement(
    jacobian,
    delta_p,
    *,
    atomic_metric: float = 1.0,
    strain_metric: float = 25.0,
) -> np.ndarray:
    """Return the minimum-metric generalized displacement for delta_p."""

    values = np.asarray(jacobian, dtype=float)
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] < 6:
        raise ValueError("jacobian must have shape (3, 3N+6)")
    if atomic_metric <= 0 or strain_metric <= 0:
        raise ValueError("metric weights must be positive")
    weights = np.concatenate(
        (
            np.full(values.shape[1] - 6, float(atomic_metric)),
            np.full(6, float(strain_metric)),
        )
    )
    inverse_metric = 1.0 / weights
    gram = (values * inverse_metric[None, :]) @ values.T
    return (
        inverse_metric[:, None]
        * values.T
        @ np.linalg.pinv(gram)
        @ np.asarray(delta_p, dtype=float).reshape(3)
    )
