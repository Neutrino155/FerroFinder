"""Periodic geometry operations used by endpoint recovery."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

from .polarization import cell_volume


def clean_copy(atoms):
    """Copy an ASE structure without its calculator or transient metadata."""

    from ase import Atoms

    return Atoms(
        numbers=np.asarray(atoms.get_atomic_numbers(), dtype=int),
        positions=np.asarray(atoms.get_positions(), dtype=float),
        cell=np.asarray(atoms.get_cell(), dtype=float),
        pbc=np.asarray(atoms.get_pbc(), dtype=bool),
    )


def require_periodic(atoms) -> None:
    """Require a non-empty structure with a full three-dimensional cell."""

    if len(atoms) == 0 or not np.all(np.asarray(atoms.pbc, dtype=bool)):
        raise ValueError("ferrofinder requires a non-empty fully periodic structure")
    cell_volume(atoms.get_cell())


def fractional_displacement(first, second) -> np.ndarray:
    """Return componentwise minimum-image fractional displacement."""

    delta = np.asarray(second.get_scaled_positions(wrap=False)) - np.asarray(
        first.get_scaled_positions(wrap=False)
    )
    return delta - np.rint(delta)


def _species_match(first, second) -> np.ndarray:
    """Match each atom to a same-species atom by periodic distance."""

    numbers0 = np.asarray(first.get_atomic_numbers())
    numbers1 = np.asarray(second.get_atomic_numbers())
    if len(numbers0) != len(numbers1) or sorted(numbers0) != sorted(numbers1):
        raise ValueError("structures must have identical species and atom counts")
    cell = np.asarray(first.get_cell(), dtype=float)
    fractions0 = np.asarray(first.get_scaled_positions(wrap=True), dtype=float)
    fractions1 = np.asarray(second.get_scaled_positions(wrap=True), dtype=float)
    order = np.empty(len(first), dtype=int)
    for species in np.unique(numbers0):
        rows = np.flatnonzero(numbers0 == species)
        columns = np.flatnonzero(numbers1 == species)
        delta = fractions0[rows, None, :] - fractions1[None, columns, :]
        delta -= np.rint(delta)
        row, column = linear_sum_assignment(np.linalg.norm(delta @ cell, axis=-1))
        order[rows[row]] = columns[column]
    return order


def align_periodic(reference, trial):
    """Reorder a structure by species and align its periodic origin."""

    order = _species_match(reference, trial)
    result = clean_copy(trial[order])
    reference_fractional = np.asarray(reference.get_scaled_positions(wrap=True))
    trial_fractional = np.asarray(result.get_scaled_positions(wrap=True))
    anchor = np.flatnonzero(np.asarray(reference.get_atomic_numbers()) == reference.get_atomic_numbers()[0])
    shift = reference_fractional[anchor] - trial_fractional[anchor]
    shift = np.mean(shift - np.rint(shift), axis=0)
    result.set_scaled_positions(trial_fractional + shift)
    result.wrap()
    return result


def periodic_rmsd(reference, trial) -> float:
    """Return the species-aware minimum-image RMS displacement in Å."""

    aligned = align_periodic(reference, trial)
    cartesian = fractional_displacement(reference, aligned) @ np.asarray(
        reference.get_cell(), dtype=float
    )
    return float(np.sqrt(np.mean(np.sum(cartesian**2, axis=1))))


def cell_rmsd(first, second) -> float:
    """Return a rotation-invariant RMS difference between cell metrics in Å."""

    a, b = np.asarray(first.get_cell(), dtype=float), np.asarray(second.get_cell(), dtype=float)
    if a.shape != b.shape:
        return float("inf")
    metric_a, metric_b = a @ a.T, b @ b.T
    scale = max(float(np.sqrt(np.mean(np.diag(metric_a)))), 1.0e-12)
    return float(np.sqrt(np.mean((metric_a - metric_b) ** 2)) / scale)


def deformation_from(reference_cell, cell) -> np.ndarray:
    """Return row-vector deformation F defined by H = H0 @ F."""

    return np.linalg.solve(np.asarray(reference_cell), np.asarray(cell))


def cell_domain(
    reference_cell,
    cell,
    *,
    stretch_min: float,
    stretch_max: float,
    volume_min: float,
    volume_max: float,
) -> bool:
    """Check principal-stretch and volume limits for a trial cell."""

    deformation = deformation_from(reference_cell, cell)
    stretches = np.linalg.svd(deformation, compute_uv=False)
    volume_ratio = abs(float(np.linalg.det(deformation)))
    return bool(
        np.all(stretches >= stretch_min)
        and np.all(stretches <= stretch_max)
        and volume_min <= volume_ratio <= volume_max
    )


def apply_generalized_displacement(atoms, displacement):
    """Apply 3N Cartesian displacements and six symmetric log strains."""

    vector = np.asarray(displacement, dtype=float).reshape(-1)
    n_atoms = len(atoms)
    if vector.size != 3 * n_atoms + 6:
        raise ValueError("generalized displacement must contain 3N+6 values")
    result = clean_copy(atoms)
    fractions = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    strain = np.zeros((3, 3), dtype=float)
    for index, value in enumerate(vector[3 * n_atoms :]):
        if index < 3:
            strain[index, index] = value
        else:
            first, second = ((1, 2), (0, 2), (0, 1))[index - 3]
            strain[first, second] = strain[second, first] = value
    result.set_cell(np.asarray(atoms.get_cell()) @ expm(strain), scale_atoms=False)
    result.set_scaled_positions(fractions)
    result.set_positions(result.get_positions() + vector[: 3 * n_atoms].reshape(-1, 3))
    result.wrap()
    return result
