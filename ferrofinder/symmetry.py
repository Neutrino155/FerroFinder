"""spglib analysis used to rank candidate nonpolar parent structures."""

from __future__ import annotations

import warnings

import numpy as np

from .polarization import reduced_distance, transform_reduced_vector


def _cell_tuple(atoms):
    return (
        np.asarray(atoms.get_cell(), dtype=float),
        np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float),
        np.asarray(atoms.get_atomic_numbers(), dtype=int),
    )


def _dataset_value(dataset, key: str, default):
    value = getattr(dataset, key, None)
    if value is not None:
        return value
    getter = getattr(dataset, "get", None)
    return default if getter is None else getter(key, default)


def analyse(atoms, symprec: float = 0.01) -> dict:
    """Return JSON-ready space-group and point-group information."""

    import spglib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dataset = spglib.get_symmetry_dataset(_cell_tuple(atoms), symprec=float(symprec))
        operations = spglib.get_symmetry(_cell_tuple(atoms), symprec=float(symprec))
    if dataset is None:
        return {
            "number": 0,
            "international": "?",
            "hall": "?",
            "pointgroup": "?",
            "is_polar_point_group": False,
            "has_inversion": False,
            "n_operations": 0,
            "symprec": float(symprec),
        }
    pointgroup = str(_dataset_value(dataset, "pointgroup", "?"))
    rotations = [] if operations is None else operations["rotations"]
    has_inversion = any(
        np.array_equal(rotation, -np.eye(3, dtype=int)) for rotation in rotations
    )
    polar_groups = {"1", "2", "m", "mm2", "3", "3m", "4", "4mm", "6", "6mm"}
    return {
        "number": int(_dataset_value(dataset, "number", 0)),
        "international": str(_dataset_value(dataset, "international", "?")),
        "hall": str(_dataset_value(dataset, "hall", "?")),
        "pointgroup": pointgroup,
        "is_polar_point_group": pointgroup in polar_groups,
        "has_inversion": bool(has_inversion),
        "n_operations": len(rotations),
        "symprec": float(symprec),
    }


def proper_rotation_operations(
    atoms,
    *,
    symprec: float = 0.01,
    reduced_polarization=None,
    polarization_tolerance: float = 0.08,
) -> list[dict]:
    """Return proper parent rotations that do not identify opposite domains."""

    if polarization_tolerance <= 0:
        raise ValueError("polarization_tolerance must be positive")
    import spglib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        data = spglib.get_symmetry(_cell_tuple(atoms), symprec=float(symprec))
    rotations = [] if data is None else data["rotations"]
    translations = [] if data is None else data["translations"]
    reduced = (
        None
        if reduced_polarization is None
        else np.asarray(reduced_polarization, dtype=float).reshape(3)
    )
    cell = np.asarray(atoms.get_cell(), dtype=float)
    result = []
    for index, (rotation, translation) in enumerate(zip(rotations, translations, strict=True)):
        rotation = np.asarray(rotation, dtype=int).reshape(3, 3)
        if not np.isclose(np.linalg.det(rotation), 1.0):
            continue
        if reduced is not None:
            transformed = transform_reduced_vector(reduced, rotation)
            if reduced_distance(transformed, -reduced) <= polarization_tolerance:
                continue
        cartesian = np.linalg.solve(cell, rotation.T @ cell)
        result.append(
            {
                "index": index,
                "rotation": rotation,
                "translation": np.asarray(translation, dtype=float),
                "cartesian_rotation": cartesian,
            }
        )
    if result:
        return result
    return [
        {
            "index": 0,
            "rotation": np.eye(3, dtype=int),
            "translation": np.zeros(3),
            "cartesian_rotation": np.eye(3),
        }
    ]
