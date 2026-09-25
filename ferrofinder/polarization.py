"""Reduced Berry-polarisation coordinates and branch arithmetic.

MACE-Field returns an intensive Cartesian polarisation in ``e/A^2``.  For a
cell whose row vectors form ``H``, the three reduced coordinates are

``p = volume * inv(H).T @ P``.

An integer change in ``p`` is one polarization quantum.  This representation
is especially useful here because the constraint is periodic: two values that
differ by an integer vector describe the same Berry-polarisation branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cell_volume(cell) -> float:
    """Return the positive volume of a full-rank cell."""

    value = abs(float(np.linalg.det(np.asarray(cell, dtype=float))))
    if value <= 1.0e-12:
        raise ValueError("a full-rank periodic cell is required")
    return value


def cartesian_to_reduced(polarization, cell) -> np.ndarray:
    """Convert Cartesian ``P`` in ``e/A^2`` to dimensionless reduced ``p``."""

    matrix = np.asarray(cell, dtype=float).reshape(3, 3)
    volume = cell_volume(matrix)
    # ASE stores lattice vectors as rows.  With column vectors,
    # P = H.T @ p / Omega, hence p = Omega H^{-T} @ P.
    return volume * np.linalg.solve(
        matrix.T, np.asarray(polarization, dtype=float).reshape(3)
    )


def reduced_to_cartesian(reduced, cell) -> np.ndarray:
    """Convert reduced polarization coordinates back to Cartesian ``P``."""

    matrix = np.asarray(cell, dtype=float).reshape(3, 3)
    volume = cell_volume(matrix)
    # The transpose is essential for non-orthogonal row-vector cells.
    return matrix.T @ np.asarray(reduced, dtype=float).reshape(3) / volume


def polarization_quantum(cell) -> np.ndarray:
    """Return Cartesian polarization-quantum vectors as columns.

    Integer shifts in reduced coordinates produce ``H.T @ n / Omega``.
    """

    matrix = np.asarray(cell, dtype=float).reshape(3, 3)
    return matrix.T / cell_volume(matrix)


def nearest_branch_delta(reference, trial) -> tuple[np.ndarray, np.ndarray]:
    """Return ``trial-reference`` after subtracting its nearest integer branch."""

    delta = np.asarray(trial, dtype=float) - np.asarray(reference, dtype=float)
    shifts = np.rint(delta).astype(int)
    return delta - shifts, shifts


def branch_match(reference, trial) -> np.ndarray:
    """Return the image of ``trial`` nearest to ``reference`` modulo quanta."""

    delta, _ = nearest_branch_delta(reference, trial)
    return np.asarray(reference, dtype=float) + delta


def unwrap_reduced_path(values) -> tuple[np.ndarray, np.ndarray]:
    """Continuously unwrap a sampled reduced-polarisation path.

    The first point is kept unchanged.  Every following point is shifted by
    the integer vector that makes it closest to the preceding unwrapped point.
    The returned integer array records the per-image integer added to the raw
    reduced polarisation, so ``unwrapped == raw + shifts``.
    """

    raw = np.asarray(values, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) == 0:
        raise ValueError("path polarization must have shape (n, 3) and be non-empty")
    result = np.empty_like(raw)
    shifts = np.zeros_like(raw, dtype=int)
    result[0] = raw[0]
    for index in range(1, len(raw)):
        delta, jump = nearest_branch_delta(result[index - 1], raw[index])
        result[index] = result[index - 1] + delta
        # ``jump`` is measured from the raw image to the already-unwrapped
        # previous image.  Its negative is therefore the global integer shift
        # to apply to this raw image.  Accumulating jumps here double-counts
        # repeated wraps after the path has crossed a Berry quantum.
        shifts[index] = -jump
    return result, shifts


def formal_inversion_classes() -> list[np.ndarray]:
    """Return the eight ``{0, 1/2}^3`` inversion-compatible classes."""

    return [
        np.asarray([x, y, z], dtype=float)
        for x in (0.0, 0.5)
        for y in (0.0, 0.5)
        for z in (0.0, 0.5)
    ]


def nearest_formal_target(polarization, cell) -> tuple[np.ndarray, np.ndarray]:
    """Find the closest branch representative of every formal class."""

    reduced = cartesian_to_reduced(polarization, cell)
    targets = []
    classes = []
    for formal in formal_inversion_classes():
        classes.append(formal)
        targets.append(formal + np.rint(reduced - formal))
    order = np.argsort([np.linalg.norm(item - reduced) for item in targets])
    return np.asarray([targets[i] for i in order]), np.asarray(
        [classes[i] for i in order]
    )


def transform_reduced_vector(vector, rotation) -> np.ndarray:
    """Apply a fractional-space point-group rotation to reduced ``p``.

    Reduced Berry coordinates transform like fractional coordinates.  Keeping
    this operation here avoids having several workflow modules guess whether a
    spglib rotation should multiply from the left or right.  The public
    representation is a column vector, so a spglib rotation ``R`` acts as
    ``R @ p``.
    """

    return np.asarray(rotation, dtype=float).reshape(3, 3) @ np.asarray(
        vector, dtype=float
    ).reshape(3)


@dataclass(frozen=True)
class FormalTargetOrbit:
    """One symmetry-inequivalent group of formal target branches.

    ``representative`` is the target actually sent to the continuation solver.
    ``formal_members`` are the original ``{0, 1/2}^3`` classes represented by
    this orbit.  The corresponding branch representatives are retained too,
    because integer Berry-branch shifts matter for provenance and debugging.
    """

    orbit_id: int
    representative: np.ndarray
    representative_formal_member: tuple[float, float, float]
    formal_members: tuple[tuple[float, float, float], ...]
    target_members: tuple[tuple[float, float, float], ...]
    symmetry_operations: tuple[int, ...]
    distance_from_parent: float

    def as_dict(self) -> dict:
        """Return a JSON- and checkpoint-friendly description."""

        return {
            "orbit_id": int(self.orbit_id),
            "representative_target": self.representative.tolist(),
            "representative_formal_member": list(self.representative_formal_member),
            "formal_members": [list(item) for item in self.formal_members],
            "symmetry_equivalent_to": [
                list(item)
                for item in self.formal_members
                if item != self.representative_formal_member
            ],
            "target_members": [list(item) for item in self.target_members],
            "symmetry_operations": list(self.symmetry_operations),
            "distance_from_parent": float(self.distance_from_parent),
            "attempted": False,
            "skipped_reason": None,
        }


def formal_target_orbits(
    polarization,
    cell,
    operations=None,
    *,
    equivalence_tolerance: float = 1.0e-8,
) -> list[FormalTargetOrbit]:
    """Group the eight formal targets under a supplied proper-rotation group.

    The eight inversion-compatible classes are topological possibilities, not
    automatically distinct structural searches.  A parent point-group
    rotation can permute their components, so this function computes the
    orbits before any MACE relaxation is launched.  Integer polarization
    quanta are removed when comparing targets, which is essential because the
    continuation targets are branch representatives rather than canonical
    values in ``[0, 1)``.

    ``operations`` is deliberately a list of plain dictionaries containing a
    ``rotation`` matrix.  That keeps the helper independent of spglib's
    version-specific dataset classes and makes its output easy to checkpoint.
    With no operations, only the identity is used and all eight classes remain
    available.
    """

    if equivalence_tolerance <= 0:
        raise ValueError("formal target equivalence tolerance must be positive")
    targets, classes = nearest_formal_target(polarization, cell)
    operations = list(operations or [{"index": 0, "rotation": np.eye(3)}])
    normalised_operations = []
    for fallback_index, operation in enumerate(operations):
        normalised_operations.append(
            (
                int(operation.get("index", fallback_index)),
                np.asarray(operation["rotation"], dtype=float).reshape(3, 3),
            )
        )

    # Build connected components instead of assuming the operation list is
    # perfectly closed.  This remains robust with older spglib releases and
    # with a caller-provided subset of operations.
    remaining = set(range(len(targets)))
    components = []
    while remaining:
        seed = min(remaining)
        component = {seed}
        changed = True
        while changed:
            changed = False
            for source in tuple(component):
                for operation_index, rotation in normalised_operations:
                    transformed = transform_reduced_vector(targets[source], rotation)
                    for trial in tuple(remaining - component):
                        if (
                            reduced_distance(transformed, targets[trial])
                            <= equivalence_tolerance
                        ):
                            component.add(trial)
                            changed = True
        remaining -= component
        components.append(component)

    result = []
    for component in components:
        member_indices = sorted(component)
        representative_index = min(
            member_indices,
            key=lambda index: (
                float(
                    np.linalg.norm(
                        targets[index] - cartesian_to_reduced(polarization, cell)
                    )
                ),
                tuple(np.asarray(classes[index], dtype=float)),
                tuple(np.asarray(targets[index], dtype=float)),
            ),
        )
        used_operations = []
        representative_target = targets[representative_index]
        for operation_index, rotation in normalised_operations:
            transformed = transform_reduced_vector(representative_target, rotation)
            if any(
                reduced_distance(transformed, targets[index]) <= equivalence_tolerance
                for index in member_indices
            ):
                used_operations.append(operation_index)
        result.append(
            FormalTargetOrbit(
                orbit_id=-1,
                representative=np.asarray(representative_target, dtype=float),
                representative_formal_member=tuple(
                    float(value) for value in classes[representative_index]
                ),
                formal_members=tuple(
                    tuple(float(value) for value in classes[index])
                    for index in member_indices
                ),
                target_members=tuple(
                    tuple(float(value) for value in targets[index])
                    for index in member_indices
                ),
                symmetry_operations=tuple(sorted(set(used_operations))),
                distance_from_parent=float(
                    np.linalg.norm(
                        representative_target - cartesian_to_reduced(polarization, cell)
                    )
                ),
            )
        )

    result.sort(
        key=lambda orbit: (
            orbit.distance_from_parent,
            tuple(orbit.representative),
            orbit.formal_members,
        )
    )
    return [
        FormalTargetOrbit(
            orbit_id=index,
            representative=orbit.representative,
            representative_formal_member=orbit.representative_formal_member,
            formal_members=orbit.formal_members,
            target_members=orbit.target_members,
            symmetry_operations=orbit.symmetry_operations,
            distance_from_parent=orbit.distance_from_parent,
        )
        for index, orbit in enumerate(result)
    ]


def reduced_distance(first, second) -> float:
    """Return branch-aware Euclidean distance between reduced vectors."""

    delta, _ = nearest_branch_delta(first, second)
    return float(np.linalg.norm(delta))
