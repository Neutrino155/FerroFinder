import numpy as np

from ferrofinder.polarization import (
    cartesian_to_reduced,
    formal_inversion_classes,
    formal_target_orbits,
    polarization_quantum,
    reduced_to_cartesian,
    unwrap_reduced_path,
)


def test_variable_cell_polarization_conversion_is_inverse():
    cell = np.array([[3.0, 0.2, 0.0], [0.0, 4.0, 0.1], [0.1, 0.0, 5.0]])
    polarization = np.array([0.02, -0.01, 0.03])
    reduced = cartesian_to_reduced(polarization, cell)
    np.testing.assert_allclose(reduced_to_cartesian(reduced, cell), polarization)


def test_row_vector_cell_uses_transposed_analytic_mapping():
    """The direct physical mapping must hold for a strongly skewed cell."""

    cell = np.array([[3.1, 0.7, -0.4], [0.2, 2.4, 0.8], [-0.6, 0.3, 4.2]])
    reduced = np.array([0.37, -0.21, 0.49])
    volume = abs(np.linalg.det(cell))
    expected = cell.T @ reduced / volume
    np.testing.assert_allclose(reduced_to_cartesian(reduced, cell), expected)
    np.testing.assert_allclose(cartesian_to_reduced(expected, cell), reduced)


def test_bifeo3_hexagonal_cell_mixes_cartesian_and_reduced_axes_correctly():
    """Use the benchmark's oblique basal lattice with a general Cartesian P."""

    cell = np.array(
        [
            [5.68376, 0.0, 0.0],
            [-2.84188, 4.92228, 0.0],
            [0.0, 0.0, 7.2514],
        ]
    )
    polarization = np.array([0.012, -0.018, 0.027])
    volume = abs(np.linalg.det(cell))
    expected_reduced = volume * np.linalg.solve(cell.T, polarization)

    reduced = cartesian_to_reduced(polarization, cell)

    np.testing.assert_allclose(reduced, expected_reduced, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(
        reduced_to_cartesian(reduced, cell), polarization, rtol=1.0e-13, atol=1.0e-13
    )
    assert not np.allclose(reduced, volume * np.linalg.solve(cell, polarization))


def test_polarization_quantum_columns_follow_row_lattice_vectors():
    cell = np.array([[3.1, 0.7, -0.4], [0.2, 2.4, 0.8], [-0.6, 0.3, 4.2]])
    np.testing.assert_allclose(
        polarization_quantum(cell), cell.T / abs(np.linalg.det(cell))
    )


def test_branch_unwrapping_uses_nearest_integer_quantum():
    raw = np.array([[0.4, 0.0, 0.0], [0.91, 0.0, 0.0], [-0.9, 0.0, 0.0]])
    unwrapped, shifts = unwrap_reduced_path(raw)
    np.testing.assert_allclose(unwrapped[:, 0], [0.4, -0.09, 0.1])
    np.testing.assert_array_equal(shifts[:, 0], [0, -1, 1])
    np.testing.assert_allclose(unwrapped, raw + shifts)


def test_formal_classes_are_eight_deterministic_vectors():
    classes = formal_inversion_classes()
    assert len(classes) == 8
    assert {tuple(value) for value in classes} == {
        (x, y, z) for x in (0.0, 0.5) for y in (0.0, 0.5) for z in (0.0, 0.5)
    }


def test_tetragonal_proper_rotations_reduce_formal_targets():
    quarter_turns = []
    for index, rotation in enumerate(
        (
            np.eye(3, dtype=int),
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        )
    ):
        quarter_turns.append({"index": index, "rotation": np.asarray(rotation)})
    cell = np.diag([4.0, 4.0, 4.0])
    polarization = reduced_to_cartesian([0.1, 0.1, 0.2], cell)
    orbits = formal_target_orbits(
        polarization, cell, quarter_turns, equivalence_tolerance=0.08
    )
    assert len(orbits) == 6
    assert sum(len(orbit.formal_members) for orbit in orbits) == 8
    assert any(len(orbit.formal_members) == 2 for orbit in orbits)
    for orbit in orbits:
        record = orbit.as_dict()
        assert record["representative_formal_member"] in record["formal_members"]
        assert (
            record["representative_formal_member"]
            not in record["symmetry_equivalent_to"]
        )


def test_identity_only_keeps_all_formal_targets():
    cell = np.diag([4.0, 4.0, 4.0])
    polarization = reduced_to_cartesian([0.1, 0.1, 0.2], cell)
    orbits = formal_target_orbits(
        polarization,
        cell,
        [{"index": 0, "rotation": np.eye(3, dtype=int)}],
    )
    assert len(orbits) == 8
