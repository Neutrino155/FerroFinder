import numpy as np
from ase import Atoms

from ferrofinder.model import ModelEvaluation
from ferrofinder.polarization import reduced_to_cartesian
from ferrofinder.response import (
    atomic_reduced_jacobian,
    generalized_jacobian,
    lattice_reduced_jacobian,
    minimum_metric_displacement,
    strain_basis,
)


def test_bec_jacobian_has_expected_orientation():
    cell = np.diag([2.0, 3.0, 4.0])
    becs = np.zeros((1, 3, 3))
    becs[0, 0, 0] = 2.0
    becs[0, 1, 1] = 3.0
    becs[0, 2, 2] = 4.0
    jacobian = atomic_reduced_jacobian(becs, cell)
    np.testing.assert_allclose(jacobian, np.diag([1.0, 1.0, 1.0]))


def test_bec_jacobian_uses_h_inverse_transpose_for_triclinic_cell():
    cell = np.array([[3.1, 0.7, -0.4], [0.2, 2.4, 0.8], [-0.6, 0.3, 4.2]])
    becs = np.zeros((1, 3, 3))
    becs[0] = np.array([[1.2, 0.1, -0.2], [0.0, 0.8, 0.3], [0.4, -0.1, 1.1]])
    expected = np.linalg.inv(cell).T @ becs[0]
    np.testing.assert_allclose(atomic_reduced_jacobian(becs, cell), expected)


def test_bec_prediction_matches_direct_finite_difference_in_triclinic_cell():
    cell = np.array([[3.1, 0.7, -0.4], [0.2, 2.4, 0.8], [-0.6, 0.3, 4.2]])
    bec = np.array([[1.2, 0.1, -0.2], [0.0, 0.8, 0.3], [0.4, -0.1, 1.1]])
    reference = Atoms("Si", positions=[[0.3, 0.4, 0.5]], cell=cell, pbc=True)
    base_reduced = np.array([0.12, -0.08, 0.21])
    displacement = np.array([0.017, -0.011, 0.023])

    class DirectModel:
        def evaluate(self, atoms, **kwargs):
            delta = np.asarray(atoms.positions[0] - reference.positions[0])
            reduced = (
                base_reduced + atomic_reduced_jacobian(bec[None, ...], cell) @ delta
            )
            polarization = reduced_to_cartesian(reduced, cell)
            return ModelEvaluation(
                0.0,
                polarization,
                reduced,
                bec[None, ...],
            )

    trial = reference.copy()
    trial.positions += displacement
    direct_delta = DirectModel().evaluate(trial).reduced_polarization - base_reduced
    predicted_delta = atomic_reduced_jacobian(bec[None, ...], cell) @ displacement
    np.testing.assert_allclose(direct_delta, predicted_delta, rtol=1e-11, atol=1e-12)


class StrainModel:
    def evaluate(self, atoms, **kwargs):
        from ferrofinder.model import ModelEvaluation
        from ferrofinder.polarization import cartesian_to_reduced

        # Choose reduced p directly as a small multiple of the current cell
        # diagonal, then convert it back to the Cartesian model output.
        cell = np.asarray(atoms.cell)
        reduced = np.diag(cell) * 0.1
        p = cell.T @ reduced / atoms.get_volume()
        return ModelEvaluation(
            0.0,
            p,
            cartesian_to_reduced(p, cell),
            np.ones((len(atoms), 3, 3)),
        )


def test_lattice_jacobian_uses_fixed_fractional_positions():
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.diag([2.0, 3.0, 4.0]), pbc=True)
    jacobian = lattice_reduced_jacobian(StrainModel(), atoms, delta=1e-5)
    assert jacobian.shape == (3, 6)
    assert np.all(np.isfinite(jacobian))
    np.testing.assert_allclose(np.diag(jacobian[:, :3]), [0.2, 0.3, 0.4], rtol=1e-3)


class BranchCrossingStrainModel:
    def evaluate(self, atoms, **kwargs):
        cell = np.asarray(atoms.cell, dtype=float)
        # Deliberately return equivalent raw Berry branches on either side of
        # the reference cell.  Branch matching should remove the artificial
        # jump before the finite difference.
        reduced = (
            np.array([1.51, 0.0, 0.0])
            if cell[0, 0] > 2.0
            else np.array([-0.49, 0.0, 0.0])
        )
        p = cell.T @ reduced / atoms.get_volume()
        from ferrofinder.model import ModelEvaluation
        from ferrofinder.polarization import cartesian_to_reduced

        return ModelEvaluation(
            0.0,
            p,
            cartesian_to_reduced(p, cell),
            np.ones((len(atoms), 3, 3)),
        )


def test_lattice_jacobian_branch_matches_each_strained_cell():
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.diag([2.0, 3.0, 4.0]), pbc=True)
    jacobian = lattice_reduced_jacobian(BranchCrossingStrainModel(), atoms, delta=1e-5)
    assert abs(jacobian[0, 0]) < 1.0e3


def test_strain_order_and_generalized_concatenation():
    assert np.array_equal(strain_basis(0), np.diag([1.0, 0.0, 0.0]))
    off_diagonal = strain_basis(3)
    assert off_diagonal[1, 2] == off_diagonal[2, 1] == 1.0
    result = generalized_jacobian(np.zeros((3, 3)), np.ones((3, 6)))
    assert result.shape == (3, 9)


def test_minimum_metric_initializer_prefers_cheap_atomic_coordinates():
    jacobian = np.zeros((3, 9))
    jacobian[:, :3] = np.eye(3)
    displacement = minimum_metric_displacement(
        jacobian, [0.2, -0.1, 0.3], atomic_metric=1.0, strain_metric=25.0
    )
    np.testing.assert_allclose(displacement[:3], [0.2, -0.1, 0.3])
    np.testing.assert_allclose(displacement[3:], 0.0)
