import numpy as np
from ase import Atoms

from ferrofinder.structures import (
    cell_domain,
    cell_rmsd,
    deformation_from,
)


def test_row_cell_deformation_is_consistent_for_nonorthogonal_cells():
    reference = np.array([[3.0, 0.2, 0.0], [0.0, 4.0, 0.1], [0.1, 0.0, 5.0]])
    deformation = np.array([[1.02, 0.01, 0.0], [0.0, 0.98, 0.02], [0.0, 0.0, 1.01]])
    np.testing.assert_allclose(
        deformation_from(reference, reference @ deformation), deformation, atol=1.0e-14
    )


def test_cell_domain_rejects_stretch_outside_bounds():
    reference = np.diag([3.0, 3.0, 3.0])
    assert cell_domain(
        reference,
        reference @ np.diag([1.1, 1.0, 1.0]),
        stretch_min=0.8,
        stretch_max=1.25,
        volume_min=0.5,
        volume_max=2.0,
    )
    assert not cell_domain(
        reference,
        reference @ np.diag([1.3, 1.0, 1.0]),
        stretch_min=0.8,
        stretch_max=1.25,
        volume_min=0.5,
        volume_max=2.0,
    )


def test_cell_rmsd_ignores_global_lattice_rotation():
    first_cell = np.diag([3.0, 4.0, 5.0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    first = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=first_cell, pbc=True)
    second = Atoms(
        "Si",
        positions=[[0.0, 0.0, 0.0]],
        cell=first_cell @ rotation.T,
        pbc=True,
    )
    assert cell_rmsd(first, second) < 1.0e-12
