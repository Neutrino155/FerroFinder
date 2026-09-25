import numpy as np
from ase import Atoms

from ferrofinder.symmetry import analyse, proper_rotation_operations


def test_symmetry_analysis_reports_space_and_point_groups():
    atoms = Atoms(
        "Si2",
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        cell=np.diag([5.4, 5.4, 5.4]),
        pbc=True,
    )

    info = analyse(atoms, symprec=0.01)

    assert info["number"] > 0
    assert info["international"]
    assert info["pointgroup"]
    assert info["has_inversion"]


def test_proper_rotations_do_not_merge_opposite_polarization_domains():
    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 3.0, pbc=True)
    polarization = np.array([0.2, 0.0, 0.0])
    operations = proper_rotation_operations(
        atoms,
        reduced_polarization=polarization,
        polarization_tolerance=1.0e-8,
    )

    assert operations
    for operation in operations:
        rotation = operation["rotation"]
        assert round(np.linalg.det(rotation)) == 1
        transformed = rotation @ polarization
        assert not np.allclose(transformed, -polarization)
