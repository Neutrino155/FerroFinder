from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.io import write

from ferrofinder.config import RecoveryConfig
from ferrofinder.parent_recovery import (
    _bec_backprojection,
    _clean_atoms,
    _inversion_projected_geometries,
    fold_reduced_branch,
)
from screenings.mp_ferroelectric_250.scripts.prepare_blind_inputs import (
    prepare,
)
from screenings.mp_ferroelectric_250.scripts.run_endpoint_recovery import (
    _json_safe,
    _nonfinite_paths,
)
from screenings.mp_ferroelectric_250.scripts.validate_recovery import (
    _finite_values,
)


def test_fold_reduced_branch_records_integer_quantum_shifts():
    result = fold_reduced_branch(
        [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [-0.1, 0.0, 0.0]]
    )

    assert np.allclose(
        result["unwrapped_reduced_polarization"],
        [[0.4, 0.0, 0.0], [0.6, 0.0, 0.0], [0.9, 0.0, 0.0]],
    )
    assert result["berry_integer_shifts"] == [[0, 0, 0], [1, 0, 0], [1, 0, 0]]
    assert np.allclose(
        np.asarray(result["reduced_polarization"])
        + np.asarray(result["berry_integer_shifts"]),
        result["unwrapped_reduced_polarization"],
    )
    assert np.allclose(
        result["folded_reduced_polarization"],
        [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [-0.1, 0.0, 0.0]],
    )


def test_nonfinite_diagnostics_are_tagged_and_excluded_from_aggregates():
    report = {"energy": np.inf, "polarization": [0.1, np.nan, 0.0]}

    assert _nonfinite_paths(report) == ["energy", "polarization[1]"]
    assert _json_safe(report) == {
        "energy": "Infinity",
        "polarization": [0.1, "NaN", 0.0],
    }
    assert _finite_values([0.1, "NaN", "Infinity", None]) == [0.1]


def test_clean_atoms_removes_all_source_metadata_and_extra_arrays():
    atoms = Atoms(
        "NaCl",
        positions=[[0, 0, 0], [2.8, 2.8, 2.8]],
        cell=[5.6, 5.6, 5.6],
        pbc=True,
    )
    atoms.info["polar_mpid"] = "mp-hidden"
    atoms.info["REF_polarization"] = [0.0, 0.0, 0.1]
    atoms.new_array("tags", np.asarray([0, 1]))

    clean = _clean_atoms(atoms)

    assert clean.info == {}
    assert set(clean.arrays) == {"numbers", "positions"}
    assert np.array_equal(clean.numbers, atoms.numbers)
    assert np.allclose(clean.positions, atoms.positions)


def test_bec_backprojection_moves_endpoint_to_formal_target():
    class LinearResponseModel:
        def evaluate(self, atoms, **_kwargs):
            reduced = np.asarray(atoms.positions[0], dtype=float)
            return SimpleNamespace(
                energy=0.0,
                polarization=reduced,
                reduced_polarization=reduced,
                becs=np.eye(3, dtype=float)[None, :, :],
            )

    atoms = Atoms(
        "Na",
        positions=[[0.10, 0.10, 0.10]],
        cell=np.eye(3),
        pbc=True,
    )
    config = RecoveryConfig(
        polarization_tolerance=1.0e-10,
        atomic_metric=1.0,
        strain_metric=25.0,
    )

    recovered, response, history = _bec_backprojection(
        LinearResponseModel(),
        atoms,
        np.asarray([0.25, 0.12, 0.20]),
        config,
        max_iterations=4,
        max_step_A=0.12,
    )

    assert np.allclose(response.reduced_polarization, [0.25, 0.12, 0.20])
    assert np.allclose(recovered.positions[0], [0.25, 0.12, 0.20])
    assert history[-1]["target_residual"] <= config.polarization_tolerance
    assert len(history) > 1


def test_inversion_projection_recovers_same_species_centrosymmetric_geometry():
    from ferrofinder.symmetry import analyse

    center = np.asarray([0.25, 0.25, 0.25])
    offsets = np.asarray(
        [
            [0.050, 0.100, 0.100],
            [0.101, 0.050, 0.050],
        ]
    )
    fractions = np.concatenate(
        (center + offsets, center - offsets), axis=0
    )
    atoms = Atoms(
        numbers=[11, 17, 11, 17],
        scaled_positions=fractions,
        cell=np.eye(3) * 6.0,
        pbc=True,
    )
    atoms.info["hidden_source_id"] = "must_not_propagate"

    candidates = _inversion_projected_geometries(
        atoms, max_centers=24, max_candidates=3
    )

    assert candidates
    assert any(analyse(candidate, symprec=0.03)["has_inversion"] for candidate in candidates)
    assert all(candidate.info == {} for candidate in candidates)


def test_prepare_separates_anonymous_endpoint_inputs_from_oracle(tmp_path):
    source = tmp_path / "source.extxyz"
    frames = []
    for pair_index in range(2):
        for step in range(3):
            atoms = Atoms(
                "NaCl",
                positions=[[0, 0, 0], [2.8, 2.8, 2.8 + 0.01 * step]],
                cell=[5.6, 5.6, 5.6],
                pbc=True,
            )
            atoms.info["nonpolar_mpid"] = f"np-{pair_index}"
            atoms.info["polar_mpid"] = f"p-{pair_index}"
            atoms.info["REF_polarization"] = [0.0, 0.0, 0.01 * step]
            frames.append(atoms)
    write(source, frames, format="extxyz")

    root = tmp_path / "split"
    result = prepare(source, root, seed=4)

    from ase.io import iread, read

    assert result["query_count"] == 2
    assert result["oracle_frame_count"] == 6
    input_files = sorted((root / "model_inputs" / "inputs").glob("*.extxyz"))
    assert len(input_files) == 2
    for input_file in input_files:
        endpoint = read(input_file)
        assert endpoint.info == {}
        assert set(endpoint.arrays) == {"numbers", "positions"}
    input_root_files = {path.name for path in (root / "model_inputs").iterdir()}
    assert input_root_files == {"input_manifest.json", "inputs"}
    oracle_root_files = {path.name for path in (root / "private_validation").iterdir()}
    assert oracle_root_files == {"answer_key.json", "oracle.extxyz"}
    oracle = list(iread(str(root / "private_validation" / "oracle.extxyz"), index=":"))
    assert len(oracle) == 6
    assert all("oracle_query_id" in frame.info for frame in oracle)
