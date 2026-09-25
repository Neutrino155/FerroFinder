"""Endpoint-only reconstruction of nonpolar parents and Berry branches.

Search functions accept one anonymous polar endpoint and MACEField predictions.
Known nonpolar parents and sampled path frames belong only to validation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .config import RecoveryConfig
from .polarization import (
    branch_match,
    cartesian_to_reduced,
    formal_target_orbits,
    nearest_formal_target,
    unwrap_reduced_path,
)
from .response import (
    atomic_reduced_jacobian,
    generalized_jacobian,
    lattice_reduced_jacobian,
    minimum_metric_displacement,
)
from .structures import (
    align_periodic,
    apply_generalized_displacement,
    cell_domain,
    cell_rmsd,
    deformation_from,
    periodic_rmsd,
)
from .symmetry import analyse, proper_rotation_operations


@dataclass
class ParentRecoveryResult:
    """Model-only candidates, branches, and portable diagnostics."""

    report: dict
    structures: list[tuple[str, object]]


def fold_reduced_branch(values) -> dict:
    """Unwrap reduced Berry coordinates, then fold them to [-1/2, 1/2)."""

    raw = np.asarray(values, dtype=float).reshape(-1, 3)
    unwrapped, integer_shifts = unwrap_reduced_path(raw)
    folded = (unwrapped + 0.5) % 1.0 - 0.5
    return {
        "reduced_polarization": raw.tolist(),
        "unwrapped_reduced_polarization": unwrapped.tolist(),
        "berry_integer_shifts": integer_shifts.tolist(),
        "folded_reduced_polarization": folded.tolist(),
    }


def _clean_atoms(atoms):
    """Copy only structure fields so source metadata cannot enter the model."""

    from ase import Atoms

    return Atoms(
        numbers=np.asarray(atoms.get_atomic_numbers(), dtype=int),
        positions=np.asarray(atoms.get_positions(), dtype=float),
        cell=np.asarray(atoms.get_cell(), dtype=float),
        pbc=np.asarray(atoms.get_pbc(), dtype=bool),
    )


def _symmetry_info(atoms, symprec):
    return analyse(atoms, symprec=float(symprec))


def _candidate_score(distortion, cell_distortion, residual, is_polar):
    # Give a nonpolar point group a modest preference, while making response
    # residual and endpoint distortion the dominant measurable quantities.
    return float(
        distortion
        + 0.25 * cell_distortion
        + 2.0 * residual
        + (0.5 if is_polar else 0.0)
    )


def _inversion_projected_geometries(
    atoms, *, max_centers=12, max_candidates=3
):
    """Project a polar geometry onto nearby inversion-symmetric geometries.

    Candidate inversion centers are inferred from same-species pair midpoints
    and a coarse fractional grid. For each center, same-species atoms are
    paired with a minimum-weight general matching, allowing atoms on special
    inversion sites as singletons. The returned structures contain no source
    metadata and preserve the original cell and composition.
    """

    import networkx as nx
    from ase import Atoms

    source = _clean_atoms(atoms)
    fractions = np.asarray(source.get_scaled_positions(wrap=True), dtype=float)
    numbers = np.asarray(source.get_atomic_numbers(), dtype=int)
    cell = np.asarray(source.get_cell(), dtype=float)
    center_keys = {tuple(np.round(center, 5)) for center in itertools.product(
        (0.0, 0.125, 0.25, 0.375), repeat=3
    )}
    species_groups = [np.flatnonzero(numbers == species) for species in np.unique(numbers)]
    if len(source) > 96:
        center_species_groups = [min(species_groups, key=len)]
    else:
        center_species_groups = species_groups
    for atom_indices in center_species_groups:
        positions = fractions[atom_indices]
        for first in range(len(positions)):
            for second in range(first, len(positions)):
                center = np.mod(0.5 * (positions[first] + positions[second]), 0.5)
                center_keys.add(tuple(np.round(center, 5)))
    centers = np.asarray(sorted(center_keys), dtype=float)

    # Rank centers with the sum of each atom's nearest same-species inversion
    # mate residual. This inexpensive relaxed assignment narrows the set that
    # receives an exact one-to-one matching below.
    center_scores = np.zeros(len(centers), dtype=float)
    for atom_indices in species_groups:
        positions = fractions[atom_indices]
        sample_indices = np.unique(
            np.linspace(
                0,
                len(positions) - 1,
                num=min(len(positions), 32),
                dtype=int,
            )
        )
        sample_positions = positions[sample_indices]
        for start in range(0, len(centers), 64):
            stop = min(len(centers), start + 64)
            trial_centers = centers[start:stop]
            reflected = (
                2.0 * trial_centers[:, None, :] - sample_positions[None, :, :]
            )
            delta = reflected[:, :, None, :] - positions[None, None, :, :]
            delta -= np.rint(delta)
            cartesian = delta @ cell
            distances = np.linalg.norm(cartesian, axis=-1)
            center_scores[start:stop] += np.sum(
                np.min(distances, axis=-1) ** 2, axis=1
            )
    selected_indices = np.argsort(center_scores)[: max(1, int(max_centers))]

    projected = []
    for center_index in selected_indices:
        center = centers[int(center_index)]
        result_fractions = fractions.copy()
        for species in np.unique(numbers):
            atom_indices = np.flatnonzero(numbers == species)
            positions = fractions[atom_indices]
            count = len(atom_indices)
            graph = nx.Graph()
            dummy_nodes = list(range(count, 2 * count))
            graph.add_nodes_from(range(2 * count))
            pair_error = positions[:, None, :] + positions[None, :, :] - 2.0 * center
            pair_error -= np.rint(pair_error)
            pair_cart = pair_error @ cell
            pair_cost = 0.5 * np.sum(pair_cart**2, axis=-1)
            self_error = 2.0 * (positions - center)
            self_error -= np.rint(self_error)
            self_cart = 0.5 * (self_error @ cell)
            self_cost = np.sum(self_cart**2, axis=1)
            for first in range(count):
                for second in range(first + 1, count):
                    graph.add_edge(
                        first,
                        second,
                        weight=-float(pair_cost[first, second]),
                    )
                for dummy in dummy_nodes:
                    graph.add_edge(
                        first, dummy, weight=-float(self_cost[first])
                    )
            for first in range(count, 2 * count):
                for second in range(first + 1, 2 * count):
                    graph.add_edge(first, second, weight=0.0)
            matching = nx.max_weight_matching(
                graph, maxcardinality=True, weight="weight"
            )
            if len(matching) != count:
                break
            for first_node, second_node in matching:
                if first_node >= count and second_node >= count:
                    continue
                if first_node >= count:
                    first_node, second_node = second_node, first_node
                first = int(first_node)
                if second_node >= count:
                    special_error = 2.0 * (positions[first] - center)
                    special_error -= np.rint(special_error)
                    special = positions[first] - 0.5 * special_error
                    result_fractions[atom_indices[first]] = special
                else:
                    second = int(second_node)
                    if first > second:
                        first, second = second, first
                    pair_error = positions[first] + positions[second] - 2.0 * center
                    pair_error -= np.rint(pair_error)
                    correction = 0.5 * pair_error
                    result_fractions[atom_indices[first]] = (
                        positions[first] - correction
                    )
                    result_fractions[atom_indices[second]] = (
                        positions[second] - correction
                    )
        candidate = Atoms(
            numbers=numbers,
            scaled_positions=result_fractions,
            cell=cell,
            pbc=True,
        )
        try:
            if any(
                periodic_rmsd(previous, candidate) <= 0.02
                and cell_rmsd(previous, candidate) <= 0.02
                for _score, previous in projected
            ):
                continue
            distortion = periodic_rmsd(source, candidate)
        except (ValueError, np.linalg.LinAlgError):
            continue
        projected.append((distortion, candidate))
    projected.sort(key=lambda item: item[0])
    return [atoms for _distortion, atoms in projected[: max(1, int(max_candidates))]]


def _make_candidate_record(
    model,
    polar,
    candidate,
    *,
    candidate_id,
    source,
    target,
    symprec,
    tolerance=None,
    response=None,
):
    candidate = _clean_atoms(candidate)
    symmetry = _symmetry_info(candidate, symprec)
    if response is None:
        energy, polarization, reduced = model.evaluate_energy_polarization(candidate)
    else:
        energy = float(response.energy)
        polarization = np.asarray(response.polarization, dtype=float)
        reduced = np.asarray(response.reduced_polarization, dtype=float)
    reduced = np.asarray(reduced, dtype=float).reshape(3)
    if target is None:
        targets, classes = nearest_formal_target(polarization, candidate.get_cell())
        residuals = np.linalg.norm(targets - reduced[None, :], axis=1)
        index = int(np.argmin(residuals))
        selected_target = np.asarray(targets[index], dtype=float)
        formal_class = np.asarray(classes[index], dtype=float)
    else:
        selected_target = np.asarray(target, dtype=float).reshape(3)
        aligned = branch_match(selected_target, reduced)
        residuals = np.asarray([np.linalg.norm(aligned - selected_target)])
        formal_class = np.mod(selected_target, 1.0)
        formal_class[np.isclose(formal_class, 1.0)] = 0.0
    aligned = branch_match(selected_target, reduced)
    residual = float(np.linalg.norm(aligned - selected_target))
    distortion = float(periodic_rmsd(polar, candidate))
    strain = float(cell_rmsd(polar, candidate))
    is_polar = bool(symmetry.get("is_polar_point_group", False))
    record = {
        "candidate_id": candidate_id,
        "source": source,
        "spacegroup": symmetry.get("international"),
        "spacegroup_number": int(symmetry.get("number", 0)),
        "pointgroup": symmetry.get("pointgroup"),
        "has_inversion": bool(symmetry.get("has_inversion", False)),
        "is_polar_point_group": is_polar,
        "nonpolar_parent_hypothesis": not is_polar,
        "symmetry_tolerance_A": None if tolerance is None else float(tolerance),
        "target_reduced_polarization": selected_target.tolist(),
        "formal_class": np.asarray(formal_class, dtype=float).tolist(),
        "target_residual": residual,
        "parent_symmetrization_rmsd_A": distortion,
        "parent_cell_rmsd_A": strain,
        "energy_eV": float(energy),
        "reduced_polarization": reduced.tolist(),
        "cartesian_polarization_e_A2": np.asarray(polarization, dtype=float).tolist(),
        "ranking_score": _candidate_score(distortion, strain, residual, is_polar),
    }
    return record, candidate


def _symmetry_projected_candidates(
    model,
    polar,
    source_atoms,
    config,
    *,
    target,
    candidate_prefix,
    source_label,
    levels,
    max_candidates=4,
):
    """Snap a source geometry to nearby nonpolar spglib symmetry refinements."""

    try:
        import spglib
        from ase import Atoms
    except ImportError:  # pragma: no cover
        return [], []

    source = _clean_atoms(source_atoms)
    source_cell = np.asarray(source.get_cell(), dtype=float)
    source_positions = np.asarray(source.get_scaled_positions(wrap=True), dtype=float)
    source_numbers = np.asarray(source.get_atomic_numbers(), dtype=int)
    tolerances = np.geomspace(
        max(float(config.symprec), 1.0e-3), 0.35, num=max(2, int(levels))
    )
    geometries = []
    for tolerance in tolerances:
        refined = spglib.refine_cell(
            (source_cell, source_positions, source_numbers),
            symprec=float(tolerance),
        )
        if refined is None:
            continue
        lattice, positions, numbers = refined
        numbers = np.asarray(numbers, dtype=int)
        if len(numbers) != len(source) or sorted(numbers.tolist()) != sorted(
            source_numbers.tolist()
        ):
            continue
        candidate = Atoms(
            numbers=numbers,
            scaled_positions=np.asarray(positions, dtype=float),
            cell=np.asarray(lattice, dtype=float),
            pbc=True,
        )
        tolerance_for_analysis = max(
            float(config.symprec), min(float(tolerance), 0.05)
        )
        symmetry = _symmetry_info(candidate, tolerance_for_analysis)
        if symmetry.get("is_polar_point_group", False):
            continue
        try:
            if any(
                periodic_rmsd(previous, candidate) <= 0.025
                and cell_rmsd(previous, candidate) <= 0.025
                for previous in geometries
            ):
                continue
        except (ValueError, np.linalg.LinAlgError):
            continue
        geometries.append(candidate)

    # Evaluate the nearest few symmetry projections. The tolerance ladder may
    # produce many identical refinements, so this cap also bounds MACE calls.
    geometries.sort(key=lambda atoms: periodic_rmsd(source, atoms))
    geometries = geometries[: max(1, int(max_candidates))]
    records = []
    structures = []
    for index, candidate in enumerate(geometries):
        record, candidate = _make_candidate_record(
            model,
            polar,
            candidate,
            candidate_id=f"{candidate_prefix}_{index:02d}",
            source=source_label,
            target=target,
            symprec=max(float(config.symprec), 0.01),
            response=None,
        )
        records.append(record)
        structures.append((record["candidate_id"], candidate))
    order = sorted(
        range(len(records)),
        key=lambda index: float(records[index]["ranking_score"]),
    )
    return [records[i] for i in order], [structures[i] for i in order]


def _bec_backprojection(
    model,
    atoms,
    target,
    config,
    *,
    max_iterations,
    max_step_A,
    variable_cell=False,
    max_strain_step=0.02,
    strain_response_refresh=4,
    cell_stretch_min=0.85,
    cell_stretch_max=1.15,
    cell_volume_min=0.70,
    cell_volume_max=1.30,
    initial_strain_jacobian=None,
    initial_response=None,
):
    """Use atomic and optional strain response for damped formal-P backprojection."""

    if max_step_A <= 0:
        raise ValueError("max_step_A must be positive")
    if variable_cell:
        if max_strain_step <= 0:
            raise ValueError("max_strain_step must be positive for variable-cell backprojection")
        if strain_response_refresh < 1:
            raise ValueError("strain_response_refresh must be positive")
        if cell_stretch_min <= 0 or cell_stretch_min >= cell_stretch_max:
            raise ValueError("cell principal-stretch bounds are invalid")
        if cell_volume_min <= 0 or cell_volume_min >= cell_volume_max:
            raise ValueError("cell volume-ratio bounds are invalid")
    current = _clean_atoms(atoms)
    reference_cell = np.asarray(current.get_cell(), dtype=float).copy()
    history = []
    response = None
    lattice = (
        np.zeros((3, 6), dtype=float)
        if initial_strain_jacobian is None
        else np.asarray(initial_strain_jacobian, dtype=float).reshape(3, 6)
    )
    last_lattice_refresh = 0 if initial_strain_jacobian is not None else None
    target = np.asarray(target, dtype=float).reshape(3)
    for iteration in range(int(max_iterations) + 1):
        if iteration == 0 and initial_response is not None:
            response = initial_response
        else:
            response = model.evaluate(
                current,
                include_response=True,
            )
        aligned = branch_match(target, response.reduced_polarization)
        residual = target - aligned
        residual_norm = float(np.linalg.norm(residual))
        history.append(
            {
                "iteration": iteration,
                "reduced_polarization": np.asarray(
                    response.reduced_polarization
                ).tolist(),
                "branch_aligned_reduced_polarization": aligned.tolist(),
                "target_residual": residual_norm,
                "energy_eV": float(response.energy),
            }
        )
        if residual_norm <= config.polarization_tolerance:
            break
        if iteration >= int(max_iterations):
            break
        lattice_refreshed = False
        if variable_cell and (
            last_lattice_refresh is None
            or iteration - last_lattice_refresh >= max(1, int(strain_response_refresh))
        ):
            lattice = lattice_reduced_jacobian(
                model,
                current,
                delta=config.strain_delta,
                reference_reduced_polarization=response.reduced_polarization,
            )
            last_lattice_refresh = iteration
            lattice_refreshed = True
        atomic = atomic_reduced_jacobian(response.becs, current.get_cell())
        jacobian = generalized_jacobian(atomic, lattice)
        displacement = minimum_metric_displacement(
            jacobian,
            residual,
            atomic_metric=config.atomic_metric,
            strain_metric=config.strain_metric,
        )
        if not np.all(np.isfinite(displacement)):
            break
        atomic_displacement = displacement[: 3 * len(current)].reshape(-1, 3)
        strain_displacement = displacement[3 * len(current) :]
        maximum = (
            float(np.max(np.linalg.norm(atomic_displacement, axis=1)))
            if len(current)
            else 0.0
        )
        maximum_strain_component = (
            float(np.max(np.abs(strain_displacement)))
            if strain_displacement.size
            else 0.0
        )
        if maximum <= 1.0e-12 and maximum_strain_component <= 1.0e-12:
            break
        scale = (
            min(1.0, float(max_step_A) / maximum)
            if maximum > 0.0
            else 1.0
        )
        if variable_cell and maximum_strain_component > 0.0:
            scale = min(scale, float(max_strain_step) / maximum_strain_component)
        scale = max(0.0, scale)
        trial = apply_generalized_displacement(current, displacement * scale)
        if variable_cell:
            while scale > 1.0e-7 and not cell_domain(
                reference_cell,
                trial.get_cell(),
                stretch_min=cell_stretch_min,
                stretch_max=cell_stretch_max,
                volume_min=cell_volume_min,
                volume_max=cell_volume_max,
            ):
                scale *= 0.5
                trial = apply_generalized_displacement(current, displacement * scale)
            if scale <= 1.0e-7:
                history[-1]["cell_bound_blocked"] = True
                history[-1]["strain_jacobian_refreshed"] = lattice_refreshed
                break
        current = trial
        history[-1]["max_atomic_step_A"] = maximum * scale
        history[-1]["max_log_strain_component"] = maximum_strain_component * scale
        history[-1]["step_scale"] = scale
        history[-1]["strain_jacobian_refreshed"] = lattice_refreshed
        history[-1]["strain_jacobian_frobenius_norm"] = float(
            np.linalg.norm(lattice)
        )
        deformation = deformation_from(reference_cell, current.get_cell())
        history[-1]["cell_principal_stretches"] = np.linalg.svd(
            deformation, compute_uv=False
        ).tolist()
        history[-1]["cell_volume_ratio"] = abs(float(np.linalg.det(deformation)))
        history[-1]["cell_rmsd_from_input_A"] = float(
            cell_rmsd(atoms, current)
        )
    return current, response, history


def _reflect_about_parent(parent, image):
    """Reflect one image around the parent, adapting to that image's cell."""

    from ase.geometry import find_mic

    result = _clean_atoms(image)
    parent_on_cell = _clean_atoms(parent)
    image_cell = np.asarray(image.get_cell(), dtype=float)
    parent_on_cell.set_cell(image_cell, scale_atoms=True)
    parent_positions = np.asarray(parent_on_cell.get_positions(), dtype=float)
    image_positions = np.asarray(image.get_positions(), dtype=float)
    displacement, _lengths = find_mic(
        image_positions - parent_positions, image_cell, pbc=True
    )
    result.set_positions(parent_positions - displacement)
    result.wrap()
    return result


def _prepare_branch_path(polar, parent, n_images=9):
    """Make the variable-cell geometric path used by parent reflection."""

    if n_images < 2:
        raise ValueError("a branch path needs at least two images")
    aligned = _clean_atoms(align_periodic(polar, parent))
    fractions0 = np.asarray(polar.get_scaled_positions(wrap=False), dtype=float)
    fractions1 = np.asarray(aligned.get_scaled_positions(wrap=False), dtype=float)
    displacement = fractions1 - fractions0
    displacement -= np.rint(displacement)
    cell0 = np.asarray(polar.get_cell(), dtype=float)
    cell1 = np.asarray(aligned.get_cell(), dtype=float)
    deformation = np.linalg.solve(cell0, cell1)
    left, _singular, right = np.linalg.svd(deformation)
    rotation = left @ right
    stretch = rotation.T @ deformation
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (stretch + stretch.T))
    log_strain = eigenvectors @ np.diag(np.log(np.maximum(eigenvalues, 1.0e-12))) @ eigenvectors.T

    parameters = np.linspace(0.0, 1.0, n_images)
    images = []
    distances = []
    for parameter in parameters:
        image = _clean_atoms(polar)
        cell = (
            cell1
            if parameter >= 1.0 - 1.0e-12
            else cell0 @ expm(parameter * log_strain)
        )
        image.set_cell(cell, scale_atoms=False)
        image.set_scaled_positions(fractions0 + parameter * displacement)
        image.wrap()
        images.append(image)
        pair_distances = np.asarray(image.get_all_distances(mic=True), dtype=float)
        np.fill_diagonal(pair_distances, np.inf)
        distances.append(float(np.min(pair_distances)) if pair_distances.size else float("inf"))
    diagnostics = {
        "collision_aware": False,
        "permutations_tested": 1,
        "selected_order": _species_order(polar, parent),
        "endpoint_rmsd_A": float(
            np.sqrt(np.mean(np.sum((displacement @ cell0) ** 2, axis=1)))
        ),
        "minimum_distance_A_by_image": distances,
        "minimum_distance_A": float(np.min(distances)),
        "minimum_distance_floor_A": 0.0,
        "geometry_valid": True,
        "parameters": parameters.tolist(),
        "cell_mode": "variable_cell",
        "cell_volume_A3_by_image": [float(image.get_volume()) for image in images],
        "cell_deformation_log_strain_norm": float(np.linalg.norm(log_strain)),
    }
    return images, diagnostics, aligned


def _species_order(reference, trial):
    """Record the species-aware assignment used by :func:`align_periodic`."""

    from .structures import _species_match

    return np.asarray(_species_match(reference, trial), dtype=int).tolist()


def _evaluate_geometry_path(model, images):
    energies = []
    reduced = []
    cartesian = []
    for image in images:
        energy, polarization, reduced_p = model.evaluate_energy_polarization(image)
        energies.append(float(energy))
        cartesian.append(np.asarray(polarization, dtype=float))
        reduced.append(np.asarray(reduced_p, dtype=float))
    return np.asarray(energies), np.asarray(cartesian), np.asarray(reduced)


def _construct_branch(
    model, polar, parent, orbit_id, candidate_id, target, *, num_images=17
):
    """Build a linear P→N path and its parent-reflected N→-P branch."""

    if num_images < 3 or num_images % 2 != 1:
        raise ValueError("num_images must be an odd integer of at least three")
    images_per_leg = num_images // 2 + 1
    incoming_images, geometry, aligned_parent = _prepare_branch_path(
        polar, parent, n_images=images_per_leg
    )
    incoming_energy, incoming_cartesian, incoming_raw = _evaluate_geometry_path(
        model, incoming_images
    )
    incoming_unwrapped, incoming_shifts = unwrap_reduced_path(incoming_raw)

    reflected_images = [
        _reflect_about_parent(aligned_parent, image)
        for image in reversed(incoming_images)
    ]
    reflected_energy, reflected_cartesian, reflected_raw = _evaluate_geometry_path(
        model, reflected_images
    )
    # Both traces must use one Berry gauge at the shared parent image.
    reflected_unwrapped, _reflected_shifts = unwrap_reduced_path(reflected_raw)
    reflected_gauge = np.rint(
        incoming_unwrapped[-1] - reflected_unwrapped[0]
    ).astype(int)
    reflected_unwrapped += reflected_gauge[None, :]
    combined_raw = np.concatenate((incoming_raw, reflected_raw[1:]), axis=0)
    combined = fold_reduced_branch(combined_raw)

    expected_reflected_cartesian = (
        2.0 * incoming_cartesian[-1] - incoming_cartesian[0]
    )
    aligned_reflected_target = cartesian_to_reduced(
        expected_reflected_cartesian, reflected_images[-1].get_cell()
    )
    reflected_endpoint_aligned = branch_match(
        aligned_reflected_target, reflected_unwrapped[-1]
    )
    result = {
        "orbit_id": int(orbit_id),
        "candidate_id": candidate_id,
        "formal_class": np.mod(np.asarray(target, dtype=float), 1.0).tolist(),
        "construction": "MACEField-BEC-parent plus variable-cell linear path and parent reflection",
        "geometry_diagnostics": geometry,
        "parent_reduced_polarization": incoming_raw[-1].tolist(),
        "incoming": {
            "reached": True,
            "construction": "variable_cell_linear_interpolation",
            "continuation_fractions": np.linspace(0.0, 1.0, len(incoming_images)).tolist(),
            "energies_eV": incoming_energy.tolist(),
            "cartesian_polarizations_e_A2": incoming_cartesian.tolist(),
            "raw_reduced_polarization": incoming_raw.tolist(),
            "unwrapped_reduced_polarization": incoming_unwrapped.tolist(),
            "berry_integer_shifts": incoming_shifts.tolist(),
            "folded_reduced_polarization": (
                (incoming_unwrapped + 0.5) % 1.0 - 0.5
            ).tolist(),
        },
        "reflected": {
            "reached": True,
            "construction": "reflect_displacements_about_reconstructed_parent",
            "energies_eV": reflected_energy.tolist(),
            "cartesian_polarizations_e_A2": reflected_cartesian.tolist(),
            "raw_reduced_polarization": reflected_raw.tolist(),
            "unwrapped_reduced_polarization": reflected_unwrapped.tolist(),
            "gauge_integer_shift": reflected_gauge.tolist(),
            "expected_opposite_endpoint_reduced_polarization": (
                aligned_reflected_target.tolist()
            ),
            "actual_opposite_endpoint_branch_residual": float(
                np.linalg.norm(
                    reflected_endpoint_aligned - aligned_reflected_target
                )
            ),
        },
        "pnp_branch": {
            "construction": "incoming_path_concatenated_with_parent_reflection",
            **combined,
        },
        "incoming_structure_ids": [],
        "reflected_structure_ids": [],
    }
    structures = []
    for index, image in enumerate(incoming_images):
        structure_id = f"incoming_{orbit_id:02d}_{index:03d}"
        result["incoming_structure_ids"].append(structure_id)
        structures.append((structure_id, _clean_atoms(image)))
    # The parent image is already the final incoming image.
    for index, image in enumerate(reflected_images[1:], start=1):
        structure_id = f"reflected_{orbit_id:02d}_{index:03d}"
        result["reflected_structure_ids"].append(structure_id)
        structures.append((structure_id, _clean_atoms(image)))
    return result, structures


def recover_parent_branches(
    model,
    polar_atoms,
    config: RecoveryConfig,
    *,
    max_formal_targets: int = 8,
    parent_symmetry_levels: int = 8,
    bec_max_iterations: int = 5,
    bec_max_step_A: float = 0.12,
    bec_variable_cell: bool = True,
    bec_max_strain_step: float = 0.02,
    bec_strain_response_refresh: int = 4,
    bec_cell_stretch_min: float = 0.85,
    bec_cell_stretch_max: float = 1.15,
    bec_cell_volume_min: float = 0.70,
    bec_cell_volume_max: float = 1.30,
    max_branch_candidates: int = 2,
    num_images: int = 17,
    max_inversion_centers: int = 12,
    max_inversion_candidates: int = 3,
    max_symmetry_candidates: int = 3,
) -> ParentRecoveryResult:
    """Infer parent candidates and polarization branches from a polar endpoint.

    The model sees only the polar endpoint. BEC back-projection proposes
    structures at inversion-invariant formal polarization classes, using the
    MACEField strain response by default; symmetry refinement supplies
    nonpolar parent hypotheses. Parent-reflected paths are sampled after
    ranking these hypotheses.
    """

    if int(num_images) != num_images or num_images < 3 or int(num_images) % 2 != 1:
        raise ValueError("num_images must be an odd integer of at least three")
    num_images = int(num_images)

    polar = _clean_atoms(polar_atoms)
    initial = model.evaluate(
        polar,
        include_response=True,
    )
    initial_reduced = np.asarray(initial.reduced_polarization, dtype=float)
    symmetry = _symmetry_info(polar, config.symprec)
    parent_operations = proper_rotation_operations(
        polar,
        symprec=config.symprec,
        reduced_polarization=initial_reduced,
        polarization_tolerance=max(
            config.polarization_distance_tolerance,
            config.polarization_tolerance,
        ),
    )
    target_orbits = formal_target_orbits(
        initial.polarization,
        polar.get_cell(),
        parent_operations,
        equivalence_tolerance=max(config.polarization_distance_tolerance, 1.0e-8),
    )[: max(0, int(max_formal_targets))]

    structures = [("polar_input", polar)]
    candidate_records = []
    candidate_atoms = {}
    used_ids = set()

    def add_candidate(record, atoms):
        identifier = str(record["candidate_id"])
        if identifier in used_ids:
            return
        used_ids.add(identifier)
        candidate_records.append(record)
        candidate_atoms[identifier] = _clean_atoms(atoms)

    # Search inversion centers directly from the endpoint's species/geometry.
    # This supplies a nonpolar-parent route even when a formal-polarization
    # projection alone leaves the endpoint's polar point group intact.
    inversion_geometries = _inversion_projected_geometries(
        polar,
        max_centers=max_inversion_centers,
        max_candidates=max_inversion_candidates,
    )
    for index, candidate in enumerate(inversion_geometries):
        record, candidate = _make_candidate_record(
            model,
            polar,
            candidate,
            candidate_id=f"inversion_parent_endpoint_{index:02d}",
            source="same_species_inversion_center_projection_from_endpoint",
            target=None,
            symprec=config.symprec,
        )
        add_candidate(record, candidate)

    # Fast symmetry snapping from the endpoint provides candidates even when
    # the polarization response is poorly conditioned.
    initial_targets, _initial_classes = nearest_formal_target(
        initial.polarization, polar.get_cell()
    )
    best_initial_index = int(
        np.argmin(np.linalg.norm(initial_targets - initial_reduced[None, :], axis=1))
    )
    initial_target = np.asarray(initial_targets[best_initial_index], dtype=float)
    initial_projection_records, initial_projection_structures = (
        _symmetry_projected_candidates(
            model,
            polar,
            polar,
            config,
            target=initial_target,
            candidate_prefix="symmetry_projection",
            source_label="spglib_refine_cell_from_polar_endpoint",
            levels=parent_symmetry_levels,
            max_candidates=max_symmetry_candidates,
        )
    )
    for record, (structure_id, candidate) in zip(
        initial_projection_records, initial_projection_structures, strict=True
    ):
        add_candidate(record, candidate)

    formal_branches = []
    initial_strain_jacobian = None
    if bec_variable_cell and target_orbits:
        initial_strain_jacobian = lattice_reduced_jacobian(
            model,
            polar,
            delta=config.strain_delta,
            reference_reduced_polarization=initial.reduced_polarization,
        )
    for orbit in target_orbits:
        target = np.asarray(orbit.representative, dtype=float)
        orbit_id = int(orbit.orbit_id)
        projected, response, backprojection_history = _bec_backprojection(
            model,
            polar,
            target,
            config,
            max_iterations=bec_max_iterations,
            max_step_A=bec_max_step_A,
            variable_cell=bec_variable_cell,
            max_strain_step=bec_max_strain_step,
            strain_response_refresh=bec_strain_response_refresh,
            cell_stretch_min=bec_cell_stretch_min,
            cell_stretch_max=bec_cell_stretch_max,
            cell_volume_min=bec_cell_volume_min,
            cell_volume_max=bec_cell_volume_max,
            initial_strain_jacobian=initial_strain_jacobian,
            initial_response=initial,
        )
        raw_id = f"bec_projection_{orbit_id:02d}"
        raw_record, raw_candidate = _make_candidate_record(
            model,
            polar,
            projected,
            candidate_id=raw_id,
            source="MACEField_BEC_backprojection",
            target=target,
            symprec=config.symprec,
            response=response,
        )
        add_candidate(raw_record, raw_candidate)

        if orbit_id == int(target_orbits[0].orbit_id):
            inversion_geometries = _inversion_projected_geometries(
                projected,
                max_centers=max(1, max_inversion_centers * 2 // 3),
                max_candidates=max(1, max_inversion_candidates - 1),
            )
            for index, candidate in enumerate(inversion_geometries):
                record, candidate = _make_candidate_record(
                    model,
                    polar,
                    candidate,
                    candidate_id=f"inversion_parent_bec_{orbit_id:02d}_{index:02d}",
                    source="same_species_inversion_center_projection_after_BEC",
                    target=target,
                    symprec=config.symprec,
                )
                add_candidate(record, candidate)

        snapped_records, snapped_structures = _symmetry_projected_candidates(
            model,
            polar,
            projected,
            config,
            target=target,
            candidate_prefix=f"symmetry_parent_{orbit_id:02d}",
            source_label="spglib_refine_cell_after_MACEField_BEC_backprojection",
            levels=parent_symmetry_levels,
            max_candidates=max_symmetry_candidates,
        )
        for record, (_structure_id, candidate) in zip(
            snapped_records, snapped_structures, strict=True
        ):
            add_candidate(record, candidate)

        # Re-rank every available geometry against this target. A candidate
        # generated for another formal class can still be a valid parent, but
        # its branch residual must be recomputed rather than reusing its old
        # score.
        options = []
        for item in candidate_records:
            residual = float(
                np.linalg.norm(
                    branch_match(target, item["reduced_polarization"]) - target
                )
            )
            is_polar = bool(item["is_polar_point_group"])
            score = _candidate_score(
                float(item["parent_symmetrization_rmsd_A"]),
                float(item["parent_cell_rmsd_A"]),
                residual,
                is_polar,
            )
            options.append((score, residual, item))
        options.sort(key=lambda item: (item[0], item[1]))
        if not options:
            parent_record = raw_record
        else:
            parent_record = dict(options[0][2])
            parent_record["target_residual"] = options[0][1]
            parent_record["target_reduced_polarization"] = target.tolist()
            parent_record["formal_class"] = np.mod(target, 1.0).tolist()
            parent_record["ranking_score"] = options[0][0]
        formal_branches.append(
            {
                "orbit_id": orbit_id,
                "formal_class": list(orbit.representative_formal_member),
                "formal_classes_in_orbit": [
                    list(item) for item in orbit.formal_members
                ],
                "target_reduced_polarization": target.tolist(),
                "bec_backprojection": {
                    "converged_to_target": bool(
                        backprojection_history
                        and backprojection_history[-1]["target_residual"]
                        <= config.polarization_tolerance
                    ),
                    "history": backprojection_history,
                    "iterations": max(
                        0, len(backprojection_history) - 1
                    ),
                },
                "parent": parent_record,
                "branch_status": "awaiting_branch_sampling",
            }
        )

    # A limited number of top-ranked parent hypotheses receive full reflected
    # paths, keeping the endpoint-only search inexpensive at dataset scale.
    ranked = sorted(
        candidate_records,
        key=lambda item: (
            float(item["ranking_score"]),
            float(item["target_residual"]),
        ),
    )
    branchable = []
    used_orbits = set()
    for record in ranked if int(max_branch_candidates) > 0 else []:
        candidate = candidate_atoms[record["candidate_id"]]
        if not formal_branches:
            continue
        selected_parent_branch = min(
            formal_branches,
            key=lambda item: np.linalg.norm(
                branch_match(
                    np.asarray(item["target_reduced_polarization"]),
                    np.asarray(record["reduced_polarization"]),
                )
                - np.asarray(item["target_reduced_polarization"])
            ),
        )
        selected_orbit_id = int(selected_parent_branch["orbit_id"])
        if selected_orbit_id in used_orbits:
            continue
        if any(
            periodic_rmsd(candidate, prior_atoms) <= 0.03
            and cell_rmsd(candidate, prior_atoms) <= 0.03
            for _prior, prior_atoms in branchable
        ):
            continue
        branchable.append((record, candidate))
        used_orbits.add(selected_orbit_id)
        if len(branchable) >= max(0, int(max_branch_candidates)):
            break

    branch_records = []
    for record, parent in branchable:
        parent_branch = next(
            (
                item
                for item in formal_branches
                if item["parent"]["candidate_id"] == record["candidate_id"]
            ),
            None,
        )
        if parent_branch is None:
            target = np.asarray(record["target_reduced_polarization"], dtype=float)
            parent_branch = min(
                formal_branches,
                key=lambda item: np.linalg.norm(
                    branch_match(
                        np.asarray(item["target_reduced_polarization"]),
                        np.asarray(record["reduced_polarization"]),
                    )
                    - np.asarray(item["target_reduced_polarization"])
                ),
            )
            orbit_id = int(parent_branch["orbit_id"])
            target = np.asarray(
                parent_branch["target_reduced_polarization"], dtype=float
            )
        else:
            target = np.asarray(
                parent_branch["target_reduced_polarization"], dtype=float
            )
            orbit_id = int(parent_branch["orbit_id"])
        branch, path_structures = _construct_branch(
            model,
            polar,
            parent,
            orbit_id,
            record["candidate_id"],
            target,
            num_images=num_images,
        )
        branch["parent"] = record
        branch_records.append(branch)
        structures.extend(path_structures)
        parent_branch["parent"] = record
        parent_branch["branch_status"] = "sampled"
        parent_branch["incoming"] = branch["incoming"]
        parent_branch["reflected"] = branch["reflected"]
        parent_branch["pnp_branch"] = branch["pnp_branch"]
        parent_branch["incoming_structure_ids"] = branch["incoming_structure_ids"]
        parent_branch["reflected_structure_ids"] = branch["reflected_structure_ids"]

    for record in candidate_records:
        structures.append((record["candidate_id"], candidate_atoms[record["candidate_id"]]))

    report = {
        "schema": (
            "ferrofinder.endpoint_parent_recovery.v4_variable_cell"
            if bec_variable_cell
            else "ferrofinder.endpoint_parent_recovery.v3"
        ),
        "status": "complete",
        "input_atom_count": len(polar),
        "input_formula": polar.get_chemical_formula(mode="hill"),
        "input_spacegroup": symmetry.get("international"),
        "input_pointgroup": symmetry.get("pointgroup"),
        "input_is_polar_point_group": bool(
            symmetry.get("is_polar_point_group", False)
        ),
        "input_energy_eV": float(initial.energy),
        "input_cartesian_polarization_e_A2": np.asarray(
            initial.polarization
        ).tolist(),
        "input_reduced_polarization": initial_reduced.tolist(),
        "input_formal_targets": [
            {
                "orbit_id": int(orbit.orbit_id),
                "representative_formal_class": list(
                    orbit.representative_formal_member
                ),
                "target_reduced_polarization": np.asarray(
                    orbit.representative, dtype=float
                ).tolist(),
            }
            for orbit in target_orbits
        ],
        "parent_candidates": candidate_records,
        "formal_branches": formal_branches,
        "sampled_polarization_branches": branch_records,
        "path_sampling": {
            "num_images": int(num_images),
            "images_per_leg_including_parent": int(num_images // 2 + 1),
            "construction": "linear polar-to-parent interpolation plus parent-reflected second leg",
        },
        "branch_folding": "nearest-quantum unwrap with per-image integer shifts, then Wigner-Seitz fold [-0.5, 0.5)",
        "bec_backprojection_settings": {
            "variable_cell": bool(bec_variable_cell),
            "max_strain_step_log_component": float(bec_max_strain_step),
            "strain_response_refresh_iterations": int(bec_strain_response_refresh),
            "cell_principal_stretch_bounds": [
                float(bec_cell_stretch_min),
                float(bec_cell_stretch_max),
            ],
            "cell_volume_ratio_bounds": [
                float(bec_cell_volume_min),
                float(bec_cell_volume_max),
            ],
            "strain_jacobian": "branch-matched finite differences of reduced polarization in six symmetric log-strain coordinates at fixed fractional positions",
        },
        "model_input_contract": "one polar endpoint structure; no reference parent or path frame",
    }
    return ParentRecoveryResult(report, structures)
