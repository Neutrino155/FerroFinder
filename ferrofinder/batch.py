"""Batch endpoint recovery and all-image response enrichment."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_WORKER_MODEL = None
_WORKER_SETTINGS = None


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atoms_sha256(atoms) -> str:
    digest = hashlib.sha256()
    for value in (
        np.asarray(atoms.get_atomic_numbers(), dtype="<i8"),
        np.asarray(atoms.get_positions(), dtype="<f8"),
        np.asarray(atoms.get_cell(), dtype="<f8"),
        np.asarray(atoms.get_pbc(), dtype=np.uint8),
    ):
        digest.update(np.asarray(value).tobytes(order="C"))
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "material"


def _discover_files(source: Path, recursive: bool) -> list[Path]:
    supported = {".cif", ".mcif", ".xyz", ".extxyz"}
    if source.is_file():
        files = [source]
    elif source.is_dir():
        iterator = source.rglob("*") if recursive else source.iterdir()
        files = [path for path in iterator if path.is_file()]
    else:
        raise FileNotFoundError(f"input path not found: {source}")
    files = sorted(
        (path.resolve() for path in files if path.suffix.lower() in supported),
        key=lambda item: item.as_posix().lower(),
    )
    if not files:
        raise ValueError(
            f"no CIF, MCIF, XYZ, or extended XYZ structures found in {source}"
        )
    unsupported = source.is_file() and source.suffix.lower() not in supported
    if unsupported:
        raise ValueError(f"unsupported structure extension: {source.suffix}")
    return files


def _collect_inputs(source: Path, recursive: bool, cell):
    from ase.geometry import cellpar_to_cell
    from ase.io import iread

    from .structures import clean_copy, require_periodic

    files = _discover_files(source, recursive)
    entries = []
    query_index = 0
    for path in files:
        try:
            frames = list(iread(str(path), index=":"))
            if not frames:
                raise ValueError("file contains no structures")
        except Exception as exc:
            query_id = f"query_{query_index:04d}"
            query_index += 1
            entries.append(
                {
                    "query_id": query_id,
                    "display_name": path.name,
                    "source": str(path),
                    "frame_index": None,
                    "input_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        try:
            source_label = path.relative_to(source).as_posix() if source.is_dir() else path.name
        except ValueError:
            source_label = path.name
        for frame_index, atoms in enumerate(frames):
            query_id = f"query_{query_index:04d}"
            query_index += 1
            display_name = source_label
            if len(frames) > 1:
                display_name = f"{source_label} · frame {frame_index + 1}"
            entry = {
                "query_id": query_id,
                "display_name": display_name,
                "source": str(path),
                "frame_index": int(frame_index),
            }
            try:
                if cell is not None:
                    atoms.set_cell(cellpar_to_cell(cell), scale_atoms=False)
                    atoms.set_pbc(True)
                require_periodic(atoms)
                clean = clean_copy(atoms)
                entry["formula"] = clean.get_chemical_formula(mode="hill")
                entry["atom_count"] = int(len(clean))
                entry["atoms"] = clean
                entry["input_sha256"] = _atoms_sha256(clean)
            except Exception as exc:
                entry["input_error"] = f"{type(exc).__name__}: {exc}"
            entries.append(entry)
    if not entries:
        raise ValueError("input files contained no structure frames")

    return entries


def _initialize_worker(model_path, head, device, dtype, mace_checkout, threads):
    global _WORKER_MODEL
    import torch
    from threadpoolctl import threadpool_limits

    from .model import MACEFieldModel

    torch.set_num_threads(threads)
    threadpool_limits(limits=threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    _WORKER_MODEL = MACEFieldModel(
        model_path,
        head=head,
        device=device,
        dtype=dtype,
        mace_checkout=mace_checkout,
    )
    _WORKER_MODEL.make_calculator()


def _relax_initial_structure(atoms, *, variable_cell, relaxation, input_sha256):
    """Rattle and relax one polar endpoint with the worker's MACE-Field model."""

    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS

    from .structures import clean_copy

    work = clean_copy(atoms)
    original_cell = np.asarray(work.get_cell(), dtype=float).copy()
    seed = int(input_sha256[:8], 16)
    work.rattle(stdev=relaxation["rattle_stdev_A"], seed=seed)
    # MACE-Field's ASE calculator accepts the per-structure field from atoms.info.
    work.info = {"electric_field": [0.0, 0.0, 0.0]}
    work.calc = _WORKER_MODEL.make_calculator()
    rattled_energy = float(work.get_potential_energy())
    target = FrechetCellFilter(work) if variable_cell else work
    optimizer = BFGS(target, logfile=None)
    converged = bool(
        optimizer.run(
            fmax=relaxation["fmax_eV_per_A"],
            steps=relaxation["max_steps"],
        )
    )
    final_energy = float(work.get_potential_energy())
    final_forces = np.asarray(work.get_forces(), dtype=float)
    max_atomic_force = (
        float(np.max(np.linalg.norm(final_forces, axis=1)))
        if len(final_forces)
        else 0.0
    )
    relaxed_hash = _atoms_sha256(work)
    final_cell = np.asarray(work.get_cell(), dtype=float).copy()
    work.calc = None
    return work, {
        "enabled": True,
        "optimizer": "ASE BFGS",
        "cell_mode": "variable" if variable_cell else "fixed",
        "cell_filter": "FrechetCellFilter" if variable_cell else None,
        "rattle_distribution": "independent Gaussian Cartesian displacements",
        "rattle_stdev_A": float(relaxation["rattle_stdev_A"]),
        "rattle_seed": seed,
        "electric_field_V_per_A": [0.0, 0.0, 0.0],
        "fmax_eV_per_A": float(relaxation["fmax_eV_per_A"]),
        "max_steps": int(relaxation["max_steps"]),
        "steps_taken": int(optimizer.nsteps),
        "converged": converged,
        "energy_after_rattle_before_relax_eV": rattled_energy,
        "energy_after_relax_eV": final_energy,
        "max_atomic_force_after_relax_eV_per_A": max_atomic_force,
        "input_cell_A": original_cell.tolist(),
        "relaxed_cell_A": final_cell.tolist(),
        "relaxed_endpoint_sha256": relaxed_hash,
    }


def _evaluate_branch(model, query_id, branch_index, branch, by_id, settings):
    from .polarization import branch_match, reduced_to_cartesian

    structure_ids = [
        *branch["incoming_structure_ids"],
        *branch["reflected_structure_ids"],
    ]
    structures = [by_id[structure_id] for structure_id in structure_ids]
    stored_energies = [
        *branch["incoming"]["energies_eV"],
        *branch["reflected"]["energies_eV"][1:],
    ]
    references = np.asarray(
        branch["pnp_branch"]["unwrapped_reduced_polarization"], dtype=float
    )
    if not (
        len(structures) == len(stored_energies) == len(references)
    ):
        raise ValueError(f"{query_id}: P-N-P branch arrays do not align")

    compute_becs = settings["compute_becs"]
    compute_polarizability = settings["compute_polarizability"]
    evaluate_response = compute_becs or compute_polarizability
    frames = []
    for image_index, (structure_id, atoms, stored_energy, reference) in enumerate(
        zip(structure_ids, structures, stored_energies, references, strict=True)
    ):
        evaluation = None
        if evaluate_response:
            evaluation = model.evaluate(
                atoms,
                include_response=True,
                include_polarizability=compute_polarizability,
            )
            reduced = branch_match(reference, evaluation.reduced_polarization)
            energy = float(evaluation.energy)
        else:
            reduced = np.asarray(reference, dtype=float)
            energy = float(stored_energy)
        frame = {
            "image_index": image_index,
            "structure_id": structure_id,
            "energy_eV": energy,
            "stored_path_energy_eV": float(stored_energy),
            "reduced_polarization": np.asarray(reduced, dtype=float).tolist(),
            "polarization_e_A2": reduced_to_cartesian(reduced, atoms.get_cell()).tolist(),
        }
        if compute_becs and evaluation is not None:
            frame["becs_e"] = np.asarray(evaluation.becs, dtype=float).tolist()
        if compute_polarizability and evaluation is not None:
            frame["polarizability_e_per_V_A"] = np.asarray(
                evaluation.polarizability, dtype=float
            ).tolist()
        frames.append(frame)
    return structures, frames


def _run_material(task: dict) -> dict:
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        raise RuntimeError("worker model was not initialized")
    from ase.io import read, write

    from .config import RecoveryConfig
    from .parent_recovery import recover_parent_branches
    from .visualization.recovery import payload_from_path

    started = time.monotonic()
    material_dir = Path(task["material_dir"])
    output_root = Path(task["output_root"])
    atoms = read(task["input_path"])
    relaxation_result = {"enabled": False}
    relaxed_input_file = None
    if task["relaxation"]["enabled"]:
        atoms, relaxation_result = _relax_initial_structure(
            atoms,
            variable_cell=task["variable_cell"],
            relaxation=task["relaxation"],
            input_sha256=task["input_sha256"],
        )
        relaxed_input = material_dir / "relaxed_input.extxyz"
        write(relaxed_input, atoms, format="extxyz")
        relaxed_input_file = str(relaxed_input.relative_to(output_root))
        relaxation_result["output_file"] = relaxed_input_file
    before = _WORKER_MODEL.evaluation_statistics()
    config = RecoveryConfig(**task["recovery_config"])
    result = recover_parent_branches(
        _WORKER_MODEL,
        atoms,
        config,
        **task["recovery_options"],
    )
    after_recovery = _WORKER_MODEL.evaluation_statistics()
    recovery_delta = {
        key: int(after_recovery.get(key, 0)) - int(before.get(key, 0))
        for key in after_recovery
    }
    report = dict(result.report)
    report["input_source"] = {
        "query_id": task["query_id"],
        "display_name": task["display_name"],
        "source": task["source"],
        "frame_index": task["frame_index"],
        "input_sha256": task["input_sha256"],
        "relaxed_input_file": relaxed_input_file,
    }
    report["run"] = {
        "model_file": Path(task["model_path"]).name,
        "model_sha256": task["model_sha256"],
        "model_head": task["head"],
        "device": task["device"],
        "dtype": task["dtype"],
        "num_images": task["recovery_options"]["num_images"],
        "compute_becs_for_every_image": task["settings"]["compute_becs"],
        "compute_polarizability_for_every_image": task["settings"]["compute_polarizability"],
        "initial_structure_relaxation": relaxation_result,
        "model_evaluations_for_recovery": recovery_delta,
    }
    _write_json(material_dir / "report.json", report)
    by_id = {}
    stored_structures = []
    for structure_id, structure in result.structures:
        frame = structure.copy()
        frame.calc = None
        frame.info = {"ferrofinder_structure_id": str(structure_id)}
        stored_structures.append(frame)
        by_id[str(structure_id)] = structure
    write(material_dir / "structures.extxyz", stored_structures, format="extxyz")

    payload_files = []
    branch_rows = []
    for branch_index, branch in enumerate(
        report.get("sampled_polarization_branches", []), start=1
    ):
        structures, frames = _evaluate_branch(
            _WORKER_MODEL,
            task["query_id"],
            branch_index,
            branch,
            by_id,
            task["settings"],
        )
        branch_id = f"{task['query_id']}_branch_{branch_index:02d}"
        property_path = material_dir / f"path_properties_branch_{branch_index:02d}.json"
        _write_json(
            property_path,
            {
                "schema": "ferrofinder.reconstructed_branch_response.v1",
                "status": "complete",
                "query_id": task["query_id"],
                "branch_id": branch_id,
                "formula": report["input_formula"],
                "image_count": len(frames),
                "computed": task["settings"],
                "units": {
                    "energy": "eV",
                    "polarization": "e/angstrom^2",
                    "reduced_polarization": "polarization quanta",
                    "born_effective_charges": "e",
                    "polarizability": "e/(V angstrom)",
                },
                "frames": frames,
            },
        )
        payload = payload_from_path(
            query_id=branch_id,
            formula=report["input_formula"],
            path={
                "energies": [frame["energy_eV"] for frame in frames],
                "unwrapped_reduced_polarizations": [
                    frame["reduced_polarization"] for frame in frames
                ],
            },
            structures=structures,
            parent=branch.get("parent"),
            frame_properties=frames,
        )
        payload["name"] = f"{task['display_name']} · branch {branch_index}"
        payload_path = material_dir / f"payload_branch_{branch_index:02d}.json"
        _write_json(payload_path, payload)
        payload_files.append(str(payload_path.relative_to(output_root)))
        branch_rows.append(
            {
                "branch_id": branch_id,
                "branch_index": branch_index,
                "candidate_id": branch.get("candidate_id"),
                "image_count": len(frames),
                "payload_file": str(payload_path.relative_to(output_root)),
                "properties_file": str(property_path.relative_to(output_root)),
            }
        )

    after = _WORKER_MODEL.evaluation_statistics()
    model_delta = {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in after}
    elapsed = time.monotonic() - started
    report["run"]["model_evaluations_for_recovery_and_path_properties"] = model_delta
    report["run"]["elapsed_seconds"] = elapsed
    _write_json(material_dir / "report.json", report)
    status = "complete" if branch_rows else "no_branch"
    row = {
        "query_id": task["query_id"],
        "display_name": task["display_name"],
        "formula": report["input_formula"],
        "atom_count": int(report["input_atom_count"]),
        "status": status,
        "parent_candidate_count": len(report.get("parent_candidates", [])),
        "branch_count": len(branch_rows),
        "elapsed_seconds": elapsed,
        "material_dir": str(material_dir.relative_to(output_root)),
        "relaxed_input_file": relaxed_input_file,
        "relaxation_converged": relaxation_result.get("converged"),
        "branches": branch_rows,
    }
    marker = {
        "status": status,
        "input_sha256": task["input_sha256"],
        "configuration_sha256": task["configuration_sha256"],
        "model_sha256": task["model_sha256"],
        "relaxed_input_file": relaxed_input_file,
        "row": row,
        "payload_files": payload_files,
    }
    _write_json(material_dir / "complete.json", marker)
    return {"row": row, "payload_files": payload_files}


def _write_dashboard(path: Path, rows: list[dict], settings: dict):
    table_rows = []
    for row in rows:
        branch_links = " · ".join(
            f'<a href="explorer/index.html?query={html.escape(branch["branch_id"], quote=True)}">branch {branch["branch_index"]}</a>'
            for branch in row.get("branches", [])
        )
        report_path = Path(row.get("material_dir", ".")) / "report.json"
        report_href = html.escape(report_path.as_posix(), quote=True)
        relaxed_path = row.get("relaxed_input_file")
        relaxed_href = (
            f'<a href="{html.escape(relaxed_path, quote=True)}">download</a>'
            if relaxed_path
            else "—"
        )
        if not relaxed_path:
            relaxation_status = "not requested"
        elif row.get("relaxation_converged") is True:
            relaxation_status = "converged"
        else:
            relaxation_status = "not converged"
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row.get('display_name', row['query_id']))}</td>"
            f"<td>{html.escape(row.get('formula', ''))}</td>"
            f"<td>{row.get('atom_count', '—')}</td>"
            f"<td>{row.get('parent_candidate_count', 0)}</td>"
            f"<td>{row.get('branch_count', 0)}</td>"
            f"<td>{html.escape(row.get('status', 'unknown'))}</td>"
            f"<td>{branch_links or '—'}</td>"
            f"<td><a href=\"{report_href}\">report</a></td>"
            f"<td>{relaxation_status}</td>"
            f"<td>{relaxed_href}</td>"
            "</tr>"
        )
    html_text = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FerroFinder batch recovery</title>
<style>body{{font:15px system-ui,sans-serif;margin:2rem auto;max-width:1500px;padding:0 1rem;color:#193346}}h1{{margin-bottom:.3rem}}.muted{{color:#627989}}.summary{{padding:1rem;background:#eff5f7;border-radius:8px;margin:1rem 0}}input{{padding:.6rem;width:min(100%,32rem);border:1px solid #9badb8;border-radius:5px}}table{{border-collapse:collapse;width:100%;margin-top:1rem}}th,td{{border-bottom:1px solid #d7e0e5;padding:.55rem;text-align:left;vertical-align:top}}th{{background:#f4f7f8;position:sticky;top:0}}a{{color:#126e75}}</style>
<h1>FerroFinder recovery results</h1>
<p class="muted">Endpoint-only parent hypotheses and sampled P → N → −P branches. Paths are geometric reconstructions, not relaxed barriers.</p>
<div class="summary">Materials: {len(rows)} · variable cell: {settings['variable_cell']} · initial relaxation: {settings['initial_structure_relaxation']['enabled']} · images per branch: {settings['num_images']} · BECs per image: {settings['compute_becs']} · polarizability per image: {settings['compute_polarizability']}</div>
<label>Filter materials <input id="filter" type="search" placeholder="formula, filename, status"></label>
<table><thead><tr><th>Input</th><th>Formula</th><th>Atoms</th><th>Parent proposals</th><th>Branches</th><th>Status</th><th>Explore</th><th>Report</th><th>Initial relaxation</th><th>Relaxed input</th></tr></thead><tbody>{''.join(table_rows)}</table>
<script>document.getElementById('filter').addEventListener('input',event=>{{const q=event.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(row=>row.hidden=!row.textContent.toLowerCase().includes(q));}});</script>
</html>"""
    path.write_text(html_text, encoding="utf-8")


def _write_run_manifest(path: Path, data: dict):
    _write_json(path, data)


def run_batch(args) -> Path:
    """Run recovery, path response evaluation, and shared explorer generation."""

    from .config import RecoveryConfig
    from .visualization.recovery import write_explorer_bundle

    source = Path(args.structure).expanduser().resolve()
    if args.output_dir is not None:
        output_root = args.output_dir.expanduser().resolve()
    else:
        output_root = source.with_name(f"{source.stem if source.is_file() else source.name}_ferrofinder")
    if not source.is_dir() and source.suffix.lower() not in {".cif", ".mcif", ".xyz", ".extxyz"}:
        raise ValueError(f"unsupported structure extension: {source.suffix}")
    if args.num_images < 3 or args.num_images % 2 != 1:
        raise ValueError("--num-images must be an odd integer of at least three")
    if args.workers < 1 or args.threads < 1:
        raise ValueError("--workers and --threads must be positive")
    if args.relax:
        if (
            not np.isfinite(args.relax_rattle_stdev)
            or args.relax_rattle_stdev <= 0
        ):
            raise ValueError(
                "--relax-rattle-stdev must be finite and positive when --relax is enabled"
            )
        if not np.isfinite(args.relax_fmax) or args.relax_fmax <= 0:
            raise ValueError("--relax-fmax must be finite and positive")
        if args.relax_max_steps < 1:
            raise ValueError("--relax-max-steps must be positive")
    if args.device.startswith("cuda") and args.workers > 1:
        raise ValueError("use --workers 1 for CUDA; each worker loads its own model")
    if args.max_branch_candidates < 0:
        raise ValueError("--max-branch-candidates cannot be negative")
    if args.max_formal_targets < 0 or args.parent_symmetry_levels < 0:
        raise ValueError("formal-target and symmetry-level limits cannot be negative")
    if args.bec_max_iterations < 0 or args.bec_strain_response_refresh < 1:
        raise ValueError("BEC iteration count must be nonnegative and refresh interval positive")
    if args.bec_max_step_a <= 0 or args.bec_max_strain_step <= 0:
        raise ValueError("BEC atomic and strain step limits must be positive")
    stretch_min, stretch_max = args.bec_cell_stretch_bounds
    volume_min, volume_max = args.bec_cell_volume_bounds
    if stretch_min <= 0 or stretch_max < stretch_min:
        raise ValueError("cell stretch bounds must be positive and ordered")
    if volume_min <= 0 or volume_max < volume_min:
        raise ValueError("cell volume bounds must be positive and ordered")
    if min(
        args.max_inversion_centers,
        args.max_inversion_candidates,
        args.max_symmetry_candidates,
    ) < 1:
        raise ValueError("inversion and symmetry candidate limits must be positive")
    if args.compute_polarizability and not args.compute_becs:
        raise ValueError(
            "MACE-Field polarizability evaluation also computes BECs; enable --compute-becs"
        )
    if output_root.exists():
        if not output_root.is_dir():
            raise FileExistsError(f"output path is not a directory: {output_root}")
        if any(output_root.iterdir()) and not args.resume:
            raise FileExistsError(
                f"output directory is not empty; choose a new --output-dir or use --resume: {output_root}"
            )
    if source.is_dir() and (output_root == source or source in output_root.parents):
        raise ValueError("output directory must be outside the input folder")
    args.output_dir = output_root
    entries = _collect_inputs(source, args.recursive, args.cell)
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MACE-Field checkpoint not found: {model_path}")
    model_sha = _sha256(model_path)
    recovery_config = {
        "symprec": args.symprec,
        "polarization_distance_tolerance": args.polarization_distance_tolerance,
        "polarization_tolerance": args.polarization_tolerance,
        "strain_delta": args.strain_delta,
        "atomic_metric": args.atomic_metric,
        "strain_metric": args.strain_metric,
    }
    RecoveryConfig(**recovery_config).validate()
    recovery_options = {
        "max_formal_targets": args.max_formal_targets,
        "parent_symmetry_levels": args.parent_symmetry_levels,
        "bec_max_iterations": args.bec_max_iterations,
        "bec_max_step_A": args.bec_max_step_a,
        "bec_variable_cell": args.variable_cell,
        "bec_max_strain_step": args.bec_max_strain_step,
        "bec_strain_response_refresh": args.bec_strain_response_refresh,
        "bec_cell_stretch_min": args.bec_cell_stretch_bounds[0],
        "bec_cell_stretch_max": args.bec_cell_stretch_bounds[1],
        "bec_cell_volume_min": args.bec_cell_volume_bounds[0],
        "bec_cell_volume_max": args.bec_cell_volume_bounds[1],
        "max_branch_candidates": args.max_branch_candidates,
        "num_images": args.num_images,
        "max_inversion_centers": args.max_inversion_centers,
        "max_inversion_candidates": args.max_inversion_candidates,
        "max_symmetry_candidates": args.max_symmetry_candidates,
    }
    settings = {
        "variable_cell": bool(args.variable_cell),
        "num_images": int(args.num_images),
        "compute_becs": bool(args.compute_becs),
        "compute_polarizability": bool(args.compute_polarizability),
        "max_branch_candidates": int(args.max_branch_candidates),
        "recovery_config": recovery_config,
        "recovery_options": recovery_options,
        "head": args.head,
        "device": args.device,
        "dtype": args.dtype,
        "threads": args.threads,
        "initial_structure_relaxation": {
            "enabled": bool(args.relax),
            "rattle_stdev_A": float(args.relax_rattle_stdev),
            "fmax_eV_per_A": float(args.relax_fmax),
            "max_steps": int(args.relax_max_steps),
            "optimizer": "ASE BFGS",
            "cell_filter": (
                "FrechetCellFilter"
                if args.relax and args.variable_cell
                else None
            ),
        },
    }
    configuration_settings = dict(settings)
    if not args.relax:
        # Preserve resume compatibility with runs made before optional
        # preprocessing relaxation was introduced.
        configuration_settings.pop("initial_structure_relaxation")
    configuration_sha = hashlib.sha256(
        json.dumps(configuration_settings, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for entry in entries:
        if "input_sha256" not in entry:
            entry["input_sha256"] = ""
    identity = {
        "model_sha256": model_sha,
        "configuration_sha256": configuration_sha,
        "inputs": [
            {
                key: entry.get(key)
                for key in (
                    "query_id", "display_name", "source", "frame_index",
                    "formula", "atom_count", "input_sha256", "input_error",
                )
            }
            for entry in entries
        ],
    }
    run_manifest_path = output_root / "run_manifest.json"
    if args.resume:
        if not run_manifest_path.is_file():
            raise FileExistsError("--resume requires a prior run_manifest.json")
        previous = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if previous.get("identity") != identity:
            raise ValueError("resume inputs, model, or settings differ from the saved run")
    from ase.io import write

    for entry in entries:
        atoms = entry.pop("atoms", None)
        if atoms is None:
            continue
        input_path = output_root / "inputs" / f"{entry['query_id']}.extxyz"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        write(input_path, atoms, format="extxyz")
        entry["input_file"] = str(input_path.relative_to(output_root))
    run_data = {
        "schema": "ferrofinder.batch_recovery.v1",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "head": args.head,
        "identity": identity,
        "settings": settings,
        "input_count": len(entries),
        "completed_count": 0,
        "error_count": 0,
        "no_branch_count": 0,
        "relaxation_not_converged_count": 0,
        "materials": [],
    }
    _write_run_manifest(run_manifest_path, run_data)

    multi_input = len(entries) > 1
    model_kwargs = {
        "model_path": str(model_path),
        "head": args.head,
        "device": args.device,
        "dtype": args.dtype,
        "mace_checkout": str(args.mace_checkout.expanduser().resolve())
        if args.mace_checkout
        else None,
        "threads": int(args.threads),
    }
    tasks = []
    rows_by_id = {}
    payloads = {}
    for entry in entries:
        material_dir = (
            output_root / "materials" / entry["query_id"] if multi_input else output_root
        )
        material_dir.mkdir(parents=True, exist_ok=True)
        marker_path = material_dir / "complete.json"
        if args.resume and marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            relaxed_input_file = marker.get("relaxed_input_file")
            relaxed_input_available = (
                not args.relax
                or (
                    relaxed_input_file is not None
                    and (output_root / relaxed_input_file).is_file()
                )
            )
            if (
                marker.get("input_sha256") == entry.get("input_sha256")
                and marker.get("configuration_sha256") == configuration_sha
                and marker.get("model_sha256") == model_sha
                and marker.get("status") in {"complete", "no_branch"}
                and relaxed_input_available
            ):
                rows_by_id[entry["query_id"]] = marker["row"]
                for relative in marker.get("payload_files", []):
                    payload = json.loads((output_root / relative).read_text(encoding="utf-8"))
                    payloads[payload["query_id"]] = payload
                continue
        if entry.get("input_error"):
            row = {
                "query_id": entry["query_id"],
                "display_name": entry["display_name"],
                "formula": entry.get("formula", ""),
                "atom_count": entry.get("atom_count", 0),
                "status": "error",
                "error": entry["input_error"],
                "parent_candidate_count": 0,
                "branch_count": 0,
                "branches": [],
            }
            rows_by_id[entry["query_id"]] = row
            continue
        tasks.append(
            {
                **model_kwargs,
                "query_id": entry["query_id"],
                "display_name": entry["display_name"],
                "source": entry["source"],
                "frame_index": entry["frame_index"],
                "input_sha256": entry["input_sha256"],
                "input_path": str(output_root / entry["input_file"]),
                "material_dir": str(material_dir),
                "output_root": str(output_root),
                "model_sha256": model_sha,
                "configuration_sha256": configuration_sha,
                "settings": {
                    "compute_becs": settings["compute_becs"],
                    "compute_polarizability": settings["compute_polarizability"],
                },
                "variable_cell": settings["variable_cell"],
                "relaxation": settings["initial_structure_relaxation"],
                "recovery_config": recovery_config,
                "recovery_options": recovery_options,
            }
        )

    outcomes = []
    if tasks:
        if args.workers == 1:
            _initialize_worker(**model_kwargs)
            for task in tasks:
                try:
                    outcomes.append(_run_material(task))
                except Exception as exc:  # preserve per-material failure information
                    outcomes.append(
                        {
                            "row": {
                                "query_id": task["query_id"],
                                "display_name": task["display_name"],
                                "formula": "",
                                "atom_count": 0,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "parent_candidate_count": 0,
                                "branch_count": 0,
                                "branches": [],
                            },
                            "payload_files": [],
                        }
                    )
                row = outcomes[-1]["row"]
                rows_by_id[row["query_id"]] = row
                for relative in outcomes[-1].get("payload_files", []):
                    payload = json.loads((output_root / relative).read_text(encoding="utf-8"))
                    payloads[payload["query_id"]] = payload
                run_data["materials"] = [rows_by_id[key] for key in sorted(rows_by_id)]
                run_data["completed_count"] = sum(
                    item["status"] in {"complete", "no_branch"}
                    for item in rows_by_id.values()
                )
                run_data["error_count"] = sum(item["status"] == "error" for item in rows_by_id.values())
                run_data["no_branch_count"] = sum(item["status"] == "no_branch" for item in rows_by_id.values())
                run_data["relaxation_not_converged_count"] = sum(
                    item.get("relaxation_converged") is False
                    for item in rows_by_id.values()
                )
                _write_run_manifest(run_manifest_path, run_data)
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_initialize_worker,
                initargs=(
                    model_kwargs["model_path"], model_kwargs["head"],
                    model_kwargs["device"], model_kwargs["dtype"],
                    model_kwargs["mace_checkout"], model_kwargs["threads"],
                ),
            ) as pool:
                futures = {pool.submit(_run_material, task): task for task in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        outcome = {
                            "row": {
                                "query_id": task["query_id"],
                                "display_name": task["display_name"],
                                "formula": "",
                                "atom_count": 0,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "parent_candidate_count": 0,
                                "branch_count": 0,
                                "branches": [],
                            },
                            "payload_files": [],
                        }
                    outcomes.append(outcome)
                    row = outcome["row"]
                    rows_by_id[row["query_id"]] = row
                    for relative in outcome.get("payload_files", []):
                        payload = json.loads((output_root / relative).read_text(encoding="utf-8"))
                        payloads[payload["query_id"]] = payload
                    run_data["materials"] = [rows_by_id[key] for key in sorted(rows_by_id)]
                    run_data["completed_count"] = sum(
                        item["status"] in {"complete", "no_branch"}
                        for item in rows_by_id.values()
                    )
                    run_data["error_count"] = sum(item["status"] == "error" for item in rows_by_id.values())
                    run_data["no_branch_count"] = sum(item["status"] == "no_branch" for item in rows_by_id.values())
                    run_data["relaxation_not_converged_count"] = sum(
                        item.get("relaxation_converged") is False
                        for item in rows_by_id.values()
                    )
                    _write_run_manifest(run_manifest_path, run_data)

    rows = [rows_by_id[key] for key in sorted(rows_by_id)]
    write_explorer_bundle(output_root / "explorer", payloads)
    _write_dashboard(output_root / "index.html", rows, settings)
    run_data["materials"] = rows
    run_data["completed_count"] = sum(
        item["status"] in {"complete", "no_branch"} for item in rows
    )
    run_data["error_count"] = sum(item["status"] == "error" for item in rows)
    run_data["no_branch_count"] = sum(item["status"] == "no_branch" for item in rows)
    run_data["relaxation_not_converged_count"] = sum(
        item.get("relaxation_converged") is False for item in rows
    )
    run_data["status"] = (
        "complete"
        if run_data["error_count"] == 0
        and run_data["no_branch_count"] == 0
        and run_data["relaxation_not_converged_count"] == 0
        else "partial"
    )
    run_data["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write_run_manifest(run_manifest_path, run_data)
    return output_root
