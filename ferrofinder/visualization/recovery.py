"""Compact, offline browser viewer for reconstructed polarization branches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def payload_from_path(
    *,
    query_id: str,
    formula: str,
    path: dict,
    structures: list,
    parent: dict | None = None,
    validation: dict | None = None,
    frame_properties: list[dict] | None = None,
) -> dict:
    """Convert one sampled branch and its structures into browser data."""

    energies = np.asarray(path["energies"], dtype=float).reshape(-1)
    polarization_values = path.get("unwrapped_reduced_polarizations")
    if polarization_values is None:
        polarization_values = path["reduced_polarizations"]
    polarization = np.asarray(polarization_values, dtype=float).reshape(-1, 3)
    if len(structures) != len(energies) or len(energies) != len(polarization):
        raise ValueError("branch structures, energies, and polarizations must align")
    if frame_properties is not None and len(frame_properties) != len(structures):
        raise ValueError("per-image properties must align with branch structures")

    frames = []
    for index, (atoms, energy, reduced) in enumerate(
        zip(structures, energies, polarization, strict=True)
    ):
        frame = {
                "cell": np.asarray(atoms.get_cell(), dtype=float).round(8).tolist(),
                "fractional_positions": np.asarray(
                    atoms.get_scaled_positions(wrap=True), dtype=float
                ).round(8).tolist(),
                "symbols": atoms.get_chemical_symbols(),
                "energy_eV": float(energy),
                "reduced_polarization": reduced.round(8).tolist(),
                "coordinate": float(np.linspace(1.0, -1.0, len(energies))[index]),
            }
        if frame_properties is not None:
            properties = frame_properties[index]
            frame["energy_eV"] = float(properties["energy_eV"])
            frame["reduced_polarization"] = np.asarray(
                properties["reduced_polarization"], dtype=float
            ).round(8).tolist()
            for key, digits in (
                ("becs_e", 5),
                ("polarizability_e_per_V_A", 6),
            ):
                if properties.get(key) is not None:
                    frame[key] = np.asarray(properties[key], dtype=float).round(
                        digits
                    ).tolist()
            if properties.get("structure_id") is not None:
                frame["structure_id"] = str(properties["structure_id"])
        frames.append(frame)

    validation = validation or {}
    validation_payload = {}
    for output_key, input_key in (
        ("polar_mpid", "polar_mpid"),
        ("nonpolar_mpid", "nonpolar_mpid"),
        ("rank1_hit", "parent_top1_within_0p25A"),
        ("rank1_rmsd_A", "parent_top1_rmsd_A"),
        ("rank1_cell_rmsd_A", "parent_top1_cell_rmsd_A"),
        ("best_rmsd_A", "parent_best_candidate_rmsd_A"),
        ("path_dtw_rmsd_A", "path_dtw_mean_rmsd_A"),
        ("branch_rmse", "branch_reduced_rmse_mod_quantum"),
    ):
        if validation.get(input_key) is not None:
            validation_payload[output_key] = validation[input_key]
    return {
        "query_id": str(query_id),
        "formula": str(formula),
        "atom_count": len(structures[0]) if structures else 0,
        "parent": {
            "candidate_id": (parent or {}).get("candidate_id"),
            "source": (parent or {}).get("source"),
            "pointgroup": (parent or {}).get("pointgroup"),
        },
        "validation": validation_payload,
        "frames": frames,
    }


def payload_from_result(result, validation: dict | None = None) -> dict:
    """Build viewer data from a :class:`ParentRecoveryResult`."""

    report = result.report
    structures = dict(result.structures)
    branches = report.get("sampled_polarization_branches", [])
    if not branches:
        raise ValueError("recovery result has no sampled polarization branch")
    branch = branches[0]
    branch_structures = [
        structures[key]
        for key in (
            *branch["incoming_structure_ids"],
            *branch["reflected_structure_ids"],
        )
    ]
    energy = [
        *branch["incoming"]["energies_eV"],
        *branch["reflected"]["energies_eV"][1:],
    ]
    path = {
        "energies": energy,
        "unwrapped_reduced_polarizations": branch["pnp_branch"][
            "unwrapped_reduced_polarization"
        ],
    }
    return payload_from_path(
        query_id="recovery",
        formula=report["input_formula"],
        path=path,
        structures=branch_structures,
        parent=branch.get("parent"),
        validation=validation,
    )


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FerroFinder branch explorer</title><style>
:root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#182c3c;background:#f1f5f7;font-synthesis:none}*{box-sizing:border-box}body{margin:0}header{padding:20px clamp(14px,3vw,38px);background:#173348;color:#f4f8fa}header h1{margin:3px 0;font-size:clamp(20px,3vw,30px)}header p{margin:5px 0 0;color:#c6d6df;font-size:13px}.eyebrow{color:#8bdfc9;font-size:10px;font-weight:800;letter-spacing:.14em}.wrap{max-width:1500px;margin:auto;padding:14px}.toolbar,.summary,.panel{background:#fff;border:1px solid #d9e3e9;border-radius:11px;box-shadow:0 3px 14px #18364b0c}.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:9px 16px;margin-bottom:12px;padding:9px 12px}.toolbar label{display:flex;align-items:center;gap:6px;color:#4d6374;font-size:12px;font-weight:650}.toolbar input[type=range]{width:115px;margin:0;accent-color:#238178}.toolbar output{min-width:36px;color:#496274;font-variant-numeric:tabular-nums}.toolbar button,.toolbar select{padding:6px 9px;border:1px solid #cbd8e0;border-radius:7px;background:white;color:#20394b;font:inherit;font-size:12px}.toolbar button{cursor:pointer}.summary{display:flex;flex-wrap:wrap;gap:8px 20px;margin-bottom:12px;padding:10px 13px}.summary div{display:grid;gap:2px}.summary strong{font-size:13px;color:#183c50}.summary span{font-size:9px;color:#718494;text-transform:uppercase;letter-spacing:.06em}.grid{display:grid;grid-template-columns:minmax(360px,1fr) minmax(340px,1fr);gap:12px}.panel{padding:12px;min-width:0}.panel h2{margin:0 0 5px;color:#27495d;font-size:15px}.hint{margin:0 0 7px;color:#768895;font-size:11px}.chart{display:block;width:100%;height:205px;touch-action:none;cursor:crosshair}.stage{position:relative;height:min(56vh,520px);min-height:340px;border-radius:8px;background:radial-gradient(ellipse at 50% 40%,#fff 0,#f7fafb 68%,#eff4f6 100%);overflow:hidden}.stage canvas{display:block;width:100%;height:100%;touch-action:none;cursor:grab}.stage canvas:active{cursor:grabbing}.viewer-caption{position:absolute;left:12px;bottom:10px;color:#526a7b;font-size:11px;pointer-events:none}.legend{display:flex;gap:5px;flex-wrap:wrap;justify-content:flex-end}.legend span{padding:3px 6px;border-radius:5px;background:#314757;color:white;font-size:10px;font-weight:700}.subline{margin:8px 1px 0;color:#536b7d;font-size:12px;font-variant-numeric:tabular-nums}.error{padding:24px;color:#9b263d}.muted{color:#748796}@media(max-width:850px){.grid{grid-template-columns:1fr}.stage{height:440px}}@media(max-width:520px){.wrap{padding:8px}.panel{padding:9px}.summary{gap:7px 13px}.stage{height:390px;min-height:300px}.chart{height:185px;touch-action:none}}
</style><style>
.structure-header{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}.structure-tools{display:flex;align-items:flex-end;gap:8px;flex-direction:column}.view-buttons{display:flex;gap:3px;padding:3px;border:1px solid #d4dfe6;border-radius:8px;background:#f5f8fa}.view-buttons button{min-width:34px;padding:5px 8px;border:0;border-radius:5px;background:transparent;color:#385468;font:inherit;font-size:11px;font-weight:700;cursor:pointer}.view-buttons button[aria-pressed="true"]{background:#1d6570;color:#fff;box-shadow:0 1px 3px #18364b33}.view-buttons button:focus-visible{outline:2px solid #e39b35;outline-offset:2px}@media(max-width:520px){.structure-header{flex-wrap:wrap}.structure-tools{flex-direction:row;align-items:center;flex-wrap:wrap}}
</style></head><body>
<header><div class="eyebrow">FERROFINDER · ENDPOINT-ONLY RECONSTRUCTION</div><h1 id="title">Polarization branch explorer</h1><p>Sampled P → N → −P geometric reconstruction. It is not a relaxed switching path or an activation barrier.</p></header>
<main class="wrap"><section class="toolbar" aria-label="Structure viewer controls">
<label id="material-selector" hidden>Material <select id="material"></select></label>
<label>Frame <input id="frame" type="range" min="0" max="1" value="0"><output id="frame-label">1 / 1</output></label>
<button id="play" type="button">Play</button><button id="reset" type="button">Reset view</button><button id="download" type="button">Download frame</button>
<label><input id="cell" type="checkbox" checked> cell</label><label><input id="bonds" type="checkbox"> bonds</label><label><input id="motion" type="checkbox" checked> motion guide</label><label><input id="labels" type="checkbox"> labels</label>
<label>Atom size <input id="atom-size" type="range" min="0.5" max="3" step="0.05" value="2"><output id="atom-size-value">2.00×</output></label>
</section>
<section class="summary" id="summary"></section>
<section class="grid"><article class="panel"><h2>Sampled branch</h2><p class="hint">Drag or click either graph to move through the images. The path is geometric, not a relaxed barrier.</p><canvas class="chart" id="energy" aria-label="Energy along branch"></canvas><canvas class="chart" id="polarization" aria-label="Cartesian polarization in microcoulombs per square centimeter"></canvas></article>
<article class="panel"><div class="structure-header"><div><h2>Structure</h2><p class="hint" id="structure-hint">Drag to rotate · Shift-drag to pan · scroll to zoom · click an atom to inspect its Born effective charge.</p></div><div class="structure-tools"><div class="view-buttons" role="group" aria-label="Structure projection"><button type="button" data-structure-view="3d" aria-pressed="true" title="Perspective 3D view">3D</button><button type="button" data-structure-view="x" aria-pressed="false" title="Look along x, showing the y-z projection">x</button><button type="button" data-structure-view="y" aria-pressed="false" title="Look along y, showing the x-z projection">y</button><button type="button" data-structure-view="z" aria-pressed="false" title="Look along z, showing the x-y projection">z</button></div><div class="legend" id="legend"></div></div></div><div class="stage"><canvas id="structure"></canvas><div class="viewer-caption" id="caption"></div></div><div class="subline" id="metrics"></div></article></section>
<section class="panel report" id="details" aria-live="polite"></section><div id="chart-tooltip" role="status"></div></main>
<style>
.report{margin-top:12px}.report-head{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}.report h3{margin:12px 0 6px;color:#27495d;font-size:13px}.report-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.report-block{min-width:0}.report table{border-collapse:collapse;font-size:11px;width:100%;font-variant-numeric:tabular-nums}.report th,.report td{border:1px solid #dce5ea;padding:4px 6px;text-align:right}.report th:first-child,.report td:first-child{text-align:left}.report th{background:#f3f7f9;color:#506677}.report details{margin-top:10px;max-width:100%;overflow-x:auto}.report summary{cursor:pointer;color:#355b70;font-size:12px;font-weight:650}.atom-select{padding:6px 9px;border:1px solid #cbd8e0;border-radius:7px;background:white;color:#20394b;font:inherit;font-size:12px}.pol-vector{font-variant-numeric:tabular-nums;color:#244d61}.barrier-note{color:#738696;font-size:11px}.sample-tooltip{position:fixed;z-index:10;pointer-events:none;background:#153347;color:#f5f8fa;border:1px solid #688294;border-radius:7px;padding:8px 10px;white-space:pre-line;box-shadow:0 4px 18px #132e4055;font:11px/1.5 ui-monospace,monospace;max-width:330px}#chart-tooltip{display:none}@media(max-width:900px){.report-grid{grid-template-columns:1fr 1fr}}@media(max-width:600px){.report-grid{grid-template-columns:1fr}}
</style>
<script src="explorer.js"></script></body></html>'''


_JS = Path(__file__).with_name("recovery_explorer.js").read_text(encoding="utf-8")


def write_explorer_bundle(output_dir: str | Path, payloads: dict[str, dict]) -> dict:
    """Write one shared explorer app and one small data file per material."""

    root = Path(output_dir)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    queries = []
    for query_id, payload in sorted(payloads.items()):
        safe_id = str(query_id)
        target = data_dir / f"{safe_id}.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            .replace("<", "\\u003c")
            + "\n",
            encoding="utf-8",
        )
        queries.append(
            {
                "query_id": safe_id,
                "formula": payload.get("formula", ""),
                "name": payload.get("name", safe_id),
            }
        )
    (root / "manifest.json").write_text(
        json.dumps({"schema": "ferrofinder.recovery_explorer.v1", "queries": queries}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (root / "index.html").write_text(_HTML, encoding="utf-8")
    (root / "explorer.js").write_text(_JS, encoding="utf-8")
    return {"query_count": len(queries), "query_ids": [item["query_id"] for item in queries]}
