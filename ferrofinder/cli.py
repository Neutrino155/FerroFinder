"""Command line interface for endpoint-only ferroelectric parent recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _cell_parameters(value: str) -> np.ndarray:
    """Parse ``a,b,c,alpha,beta,gamma`` in angstroms and degrees."""

    try:
        values = np.asarray([float(part) for part in value.split(",")])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--cell must be six comma-separated values: a,b,c,alpha,beta,gamma"
        ) from exc
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError(
            "--cell must be six finite comma-separated values: "
            "a,b,c,alpha,beta,gamma"
        )
    if np.any(values[:3] <= 0) or np.any(values[3:] <= 0) or np.any(values[3:] >= 180):
        raise argparse.ArgumentTypeError("cell lengths must be positive and angles in (0, 180)")
    return values


def build_parser() -> argparse.ArgumentParser:
    """Create the public single-structure and batch recovery CLI."""

    parser = argparse.ArgumentParser(
        prog="ferrofinder",
        description=(
            "Infer nonpolar-parent candidates and sampled P-N--P branches from "
            "one periodic polar endpoint, a multi-frame structure file, or a folder."
        ),
        epilog=(
            "CIF and extended XYZ can carry periodic cells. Plain XYZ has no cell; "
            "supply --cell a,b,c,alpha,beta,gamma. A multi-frame XYZ is processed "
            "as independent polar endpoints."
        ),
    )
    parser.add_argument(
        "structure",
        help="periodic polar CIF/XYZ/extended XYZ file or a directory of structures",
    )
    parser.add_argument("--model", required=True, type=Path, help="field-aware MACE-Field checkpoint")
    parser.add_argument("--head", default="mp-ferroelectric", help="checkpoint head")
    parser.add_argument("--mace-checkout", type=Path, help="optional MACE-Field source checkout")
    parser.add_argument("--device", default="cpu", help="MACE device, for example cpu or cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (default: 1)")
    parser.add_argument("--workers", type=int, default=1, help="parallel material workers (CPU only)")
    parser.add_argument("--recursive", action="store_true", help="search input folders recursively")
    parser.add_argument(
        "--cell",
        type=_cell_parameters,
        help="assign this common cell to XYZ frames: a,b,c,alpha,beta,gamma (Å, degrees)",
    )
    parser.add_argument("--output-dir", type=Path, help="output directory (default: sibling <input>_ferrofinder)")
    parser.add_argument("--resume", action="store_true", help="resume matching completed material results")
    parser.add_argument(
        "--relax",
        action="store_true",
        help="rattle slightly and relax each input structure before recovery",
    )
    parser.add_argument(
        "--relax-rattle-stdev",
        type=float,
        default=0.01,
        help="Cartesian Gaussian rattle standard deviation in Å before relaxation (default: 0.01)",
    )
    parser.add_argument(
        "--relax-fmax",
        type=float,
        default=0.05,
        help="ASE BFGS force convergence target in eV/Å (default: 0.05)",
    )
    parser.add_argument(
        "--relax-max-steps",
        type=int,
        default=500,
        help="maximum ASE BFGS steps for each input relaxation (default: 500)",
    )
    parser.add_argument("--symprec", type=float, default=0.01, help="spglib symmetry tolerance (Å)")
    parser.add_argument("--polarization-tolerance", type=float, default=0.001, help="BEC back-projection convergence tolerance")
    parser.add_argument("--polarization-distance-tolerance", type=float, default=0.08, help="formal-polarization orbit equivalence tolerance")
    parser.add_argument("--strain-delta", type=float, default=5.0e-4, help="finite-difference log-strain step")
    parser.add_argument("--atomic-metric", type=float, default=1.0, help="atomic-displacement metric weight")
    parser.add_argument("--strain-metric", type=float, default=25.0, help="cell-strain metric weight")
    parser.add_argument("--max-formal-targets", type=int, default=8, help="maximum formal polarization targets")
    parser.add_argument("--parent-symmetry-levels", type=int, default=8, help="symmetry refinement levels for parent proposals")
    parser.add_argument("--max-branch-candidates", type=int, default=1, help="maximum ranked parent proposals to turn into paths")
    parser.add_argument("--num-images", type=int, default=17, help="total P-N--P images; must be odd and at least 3")
    parser.add_argument("--bec-max-iterations", type=int, default=5)
    parser.add_argument("--bec-max-step-a", type=float, default=0.12, help="maximum atomic inverse-distortion step (Å)")
    parser.add_argument("--bec-max-strain-step", type=float, default=0.02, help="maximum inverse log-strain step")
    parser.add_argument("--bec-strain-response-refresh", type=int, default=4)
    parser.add_argument("--bec-cell-stretch-bounds", type=float, nargs=2, metavar=("MIN", "MAX"), default=(0.85, 1.15))
    parser.add_argument("--bec-cell-volume-bounds", type=float, nargs=2, metavar=("MIN", "MAX"), default=(0.70, 1.30))
    parser.add_argument("--max-inversion-centers", type=int, default=12)
    parser.add_argument("--max-inversion-candidates", type=int, default=3)
    parser.add_argument("--max-symmetry-candidates", type=int, default=3)
    parser.add_argument(
        "--variable-cell",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include six-component MACE-Field strain response (default: enabled)",
    )
    parser.add_argument("--fixed-cell", dest="variable_cell", action="store_false", help="alias for --no-variable-cell")
    parser.add_argument("--compute-becs", action=argparse.BooleanOptionalAction, default=True, help="evaluate BECs for every sampled branch image")
    parser.add_argument("--compute-polarizability", action=argparse.BooleanOptionalAction, default=True, help="evaluate polarizability for every sampled branch image")
    return parser


def run(args: argparse.Namespace) -> Path:
    """Run endpoint recovery for one or many independent polar structures."""

    from .batch import run_batch

    return run_batch(args)


def main(argv=None) -> int:
    """Parse arguments, run the model, and report the result directory."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = run(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    print(
        f"FerroFinder {manifest['status']}: {manifest['input_count']} input structure(s), "
        f"{manifest['completed_count']} completed, {manifest['error_count']} errors, "
        f"{manifest['no_branch_count']} without a sampled branch, "
        f"{manifest.get('relaxation_not_converged_count', 0)} initial relaxations not converged"
    )
    print(f"  index: {output / 'index.html'}")
    print(f"  explorer: {output / 'explorer' / 'index.html'}")
    print(f"  manifest: {output / 'run_manifest.json'}")
    return 0 if manifest["status"] == "complete" else 2
