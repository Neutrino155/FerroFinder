# User guide

FerroFinder takes one or more periodic polar endpoints and a field-aware
MACE-Field checkpoint. For each endpoint it proposes nonpolar-parent
hypotheses, ranks them, constructs a sampled P → N → −P branch, and can
evaluate response properties on every image.

## Structure input

Pass a single structure file, a multi-frame XYZ/extended XYZ file, or a folder
of structure files. Every frame is treated as an independent polar endpoint.
Folders include CIF, MCIF, XYZ, and extended XYZ files; use `--recursive` for
nested folders.

- CIF supplies the unit cell and periodic boundaries.
- Extended XYZ should include its `Lattice` and periodic-boundary metadata.
- Plain XYZ contains positions only. Supply `--cell a,b,c,alpha,beta,gamma`;
  the same cell is assigned to every frame. Lengths are in Å and angles in
  degrees.
- Every endpoint must have a full periodic cell with nonzero volume.

Example for a CIF:

```bash
ferrofinder polar.cif \
  --model /models/MACEField.model \
  --head mp-ferroelectric \
  --output-dir results/polar
```

Example for a plain XYZ file:

```bash
ferrofinder polar.xyz \
  --cell 5.6,5.6,7.9,90,90,120 \
  --model /models/MACEField.model \
  --output-dir results/polar
```

For a folder of CIFs, run the batch once. A shared model is loaded by each
worker, and the output includes one combined index and explorer:

```bash
ferrofinder polar_endpoints/ \
  --model /models/MACEField.model \
  --output-dir results/screen \
  --workers 8 \
  --variable-cell \
  --num-images 17
```

A multi-frame extended XYZ file is processed directly in frame order:

```bash
ferrofinder endpoints.extxyz \
  --model /models/MACEField.model \
  --output-dir results/screen \
  --workers 8
```

## Options

Run `ferrofinder --help` for the current option list. The main options are:

- `--head`: MACE-Field checkpoint head; default `mp-ferroelectric`.
- `--device`: model device; default `cpu`.
- `--dtype`: `float64` by default, or `float32`.
- `--threads`: PyTorch thread count; default `1`.
- `--workers`: number of parallel material processes; default `1`. Each worker
  loads its own model. For CUDA, use one worker.
- `--num-images`: total points in each symmetric P → N → −P branch; default
  `17`. It must be odd and at least three. The two legs share the parent image.
- `--compute-becs` / `--no-compute-becs`: evaluate or skip Born effective
  charges at every branch image; enabled by default.
- `--compute-polarizability` / `--no-compute-polarizability`: evaluate or
  skip polarizability at every branch image; enabled by default. MACE-Field's
  polarizability calculation also requires response/BEC evaluation.
- `--variable-cell` / `--fixed-cell`: enable or disable the six-component
  strain-response block during inverse-distortion parent recovery. Variable
  cell is the default.
- `--relax`: before parent recovery, apply a deterministic Gaussian rattle
  (0.01 Å standard deviation by default) and relax each input with ASE BFGS.
  This uses `FrechetCellFilter` for a full cell and position relaxation when
  variable-cell mode is enabled, or relaxes positions only in fixed-cell mode.
  The default force target is 0.05 eV/Å with up to 500 steps. Tune these with
  `--relax-rattle-stdev`, `--relax-fmax`, and `--relax-max-steps`.
- `--symprec`: spglib symmetry tolerance in Å; default `0.01`.
- `--polarization-tolerance` and `--polarization-distance-tolerance`: control
  polarization convergence and formal-target equivalence.
- `--strain-delta`, `--atomic-metric`, `--strain-metric`,
  `--bec-max-iterations`, `--bec-max-step-a`, `--bec-max-strain-step`, and
  `--bec-cell-{stretch,volume}-bounds`: configure inverse-distortion finite
  differences, step sizes, and allowed lattice changes.
- `--max-formal-targets`, `--parent-symmetry-levels`,
  `--max-inversion-centers`, `--max-inversion-candidates`,
  `--max-symmetry-candidates`, and `--max-branch-candidates`: limit search
  breadth and the number of paths constructed.
- `--resume`: continue a matching run. It checks input geometry, checkpoint,
  and search settings before reusing completed results.
- A relaxed endpoint is saved as `relaxed_input.extxyz` in the corresponding
  material directory. Its report records the rattle seed, optimizer settings,
  convergence, energy change, and relaxed cell.

An existing output directory is rejected unless `--resume` is used. A changed
input set, checkpoint, or setting requires a new output directory. The 0.25 Å
and 0.50 Å structural match thresholds from the MP-Ferroelectric validation
report are not search cutoffs: without a known parent, FerroFinder reports
ranked hypotheses rather than accepting or rejecting one by reference RMSD.

## Output files

- `report.json` contains model inputs, polarization targets, ranked parent
  candidates, back-projection history, and sampled branch data.
- `structures.extxyz` contains the endpoint, parent candidates, and sampled
  path images, each labeled by `ferrofinder_structure_id`.
- `path_properties_branch_*.json` contains per-image energies, polarization,
  BECs, and polarizability for each sampled branch.
- For a multi-structure run, `materials/<query_id>/` contains those per-input
  artifacts; `run_manifest.json` records settings, completion state, and
  input provenance.
- The output-root `index.html` lists inputs and links to branches. The shared
  `explorer/` app switches between all branch payloads and includes one data
  JSON per branch.

Serve the output directory with `python -m http.server 8000`, then open
`http://localhost:8000/` for the results table or
`http://localhost:8000/explorer/` for the branch explorer.

In the viewer, choose a material, drag to rotate, use the wheel to zoom, scrub
or play the branch, select an atom to inspect its BEC, and optionally show
bonds, cell edges, atom labels, or the motion guide. The report includes cell
geometry, distortion, polarization quanta, Cartesian/reduced polarization,
BECs, and polarizability when requested. Download a selected frame as extended
XYZ.

## Interpretation

Candidate ranking reflects model polarization, geometric distortion, cell
change, and symmetry. A nonpolar point group is a useful candidate feature;
it is not proof of a stable nonpolar ground state. Validate candidates with
appropriate structural relaxation and higher-level calculations. The sampled
branch is a geometric reconstruction and must not be read as an activation
barrier or minimum-energy path.
