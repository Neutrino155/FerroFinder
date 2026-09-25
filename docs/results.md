# Result files

A successful single-structure run writes:

```text
OUTPUT_DIR/
├── report.json
├── structures.extxyz
├── relaxed_input.extxyz       # only when --relax is enabled
├── path_properties_branch_01.json
├── run_manifest.json
├── index.html
└── explorer/
    ├── index.html
    ├── explorer.js
    ├── manifest.json
    └── data/
        └── query_0000_branch_01.json
```

`report.json` contains input and model metadata, formal polarization targets,
candidate rankings, BEC back-projection histories, and sampled branch values.
`structures.extxyz` stores the polar endpoint, parent candidates, and each
sampled path frame with a `ferrofinder_structure_id` label. The branch response
file stores per-image energy and polarization and, when enabled, every atom's
BEC and the polarizability tensor. The browser explorer reads compact JSON and
does not load MACE-Field.

A multi-structure run uses the same root-level index, manifest, and explorer,
and adds one material directory per input:

```text
OUTPUT_DIR/
├── index.html
├── run_manifest.json
├── inputs/
├── materials/
│   ├── query_0000/
│   │   ├── report.json
│   │   ├── structures.extxyz
│   │   ├── relaxed_input.extxyz  # only when --relax is enabled
│   │   └── path_properties_branch_01.json
│   └── ...
└── explorer/
    ├── index.html
    ├── explorer.js
    ├── manifest.json
    └── data/
```

The root index lists inputs and links to their branches. The shared explorer
lets you switch between materials and branch candidates. `run_manifest.json`
records the checkpoint, settings, source frames, completion status, and
per-material outcomes. Failed or unbranchable inputs remain visible there and
in the index.

Serve the output directory locally so the index and explorer can fetch their
data:

```bash
cd OUTPUT_DIR
python -m http.server 8000
```

Open `http://localhost:8000/` for the input table and
`http://localhost:8000/explorer/` for the branch viewer. The viewer can export
the displayed frame as extended XYZ. The saved structures file remains the
complete candidate and branch record.

The model evaluation is not a structural relaxation. Candidate point-group
symmetry, energy, and polarization are model predictions. The generated path
is a geometric interpolation and reflection, not an optimized switching path
or a barrier calculation.
