# Python API

The command line interface is the shortest way to run FerroFinder. The same
endpoint-only method is available from Python:

```python
from ase.io import read

from ferrofinder import MACEFieldModel, RecoveryConfig, recover_parent_branches

polar = read("polar.cif")
model = MACEFieldModel("/models/MACEField.model", head="mp-ferroelectric")
model.make_calculator()
result = recover_parent_branches(
    model,
    polar,
    RecoveryConfig(),
    max_branch_candidates=1,
    num_images=17,
)

print(result.report["input_formula"])
print(result.report["sampled_polarization_branches"][0]["parent"]["candidate_id"])
```

`RecoveryConfig` contains the symmetry and polarization tolerances and the
atomic/strain metric weights. Variable-cell inverse distortion is enabled by
default; pass `bec_variable_cell=False` for a fixed-cell comparison. Its strain
bounds and candidate counts are keyword arguments to `recover_parent_branches`.
`num_images` is the total number of points on the symmetric P → N → −P
branch; it must be odd and at least three. The command line exposes the same
search bounds, plus per-image BEC and polarizability evaluation.

For plain XYZ input, set the unit cell and periodic boundaries before calling
the API. For CIF and extended XYZ, ASE reads the cell and periodic metadata.
`result.structures` is a list of `(structure_id, ase.Atoms)` pairs, and
`result.report` is JSON-ready scientific output. Use the CLI if you also want
standardized `report.json`, `structures.extxyz`, and an interactive explorer.
